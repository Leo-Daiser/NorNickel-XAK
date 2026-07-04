"""Controlled batch ingestion for large local corpora.

This script is intentionally API-based: it exercises the same ingestion path as
the Streamlit UI, but sends one file per request and records resumable state.
It does not use LLM extraction and does not require embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

os.environ.setdefault("RUNTIME_PROFILE", "economy_core")
os.environ.setdefault("ENABLE_LLM", "false")
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("ENABLE_LOCAL_EMBEDDINGS", "false")
os.environ.setdefault("RETRIEVAL_MODE", "bm25")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parsing.file_profile import ARCHIVE_EXTENSIONS, IMAGE_EXTENSIONS, LEGACY_OFFICE_EXTENSIONS  # noqa: E402
from app.parsing.text_quality import SUPPORTED_FILE_EXTENSIONS  # noqa: E402


DEFAULT_API_BASE = os.getenv("API_BASE", "http://localhost:8000")
DEFAULT_STATE_PATH = ROOT / "artifacts" / "batch_ingest_state.json"
DEFAULT_REPORT_PATH = ROOT / "artifacts" / "batch_ingest_report.json"


@dataclass(frozen=True)
class PlannedFile:
    path: Path
    relative_path: str
    extension: str
    file_size_bytes: int
    file_size_mb: float
    source_group: str | None
    planned_status: str
    planned_reason: str

    def fingerprint(self) -> str:
        stat = self.path.stat()
        raw = f"{self.relative_path}|{self.file_size_bytes}|{int(stat.st_mtime)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def discover_files(root: str | Path, *, max_files: int | None = None, sample_per_group: int | None = None) -> tuple[list[Path], dict[str, Any]]:
    root_path = Path(root)
    if root_path.is_file():
        return [root_path], {"total_files_found": 1, "selected_files": 1, "skipped_files": 0}
    all_files = sorted(path for path in root_path.rglob("*") if path.is_file() and not path.name.startswith("."))
    selected = all_files
    if sample_per_group and sample_per_group > 0:
        grouped: dict[str, list[Path]] = {}
        for path in all_files:
            grouped.setdefault(source_group_for(path, root_path) or "root", []).append(path)
        selected = []
        for paths in grouped.values():
            selected.extend(paths[:sample_per_group])
        selected = sorted(selected)
    if max_files and max_files > 0:
        selected = selected[:max_files]
    return selected, {
        "total_files_found": len(all_files),
        "selected_files": len(selected),
        "skipped_files": max(0, len(all_files) - len(selected)),
        "max_files": max_files,
        "sample_per_group": sample_per_group,
    }


def build_ingest_plan(
    root: str | Path,
    *,
    max_file_mb: float,
    max_files: int | None = None,
    sample_per_group: int | None = None,
) -> tuple[list[PlannedFile], dict[str, Any]]:
    root_path = Path(root)
    files, selection = discover_files(root_path, max_files=max_files, sample_per_group=sample_per_group)
    return [plan_file(path, root_path, max_file_mb=max_file_mb) for path in files], selection


def plan_file(path: Path, root: Path, *, max_file_mb: float) -> PlannedFile:
    ext = path.suffix.lower()
    size = path.stat().st_size
    size_mb = round(size / (1024 * 1024), 3)
    status = "ready"
    reason = "supported"
    if ext in ARCHIVE_EXTENSIONS:
        status, reason = "skip", "archive_needs_extraction"
    elif ext in LEGACY_OFFICE_EXTENSIONS:
        status, reason = "skip", "legacy_format_needs_conversion"
    elif ext in IMAGE_EXTENSIONS:
        status, reason = "skip", "ocr_required"
    elif ext not in SUPPORTED_FILE_EXTENSIONS:
        status, reason = "skip", "unsupported_format"
    elif size_mb > max_file_mb:
        status, reason = "skip", "file_too_large_for_batch"
    return PlannedFile(
        path=path,
        relative_path=relative_path(path, root),
        extension=ext,
        file_size_bytes=size,
        file_size_mb=size_mb,
        source_group=source_group_for(path, root),
        planned_status=status,
        planned_reason=reason,
    )


def run_batch(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    plan, selection = build_ingest_plan(
        args.input,
        max_file_mb=args.max_file_mb,
        max_files=args.max_files,
        sample_per_group=args.sample_per_group,
    )
    state = load_json(args.state)
    completed = state.setdefault("completed", {})
    rows: list[dict[str, Any]] = []
    for item in plan:
        row = planned_row(item)
        if item.planned_status != "ready":
            row["status"] = "skipped"
            rows.append(row)
            continue
        fingerprint = item.fingerprint()
        previous = completed.get(item.relative_path)
        if previous and previous.get("fingerprint") == fingerprint and not args.force:
            rows.append({**row, **previous, "status": "already_ingested"})
            continue
        if args.dry_run:
            rows.append({**row, "status": "planned"})
            continue
        result = ingest_one_file(
            item,
            api_base=args.api_base,
            timeout=args.timeout,
            sync_graph=args.sync_graph,
            timeout_retries=getattr(args, "timeout_retries", 0),
            retry_timeout_multiplier=getattr(args, "retry_timeout_multiplier", 2.0),
        )
        row.update(result)
        row["fingerprint"] = fingerprint
        if result.get("status") in {"ingested", "partial", "ocr_required", "empty_or_parse_failed"}:
            completed[item.relative_path] = row
            save_json(args.state, state)
        rows.append(row)
    report = summarize(
        rows,
        selection=selection,
        dry_run=args.dry_run,
        api_base=args.api_base,
        fail_on_file_error=getattr(args, "fail_on_file_error", False),
    )
    save_json(args.report, report)
    if not args.dry_run:
        save_json(args.state, state)
    return report, 1 if report["summary"] == "FAIL" else 0


def ingest_one_file(
    item: PlannedFile,
    *,
    api_base: str,
    timeout: int,
    sync_graph: bool,
    timeout_retries: int = 0,
    retry_timeout_multiplier: float = 2.0,
) -> dict[str, Any]:
    attempts = max(1, int(timeout_retries or 0) + 1)
    current_timeout = max(1, int(timeout))
    last_result: dict[str, Any] = {}
    for attempt_index in range(attempts):
        result = _ingest_one_file_attempt(item, api_base=api_base, timeout=current_timeout, sync_graph=sync_graph)
        result["attempts"] = attempt_index + 1
        result["final_timeout"] = current_timeout
        result["retry_attempted"] = attempt_index > 0
        last_result = result
        if result.get("error") != "read_timeout" or attempt_index >= attempts - 1:
            return result
        current_timeout = max(current_timeout + 1, int(current_timeout * max(1.0, float(retry_timeout_multiplier or 1.0))))
    return last_result


def _ingest_one_file_attempt(item: PlannedFile, *, api_base: str, timeout: int, sync_graph: bool) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/ingest/documents"
    mime = mimetypes.guess_type(item.path.name)[0] or "application/octet-stream"
    started = time.perf_counter()
    try:
        with item.path.open("rb") as handle:
            response = requests.post(
                url,
                params={"sync_graph": "true" if sync_graph else "false"},
                files=[("files", (item.path.name, handle, mime))],
                timeout=timeout,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"raw": response.text[:500]}
        if response.status_code != 200:
            return {
                "status": "failed",
                "http_status": response.status_code,
                "elapsed_ms": elapsed_ms,
                "error": safe_text(payload),
            }
        ingested = (payload.get("ingested") or [{}])[0] if isinstance(payload, dict) else {}
        return {
            "status": ingested.get("parse_status") or ingested.get("status") or "ingested",
            "http_status": response.status_code,
            "elapsed_ms": elapsed_ms,
            "doc_id": ingested.get("doc_id"),
            "document_version": ingested.get("document_version"),
            "parser": ingested.get("parser"),
            "chunks": ingested.get("chunks", 0),
            "knowledge_expansion": summarize_delta(ingested.get("knowledge_expansion") or {}),
            "strict_graph_projection": ingested.get("strict_graph_projection") or {},
            "parser_error": ingested.get("parser_error"),
            "parser_warnings": (ingested.get("parser_diagnostics") or {}).get("warnings") or [],
        }
    except requests.ReadTimeout:
        return {"status": "failed", "elapsed_ms": int((time.perf_counter() - started) * 1000), "error": "read_timeout"}
    except requests.RequestException as exc:
        return {"status": "failed", "elapsed_ms": int((time.perf_counter() - started) * 1000), "error": f"{type(exc).__name__}: {exc}"[:500]}


def summarize_delta(delta: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "status",
        "new_canonical_facts_count",
        "duplicate_facts_count",
        "corroborated_facts_count",
        "conflict_groups_added_count",
        "data_gaps_added_count",
        "new_comparison_opportunities_count",
        "facts_without_evidence",
    ]
    return {key: delta.get(key) for key in keys if key in delta}


def summarize(
    rows: list[dict[str, Any]],
    *,
    selection: dict[str, Any],
    dry_run: bool,
    api_base: str,
    fail_on_file_error: bool = False,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row.get("status") or "unknown")] = status_counts.get(str(row.get("status") or "unknown"), 0) + 1
        reason_counts[str(row.get("planned_reason") or "unknown")] = reason_counts.get(str(row.get("planned_reason") or "unknown"), 0) + 1
    failed = status_counts.get("failed", 0)
    read_timeout_count = sum(1 for row in rows if str(row.get("error") or "") == "read_timeout")
    http_error_count = sum(1 for row in rows if row.get("http_status") and int(row.get("http_status") or 0) != 200)
    parser_failed_count = sum(1 for row in rows if str(row.get("status") or "").lower() in {"parse_failed", "parser_failed"})
    facts_without_evidence = sum(int((row.get("knowledge_expansion") or {}).get("facts_without_evidence") or 0) for row in rows)
    warnings: list[str] = []
    if failed:
        warnings.append("file_ingest_failures_present")
    if read_timeout_count:
        warnings.append("read_timeouts_present")
    if parser_failed_count:
        warnings.append("parser_failures_present")
    if http_error_count:
        warnings.append("http_errors_present")
    summary_status = "PASS"
    if facts_without_evidence or (failed and fail_on_file_error):
        summary_status = "FAIL"
    elif failed or parser_failed_count or http_error_count:
        summary_status = "WARN"
    return {
        "summary": summary_status,
        "dry_run": dry_run,
        "api_base": api_base,
        "selection": selection,
        "fail_on_file_error": bool(fail_on_file_error),
        "documents_planned": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "planned_reason_counts": dict(sorted(reason_counts.items())),
        "warnings": warnings,
        "failed_files_count": failed,
        "read_timeout_count": read_timeout_count,
        "http_error_count": http_error_count,
        "parser_failed_count": parser_failed_count,
        "facts_without_evidence": facts_without_evidence,
        "new_canonical_facts_count": sum(int((row.get("knowledge_expansion") or {}).get("new_canonical_facts_count") or 0) for row in rows),
        "conflict_groups_added_count": sum(int((row.get("knowledge_expansion") or {}).get("conflict_groups_added_count") or 0) for row in rows),
        "data_gaps_added_count": sum(int((row.get("knowledge_expansion") or {}).get("data_gaps_added_count") or 0) for row in rows),
        "rows": rows,
    }


def planned_row(item: PlannedFile) -> dict[str, Any]:
    return {
        "path": item.relative_path,
        "extension": item.extension,
        "file_size_mb": item.file_size_mb,
        "source_group": item.source_group,
        "planned_status": item.planned_status,
        "planned_reason": item.planned_reason,
    }


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def source_group_for(path: Path, root: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(root.resolve())
        return relative.parts[0] if len(relative.parts) > 1 else None
    except Exception:
        return None


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_text(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False) if isinstance(payload, (dict, list)) else str(payload)
    return text.replace(os.getenv("MISTRAL_API_KEY", "") or "\0", "").replace(os.getenv("OPENROUTER_API_KEY", "") or "\0", "")[:500]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-ingest a local corpus through the API, one file per request.")
    parser.add_argument("--input", default="data_storage", help="Input corpus directory or file.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL.")
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH), help="Resume state JSON path.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Output report JSON path.")
    parser.add_argument("--max-file-mb", type=float, default=25.0, help="Skip files larger than this limit.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional total selected file cap.")
    parser.add_argument("--sample-per-group", type=int, default=None, help="Optional top-level group sample cap.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-file API timeout in seconds.")
    parser.add_argument("--timeout-retries", type=int, default=0, help="Retry read timeouts this many times. Default 0 avoids duplicate processing risk.")
    parser.add_argument("--retry-timeout-multiplier", type=float, default=2.0, help="Timeout multiplier for each opt-in read-timeout retry.")
    parser.add_argument("--sync-graph", action="store_true", help="Ask API to sync graph after each file. Default is false for speed.")
    parser.add_argument("--dry-run", action="store_true", help="Only build the plan; do not call the API.")
    parser.add_argument("--force", action="store_true", help="Re-ingest files already present in state.")
    parser.add_argument("--fail-on-file-error", action="store_true", help="Return FAIL/exit 1 on any per-file ingest failure. Default reports WARN and continues.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report, exit_code = run_batch(args)
    print(f"SUMMARY: {report['summary']}")
    print(f"dry_run: {report['dry_run']}")
    print(f"documents_planned: {report['documents_planned']}")
    print(f"status_counts: {json.dumps(report['status_counts'], ensure_ascii=False)}")
    print(f"planned_reason_counts: {json.dumps(report['planned_reason_counts'], ensure_ascii=False)}")
    print(f"warnings: {json.dumps(report.get('warnings') or [], ensure_ascii=False)}")
    print(f"failed_files_count: {report.get('failed_files_count', 0)}")
    print(f"read_timeout_count: {report.get('read_timeout_count', 0)}")
    print(f"facts_without_evidence: {report['facts_without_evidence']}")
    print(f"new_canonical_facts_count: {report['new_canonical_facts_count']}")
    print(f"report: {args.report}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

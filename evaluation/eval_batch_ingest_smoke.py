"""Live API smoke for one-file-at-a-time ingestion on the real corpus.

This evaluation is intentionally small and controlled. It verifies the robust
batch path that replaces Streamlit bulk upload for large corpora. It does not
use LLM extraction and does not require embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

os.environ.setdefault("RUNTIME_PROFILE", "economy_core")
os.environ.setdefault("ENABLE_LLM", "false")
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("ENABLE_LOCAL_EMBEDDINGS", "false")
os.environ.setdefault("RETRIEVAL_MODE", "bm25")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.batch_ingest_corpus import PlannedFile, build_ingest_plan, ingest_one_file, planned_row  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts" / "eval_batch_ingest_smoke.json"


def select_ready_sample(plan: list[PlannedFile], *, limit: int) -> list[PlannedFile]:
    """Select ready files fairly across top-level source groups."""

    groups: dict[str, list[PlannedFile]] = defaultdict(list)
    for item in plan:
        if item.planned_status == "ready":
            groups[str(item.source_group or "root")].append(item)
    selected: list[PlannedFile] = []
    group_names = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for group in group_names:
            if groups[group]:
                selected.append(groups[group].pop(0))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def health_profile_warnings(health: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    summary = health.get("runtime_profile_summary") or {}
    profile = str(health.get("runtime_profile") or summary.get("runtime_profile") or "")
    retrieval = str((health.get("retrieval") or {}).get("retrieval_mode") or summary.get("retrieval_mode") or "")
    local_embeddings = bool((health.get("retrieval") or {}).get("local_embeddings_enabled") or summary.get("local_embeddings_enabled"))
    llm = health.get("llm") or {}
    llm_enabled = bool(llm.get("enabled") or summary.get("llm_enabled"))
    llm_provider = str(llm.get("provider") or summary.get("llm_provider") or "")
    if profile == "economy_core" and (retrieval == "hybrid" or local_embeddings or llm_enabled or llm_provider not in {"", "offline"}):
        warnings.append("runtime_profile_economy_core_overridden_by_env")
    effective_retrieval = str((health.get("retrieval") or {}).get("effective_retrieval_mode") or "")
    if retrieval == "hybrid" and effective_retrieval == "hybrid_degraded_to_bm25":
        warnings.append("hybrid_degraded_to_bm25")
    return warnings


def fetch_health(api_base: str, *, timeout: int) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(f"{api_base.rstrip('/')}/health", timeout=timeout)
        if response.status_code != 200:
            return None, f"health_http_{response.status_code}"
        return response.json(), None
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"[:300]


def _check(rows: list[dict[str, Any]], condition: bool, name: str, reason: str, *, warn: bool = False) -> None:
    rows.append({"check": name, "status": "PASS" if condition else "WARN" if warn else "FAIL", "reason": reason})


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    health, health_error = fetch_health(args.api_base, timeout=args.health_timeout)
    if health_error:
        _check(checks, False, "api_available", f"API target unavailable: {health_error}", warn=True)
        return {
            "summary": "WARN",
            "checks": checks,
            "api_base": args.api_base,
            "rows": [],
            "health": None,
            "health_error": health_error,
        }
    _check(checks, True, "api_available", "API /health returned 200.")
    profile_warnings = health_profile_warnings(health or {})
    _check(
        checks,
        not profile_warnings,
        "runtime_profile_consistent",
        ",".join(profile_warnings) if profile_warnings else "profile/env settings are consistent.",
        warn=True,
    )

    plan, selection = build_ingest_plan(
        args.input,
        max_file_mb=args.max_file_mb,
        max_files=args.max_files,
        sample_per_group=args.sample_per_group,
    )
    sample = select_ready_sample(plan, limit=args.sample_size)
    _check(checks, bool(sample), "ready_sample_selected", f"ready_sample={len(sample)}")
    if args.dry_run:
        rows = [{**planned_row(item), "status": "planned"} for item in sample]
        return summarize_result(checks, rows, health=health, selection=selection, dry_run=True, api_base=args.api_base)

    rows: list[dict[str, Any]] = []
    for item in sample:
        row = planned_row(item)
        result = ingest_one_file(item, api_base=args.api_base, timeout=args.timeout, sync_graph=args.sync_graph)
        row.update(result)
        rows.append(row)
    failed = [row for row in rows if str(row.get("status") or "").lower() == "failed"]
    parser_crashes = [row for row in rows if str(row.get("status") or "").lower() in {"parse_failed", "parser_failed"}]
    read_timeouts = [row for row in rows if str(row.get("error") or "") == "read_timeout"]
    facts_without_evidence = sum(int((row.get("knowledge_expansion") or {}).get("facts_without_evidence") or 0) for row in rows)
    chunks_total = sum(int(row.get("chunks") or 0) for row in rows)
    _check(checks, not failed, "no_http_ingest_failures", f"failed={len(failed)}")
    _check(checks, not read_timeouts, "no_read_timeouts", f"read_timeouts={len(read_timeouts)}")
    _check(checks, not parser_crashes, "no_parser_crashes", f"parser_crashes={len(parser_crashes)}")
    _check(checks, facts_without_evidence == 0, "facts_have_evidence", f"facts_without_evidence={facts_without_evidence}")
    _check(checks, chunks_total > 0, "chunks_created", f"chunks_total={chunks_total}", warn=True)
    return summarize_result(checks, rows, health=health, selection=selection, dry_run=False, api_base=args.api_base)


def summarize_result(
    checks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    health: dict[str, Any] | None,
    selection: dict[str, Any] | None,
    dry_run: bool,
    api_base: str,
) -> dict[str, Any]:
    failed = any(row["status"] == "FAIL" for row in checks)
    warned = any(row["status"] == "WARN" for row in checks)
    status_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("status") or "unknown")
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "summary": "FAIL" if failed else "WARN" if warned else "PASS",
        "dry_run": dry_run,
        "api_base": api_base,
        "selection": selection or {},
        "checks": checks,
        "status_counts": dict(sorted(status_counts.items())),
        "rows": rows,
        "health_summary": safe_health_summary(health or {}),
    }


def safe_health_summary(health: dict[str, Any]) -> dict[str, Any]:
    retrieval = health.get("retrieval") or {}
    llm = health.get("llm") or {}
    return {
        "status": health.get("status"),
        "runtime_profile": health.get("runtime_profile"),
        "kg_backend_active": health.get("kg_backend_active"),
        "neo4j_available": health.get("neo4j_available"),
        "llm_enabled": llm.get("enabled"),
        "llm_provider": llm.get("provider"),
        "llm_ready": llm.get("ready"),
        "retrieval_mode": retrieval.get("retrieval_mode"),
        "effective_retrieval_mode": retrieval.get("effective_retrieval_mode"),
        "local_embeddings_enabled": retrieval.get("local_embeddings_enabled"),
        "local_embeddings_ready": retrieval.get("local_embeddings_ready"),
        "hybrid_degraded_reason": retrieval.get("hybrid_degraded_reason"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test one-file-at-a-time batch ingestion through a running API.")
    parser.add_argument("--input", default="data_storage", help="Input corpus directory or file.")
    parser.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"), help="API base URL.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON report path.")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of ready files to ingest.")
    parser.add_argument("--max-file-mb", type=float, default=10.0, help="Skip larger files in smoke mode.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional plan file cap.")
    parser.add_argument("--sample-per-group", type=int, default=None, help="Optional plan sample cap per top-level group.")
    parser.add_argument("--health-timeout", type=int, default=5, help="Health request timeout in seconds.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-file ingestion timeout in seconds.")
    parser.add_argument("--sync-graph", action="store_true", help="Ask API to sync graph after each file. Default false.")
    parser.add_argument("--dry-run", action="store_true", help="Select sample only; do not call ingestion endpoint.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_eval(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY: {result['summary']}")
    for row in result["checks"]:
        print(f"[{row['status']}] {row['check']}: {row['reason']}")
    print(f"status_counts: {json.dumps(result.get('status_counts') or {}, ensure_ascii=False)}")
    print(f"JSON report: {output}")
    return 1 if result["summary"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

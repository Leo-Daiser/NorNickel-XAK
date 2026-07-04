"""Plan OCR and large-PDF processing without running heavy work by default."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("RUNTIME_PROFILE", "economy_core")
os.environ.setdefault("ENABLE_LLM", "false")
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("ENABLE_LOCAL_EMBEDDINGS", "false")
os.environ.setdefault("RETRIEVAL_MODE", "bm25")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.batch_ingest_corpus import PlannedFile, build_ingest_plan  # noqa: E402


DEFAULT_JSON = ROOT / "artifacts" / "ocr_large_pdf_report.json"
DEFAULT_MD = ROOT / "artifacts" / "ocr_large_pdf_report.md"
DEFAULT_STAGING = ROOT / "artifacts" / "ocr_staging"


def build_ocr_large_pdf_report(
    input_path: str | Path,
    *,
    staging_dir: str | Path = DEFAULT_STAGING,
    max_file_mb: float = 25.0,
    ocrmypdf_path: str | None = None,
    tesseract_path: str | None = None,
    pdftotext_path: str | None = None,
    max_files: int | None = None,
    sample_per_group: int | None = None,
) -> dict[str, Any]:
    plan, selection = build_ingest_plan(
        input_path,
        max_file_mb=max_file_mb,
        max_files=max_files,
        sample_per_group=sample_per_group,
    )
    tools = {
        "ocrmypdf": find_tool("ocrmypdf", ocrmypdf_path),
        "tesseract": find_tool("tesseract", tesseract_path),
        "pdftotext": find_tool("pdftotext", pdftotext_path),
    }
    rows = [
        queue_row(item, staging_dir=Path(staging_dir), tools=tools)
        for item in plan
        if item.planned_reason in {"ocr_required", "file_too_large_for_batch"}
    ]
    summary = summarize(rows, selection=selection, tools=tools, max_file_mb=max_file_mb)
    return {
        "status": "ok",
        "input": str(input_path),
        "staging_dir": str(staging_dir),
        "summary": summary,
        "rows": rows,
        "resource_profile": {
            "economy_core_compatible": True,
            "llm_required": False,
            "embeddings_required": False,
            "ocr_executed": False,
            "large_pdf_parse_executed": False,
            "note": "This report plans OCR/large-PDF work only; it does not run OCR or parse large PDFs.",
        },
    }


def queue_row(item: PlannedFile, *, staging_dir: Path, tools: dict[str, str | None]) -> dict[str, Any]:
    if item.planned_reason == "ocr_required":
        action = "run_ocr_or_mark_unreadable"
        tool_ready = bool(tools.get("ocrmypdf") and tools.get("tesseract"))
        blocking_reason = None if tool_ready else "ocr_tools_missing"
        target = staging_dir / "ocr" / f"{safe_stem(item.path)}.pdf"
    else:
        action = "large_file_parse_queue"
        tool_ready = bool(tools.get("pdftotext"))
        blocking_reason = None if tool_ready else "large_pdf_text_tool_missing"
        target = staging_dir / "large_pdf_text" / f"{safe_stem(item.path)}.txt"
    return {
        "path": item.relative_path,
        "extension": item.extension,
        "source_group": item.source_group,
        "file_size_mb": item.file_size_mb,
        "reason": item.planned_reason,
        "recommended_action": action,
        "tool_ready": tool_ready,
        "blocking_reason": blocking_reason,
        "staged_output_path": str(target),
        "page_count_estimated": estimate_pdf_pages(item.path) if item.extension == ".pdf" else None,
        "provenance_policy": "Keep original file path/content hash as provenance; OCR/text output is a derived parsing artifact.",
    }


def summarize(rows: list[dict[str, Any]], *, selection: dict[str, Any], tools: dict[str, str | None], max_file_mb: float) -> dict[str, Any]:
    by_reason: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_block: dict[str, int] = {}
    for row in rows:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
        by_action[row["recommended_action"]] = by_action.get(row["recommended_action"], 0) + 1
        block = row.get("blocking_reason")
        if block:
            by_block[str(block)] = by_block.get(str(block), 0) + 1
    return {
        "documents_total_found": selection.get("total_files_found"),
        "queue_count": len(rows),
        "max_file_mb": max_file_mb,
        "reason_counts": dict(sorted(by_reason.items())),
        "recommended_action_counts": dict(sorted(by_action.items())),
        "blocking_reason_counts": dict(sorted(by_block.items())),
        "tools_available": {name: bool(path) for name, path in tools.items()},
        "tools": tools,
        "ready_to_run_count": sum(1 for row in rows if row.get("tool_ready")),
        "blocked_count": sum(1 for row in rows if row.get("blocking_reason")),
    }


def estimate_pdf_pages(path: Path) -> int | None:
    try:
        from pypdf import PdfReader

        logging.getLogger("pypdf").setLevel(logging.CRITICAL)
        with contextlib.redirect_stderr(io.StringIO()):
            return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def find_tool(name: str, explicit: str | None = None) -> str | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    found = shutil.which(name)
    if found:
        candidates.append(found)
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return None


def safe_stem(path: Path) -> str:
    raw = path.stem.replace(" ", "_")
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in raw)[:120]


def write_report(report: dict[str, Any], *, json_path: str | Path, markdown_path: str | Path | None = None) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_path:
        md = Path(markdown_path)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# OCR / Large PDF Queue Report",
        "",
        "## Summary",
        f"- documents_total_found: {summary.get('documents_total_found')}",
        f"- queue_count: {summary.get('queue_count')}",
        f"- ready_to_run_count: {summary.get('ready_to_run_count')}",
        f"- blocked_count: {summary.get('blocked_count')}",
        f"- max_file_mb: {summary.get('max_file_mb')}",
        f"- tools_available: {summary.get('tools_available')}",
        "",
        "## Reasons",
        "| reason | count |",
        "|---|---:|",
    ]
    for reason, count in (summary.get("reason_counts") or {}).items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Blocking Reasons", "| reason | count |", "|---|---:|"])
    blocks = summary.get("blocking_reason_counts") or {}
    if blocks:
        for reason, count in blocks.items():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "## Notes",
            "- OCR is not executed by this report.",
            "- Large PDFs are not parsed by this report.",
            "- OCR/text outputs must preserve original-source provenance.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan OCR and large-PDF queue work without running heavy processing.")
    parser.add_argument("--input", default="data_storage", help="Input corpus directory or file.")
    parser.add_argument("--output", default=str(DEFAULT_JSON), help="Output JSON report path.")
    parser.add_argument("--markdown", default=str(DEFAULT_MD), help="Output Markdown report path.")
    parser.add_argument("--staging-dir", default=str(DEFAULT_STAGING), help="Planned OCR/text staging directory.")
    parser.add_argument("--max-file-mb", type=float, default=25.0, help="Direct-ingest size threshold used by the shared plan.")
    parser.add_argument("--ocrmypdf-path", default=None, help="Explicit path to ocrmypdf executable.")
    parser.add_argument("--tesseract-path", default=None, help="Explicit path to tesseract executable.")
    parser.add_argument("--pdftotext-path", default=None, help="Explicit path to pdftotext executable.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional selected file cap.")
    parser.add_argument("--sample-per-group", type=int, default=None, help="Optional sample cap per top-level source group.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_ocr_large_pdf_report(
        args.input,
        staging_dir=args.staging_dir,
        max_file_mb=args.max_file_mb,
        ocrmypdf_path=args.ocrmypdf_path,
        tesseract_path=args.tesseract_path,
        pdftotext_path=args.pdftotext_path,
        max_files=args.max_files,
        sample_per_group=args.sample_per_group,
    )
    write_report(report, json_path=args.output, markdown_path=args.markdown)
    summary = report["summary"]
    print("OCR / Large PDF Queue Report")
    print(f"queue_count: {summary.get('queue_count')}")
    print(f"ready_to_run_count: {summary.get('ready_to_run_count')}")
    print(f"blocked_count: {summary.get('blocked_count')}")
    print(f"reason_counts: {json.dumps(summary.get('reason_counts') or {}, ensure_ascii=False)}")
    print(f"blocking_reason_counts: {json.dumps(summary.get('blocking_reason_counts') or {}, ensure_ascii=False)}")
    print(f"tools_available: {json.dumps(summary.get('tools_available') or {}, ensure_ascii=False)}")
    print(f"json_report: {args.output}")
    print(f"markdown_report: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

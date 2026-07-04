"""Build a conversion/OCR backlog for files not ready for direct ingestion."""

from __future__ import annotations

import argparse
import json
import os
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


DEFAULT_JSON = ROOT / "artifacts" / "conversion_backlog_report.json"
DEFAULT_MD = ROOT / "artifacts" / "conversion_backlog_report.md"


ACTION_BY_REASON = {
    "archive_needs_extraction": {
        "action": "extract_archive",
        "priority": "high",
        "owner_note": "Распаковать в контролируемую staging-папку, затем повторно запустить readiness/batch ingest по извлеченным файлам.",
    },
    "legacy_format_needs_conversion": {
        "action": "convert_legacy_office",
        "priority": "high",
        "owner_note": "Конвертировать в docx/xlsx/pptx через доверенный локальный инструмент или LibreOffice headless, сохраняя исходный файл как provenance.",
    },
    "ocr_required": {
        "action": "run_ocr_or_mark_unreadable",
        "priority": "high",
        "owner_note": "Запустить OCR в отдельном controlled pipeline; если OCR недоступен, оставить статус ocr_required, не считать документ no-data.",
    },
    "file_too_large_for_batch": {
        "action": "large_file_parse_queue",
        "priority": "medium",
        "owner_note": "Обрабатывать отдельно с увеличенным timeout/страничным parser budget, не через UI upload.",
    },
    "unsupported_format": {
        "action": "manual_source_review",
        "priority": "medium",
        "owner_note": "Проверить формат и решить: конвертация, исключение или новый parser adapter.",
    },
}


def build_conversion_backlog(
    input_path: str | Path,
    *,
    max_file_mb: float = 25.0,
    max_files: int | None = None,
    sample_per_group: int | None = None,
) -> dict[str, Any]:
    plan, selection = build_ingest_plan(
        input_path,
        max_file_mb=max_file_mb,
        max_files=max_files,
        sample_per_group=sample_per_group,
    )
    rows = [backlog_row(item) for item in plan if item.planned_status != "ready"]
    summary = summarize_backlog(rows, selection=selection, max_file_mb=max_file_mb)
    return {
        "status": "ok",
        "input": str(input_path),
        "summary": summary,
        "rows": rows,
        "resource_profile": {
            "economy_core_compatible": True,
            "llm_required": False,
            "embeddings_required": False,
            "ocr_executed": False,
            "conversion_executed": False,
            "note": "This report only plans conversion/OCR work; it does not transform files.",
        },
    }


def backlog_row(item: PlannedFile) -> dict[str, Any]:
    reason = item.planned_reason
    action = ACTION_BY_REASON.get(reason, ACTION_BY_REASON["unsupported_format"])
    return {
        "path": item.relative_path,
        "extension": item.extension,
        "source_group": item.source_group,
        "file_size_mb": item.file_size_mb,
        "reason": reason,
        "recommended_action": action["action"],
        "priority": action["priority"],
        "owner_note": action["owner_note"],
        "provenance_policy": "Keep original file path/content hash as provenance; converted/OCR text must reference the original source.",
    }


def summarize_backlog(rows: list[dict[str, Any]], *, selection: dict[str, Any], max_file_mb: float) -> dict[str, Any]:
    by_reason: dict[str, int] = {}
    by_action: dict[str, int] = {}
    by_group: dict[str, int] = {}
    for row in rows:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
        by_action[row["recommended_action"]] = by_action.get(row["recommended_action"], 0) + 1
        by_group[str(row.get("source_group") or "root")] = by_group.get(str(row.get("source_group") or "root"), 0) + 1
    return {
        "documents_total_selected": selection.get("selected_files"),
        "documents_total_found": selection.get("total_files_found"),
        "documents_skipped_by_selection": selection.get("skipped_files", 0),
        "backlog_count": len(rows),
        "ready_for_direct_ingest_count": int(selection.get("selected_files") or 0) - len(rows),
        "max_file_mb": max_file_mb,
        "reason_counts": dict(sorted(by_reason.items())),
        "recommended_action_counts": dict(sorted(by_action.items())),
        "source_group_counts": dict(sorted(by_group.items())),
        "blocking_for_full_corpus": bool(rows),
    }


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
        "# Conversion / OCR Backlog Report",
        "",
        "## Summary",
        f"- documents_total_found: {summary.get('documents_total_found')}",
        f"- documents_total_selected: {summary.get('documents_total_selected')}",
        f"- backlog_count: {summary.get('backlog_count')}",
        f"- ready_for_direct_ingest_count: {summary.get('ready_for_direct_ingest_count')}",
        f"- max_file_mb: {summary.get('max_file_mb')}",
        "",
        "## Reasons",
        "| reason | count |",
        "|---|---:|",
    ]
    for reason, count in (summary.get("reason_counts") or {}).items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Recommended Actions", "| action | count |", "|---|---:|"])
    for action, count in (summary.get("recommended_action_counts") or {}).items():
        lines.append(f"| {action} | {count} |")
    lines.extend(["", "## Source Groups", "| group | count |", "|---|---:|"])
    for group, count in (summary.get("source_group_counts") or {}).items():
        lines.append(f"| {group} | {count} |")
    lines.extend(["", "## Highest Priority Items"])
    high = [row for row in report.get("rows") or [] if row.get("priority") == "high"][:25]
    if not high:
        lines.append("- none")
    for row in high:
        lines.append(f"- {row.get('path')} -> {row.get('recommended_action')} ({row.get('reason')})")
    lines.extend(
        [
            "",
            "## Resource Policy",
            "- No LLM extraction.",
            "- No embeddings required.",
            "- OCR/conversion is not executed by this report.",
            "- Converted/OCR outputs must preserve original-source provenance.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create conversion/OCR backlog for corpus files not ready for direct ingestion.")
    parser.add_argument("--input", default="data_storage", help="Input corpus directory or file.")
    parser.add_argument("--output", default=str(DEFAULT_JSON), help="Output JSON path.")
    parser.add_argument("--markdown", default=str(DEFAULT_MD), help="Output Markdown path.")
    parser.add_argument("--max-file-mb", type=float, default=25.0, help="Files larger than this are routed to large-file queue.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional selected file cap.")
    parser.add_argument("--sample-per-group", type=int, default=None, help="Optional sample cap per top-level source group.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_conversion_backlog(
        args.input,
        max_file_mb=args.max_file_mb,
        max_files=args.max_files,
        sample_per_group=args.sample_per_group,
    )
    write_report(report, json_path=args.output, markdown_path=args.markdown)
    summary = report["summary"]
    print("Conversion / OCR Backlog Report")
    print(f"documents_total_found: {summary.get('documents_total_found')}")
    print(f"backlog_count: {summary.get('backlog_count')}")
    print(f"ready_for_direct_ingest_count: {summary.get('ready_for_direct_ingest_count')}")
    print(f"reason_counts: {json.dumps(summary.get('reason_counts') or {}, ensure_ascii=False)}")
    print(f"recommended_action_counts: {json.dumps(summary.get('recommended_action_counts') or {}, ensure_ascii=False)}")
    print(f"json_report: {args.output}")
    print(f"markdown_report: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unified readiness dashboard for the real data_storage corpus.

The dashboard is intentionally non-destructive. It builds inventory and backlog
reports without running OCR, conversion, extraction-heavy parsing, embeddings or
LLM calls. Its purpose is to show what can be ingested directly and what needs
controlled preprocessing before the product path is exercised.
"""

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

from app.parsing.file_profile import profile_corpus  # noqa: E402
from scripts.archive_staging_report import build_archive_staging_report  # noqa: E402
from scripts.batch_ingest_corpus import build_ingest_plan  # noqa: E402
from scripts.conversion_backlog_report import build_conversion_backlog  # noqa: E402
from scripts.legacy_office_conversion_report import build_legacy_conversion_report  # noqa: E402
from scripts.ocr_large_pdf_report import build_ocr_large_pdf_report  # noqa: E402


DEFAULT_JSON = ROOT / "artifacts" / "data_storage_readiness_dashboard.json"
DEFAULT_MD = ROOT / "artifacts" / "data_storage_readiness_dashboard.md"


def build_dashboard(
    input_path: str | Path,
    *,
    max_file_mb: float = 25.0,
    max_files: int | None = None,
    sample_per_group: int | None = None,
) -> dict[str, Any]:
    corpus = profile_corpus(
        input_path,
        profile_mode="inventory",
        max_parse_mb=max_file_mb,
        max_files=max_files,
        sample_per_group=sample_per_group,
    )
    plan, selection = build_ingest_plan(
        input_path,
        max_file_mb=max_file_mb,
        max_files=max_files,
        sample_per_group=sample_per_group,
    )
    backlog = build_conversion_backlog(
        input_path,
        max_file_mb=max_file_mb,
        max_files=max_files,
        sample_per_group=sample_per_group,
    )
    archives = build_archive_staging_report(
        input_path,
        max_file_mb=max_file_mb,
        max_files=max_files,
        sample_per_group=sample_per_group,
    )
    legacy = build_legacy_conversion_report(
        input_path,
        max_file_mb=max_file_mb,
        max_files=max_files,
        sample_per_group=sample_per_group,
    )
    ocr = build_ocr_large_pdf_report(
        input_path,
        max_file_mb=max_file_mb,
        max_files=max_files,
        sample_per_group=sample_per_group,
    )

    summary = summarize_dashboard(
        corpus=corpus,
        plan=plan,
        selection=selection,
        backlog=backlog,
        archives=archives,
        legacy=legacy,
        ocr=ocr,
        max_file_mb=max_file_mb,
    )
    return {
        "status": summary["status"],
        "input": str(input_path),
        "selection": selection,
        "summary": summary,
        "recommended_next_actions": recommended_next_actions(summary),
        "components": {
            "corpus_inventory": compact_corpus_inventory(corpus),
            "batch_ingest_plan": compact_ingest_plan(plan),
            "conversion_backlog": compact_component(backlog, "summary"),
            "archive_staging": compact_component(archives, "summary"),
            "legacy_office_conversion": compact_component(legacy, "summary"),
            "ocr_large_pdf_queue": compact_component(ocr, "summary"),
        },
        "resource_profile": {
            "economy_core_compatible": True,
            "llm_required": False,
            "embeddings_required": False,
            "ocr_executed": False,
            "conversion_executed": False,
            "archive_extraction_executed": False,
            "note": "Dashboard is planning/inventory only; no source transformation is executed.",
        },
    }


def summarize_dashboard(
    *,
    corpus: dict[str, Any],
    plan: list[Any],
    selection: dict[str, Any],
    backlog: dict[str, Any],
    archives: dict[str, Any],
    legacy: dict[str, Any],
    ocr: dict[str, Any],
    max_file_mb: float,
) -> dict[str, Any]:
    planned_reason_counts: dict[str, int] = {}
    extension_counts: dict[str, int] = {}
    for item in plan:
        planned_reason_counts[item.planned_reason] = planned_reason_counts.get(item.planned_reason, 0) + 1
        extension_counts[item.extension or "<none>"] = extension_counts.get(item.extension or "<none>", 0) + 1

    ready_count = planned_reason_counts.get("supported", 0)
    backlog_summary = backlog.get("summary") or {}
    archive_summary = archives.get("summary") or {}
    legacy_summary = legacy.get("summary") or {}
    ocr_summary = ocr.get("summary") or {}
    corpus_summary = corpus.get("summary") or {}
    blocked_total = int(backlog_summary.get("backlog_count") or 0)
    parser_failures = int(corpus_summary.get("parser_failures_count") or 0)
    facts_without_evidence = int(corpus_summary.get("facts_without_evidence") or 0)
    warnings: list[str] = []
    if blocked_total:
        warnings.append("preprocessing_backlog_present")
    if int(legacy_summary.get("tool_missing_count") or 0):
        warnings.append("legacy_conversion_tool_missing")
    if int(ocr_summary.get("blocked_count") or 0):
        warnings.append("ocr_or_large_pdf_tools_missing")
    if int(archive_summary.get("external_extractor_required_count") or 0):
        warnings.append("external_archive_extractor_required")
    status = "FAIL" if parser_failures or facts_without_evidence else ("WARN" if warnings else "PASS")
    return {
        "status": status,
        "documents_total_found": selection.get("total_files_found"),
        "documents_selected": selection.get("selected_files"),
        "documents_skipped_by_selection": selection.get("skipped_files", 0),
        "max_file_mb": max_file_mb,
        "files_by_extension": dict(sorted(extension_counts.items())),
        "direct_ingest_ready_count": ready_count,
        "blocked_or_preprocessing_required_count": blocked_total,
        "planned_reason_counts": dict(sorted(planned_reason_counts.items())),
        "parser_failures_count": parser_failures,
        "facts_without_evidence": facts_without_evidence,
        "archive_count": int(archive_summary.get("archives_selected") or 0),
        "zip_inventory_ok_count": int((archive_summary.get("archive_status_counts") or {}).get("zip_inventory_ok") or 0),
        "zip_supported_members_count": int(archive_summary.get("zip_supported_members_count") or 0),
        "external_extractor_required_count": int(archive_summary.get("external_extractor_required_count") or 0),
        "legacy_files_count": int(legacy_summary.get("legacy_files_selected") or 0),
        "legacy_soffice_available": bool(legacy_summary.get("soffice_available")),
        "legacy_tool_missing_count": int(legacy_summary.get("tool_missing_count") or 0),
        "ocr_large_pdf_queue_count": int(ocr_summary.get("queue_count") or 0),
        "ocr_large_pdf_ready_to_run_count": int(ocr_summary.get("ready_to_run_count") or 0),
        "ocr_large_pdf_blocked_count": int(ocr_summary.get("blocked_count") or 0),
        "ocr_tools_available": ocr_summary.get("tools_available") or {},
        "economy_core_compatible": True,
        "warnings": warnings,
    }


def recommended_next_actions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    actions = [
        {
            "priority": 1,
            "action": "direct_batch_ingest",
            "count": summary.get("direct_ingest_ready_count", 0),
            "command": "python scripts/batch_ingest_corpus.py --input data_storage --max-file-mb 25 --timeout 300 --state artifacts/batch_ingest_state.json --report artifacts/batch_ingest_report.json",
            "reason": "Files already supported by the current parser path; ingest them one-by-one through API instead of Streamlit bulk upload.",
        },
        {
            "priority": 2,
            "action": "inspect_and_stage_zip_archives",
            "count": summary.get("zip_supported_members_count", 0),
            "command": "python scripts/archive_staging_report.py --input data_storage",
            "reason": "ZIP archives contain supported member files, but extraction must be controlled and provenance-preserving.",
        },
        {
            "priority": 3,
            "action": "install_libreoffice_for_legacy_office",
            "count": summary.get("legacy_tool_missing_count", 0),
            "command": "python scripts/legacy_office_conversion_report.py --input data_storage",
            "reason": "Legacy .doc/.xls/.docm files need conversion before deterministic parsing.",
        },
        {
            "priority": 4,
            "action": "install_ocr_and_pdf_text_tools",
            "count": summary.get("ocr_large_pdf_blocked_count", 0),
            "command": "python scripts/ocr_large_pdf_report.py --input data_storage",
            "reason": "Scanned and very large PDFs require a controlled OCR/large-PDF queue, not silent no-data handling.",
        },
        {
            "priority": 5,
            "action": "external_controlled_extraction_for_rar_multipart",
            "count": summary.get("external_extractor_required_count", 0),
            "command": "python scripts/archive_staging_report.py --input data_storage",
            "reason": "RAR and multipart archives are inventory-only until an approved extractor is available.",
        },
    ]
    return [item for item in actions if int(item.get("count") or 0) > 0]


def compact_corpus_inventory(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "profile_mode": report.get("profile_mode"),
        "summary": report.get("summary"),
        "high_risk_preview": (report.get("high_risk_documents") or [])[:20],
    }


def compact_ingest_plan(plan: list[Any]) -> dict[str, Any]:
    rows = []
    status_counts: dict[str, int] = {}
    for item in plan:
        status_counts[item.planned_reason] = status_counts.get(item.planned_reason, 0) + 1
        if item.planned_status != "ready" and len(rows) < 50:
            rows.append(
                {
                    "path": item.relative_path,
                    "extension": item.extension,
                    "source_group": item.source_group,
                    "file_size_mb": item.file_size_mb,
                    "planned_reason": item.planned_reason,
                }
            )
    return {
        "planned_reason_counts": dict(sorted(status_counts.items())),
        "blocked_preview": rows,
    }


def compact_component(report: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        key: report.get(key),
        "resource_profile": report.get("resource_profile"),
    }


def write_dashboard(report: dict[str, Any], *, json_path: str | Path, markdown_path: str | Path | None = None) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_path:
        md_path = Path(markdown_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Data Storage Readiness Dashboard",
        "",
        "## Summary",
        f"- status: {summary.get('status')}",
        f"- documents_total_found: {summary.get('documents_total_found')}",
        f"- documents_selected: {summary.get('documents_selected')}",
        f"- direct_ingest_ready_count: {summary.get('direct_ingest_ready_count')}",
        f"- blocked_or_preprocessing_required_count: {summary.get('blocked_or_preprocessing_required_count')}",
        f"- parser_failures_count: {summary.get('parser_failures_count')}",
        f"- facts_without_evidence: {summary.get('facts_without_evidence')}",
        f"- economy_core_compatible: {summary.get('economy_core_compatible')}",
        "",
        "## File Type Coverage",
        "| extension | count |",
        "|---|---:|",
    ]
    for ext, count in (summary.get("files_by_extension") or {}).items():
        lines.append(f"| {ext} | {count} |")
    lines.extend(["", "## Direct Ingest / Backlog", "| reason | count |", "|---|---:|"])
    for reason, count in (summary.get("planned_reason_counts") or {}).items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Preprocessing Queues",
            f"- archive_count: {summary.get('archive_count')}",
            f"- zip_inventory_ok_count: {summary.get('zip_inventory_ok_count')}",
            f"- zip_supported_members_count: {summary.get('zip_supported_members_count')}",
            f"- external_extractor_required_count: {summary.get('external_extractor_required_count')}",
            f"- legacy_files_count: {summary.get('legacy_files_count')}",
            f"- legacy_soffice_available: {summary.get('legacy_soffice_available')}",
            f"- legacy_tool_missing_count: {summary.get('legacy_tool_missing_count')}",
            f"- ocr_large_pdf_queue_count: {summary.get('ocr_large_pdf_queue_count')}",
            f"- ocr_large_pdf_ready_to_run_count: {summary.get('ocr_large_pdf_ready_to_run_count')}",
            f"- ocr_large_pdf_blocked_count: {summary.get('ocr_large_pdf_blocked_count')}",
            f"- ocr_tools_available: {summary.get('ocr_tools_available')}",
            "",
            "## Recommended Next Actions",
        ]
    )
    actions = report.get("recommended_next_actions") or []
    if not actions:
        lines.append("- none")
    for item in actions:
        lines.append(f"{item.get('priority')}. {item.get('action')} ({item.get('count')}): {item.get('reason')}")
        lines.append(f"   - command: `{item.get('command')}`")
    warnings = summary.get("warnings") or []
    lines.extend(["", "## Warnings"])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Resource Policy",
            "- No LLM extraction.",
            "- No embeddings required.",
            "- No OCR/conversion/archive extraction executed by this dashboard.",
            "- Original files remain provenance sources; derived artifacts stay in ignored staging.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified readiness dashboard for data_storage.")
    parser.add_argument("--input", default="data_storage", help="Input corpus directory or file.")
    parser.add_argument("--output", default=str(DEFAULT_JSON), help="Output JSON dashboard path.")
    parser.add_argument("--markdown", default=str(DEFAULT_MD), help="Output Markdown dashboard path.")
    parser.add_argument("--max-file-mb", type=float, default=25.0, help="Direct-ingest size threshold.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional selected file cap.")
    parser.add_argument("--sample-per-group", type=int, default=None, help="Optional sample cap per top-level source group.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_dashboard(
        args.input,
        max_file_mb=args.max_file_mb,
        max_files=args.max_files,
        sample_per_group=args.sample_per_group,
    )
    write_dashboard(report, json_path=args.output, markdown_path=args.markdown)
    summary = report["summary"]
    print("Data Storage Readiness Dashboard")
    print(f"status: {summary.get('status')}")
    print(f"documents_total_found: {summary.get('documents_total_found')}")
    print(f"direct_ingest_ready_count: {summary.get('direct_ingest_ready_count')}")
    print(f"blocked_or_preprocessing_required_count: {summary.get('blocked_or_preprocessing_required_count')}")
    print(f"planned_reason_counts: {json.dumps(summary.get('planned_reason_counts') or {}, ensure_ascii=False)}")
    print(f"warnings: {json.dumps(summary.get('warnings') or [], ensure_ascii=False)}")
    print(f"json_report: {args.output}")
    print(f"markdown_report: {args.markdown}")
    return 1 if summary.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

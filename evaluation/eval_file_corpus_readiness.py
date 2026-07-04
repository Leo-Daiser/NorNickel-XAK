"""Evaluate corpus readiness diagnostics for the current local corpus."""

from __future__ import annotations

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

from app.parsing.file_profile import profile_corpus, write_corpus_report  # noqa: E402


def _row(name: str, status: str, reason: str) -> dict[str, str]:
    return {"check": name, "status": status, "reason": reason}


def run_eval(input_dir: str | Path = ROOT / "data_storage") -> dict[str, Any]:
    corpus_dir = Path(input_dir)
    if not corpus_dir.exists():
        corpus_dir = ROOT / "demo_data"
    report = profile_corpus(corpus_dir, profile_mode="auto", max_parse_mb=2.0, sample_per_group=10)
    summary = report.get("summary") or {}
    files = report.get("files") or []
    rows: list[dict[str, str]] = []
    rows.append(_row("documents_present", "PASS" if files else "FAIL", f"documents_total={len(files)}"))
    rows.append(
        _row(
            "each_file_has_parse_status",
            "PASS" if files and all(item.get("parse_status") for item in files) else "FAIL",
            "Every profile row has parse_status.",
        )
    )
    rows.append(
        _row(
            "no_parser_failures",
            "PASS" if int(summary.get("parser_failures_count") or 0) == 0 else "FAIL",
            f"parser_failures_count={summary.get('parser_failures_count')}",
        )
    )
    rows.append(
        _row(
            "ocr_required_detected",
            "PASS" if int(summary.get("ocr_required_count") or 0) >= 1 else "WARN",
            f"ocr_required_count={summary.get('ocr_required_count')}",
        )
    )
    text_layer_docs = [
        item for item in files
        if item.get("parse_status") not in {"failed", "unsupported", "ocr_required"} and int(item.get("text_chars") or 0) > 0
    ]
    rows.append(
        _row(
            "text_layer_documents_have_chunks",
            "PASS" if all(int(item.get("chunks_count") or 0) > 0 for item in text_layer_docs) else "FAIL",
            f"text_layer_documents={len(text_layer_docs)}",
        )
    )
    rows.append(
        _row(
            "some_canonical_facts_extracted",
            "PASS" if int(summary.get("total_canonical_facts") or 0) > 0 else "WARN",
            f"total_canonical_facts={summary.get('total_canonical_facts')}",
        )
    )
    rows.append(
        _row(
            "facts_have_evidence",
            "PASS" if int(summary.get("facts_without_evidence") or 0) == 0 else "FAIL",
            f"facts_without_evidence={summary.get('facts_without_evidence')}",
        )
    )
    rows.append(
        _row(
            "conflicts_detected",
            "PASS" if int(summary.get("conflict_groups") or 0) > 0 else "WARN",
            f"conflict_groups={summary.get('conflict_groups')}",
        )
    )
    rows.append(
        _row(
            "data_gaps_detected",
            "PASS" if int(summary.get("data_gaps") or 0) > 0 else "WARN",
            f"data_gaps={summary.get('data_gaps')}",
        )
    )
    rows.append(
        _row(
            "economy_core_compatible",
            "PASS" if summary.get("economy_core_compatible") else "FAIL",
            "Profiler does not require LLM, embeddings, Qdrant or Neo4j.",
        )
    )
    unsupported = [item for item in files if item.get("parse_status") == "unsupported"]
    if unsupported:
        rows.append(_row("unsupported_formats_controlled", "WARN", f"unsupported_files={len(unsupported)}"))
    skipped = int(summary.get("documents_skipped_by_selection") or 0)
    if skipped:
        rows.append(_row("large_corpus_sampled", "WARN", f"documents_skipped_by_selection={skipped}"))
    failed = any(row["status"] == "FAIL" for row in rows)
    warned = any(row["status"] == "WARN" for row in rows)
    return {
        "summary": "FAIL" if failed else "WARN" if warned else "PASS",
        "checks": rows,
        "report": report,
    }


def main() -> int:
    result = run_eval()
    path = ROOT / "artifacts" / "eval_file_corpus_readiness.json"
    markdown_path = ROOT / "artifacts" / "corpus_readiness_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_corpus_report(result["report"], json_path=ROOT / "artifacts" / "corpus_readiness_report.json", markdown_path=markdown_path)
    print(f"SUMMARY: {result['summary']}")
    for row in result["checks"]:
        print(f"[{row['status']}] {row['check']}: {row['reason']}")
    print(f"JSON report: {path}")
    print(f"Corpus report: {ROOT / 'artifacts' / 'corpus_readiness_report.json'}")
    return 1 if result["summary"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

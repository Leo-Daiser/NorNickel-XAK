"""Build a product readiness report for a local document corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RUNTIME_PROFILE", "economy_core")
os.environ.setdefault("ENABLE_LLM", "false")
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("ENABLE_LOCAL_EMBEDDINGS", "false")
os.environ.setdefault("RETRIEVAL_MODE", "bm25")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parsing.file_profile import profile_corpus, write_corpus_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build corpus readiness diagnostics without LLM or embeddings.")
    parser.add_argument("--input", default="data_storage", help="Input file or directory.")
    parser.add_argument("--output", default="artifacts/corpus_readiness_report.json", help="Output JSON path.")
    parser.add_argument("--markdown", default="artifacts/corpus_readiness_report.md", help="Output Markdown path.")
    parser.add_argument(
        "--profile-mode",
        choices=["auto", "full", "inventory"],
        default="auto",
        help="auto parses supported files up to --max-parse-mb; inventory is metadata-only; full parses every supported file.",
    )
    parser.add_argument(
        "--max-parse-mb",
        type=float,
        default=25.0,
        help="Maximum file size parsed in auto mode. Larger files are reported as partial with parse_skipped_large_file.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for selected files.")
    parser.add_argument("--sample-per-group", type=int, default=None, help="Optional cap per top-level source group.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = profile_corpus(
        args.input,
        profile_mode=args.profile_mode,
        max_parse_mb=args.max_parse_mb,
        max_files=args.max_files,
        sample_per_group=args.sample_per_group,
    )
    write_corpus_report(report, json_path=args.output, markdown_path=args.markdown)
    summary = report.get("summary") or {}
    print("Corpus Readiness Report")
    print(f"profile_mode: {report.get('profile_mode')}")
    print(f"selection: {json.dumps(report.get('selection') or {}, ensure_ascii=False)}")
    print(f"documents_total: {summary.get('documents_total')}")
    print(f"documents_found_total: {summary.get('documents_found_total')}")
    print(f"documents_skipped_by_selection: {summary.get('documents_skipped_by_selection')}")
    print(f"parse_status_counts: {json.dumps(summary.get('parse_status_counts') or {}, ensure_ascii=False)}")
    print(f"ocr_required_count: {summary.get('ocr_required_count')}")
    print(f"zero_fact_documents_count: {summary.get('zero_fact_documents_count')}")
    print(f"large_files_skipped_count: {summary.get('large_files_skipped_count')}")
    print(f"archive_files_count: {summary.get('archive_files_count')}")
    print(f"legacy_office_files_count: {summary.get('legacy_office_files_count')}")
    print(f"facts_without_evidence: {summary.get('facts_without_evidence')}")
    print(f"conflict_groups: {summary.get('conflict_groups')}")
    print(f"data_gaps: {summary.get('data_gaps')}")
    print(f"economy_core_compatible: {summary.get('economy_core_compatible')}")
    print(f"json_report: {Path(args.output)}")
    if args.markdown:
        print(f"markdown_report: {Path(args.markdown)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

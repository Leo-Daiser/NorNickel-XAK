from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.extraction.pipeline import ExtractionPipeline  # noqa: E402
from app.extraction.quality_report import build_extraction_quality_report  # noqa: E402
from app.storage.catalog import SQLiteCatalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build accepted/rejected/quarantine extraction lifecycle report.")
    parser.add_argument("--catalog", default=str(settings.catalog_db_path), help="SQLite catalog path.")
    parser.add_argument("--output", default="artifacts/extraction_lifecycle_report.json", help="JSON output path.")
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive documents.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional chunk limit for quick diagnostics.")
    args = parser.parse_args()

    catalog = SQLiteCatalog(args.catalog)
    pipeline = ExtractionPipeline(audit_enabled=False)
    report = build_extraction_quality_report(
        catalog,
        pipeline=pipeline,
        active_only=not args.include_inactive,
        max_chunks=args.max_chunks,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Extraction Lifecycle Report")
    print(f"documents_processed: {report['documents_processed']}")
    print(f"chunks_processed: {report['chunks_processed']}")
    print(f"tables_processed: {report['tables_processed']}")
    print(f"candidate_facts_count: {report['candidate_facts_count']}")
    print(f"accepted_facts_count: {report['accepted_facts_count']}")
    print(f"rejected_candidates_count: {report['rejected_candidates_count']}")
    print(f"quarantine_candidates_count: {report['quarantine_candidates_count']}")
    print(f"acceptance_rate: {report['acceptance_rate']}")
    print(f"facts_without_evidence: {report['facts_without_evidence']}")
    print(f"top_rejected_reasons: {report['rejected_by_reason']}")
    print(f"top_quarantine_reasons: {report['quarantine_by_reason']}")
    print(f"wrote: {output}")
    return 0 if report["facts_without_evidence"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

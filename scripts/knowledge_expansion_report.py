"""Build a deterministic knowledge expansion report from the local catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.knowledge.expansion import build_knowledge_expansion_report  # noqa: E402
from app.storage.catalog import SQLiteCatalog  # noqa: E402


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    catalog = SQLiteCatalog(settings.catalog_db_path)
    report = build_knowledge_expansion_report(
        catalog,
        document_id=args.document_id,
        active_only=not args.include_inactive,
    )
    if args.since_last_ingest:
        report.setdefault("warnings", []).append(
            "since-last-ingest is not persisted yet; report covers the current catalog state."
        )
        report["scope"]["since_last_ingest_requested"] = True
    report["neo4j_sync_status"] = "not_run"
    report["neo4j_sync_warning"] = "This report is read-only; run scripts/sync_graph_to_neo4j.py or POST /knowledge/sync-neo4j to sync."
    return report


def print_summary(report: dict[str, Any]) -> None:
    warnings = report.get("warnings") or []
    print("Knowledge Expansion Report")
    print(f"status: {report.get('status')}")
    print(f"documents: {report.get('documents_count')} active={report.get('active_documents_count')}")
    print(f"chunks: {report.get('chunks_count')} active={report.get('active_chunks_count')}")
    print(f"canonical_facts: {report.get('canonical_facts_count')} raw_facts={report.get('raw_facts_count')}")
    print(f"duplicates: groups={report.get('duplicate_groups_count')} facts={report.get('duplicate_facts_count')}")
    print(f"conflicts: {report.get('conflict_groups_count')}")
    print(f"data_gaps: {report.get('data_gaps_count')}")
    print(f"facts_without_evidence: {report.get('facts_without_evidence')}")
    print(f"comparison_opportunities: {len(report.get('comparison_opportunities') or [])}")
    print(f"neo4j_sync_status: {report.get('neo4j_sync_status')}")
    for warning in warnings:
        print(f"WARN: {warning}")
    if report.get("facts_without_evidence"):
        print("WARN: accepted facts without evidence found")
    if report.get("conflict_groups_count"):
        print("WARN: conflict groups are present; review heterogeneity before strong comparisons")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic knowledge expansion report.")
    parser.add_argument("--since-last-ingest", action="store_true", help="Request since-last-ingest mode; currently reported as limitation.")
    parser.add_argument("--document-id", default=None, help="Limit report to one document id.")
    parser.add_argument("--json", dest="json_path", default="artifacts/knowledge_expansion_report.json", help="Output JSON path.")
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive documents/chunks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    print_summary(report)
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json_report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

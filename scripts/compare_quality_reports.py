"""Compare two extraction quality reports.

Usage:
    python scripts/compare_quality_reports.py old.json new.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COUNT_KEYS = [
    "documents_processed",
    "chunks_processed",
    "tables_processed",
    "candidate_facts_count",
    "accepted_facts_count",
    "rejected_candidates_count",
    "quarantine_candidates_count",
    "acceptance_rate",
    "facts_without_evidence",
]

MAP_KEYS = [
    "accepted_by_fact_type",
    "rejected_by_reason",
    "quarantine_by_reason",
    "accepted_by_doc_type",
    "rejected_by_doc_type",
    "quarantine_by_doc_type",
    "facts_by_extractor",
    "rejected_by_extractor",
    "quarantine_by_extractor",
    "rejected_by_intended_fact_type",
    "quarantine_by_intended_fact_type",
    "missing_material_by_intended_fact_type",
    "top_accepted_entities",
    "top_accepted_properties",
    "top_suspicious_entities",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare extraction lifecycle quality reports.")
    parser.add_argument("old_report", type=Path)
    parser.add_argument("new_report", type=Path)
    parser.add_argument("--json", dest="json_output", type=Path, default=None)
    args = parser.parse_args()

    old = _load_json(args.old_report)
    new = _load_json(args.new_report)
    diff = build_diff(old, new)
    print(_format_markdown(diff, args.old_report, args.new_report))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def build_diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"counts": {}, "maps": {}, "graph_projection": {}}
    for key in COUNT_KEYS:
        old_value = old.get(key, 0)
        new_value = new.get(key, 0)
        result["counts"][key] = {
            "old": old_value,
            "new": new_value,
            "delta": _delta(old_value, new_value),
        }
    for key in MAP_KEYS:
        result["maps"][key] = _map_diff(old.get(key, {}) or {}, new.get(key, {}) or {})
    for key in ("neo4j_projection_status", "graph_projection_status", "strict_projection_status"):
        if key in old or key in new:
            result["graph_projection"][key] = {"old": old.get(key), "new": new.get(key)}
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(old: Any, new: Any) -> Any:
    try:
        return round(float(new) - float(old), 6)
    except (TypeError, ValueError):
        return None if old == new else {"from": old, "to": new}


def _map_diff(old: dict[str, Any], new: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    keys = set(old) | set(new)
    rows = []
    for key in keys:
        old_value = int(old.get(key, 0) or 0)
        new_value = int(new.get(key, 0) or 0)
        rows.append({"key": str(key), "old": old_value, "new": new_value, "delta": new_value - old_value})
    rows.sort(key=lambda row: (-abs(row["delta"]), row["key"]))
    return rows[:limit]


def _format_markdown(diff: dict[str, Any], old_path: Path, new_path: Path) -> str:
    lines = [
        "# Extraction Quality Report Diff",
        "",
        f"Old: {old_path}",
        f"New: {new_path}",
        "",
        "## Counts",
        "",
        "| Metric | Old | New | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key, row in diff["counts"].items():
        lines.append(f"| {key} | {row['old']} | {row['new']} | {row['delta']} |")
    for section, rows in diff["maps"].items():
        if not rows:
            continue
        lines.extend(["", f"## {section}", "", "| Key | Old | New | Delta |", "|---|---:|---:|---:|"])
        for row in rows:
            lines.append(f"| {row['key']} | {row['old']} | {row['new']} | {row['delta']} |")
    if diff.get("graph_projection"):
        lines.extend(["", "## Graph Projection", ""])
        for key, row in diff["graph_projection"].items():
            lines.append(f"- {key}: {row['old']} -> {row['new']}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

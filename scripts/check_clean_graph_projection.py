from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.graph_db import GraphDB  # noqa: E402


def main() -> int:
    try:
        graph_db = GraphDB()
    except Exception as exc:
        print(f"Neo4j unavailable: {exc}")
        return 2

    try:
        report = build_report(graph_db)
    finally:
        graph_db.close()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PASS" if report["status"] == "PASS" else "FAIL")
    return 0 if report["status"] == "PASS" else 1


def build_report(graph_db: GraphDB) -> dict:
    checks: dict[str, int | list[str]] = {}
    with graph_db.session() as session:
        checks["non_accepted_fact_nodes"] = _scalar(
            session,
            """
            MATCH (n)
            WHERE n:Experiment OR n:Measurement OR n:DataGap
            WITH n
            WHERE n.validation_status IN ['rejected', 'quarantine', 'quarantined']
            RETURN count(n) AS count
            """,
        )
        checks["measurements_without_evidence"] = _scalar(
            session,
            """
            MATCH (m:Measurement)
            WHERE NOT (m)-[:SUPPORTED_BY]->(:DocumentChunk)
            RETURN count(m) AS count
            """,
        )
        checks["data_gaps_without_evidence"] = _scalar(
            session,
            """
            MATCH (g:DataGap)
            WHERE NOT (g)-[:SUPPORTED_BY]->(:DocumentChunk)
            RETURN count(g) AS count
            """,
        )
        material_rows = list(
            session.run(
                """
                MATCH (m:Material)
                RETURN coalesce(m.canonical_name, m.name, '') AS name
                LIMIT 1000
                """
            )
        )
        checks["font_code_like_materials"] = [
            str(row["name"])
            for row in material_rows
            if _looks_like_font_code_material(str(row["name"]))
        ][:20]

    failures = []
    for key in ("non_accepted_fact_nodes", "measurements_without_evidence", "data_gaps_without_evidence"):
        if int(checks.get(key, 0) or 0) > 0:
            failures.append(key)
    if checks["font_code_like_materials"]:
        failures.append("font_code_like_materials")
    return {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "checks": checks,
    }


def _scalar(session, query: str) -> int:
    row = session.run(query).single()
    if not row:
        return 0
    return int(row["count"] or 0)


def _looks_like_font_code_material(value: str) -> bool:
    text = value.strip()
    if not re.fullmatch(r"[A-ZА-Я]{1,4}\d{2,5}", text):
        return False
    return not text.upper().startswith(("AISI", "7075"))


if __name__ == "__main__":
    raise SystemExit(main())

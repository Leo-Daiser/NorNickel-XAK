"""Regression eval for deterministic knowledge expansion."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("KG_BACKEND", "fallback")
os.environ.setdefault("RUNTIME_PROFILE", "economy_core")
os.environ.setdefault("ENABLE_LLM", "false")
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("ENABLE_LOCAL_EMBEDDINGS", "false")
os.environ.setdefault("RETRIEVAL_MODE", "bm25")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from tests.strict_qa_helpers import reset_api  # noqa: E402


DOC_A = "После отжига сплава ВТ6 предел прочности составил 980 MPa."
DOC_B = "Ti-6Al-4V was annealed, resulting in ultimate tensile strength of 1120 MPa."
DOC_C = "The 7075-T6 aluminum alloy showed tensile strength of 77 ksi after aging treatment."
DOC_D = "Corrosion resistance after heat treatment was discussed, but no numerical corrosion data were reported."

RAW_MARKERS = ["doc_", "chunk_", "EXP-", "SCI-", "technical_answer", "PropertyValue", "SourceChunk", "Experiment"]


def _post_doc(client: TestClient, filename: str, text: str) -> dict[str, Any]:
    response = client.post(
        "/ingest/documents",
        files=[("files", (filename, text.encode("utf-8"), "text/plain"))],
    )
    if response.status_code != 200:
        raise AssertionError(f"ingest failed for {filename}: {response.status_code} {response.text}")
    payload = response.json()["ingested"][0]
    return payload


def _report(client: TestClient) -> dict[str, Any]:
    response = client.get("/knowledge/expansion-report")
    if response.status_code != 200:
        raise AssertionError(f"report failed: {response.status_code} {response.text}")
    return response.json()


def _ask(client: TestClient, question: str) -> dict[str, Any]:
    response = client.post("/ask", json={"question": question, "preset_id": "offline_reliable"})
    if response.status_code != 200:
        raise AssertionError(f"ask failed: {response.status_code} {response.text}")
    return response.json()


def _no_raw(text: str) -> bool:
    return not any(marker in text for marker in RAW_MARKERS)


def _check(condition: bool, stage: str, reason: str, rows: list[dict[str, Any]]) -> bool:
    rows.append({"stage": stage, "status": "PASS" if condition else "FAIL", "reason": reason})
    return condition


def run_eval() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        api = reset_api(Path(tmp))
        api.settings.kg_backend = "fallback"
        client = TestClient(api.app)

        _post_doc(client, "doc_a_vt6_980.txt", DOC_A)
        _post_doc(client, "doc_b_ti64_1120.txt", DOC_B)
        initial = _report(client)
        _check("ВТ6" in initial["materials"], "initial_material_alias", "ВТ6/Ti-6Al-4V material is normalized into active report", rows)
        _check(initial["conflict_groups_count"] >= 1, "initial_conflict", "980 MPa and 1120 MPa produce heterogeneity group", rows)
        _check(initial["facts_without_evidence"] == 0, "initial_evidence", "accepted facts have evidence", rows)

        c_payload = _post_doc(client, "doc_c_7075.txt", DOC_C)
        after_c = _report(client)
        c_delta = c_payload["knowledge_expansion"]
        c_facts = c_delta.get("new_canonical_facts") or []
        c_fact = next((item for item in c_facts if item.get("material") == "7075-T6"), {})
        _check("7075-T6" in c_delta["new_materials"], "new_material", "7075-T6 appears as new material", rows)
        _check(round(float(c_fact.get("value_normalized") or 0), 1) == 530.9, "normalized_ksi", "77 ksi normalized to about 530.9 MPa", rows)
        _check(c_delta["new_comparison_opportunities_count"] >= 1, "comparison_opportunity", "ВТ6 vs 7075-T6 strength comparison opportunity is created", rows)

        answer = _ask(client, "Сравни ВТ6 и 7075-T6 по прочности.")
        answer_text = str(answer.get("answer") or answer.get("human_answer") or "")
        _check(_no_raw(answer_text), "clean_answer", "comparison answer has no raw technical ids", rows)

        d_payload = _post_doc(client, "doc_d_corrosion_gap.txt", DOC_D)
        d_delta = d_payload["knowledge_expansion"]
        _check(d_delta["data_gaps_added_count"] >= 1, "data_gap", "corrosion data gap is added", rows)
        _check(
            any(item.get("property") == "коррозионная стойкость" for item in d_delta.get("data_gaps_added") or []),
            "data_gap_property",
            "data gap is bound to corrosion resistance",
            rows,
        )

        before_reingest = _report(client)
        c_reingest = _post_doc(client, "doc_c_7075.txt", DOC_C)
        after_reingest = _report(client)
        _check(
            after_reingest["canonical_facts_count"] == before_reingest["canonical_facts_count"],
            "idempotency_count",
            "re-ingesting same document does not grow canonical facts",
            rows,
        )
        _check(c_reingest["knowledge_expansion"]["new_canonical_facts_count"] == 0, "idempotency_delta", "re-ingest delta has no new canonical facts", rows)

        doc_c_id = c_payload["doc_id"]
        off = client.patch(f"/documents/{doc_c_id}/active", json={"active": False})
        if off.status_code != 200:
            raise AssertionError(f"deactivate failed: {off.status_code} {off.text}")
        inactive = _report(client)
        _check("7075-T6" not in inactive["materials"], "deactivate_excludes", "inactive document is excluded from active expansion report", rows)

        on = client.patch(f"/documents/{doc_c_id}/active", json={"active": True})
        if on.status_code != 200:
            raise AssertionError(f"reactivate failed: {on.status_code} {on.text}")
        reactivated = _report(client)
        _check("7075-T6" in reactivated["materials"], "reactivate_restores", "reactivated document returns to active expansion report", rows)

        result = {
            "summary": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
            "stages": rows,
            "facts_before_after": {
                "initial": initial["canonical_facts_count"],
                "after_c": after_c["canonical_facts_count"],
                "after_reingest": after_reingest["canonical_facts_count"],
            },
            "graph_delta_c": c_delta,
            "graph_delta_d": d_delta,
            "conflicts": reactivated.get("conflict_groups") or [],
            "gaps": reactivated.get("data_gaps") or [],
        }
        return result


def main() -> int:
    result = run_eval()
    path = ROOT / "artifacts" / "eval_knowledge_expansion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY: {result['summary']}")
    print("stage | status | reason")
    for row in result["stages"]:
        print(f"{row['stage']} | {row['status']} | {row['reason']}")
    print(f"json_report: {path}")
    return 0 if result["summary"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

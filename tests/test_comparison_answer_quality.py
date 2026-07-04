from __future__ import annotations

from app.answering.human_answer import enhance_answer_payload


def _comparison_payload() -> dict:
    return {
        "answer": "technical_answer: 7075-T6: прочность=77.0 ksi (unknown); ВТ6: прочность=1120.0 MPa (increase)",
        "status": "ok",
        "answer_mode": "comparison",
        "analytical_intent": "material_comparison",
        "constraints": {"materials": ["ВТ6", "7075-T6"], "regimes": [], "properties": ["прочность"]},
        "facts": [
            {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 1120.0, "unit": "MPa", "effect": "increase"},
            {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 980.0, "unit": "MPa", "effect": "decrease"},
            {"material": "7075-T6", "regime": "старение", "property": "прочность", "value": 520.0, "unit": "MPa", "effect": "increase"},
            {"material": "7075-T6", "regime": "старение", "property": "прочность", "value": 77.0, "unit": "ksi", "effect": "unknown"},
        ],
        "sources": [{"source_name": "demo.csv", "quote": "strength rows"}],
        "evidence": [],
        "subgraph": {"nodes": [], "edges": []},
        "graph_context": {},
        "retrieval": {},
        "diagnostics": {},
    }


def test_comparison_answer_normalizes_units_and_warns() -> None:
    payload = enhance_answer_payload(_comparison_payload(), "expert_max")
    answer = payload["answer"]
    assert "примерно 980-1120 MPa" in answer
    assert "примерно 520-531 MPa" in answer
    assert "77 ksi ≈ 531 MPa" in answer
    assert "Сравнение ограничено" in answer


def test_comparison_answer_hides_raw_technical_terms() -> None:
    payload = enhance_answer_payload(_comparison_payload(), "expert_max")
    answer = payload["answer"]
    for forbidden in ["technical_answer", "increase", "decrease", "unknown", "прочность=77.0 ksi", "doc_", "chunk_", "EXP-", "SCI-"]:
        assert forbidden not in answer
    assert payload["technical_answer"].startswith("technical_answer:")


def test_conflict_summary_is_visible_without_raw_ids() -> None:
    payload = _comparison_payload()
    payload["facts"][0]["evidence"] = [{"document_id": "doc_secret", "chunk_id": "chunk_secret", "quote": "ВТ6 1120 MPa"}]
    payload["facts"][1]["evidence"] = [{"document_id": "doc_secret", "chunk_id": "chunk_other", "quote": "ВТ6 980 MPa"}]

    enhanced = enhance_answer_payload(payload, "expert_max")
    answer = enhanced["answer"]

    assert "В корпусе найдены разные значения прочности для ВТ6 после отжига" in answer
    assert "980 MPa" in answer
    assert "1120 MPa" in answer
    for forbidden in ["doc_secret", "chunk_secret", "chunk_other", "EXP-", "SCI-", "increase", "decrease", "unknown"]:
        assert forbidden not in answer
    assert enhanced["diagnostics"]["fact_conflicts"]

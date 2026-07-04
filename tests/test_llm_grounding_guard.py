from __future__ import annotations

from app.answering.grounding_guard import build_grounding_context, guard_llm_polished_answer, validate_text_against_payload
from app.answering.human_answer import enhance_answer_payload


def _payload(answer: str, facts: list[dict] | None = None, *, status: str = "ok") -> dict:
    return {
        "answer": answer,
        "status": status,
        "answer_mode": "comparison",
        "analytical_intent": "material_comparison",
        "constraints": {"materials": ["ВТ6", "7075-T6"], "regimes": [], "properties": ["прочность"]},
        "facts": facts
        if facts is not None
        else [
            {
                "material": "ВТ6",
                "regime": "отжиг",
                "property": "прочность",
                "value": 980.0,
                "unit": "MPa",
                "effect": "increase",
                "evidence": [{"source_name": "article_vt6.txt", "quote": "ВТ6 отжиг прочность 980 MPa"}],
            },
            {
                "material": "7075-T6",
                "regime": "старение",
                "property": "прочность",
                "value": 77.0,
                "unit": "ksi",
                "value_original": 77.0,
                "unit_original": "ksi",
                "value_normalized": 530.896289,
                "unit_normalized": "MPa",
                "normalization_family": "strength",
                "effect": "increase",
                "evidence": [{"source_name": "article_7075.txt", "quote": "tensile strength of 77 ksi"}],
            },
        ],
        "sources": [],
        "evidence": [],
        "subgraph": {"nodes": [], "edges": []},
        "graph_context": {},
        "retrieval": {},
        "diagnostics": {"llm_answer_polished": True},
    }


def test_llm_polished_answer_with_unsupported_number_is_rejected() -> None:
    payload = _payload("ВТ6 подтверждён: прочность 980 MPa, обработка 1050 °C.")

    enhanced = enhance_answer_payload(payload, "expert_max")

    guard = enhanced["diagnostics"]["llm_grounding_guard"]
    assert guard["status"] == "fallback"
    assert guard["violations_count"] >= 1
    assert "1050" not in enhanced["answer"]


def test_llm_polished_answer_with_supported_original_value_passes() -> None:
    payload = _payload("7075-T6 после aging имеет tensile strength 77 ksi.")

    enhanced = enhance_answer_payload(payload, "expert_max")

    assert enhanced["diagnostics"]["llm_grounding_guard"]["status"] == "pass"
    assert "77 ksi" in enhanced["answer"]


def test_llm_polished_answer_with_supported_normalized_value_passes() -> None:
    payload = _payload("7075-T6 после старения имеет прочность 531 MPa после пересчёта.")

    enhanced = enhance_answer_payload(payload, "expert_max")

    assert enhanced["diagnostics"]["llm_grounding_guard"]["status"] == "pass"
    assert "531 MPa" in enhanced["answer"]


def test_negative_x999_cannot_include_unrelated_numeric_values() -> None:
    payload = _payload(
        "По X999 данных нет, но рядом найдена закалка 1050 °C.",
        facts=[],
        status="partial",
    )
    payload["answer_mode"] = "overview"
    payload["analytical_intent"] = "material_overview"
    payload["constraints"] = {"materials": ["X999"], "raw_question": "Что известно о X999 при лазерной обработке?"}

    enhanced = enhance_answer_payload(payload, "expert_max")

    guard = enhanced["diagnostics"]["llm_grounding_guard"]
    assert guard["status"] == "fallback"
    assert "unsupported_numeric" in {item["kind"] for item in guard["violations"]}
    assert "1050" not in enhanced["answer"]


def test_no_facts_mode_blocks_numeric_measurements() -> None:
    result = guard_llm_polished_answer("Материал X999 показал 1200 MPa.", _payload("", facts=[], status="partial"))

    assert result.status == "fallback"
    assert any(item["kind"] == "unsupported_numeric" for item in result.violations)


def test_source_grounded_mode_does_not_promote_chunks_to_verified_facts_but_allows_quotes() -> None:
    payload = _payload("Самосжатие воздуха повышает температуру в шахте.", facts=[], status="partial")
    payload["answer_mode"] = "source_grounded_answer"
    payload["answer_is_verified"] = False
    payload["source_grounded_answer_used"] = True
    payload["sources"] = [
        {
            "source_name": "deep_mine_cooling.pdf",
            "quote": "Самосжатие воздуха повышает температуру в шахте.",
        }
    ]

    result = guard_llm_polished_answer("Самосжатие воздуха повышает температуру в шахте.", payload)

    assert result.status == "pass"
    assert result.grounding_context.no_facts_mode is True
    assert result.grounding_context.source_grounded_mode is True


def test_comparison_answer_can_include_normalized_mpa_values() -> None:
    result = guard_llm_polished_answer(
        "Сравнение: ВТ6 980 MPa; 7075-T6 531 MPa после пересчёта из 77 ksi.",
        _payload(""),
    )

    assert result.status == "pass"
    assert result.violations == []


def test_guard_fallback_keeps_answer_human_readable() -> None:
    enhanced = enhance_answer_payload(_payload("ВТ6 лучше всех: прочность 9999 MPa."), "expert_max")

    assert enhanced["diagnostics"]["llm_grounding_guard"]["status"] == "fallback"
    assert "9999" not in enhanced["answer"]
    assert "###" in enhanced["answer"]
    assert "Уверенность" in enhanced["answer"]


def test_guard_diagnostics_do_not_expose_raw_ids() -> None:
    payload = _payload("ВТ6 прочность 980 MPa.")
    payload["facts"][0]["evidence"] = [
        {
            "document_id": "doc_secret",
            "chunk_id": "chunk_secret",
            "source_name": "doc_29765440445b821ce5d3075b_synthetic_vt6_heat_treatment.csv",
            "quote": "ВТ6 отжиг прочность 980 MPa",
        }
    ]

    enhanced = enhance_answer_payload(payload, "expert_max")
    diagnostics_text = str(enhanced["diagnostics"]["llm_grounding_guard"])

    assert "doc_secret" not in diagnostics_text
    assert "chunk_secret" not in diagnostics_text
    assert "doc_29765440445b821ce5d3075b" not in diagnostics_text


def test_allowed_claims_set_contains_expected_grounding_context() -> None:
    context = build_grounding_context(_payload(""))

    assert "вт6" in context.allowed_materials
    assert "7075-t6" in context.allowed_materials
    assert "прочность" in context.allowed_properties
    assert "MPa" in context.allowed_units
    assert "ksi" in context.allowed_units


def test_exact_vt6_anneal_allows_regimes_from_composite_fact_field() -> None:
    payload = _payload(
        "ВТ6: отжиг, закалка, старение; прочность: 980 MPa.",
        facts=[
            {
                "material": "ВТ6",
                "regime": "отжиг, закалка, старение",
                "property": "прочность",
                "value": 980.0,
                "unit": "MPa",
                "evidence": [{"source_name": "article_vt6.txt", "quote": "ВТ6 отжиг прочность 980 MPa"}],
            }
        ],
    )
    payload["constraints"] = {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]}

    result = validate_text_against_payload(payload["answer"], payload)
    context = build_grounding_context(payload)

    assert {"отжиг", "закалка", "старение"}.issubset(context.allowed_regimes)
    assert result.violations == []


def test_unsupported_regime_is_still_rejected() -> None:
    payload = _payload(
        "ВТ6 после криообработки имеет прочность 980 MPa.",
        facts=[
            {
                "material": "ВТ6",
                "regime": "отжиг",
                "property": "прочность",
                "value": 980.0,
                "unit": "MPa",
                "evidence": [{"source_name": "article_vt6.txt", "quote": "ВТ6 отжиг прочность 980 MPa"}],
            }
        ],
    )
    payload["constraints"] = {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]}

    result = validate_text_against_payload(payload["answer"], payload)

    assert any(item["kind"] == "unsupported_regime" for item in result.violations)


def test_grounding_context_allows_gap_entities_without_measurements() -> None:
    payload = _payload("", facts=[], status="ok")
    payload["answer_mode"] = "gaps"
    payload["data_gaps"] = [
        {
            "material": "7075-T6",
            "regime": "старение",
            "property": "коррозионная стойкость",
            "reason": "нет данных",
        }
    ]

    context = build_grounding_context(payload)

    assert "7075-t6" in context.allowed_materials
    assert "старение" in context.allowed_regimes
    assert "коррозионная стойкость" in context.allowed_properties
    assert context.no_facts_mode is True


def test_unsupported_number_triggers_repair_and_safe_repair_passes() -> None:
    seen_requests: list[dict] = []

    def repairer(request: dict) -> str:
        seen_requests.append(request)
        return "ВТ6: 980 MPa. 7075-T6: 531 MPa после пересчёта из 77 ksi."

    enhanced = enhance_answer_payload(
        _payload("ВТ6 подтверждён: прочность 980 MPa, обработка 1050 °C."),
        "expert_max",
        llm_repairer=repairer,
    )
    guard = enhanced["diagnostics"]["llm_grounding_guard"]

    assert guard["status"] == "repaired"
    assert guard["first_pass"] is False
    assert guard["repair_attempted"] is True
    assert guard["repair_passed"] is True
    assert guard["violations_count"] >= 1
    assert guard["repaired_violations_count"] == 0
    assert guard["unsafe_answer_blocked"] is True
    assert "1050" not in enhanced["answer"]
    assert "531 MPa" in enhanced["answer"]
    assert seen_requests
    request = seen_requests[0]
    for key in [
        "allowed_materials",
        "allowed_regimes",
        "allowed_properties",
        "allowed_numeric_values_original",
        "allowed_numeric_values_normalized",
        "allowed_units",
        "allowed_source_names",
        "no_facts_mode",
        "violations",
    ]:
        assert key in request


def test_unsafe_repair_fails_and_falls_back_to_deterministic_answer() -> None:
    enhanced = enhance_answer_payload(
        _payload("ВТ6 подтверждён: прочность 980 MPa, обработка 1050 °C."),
        "expert_max",
        llm_repairer=lambda request: "ВТ6 якобы имеет 9999 MPa.",
    )
    guard = enhanced["diagnostics"]["llm_grounding_guard"]

    assert guard["status"] == "fallback"
    assert guard["repair_attempted"] is True
    assert guard["repair_passed"] is False
    assert guard["repaired_violations_count"] >= 1
    assert "1050" not in enhanced["answer"]
    assert "9999" not in enhanced["answer"]
    assert "###" in enhanced["answer"]


def test_no_facts_mode_does_not_attempt_repair_with_numeric_injection() -> None:
    calls = []

    def repairer(request: dict) -> str:
        calls.append(request)
        return "X999 имеет 1200 MPa."

    payload = _payload("X999 имеет 1200 MPa.", facts=[], status="partial")
    payload["answer_mode"] = "overview"
    payload["analytical_intent"] = "material_overview"
    payload["constraints"] = {"materials": ["X999"], "raw_question": "Что известно о X999?"}

    enhanced = enhance_answer_payload(payload, "expert_max", llm_repairer=repairer)
    guard = enhanced["diagnostics"]["llm_grounding_guard"]

    assert guard["status"] == "fallback"
    assert guard["repair_attempted"] is False
    assert calls == []
    assert "1200" not in enhanced["answer"]


def test_skipped_guard_diagnostics_when_llm_polish_is_not_used() -> None:
    payload = _payload("legacy", facts=[])
    payload["diagnostics"] = {}
    payload["answer_mode"] = "overview"

    enhanced = enhance_answer_payload(payload, "expert_max")

    guard = enhanced["diagnostics"]["llm_grounding_guard"]
    assert guard["status"] == "skipped"
    assert guard["unsafe_answer_blocked"] is False

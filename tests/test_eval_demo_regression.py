from __future__ import annotations

from evaluation.eval_demo_regression import (
    comparison_normalized_unit_errors,
    evidence_summary_has_raw_ids,
    friendly_source_warnings,
    graph_contract,
    negative_query_has_hallucinated_number,
    raw_leak_count,
    validate_case,
    DEMO_CASES,
)


def test_demo_regression_raw_leak_detector_counts_technical_tokens() -> None:
    text = "technical_answer leaked chunk_abc via PropertyValue MEASURES and increase"

    assert raw_leak_count(text) >= 5
    assert raw_leak_count("Чистый пользовательский ответ про ВТ6 и прочность.") == 0


def test_demo_regression_graph_contract_detects_clean_compact_graph() -> None:
    payload = {
        "status": "ok",
        "answer_mode": "comparison",
        "constraints": {"materials": ["ВТ6", "7075-T6"], "properties": ["прочность"]},
        "facts": [
            {"material": "ВТ6", "property": "прочность", "value": 1120, "unit": "MPa", "effect": "increase"},
            {"material": "7075-T6", "property": "прочность", "value": 77, "unit": "ksi", "effect": "unknown"},
        ],
    }

    contract = graph_contract(payload)

    assert contract["nodes"] <= 10
    assert contract["edges"] <= 12
    assert contract["raw_label_leaks"] == 0


def test_demo_regression_friendly_source_validator_rejects_raw_source_labels() -> None:
    payload = {
        "facts": [
            {
                "material": "7075-T6",
                "regime": "старение",
                "property": "прочность",
                "value": 77,
                "unit": "ksi",
                "value_normalized": 530.9,
                "unit_normalized": "MPa",
                "evidence": [
                    {
                        "source_name": "doc_29765440445b821ce5d3075b_synthetic_vt6_heat_treatment.csv",
                        "chunk_id": "chunk_abc",
                        "quote": "material: 7075-T6 | process_regime: aging | property: strength | value: 77 | unit: ksi",
                    }
                ],
            }
        ]
    }

    assert evidence_summary_has_raw_ids(payload) is False
    assert friendly_source_warnings(payload) == []


def test_demo_regression_negative_number_guard_ignores_material_code_but_catches_measurement() -> None:
    assert negative_query_has_hallucinated_number("По X999 точных данных не найдено.") is False
    assert negative_query_has_hallucinated_number("Для X999 прочность составила 980 MPa.") is True


def test_demo_regression_comparison_checker_requires_mpa_materials_and_caveat() -> None:
    good = (
        "ВТ6 и 7075-T6 сравнены в MPa. Сравнение ограничено: разные режимы и источники. "
        "Исходное значение 77 ksi ≈ 531 MPa."
    )
    bad = "ВТ6 прочнее. 7075-T6 = 77 ksi."

    assert comparison_normalized_unit_errors(good) == []
    errors = comparison_normalized_unit_errors(bad)
    assert any("MPa" in error for error in errors)
    assert any("conversion" in error or "conversion" in error.lower() or "ksi" in error for error in errors)
    assert any("caveat" in error for error in errors)


def test_demo_regression_validate_case_catches_offline_and_raw_leakage() -> None:
    case = DEMO_CASES[0]
    payload = {
        "status": "ok",
        "answer": "Офлайн-режим: leaked chunk_abc.",
        "diagnostics": {"preset_id": "offline_reliable"},
        "facts": [],
        "evidence": [],
        "subgraph": {"nodes": [], "edges": []},
    }

    result = validate_case(case, payload)

    assert result["passed"] is False
    assert result["raw_leaks_count"] >= 1
    assert any("offline" in reason.lower() or "preset" in reason.lower() for reason in result["reasons"])


def test_demo_regression_validate_case_reports_guard_stats() -> None:
    case = DEMO_CASES[1]
    payload = {
        "status": "ok",
        "answer_mode": "comparison",
        "answer": (
            "ВТ6 и 7075-T6 сравнены в MPa. Сравнение ограничено: разные режимы и источники. "
            "Исходное значение 77 ksi ≈ 531 MPa."
        ),
        "diagnostics": {
            "preset_id": "expert_max",
            "llm_answer_polished": True,
            "llm_grounding_guard": {
                "status": "repaired",
                "repair_attempted": True,
                "repair_passed": True,
                "violations_count": 2,
                "repaired_violations_count": 0,
            },
        },
        "constraints": {"materials": ["ВТ6", "7075-T6"], "properties": ["прочность"]},
        "facts": [
            {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 980, "unit": "MPa"},
            {
                "material": "7075-T6",
                "regime": "старение",
                "property": "прочность",
                "value": 77,
                "unit": "ksi",
                "value_normalized": 530.9,
                "unit_normalized": "MPa",
            },
        ],
        "evidence": [],
        "subgraph": {"nodes": [], "edges": []},
    }

    result = validate_case(case, payload)

    assert result["llm_grounding_guard_status"] == "repaired"
    assert result["guard_repair_attempted"] is True
    assert result["guard_fallback_used"] is False
    assert result["guard_violations_count"] == 2

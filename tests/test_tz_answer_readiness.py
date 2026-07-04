from __future__ import annotations

from evaluation.eval_tz_answer_readiness import classify_case, numeric_claims, unsupported_numeric_claims


CASE = {"case_id": "desalination", "question": "Какие методы обессоливания подходят?"}


def test_numeric_claims_cover_tz_units() -> None:
    claims = numeric_claims("сульфаты 200 мг/л, поток 1.5 м/с, CAPEX 10 USD/t")

    assert {"value": 200.0, "unit": "mg/L", "text": "200 мг/л"} in claims
    assert {"value": 1.5, "unit": "m/s", "text": "1.5 м/с"} in claims
    assert {"value": 10.0, "unit": "USD/t", "text": "10 USD/t"} in claims


def test_unsupported_numeric_claims_allow_query_constraints() -> None:
    answer = "Требование: сухой остаток 1000 мг/дм³. Неподтвержденное число: 777 мг/л."

    unsupported = unsupported_numeric_claims(answer, [1000.0])

    assert len(unsupported) == 1
    assert unsupported[0]["value"] == 777.0


def test_classify_case_warns_on_safe_no_evidence_answer() -> None:
    row = classify_case(
        CASE,
        {"numeric_constraints": []},
        {"status": "no_exact_match", "answer": "Точных данных в текущем корпусе не найдено.", "facts": [], "evidence": []},
        latency_ms=12,
    )

    assert row["status"] == "WARN"
    assert "no_evidence_in_current_corpus" in row["warnings"]
    assert not row["failures"]


def test_classify_case_fails_on_raw_leak() -> None:
    row = classify_case(
        CASE,
        {"numeric_constraints": []},
        {"status": "ok", "answer": "Ответ основан на chunk_123.", "facts": [], "evidence": [{"source_name": "x"}]},
        latency_ms=12,
    )

    assert row["status"] == "FAIL"
    assert "raw_leak" in row["failures"]


def test_classify_case_fails_on_unsupported_numeric() -> None:
    row = classify_case(
        CASE,
        {"numeric_constraints": [{"parameter": "сухой остаток", "value": 1000.0, "unit": "mg/L"}]},
        {"status": "ok", "answer": "Подходит метод с концентрацией 777 мг/л.", "facts": [], "evidence": [{"source_name": "x"}]},
        latency_ms=12,
    )

    assert row["status"] == "FAIL"
    assert "unsupported_numeric_claim" in row["failures"]


def test_classify_case_fails_on_constraint_mismatch_with_evidence() -> None:
    row = classify_case(
        {
            "case_id": "equipment",
            "question": "Какие ванны электроэкстракции никеля описаны?",
            "expected": {"materials": ["никель"], "equipment": ["ванна электроэкстракции"]},
        },
        {"numeric_constraints": []},
        {
            "status": "ok",
            "answer": "Найдена сводка по объекту насос.",
            "facts": [],
            "evidence": [{"source_name": "x", "quote": "насос"}],
        },
        latency_ms=12,
    )

    assert row["status"] == "FAIL"
    assert "constraint_mismatch" in row["failures"]
    assert set(row["missing_constraints"]) == {"materials", "equipment"}

from __future__ import annotations

import json
from pathlib import Path

from evaluation.eval_corpus_truthfulness import (
    CORPUS_DIR,
    QueryCase,
    evaluate_response,
    load_query_cases,
    raw_leaks,
    run_eval,
    unit_number_claims,
)


def test_synthetic_corpus_structure_is_present() -> None:
    expected_dirs = {
        "clean",
        "semi_structured",
        "dirty",
        "conflicting",
        "versioned",
        "duplicate",
        "web",
        "negative",
        "coverage_gaps",
        "structured",
    }
    actual_dirs = {path.name for path in CORPUS_DIR.iterdir() if path.is_dir()}

    assert expected_dirs <= actual_dirs
    assert any(path.suffix == ".txt" for path in CORPUS_DIR.rglob("*"))
    assert any(path.suffix == ".html" for path in CORPUS_DIR.rglob("*"))
    assert any(path.suffix == ".csv" for path in CORPUS_DIR.rglob("*"))
    assert any(path.suffix == ".json" for path in CORPUS_DIR.rglob("*"))


def test_query_bank_and_expectations_are_component_wise() -> None:
    cases = load_query_cases()
    ids = {case.case_id for case in cases}

    assert len(cases) >= 20
    assert "comparison_vt6_7075_strength" in ids
    assert "negative_x999_laser" in ids
    assert "web_vt6_anneal" in ids
    assert all(case.expectation.get("expected_raw_leaks") is not None for case in cases)


def test_raw_leak_detector_catches_user_facing_technical_tokens() -> None:
    text = "Main answer leaked doc_abc and SourceChunk plus EXP-123."

    leaks = raw_leaks(text)

    assert "doc_abc" in leaks
    assert "SourceChunk" in leaks
    assert "EXP-123" in leaks


def test_unit_number_claim_extractor_finds_supported_units() -> None:
    claims = unit_number_claims("ВТ6: 980 MPa, 7075-T6: 77 ksi, steel: 210 HV.")

    assert [claim["unit"] for claim in claims] == ["MPa", "ksi", "HV"]
    assert [claim["value"] for claim in claims] == [980.0, 77.0, 210.0]


def test_negative_numeric_hallucination_is_classified() -> None:
    case = QueryCase(
        case_id="negative_x999_laser",
        section="negative",
        question="Что известно о X999 при лазерной обработке?",
        expectation={
            "expected_mode": "negative",
            "expected_materials": ["X999"],
            "expected_regimes": ["лазерная обработка"],
            "expected_properties": [],
            "expected_numeric_values_original": [],
            "expected_numeric_values_normalized": [],
            "expected_no_data": True,
            "expected_source_presence": False,
        },
    )
    payload = {
        "status": "no_exact_match",
        "answer": "Для X999 при лазерной обработке подтверждено 1050 C.",
        "constraints": {"materials": ["X999"], "regimes": ["лазерная обработка"], "properties": []},
        "facts": [],
        "sources": [],
        "evidence": [],
    }

    row = evaluate_response(case, payload, {"active_filtering": {"passed": True}, "final_report": {}})

    assert "Hallucinated numeric value" in row["error_categories"]
    assert "No-data hallucination" in row["error_categories"]


def test_truthfulness_eval_writes_json_and_markdown(tmp_path: Path) -> None:
    result, exit_code = run_eval(profile="economy_core", limit=2, artifacts_dir=tmp_path)

    assert exit_code == 0
    assert result["summary"] in {"PASS", "WARN"}
    json_path = tmp_path / "eval_corpus_truthfulness.json"
    md_path = tmp_path / "eval_corpus_truthfulness.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["raw_leak_rate"] == 0.0
    assert "Synthetic Corpus Truthfulness Evaluation" in md_path.read_text(encoding="utf-8")

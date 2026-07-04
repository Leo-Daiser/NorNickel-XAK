from __future__ import annotations

from app.answering.human_answer import rank_facts


def test_fact_ranking_prefers_known_effect_and_unit() -> None:
    facts = [
        {
            "experiment_id": "EXP-1cc92a75d794",
            "material": "ВТ6",
            "regime": "отжиг",
            "property": "прочность",
            "value": 980,
            "unit": "MPa",
            "effect": "unknown",
            "evidence": [{"quote": "text"}],
        },
        {
            "experiment_id": "SCI-VT6-AN-900",
            "material": "ВТ6",
            "regime": "отжиг",
            "property": "прочность",
            "value": 1120,
            "unit": "MPa",
            "effect": "increase",
            "evidence": [{"quote": "Table columns: value"}],
        },
    ]
    ranked = rank_facts(facts)
    assert len(ranked["primary_facts"]) <= 5
    assert ranked["primary_facts"][0]["value"] == 1120
    assert ranked["primary_facts"][0]["effect"] == "increase"

from __future__ import annotations

from scripts.compare_quality_reports import build_diff


def test_quality_report_diff_tracks_counts_and_reason_deltas() -> None:
    old = {
        "accepted_facts_count": 10,
        "candidate_facts_count": 100,
        "rejected_by_reason": {"missing_material": 20},
    }
    new = {
        "accepted_facts_count": 14,
        "candidate_facts_count": 120,
        "rejected_by_reason": {"missing_material": 12, "unknown_property_schema": 3},
    }

    diff = build_diff(old, new)

    assert diff["counts"]["accepted_facts_count"]["delta"] == 4.0
    reason_rows = {row["key"]: row for row in diff["maps"]["rejected_by_reason"]}
    assert reason_rows["missing_material"]["delta"] == -8
    assert reason_rows["unknown_property_schema"]["delta"] == 3

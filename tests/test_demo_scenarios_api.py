from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_demo_scenarios_are_non_empty_unique_and_have_expectations(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.get("/demo/scenarios")
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert items
    ids = [item["scenario_id"] for item in items]
    assert len(ids) == len(set(ids))
    assert all("expected_intent" in item and "expected_status" in item for item in items)


def test_demo_run_scenario_returns_ask_like_response(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post("/demo/run-scenario", params={"scenario_id": "material_overview_vt6"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["answer"]
    assert payload["scenario"]["scenario_id"] == "material_overview_vt6"
    assert "facts" in payload
    assert "subgraph" in payload


def test_demo_strict_negative_returns_no_exact_match(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post("/demo/run-scenario", params={"scenario_id": "strict_negative_vt6_cryo_toughness"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "no_exact_match"
    assert payload["facts"] == []
    assert payload["scenario"]["expected_status"] == "no_exact_match"

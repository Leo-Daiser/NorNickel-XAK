from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_exact_match_returns_positive_graph_answer(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post(
        "/ask",
        params={"question": "Что делали по сплаву ВТ6 при отжиге и как изменилась прочность?", "top_k": 5},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["intent"] == "material_regime_property_effect"
    assert payload["constraints"]["materials"] == ["ВТ6"]
    assert payload["constraints"]["regimes"] == ["отжиг"]
    assert payload["constraints"]["properties"] == ["прочность"]
    assert payload["facts"]
    assert payload["facts"][0]["experiment_id"] == "EXP-VT6-AN"
    assert all(fact["property"] == "прочность" for fact in payload["facts"])
    assert "1120" in str(payload["facts"])
    assert payload["sources"]
    assert payload["subgraph"]["nodes"]
    assert payload["subgraph"]["edges"]


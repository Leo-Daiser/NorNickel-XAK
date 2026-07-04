from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_decision_history_is_material_scoped(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post("/ask", params={"question": "Покажи историю решений по ВТ6.", "top_k": 5})
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["intent"] == "decision_history"
    history = payload["decision_history"]
    assert history
    assert "7075-T6" not in str(history)
    for item in history:
        assert item["experiment_id"]
        assert item["regime"]
        assert item["measurements"]
        assert item["conclusions"] or item["evidence"]


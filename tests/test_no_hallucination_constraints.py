from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_missing_material_regime_property_does_not_use_neighboring_facts(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post(
        "/ask",
        params={"question": "Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?", "top_k": 5},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] == "no_exact_match"
    assert payload["facts"] == []
    assert "точных данных не найдено" in payload["answer"].lower()
    assert "криообработка дала" not in payload["answer"].lower()
    assert payload["data_gaps"]
    gap = payload["data_gaps"][0]
    assert gap["material"] == "ВТ6"
    assert gap["regime"] == "криообработка"
    assert gap["property"] == "вязкость"
    assert payload["partial_matches"]["same_material"]
    assert payload["sources"] == []


from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_gap_by_other_material_does_not_leak_to_vt6(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post("/ask", params={"question": "Какие пробелы в данных есть по ВТ6?", "top_k": 5})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "7075-T6" not in str(payload.get("data_gaps") or payload.get("gaps") or [])


def test_generic_gap_demo_question_is_not_clarification(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post("/ask", json={"question": "Какие есть пробелы в данных?", "top_k": 5, "preset_id": "expert_max"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["diagnostics"]["preset_id"] == "expert_max"
    assert payload["answer_mode"] != "needs_clarification"


def test_corrosion_gap_does_not_leak_to_strength_question(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post("/ask", params={"question": "Какие пробелы по 7075-T6 и прочности?", "top_k": 5})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "коррозион" not in str(payload.get("data_gaps") or payload.get("gaps") or []).lower()


def test_missing_exact_creates_inferred_gap(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post(
        "/ask",
        params={"question": "Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?", "top_k": 5},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "no_exact_match"
    assert payload["data_gaps"]
    assert payload["data_gaps"][0]["gap_id"].startswith("inferred_")

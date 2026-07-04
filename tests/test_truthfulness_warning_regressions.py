from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.strict_qa_helpers import reset_api


RAW_TOKENS = ["doc_", "chunk_", "EXP-", "SCI-", "PropertyValue", "SourceChunk", "Experiment"]


def _seed_truthfulness_warning_corpus(tmp_path: Path) -> TestClient:
    api = reset_api(tmp_path)
    client = TestClient(api.app)
    documents = [
        ("vt6_980.txt", "После отжига сплава ВТ6 предел прочности составил 980 MPa."),
        ("vt6_1120.txt", "Ti-6Al-4V was annealed, resulting in ultimate tensile strength of 1120 MPa."),
        ("vt6_931.txt", "После отжига сплава ВТ6 предел прочности составил 931 MPa."),
        ("steel_210.txt", "После закалки стали 12Х18Н10Т твердость составила 210 HV."),
        ("steel_240.txt", "После закалки стали 12Х18Н10Т твердость составила 240 HV."),
        ("gap.txt", "Коррозионная стойкость после термообработки не измерялась; численные данные не приведены."),
    ]
    for filename, text in documents:
        response = client.post(
            "/ingest/documents",
            files=[("files", (filename, text.encode("utf-8"), "text/plain"))],
        )
        assert response.status_code == 200, response.text
    return client


def test_broad_conflict_query_uses_canonical_conflict_summary(tmp_path: Path) -> None:
    client = _seed_truthfulness_warning_corpus(tmp_path)

    response = client.post(
        "/ask",
        json={"question": "Какие значения расходятся по одному и тому же режиму?", "preset_id": "expert_max", "top_k": 12},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    answer = str(payload["answer"])
    assert payload["intent"] == "conflict_analysis"
    assert payload["answer_mode"] == "graph_conflict_analysis"
    assert "Неоднородность данных" in answer
    assert "расходится между источниками" in answer
    for token in ["ВТ6", "отжиг", "прочность", "980 MPa", "1120 MPa", "931 MPa", "12Х18Н10Т", "210 HV", "240 HV"]:
        assert token in answer
    assert "Не следует считать одно значение окончательным" in answer
    assert "Параметры: Parameter" not in answer
    assert not any(token in answer for token in RAW_TOKENS)


def test_not_measured_gap_query_does_not_fall_back_to_object_summary(tmp_path: Path) -> None:
    client = _seed_truthfulness_warning_corpus(tmp_path)

    response = client.post(
        "/ask",
        json={"question": "Где явно указано, что данные не измерялись?", "preset_id": "expert_max", "top_k": 12},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    answer = str(payload["answer"])
    assert payload["intent"] == "gap_analysis"
    assert "Пробелы в данных" in answer
    assert "коррозионная стойкость" in answer
    assert "не измерялась" in answer
    assert "Параметры: Parameter" not in answer
    assert "980 MPa" not in answer
    assert "1120 MPa" not in answer
    assert "210 HV" not in answer
    assert not any(token in answer for token in RAW_TOKENS)


def test_material_strength_inventory_after_heat_treatment_is_not_partial(tmp_path: Path) -> None:
    client = _seed_truthfulness_warning_corpus(tmp_path)

    response = client.post(
        "/ask",
        json={"question": "Какие материалы имеют данные по прочности после термообработки?", "preset_id": "expert_max", "top_k": 12},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    answer = str(payload["answer"])
    assert payload["status"] == "ok"
    assert payload["intent"] == "material_inventory"
    assert "ВТ6" in answer
    assert "прочность" in answer
    assert "Подтверждённые экспериментальные данные" in answer
    assert "Структурированных фактов" not in answer
    assert not any(token in answer for token in RAW_TOKENS)

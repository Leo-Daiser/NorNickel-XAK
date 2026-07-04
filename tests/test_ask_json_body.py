from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _reset_api(tmp_path: Path):
    import app.api as api
    from app.retrieval.retrieval import RetrievalEngine
    from app.storage.catalog import SQLiteCatalog
    from app.storage.outbox import SQLiteOutbox

    api.graph_db = None
    api.catalog = SQLiteCatalog(tmp_path / "catalog.sqlite3")
    api.outbox = SQLiteOutbox(tmp_path / "outbox.sqlite3")
    api.retrieval_engine = RetrievalEngine()
    api.retrieval_engine.dense_retrieve = lambda question, top_k=20: []
    api.DOCUMENTS.clear()
    api.CHUNKS.clear()
    return api


def _load_sample(client: TestClient) -> None:
    sample = (
        "Experiment: EXP-VT6-AN. Material: сплав ВТ6. Process: отжиг at 900 C for 2 h. "
        "Предел прочности σв составил 1120 МПа."
    ).encode("utf-8")
    response = client.post("/ingest/documents", files=[("files", ("vt6.txt", sample, "text/plain"))])
    assert response.status_code == 200, response.text


def test_ask_query_params_still_work(tmp_path: Path) -> None:
    api = _reset_api(tmp_path)
    client = TestClient(api.app)
    _load_sample(client)

    response = client.post("/ask", params={"question": "Что делали по ВТ6?", "top_k": 5})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["answer"]
    assert payload["diagnostics"]["input_source"] == "query_params"


def test_ask_json_body_with_preset(tmp_path: Path) -> None:
    api = _reset_api(tmp_path)
    client = TestClient(api.app)
    _load_sample(client)

    response = client.post(
        "/ask",
        json={"question": "Что делали по ВТ6?", "top_k": 5, "preset_id": "offline_reliable"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["diagnostics"]["preset_id"] == "offline_reliable"
    assert payload["diagnostics"]["input_source"] == "json_body"


def test_ask_json_body_strict_flag_overrides_best_preset(tmp_path: Path) -> None:
    api = _reset_api(tmp_path)
    client = TestClient(api.app)
    _load_sample(client)

    response = client.post(
        "/ask",
        json={
            "question": "Что делали по ВТ6?",
            "top_k": 5,
            "preset_id": "expert_max",
            "strict_audit_mode": True,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["diagnostics"]["preset_id"] == "strict_audit"
    assert payload["diagnostics"]["effective_runtime_mode"]["strict_audit_mode"] is True


def test_ask_json_body_has_priority_over_query_params(tmp_path: Path) -> None:
    api = _reset_api(tmp_path)
    client = TestClient(api.app)
    _load_sample(client)

    response = client.post(
        "/ask",
        params={"question": "чччто там как где пупупу?", "top_k": 1},
        json={"question": "Что делали по ВТ6?", "top_k": 5, "preset_id": "strict_audit"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["diagnostics"]["query_params_ignored"] is True
    assert payload["diagnostics"]["preset_id"] == "strict_audit"


def test_ask_missing_question_returns_422() -> None:
    import app.api as api

    client = TestClient(api.app)
    response = client.post("/ask")
    assert response.status_code == 422

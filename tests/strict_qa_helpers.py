from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


STRICT_QA_SAMPLE = """experiment_id: EXP-VT6-AN. Material: сплав ВТ6. Process: отжиг 900 C 2 h. Property: прочность. Result: прочность increased to 1120 MPa. Equipment: Печь A. Laboratory: Лаборатория A. Conclusion: отжиг повысил прочность.
experiment_id: EXP-VT6-AGE. Material: сплав ВТ6. Process: старение 550 C 4 h. Property: твёрдость. Result: твёрдость increased to 360 HV. Conclusion: старение повысило твёрдость.
experiment_id: EXP-AL-GAP. Material: 7075-T6. Process: старение. Property: прочность. Result: прочность increased to 520 MPa. Data gap: нет данных по коррозионной стойкости 7075-T6 после старения."""


def reset_api(tmp_path: Path):
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


def seeded_client(tmp_path: Path) -> TestClient:
    api = reset_api(tmp_path)
    client = TestClient(api.app)
    response = client.post(
        "/ingest/documents",
        files=[("files", ("strict_qa_sample.txt", STRICT_QA_SAMPLE.encode("utf-8"), "text/plain"))],
    )
    assert response.status_code == 200, response.text
    return client


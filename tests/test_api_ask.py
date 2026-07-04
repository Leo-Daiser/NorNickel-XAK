from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_ask_returns_structured_payload_without_external_services(tmp_path: Path, monkeypatch) -> None:
    import app.api as api
    from app.retrieval.retrieval import RetrievalEngine
    from app.storage.catalog import SQLiteCatalog
    from app.storage.outbox import SQLiteOutbox

    monkeypatch.setattr(api, "graph_db", None)
    monkeypatch.setattr(api, "catalog", SQLiteCatalog(tmp_path / "catalog.sqlite3"))
    monkeypatch.setattr(api, "outbox", SQLiteOutbox(tmp_path / "outbox.sqlite3"))
    monkeypatch.setattr(api, "retrieval_engine", RetrievalEngine())
    monkeypatch.setattr(api.retrieval_engine, "dense_retrieve", lambda question, top_k=20: [])
    api.DOCUMENTS.clear()
    api.CHUNKS.clear()

    client = TestClient(api.app)
    sample = (
        "Experiment: EXP-VT6-AN. Material: сплав ВТ6. Process: отжиг at 750 C for 2 h. "
        "Property: прочность. Result: прочность decreased to 980 MPa. "
        "Equipment: Вакуумная печь SNOL-75. Laboratory: Лаборатория легких сплавов. "
        "Conclusion: отжиг ВТ6 снижает прочность."
    ).encode("utf-8")

    ingest = client.post(
        "/ingest/documents",
        files=[("files", ("vt6.txt", sample, "text/plain"))],
    )
    assert ingest.status_code == 200, ingest.text

    response = client.post(
        "/ask",
        params={"question": "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?", "top_k": 5},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["answer"]
    assert payload["facts"]
    assert payload["sources"]
    assert "gaps" in payload
    assert payload["subgraph"]["nodes"]
    assert payload["subgraph"]["edges"]
    payload_text = str(payload)
    assert "ВТ6" in payload_text
    assert "Прочность" in payload_text


def test_graph_subgraph_fallback_without_neo4j(tmp_path: Path, monkeypatch) -> None:
    import app.api as api
    from app.retrieval.retrieval import RetrievalEngine
    from app.storage.catalog import SQLiteCatalog
    from app.storage.outbox import SQLiteOutbox

    monkeypatch.setattr(api, "graph_db", None)
    monkeypatch.setattr(api, "catalog", SQLiteCatalog(tmp_path / "catalog.sqlite3"))
    monkeypatch.setattr(api, "outbox", SQLiteOutbox(tmp_path / "outbox.sqlite3"))
    monkeypatch.setattr(api, "retrieval_engine", RetrievalEngine())
    monkeypatch.setattr(api.retrieval_engine, "dense_retrieve", lambda question, top_k=20: [])
    api.DOCUMENTS.clear()
    api.CHUNKS.clear()

    client = TestClient(api.app)
    sample = "Experiment: EXP-STEEL-Q. Material: сталь 12Х18Н10Т. Process: закалка at 1050 C. Твёрдость increased to 210 HV.".encode("utf-8")
    ingest = client.post("/ingest/documents", files=[("files", ("steel.txt", sample, "text/plain"))])
    assert ingest.status_code == 200, ingest.text

    response = client.get("/graph/subgraph", params={"entity_ids": "12Х18Н10Т", "hops": 1})
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["nodes"]
    assert payload["edges"]

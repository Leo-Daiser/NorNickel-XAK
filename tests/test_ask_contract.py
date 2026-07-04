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


def test_ask_returns_required_contract(tmp_path: Path) -> None:
    api = _reset_api(tmp_path)
    client = TestClient(api.app)
    sample = (
        "Клапан DN50. Параметры: DN50, PN16, рабочая температура от -40 до +120 °C. "
        "Материал корпуса 12Х18Н10Т. Стандарт ГОСТ 33259."
    ).encode("utf-8")
    ingest = client.post("/ingest/documents", files=[("files", ("valve.txt", sample, "text/plain"))])
    assert ingest.status_code == 200
    response = client.post("/ask", params={"question": "Какие параметры указаны для клапана DN50?", "top_k": 5})
    assert response.status_code == 200
    payload = response.json()
    for key in ["answer", "facts", "technical_objects", "parts", "parameters", "standards", "materials", "requirements", "sources", "gaps", "subgraph"]:
        assert key in payload
    assert payload["facts"]
    assert payload["sources"]
    assert payload["subgraph"]["nodes"]
    assert payload["subgraph"]["edges"]
    assert "PN16" in str(payload)


def test_ask_rejects_unintelligible_question_without_random_chunks(tmp_path: Path) -> None:
    api = _reset_api(tmp_path)
    client = TestClient(api.app)
    sample = (
        "Experiment: EXP-VT6-AN. Material: сплав ВТ6. Process: отжиг at 750 C for 2 h. "
        "Property: прочность. Result: прочность decreased to 980 MPa."
    ).encode("utf-8")
    ingest = client.post("/ingest/documents", files=[("files", ("vt6.txt", sample, "text/plain"))])
    assert ingest.status_code == 200

    response = client.post("/ask", params={"question": "чччто там как где пупупу?", "top_k": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_mode"] == "needs_clarification"
    assert payload["intent"] == "clarification"
    assert payload["facts"] == []
    assert payload["sources"] == []
    assert payload["subgraph"] == {"nodes": [], "edges": []}
    assert "уточните вопрос" in payload["answer"].lower()


def test_broad_technical_retrieval_hit_uses_source_grounded_partial(tmp_path: Path) -> None:
    api = _reset_api(tmp_path)
    client = TestClient(api.app)
    sample = (
        "Охлаждение глубоких подземных рудников. Источники тепла включают самосжатие воздуха, "
        "геотермальный поток и тепло от оборудования. Для охлаждения применяют вентиляцию, "
        "холодильные установки и подачу охлаждённого воздуха."
    ).encode("utf-8")
    ingest = client.post("/ingest/documents", files=[("files", ("deep_mine_cooling.txt", sample, "text/plain"))])
    assert ingest.status_code == 200

    response = client.post("/ask", params={"question": "Какие способы охлаждения применяются для глубоких рудников?", "top_k": 5})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "partial"
    assert payload["answer_mode"] == "source_grounded_answer"
    assert payload["answer_is_verified"] is False
    assert payload["source_grounded_answer_used"] is True
    assert payload["evidence_chunks_used_count"] > 0
    assert payload["retrieval"].get("chunks_returned", 0) > 0 or payload["retrieval"].get("chunks_after_fusion", 0) > 0
    assert "Подтверждённые экспериментальные данные" not in payload["answer"]
    assert "Ответ по найденным источникам" in payload["answer"]
    assert payload["facts"] == []


def test_bare_pump_query_returns_object_overview_not_fake_experiment(tmp_path: Path) -> None:
    api = _reset_api(tmp_path)
    client = TestClient(api.app)
    sample = (
        "Технический объект: насос NPK-200. Параметры насоса NPK-200: производительность 120 м3/ч, "
        "P=10 MPa, рабочая температура T=300 C. Материал корпуса насоса: 09Г2С. "
        "Стандарт системы качества: ISO 9001. "
        "object: насос NPK-200 | part: корпус | article_number: ART-NPK-200-BODY | material: 09Г2С | parameter: P=10 MPa"
    ).encode("utf-8")
    ingest = client.post("/ingest/documents", files=[("files", ("pump.txt", sample, "text/plain"))])
    assert ingest.status_code == 200

    response = client.post("/ask", params={"question": "насос", "top_k": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "object_overview"
    assert "NPK-200" in payload["answer"]
    assert "09Г2С" in payload["answer"]
    assert "ART-NPK-200-BODY" in payload["answer"]
    assert "прочность: 10 MPa" not in str(payload)
    assert "режим обработки" not in payload["answer"].lower()


def test_url_ingestion_with_mocked_html(tmp_path: Path, monkeypatch) -> None:
    api = _reset_api(tmp_path)
    client = TestClient(api.app)
    import app.security.url_safety as url_safety
    monkeypatch.setattr(url_safety, "_resolve_host", lambda host, port: {url_safety.ipaddress.ip_address("93.184.216.34")})

    class FakeResponse:
        content = b'<html><body><h1>Valve</h1><p>Valve DN50 PN16 ISO 5208</p></body></html>'
        headers = {"content-type": "text/html"}

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(api.requests, "get", lambda *args, **kwargs: FakeResponse())
    response = client.post("/ingest/url", params={"url": "https://example.org/valve.html"})
    assert response.status_code == 200, response.text
    ask = client.post("/ask", params={"question": "Какие параметры указаны для valve DN50?", "top_k": 3})
    assert ask.status_code == 200
    assert "DN50" in str(ask.json())

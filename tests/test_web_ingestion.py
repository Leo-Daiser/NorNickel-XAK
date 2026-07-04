from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.strict_qa_helpers import reset_api


HTML_V1 = """
<html>
  <head><title>VT6 annealing study</title></head>
  <body>
    <h1>Исследование ВТ6 после отжига</h1>
    <p>После отжига сплава ВТ6 предел прочности составил 980 MPa.</p>
    <p>Источник указывает на необходимость дополнительных данных по коррозионной стойкости.</p>
  </body>
</html>
""".encode("utf-8")

HTML_V2 = """
<html>
  <head><title>VT6 annealing study</title></head>
  <body>
    <h1>Исследование ВТ6 после отжига</h1>
    <p>После отжига сплава ВТ6 предел прочности составил 1120 MPa.</p>
  </body>
</html>
""".encode("utf-8")


class FakeHtmlResponse:
    headers = {"content-type": "text/html; charset=utf-8"}
    is_redirect = False
    is_permanent_redirect = False

    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def _allow_example_org(monkeypatch) -> None:
    import app.security.url_safety as url_safety

    monkeypatch.setattr(
        url_safety,
        "_resolve_host",
        lambda host, port: {url_safety.ipaddress.ip_address("93.184.216.34")},
    )


def _install_html_fetch(api, monkeypatch, content: bytes) -> None:
    monkeypatch.setattr(api.requests, "get", lambda *args, **kwargs: FakeHtmlResponse(content))


def test_url_ingestion_saves_first_class_source_metadata_and_expansion(tmp_path: Path, monkeypatch) -> None:
    api = reset_api(tmp_path)
    _allow_example_org(monkeypatch)
    _install_html_fetch(api, monkeypatch, HTML_V1)
    client = TestClient(api.app)

    response = client.post("/ingest/url", params={"url": "https://example.org/reports/vt6-annealing.html?utm_source=demo"})

    assert response.status_code == 200, response.text
    item = response.json()["ingested"]
    assert item["source_name"] == "VT6 annealing study"
    assert item["url"] == "https://example.org/reports/vt6-annealing.html?utm_source=demo"
    assert item["chunks"] > 0
    assert item["knowledge_expansion"]["new_canonical_facts_count"] >= 1
    assert item["knowledge_expansion"]["data_gaps_added_count"] >= 1

    metadata = api.catalog.get_document_metadata(item["doc_id"])
    assert metadata["source_type"] == "url"
    assert metadata["source_url"].startswith("https://example.org/reports/vt6-annealing.html")
    assert metadata["source_name"] == "VT6 annealing study"
    assert metadata["content_hash"]
    assert metadata["ingested_at"]
    assert metadata["active"] is True

    chunks = api.catalog.list_chunks(item["doc_id"])
    assert chunks
    assert chunks[0].metadata["source_type"] == "url"
    assert chunks[0].metadata["source_url"].startswith("https://example.org/reports/vt6-annealing.html")
    assert chunks[0].metadata["source_name"] == "VT6 annealing study"

    report = client.get("/knowledge/expansion-report").json()
    assert "ВТ6" in report["materials"]
    assert report["facts_without_evidence"] == 0

    answer = client.post(
        "/ask",
        json={"question": "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?", "preset_id": "strict_audit"},
    )
    assert answer.status_code == 200
    payload = answer.json()
    rendered = json.dumps(payload, ensure_ascii=False)
    assert payload["status"] == "ok"
    assert "980 MPa" in rendered
    assert "doc_" not in str(payload.get("answer", ""))
    assert "chunk_" not in str(payload.get("answer", ""))


def test_same_url_content_is_idempotent_and_changed_content_versions(tmp_path: Path, monkeypatch) -> None:
    api = reset_api(tmp_path)
    _allow_example_org(monkeypatch)
    _install_html_fetch(api, monkeypatch, HTML_V1)
    client = TestClient(api.app)
    url = "https://example.org/reports/vt6-annealing.html"

    first = client.post("/ingest/url", params={"url": url}).json()["ingested"]
    second = client.post("/ingest/url", params={"url": url}).json()["ingested"]
    assert second["doc_id"] == first["doc_id"]
    assert second["knowledge_expansion"]["new_canonical_facts_count"] == 0
    before_count = client.get("/knowledge/summary").json()["canonical_facts_count"]

    _install_html_fetch(api, monkeypatch, HTML_V2)
    changed = client.post("/ingest/url", params={"url": url}).json()["ingested"]
    assert changed["doc_id"] != first["doc_id"]
    assert changed["document_version"] == 2
    assert client.get("/knowledge/summary").json()["canonical_facts_count"] >= before_count


def test_deactivating_url_document_excludes_and_reactivates_answer(tmp_path: Path, monkeypatch) -> None:
    api = reset_api(tmp_path)
    _allow_example_org(monkeypatch)
    _install_html_fetch(api, monkeypatch, HTML_V1)
    client = TestClient(api.app)

    ingested = client.post("/ingest/url", params={"url": "https://example.org/reports/vt6-annealing.html"}).json()["ingested"]
    question = "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?"

    positive = client.post("/ask", json={"question": question, "preset_id": "strict_audit"})
    assert positive.status_code == 200
    assert positive.json()["status"] == "ok"

    off = client.patch(f"/documents/{ingested['doc_id']}/active", json={"active": False})
    assert off.status_code == 200
    negative = client.post("/ask", json={"question": question, "preset_id": "strict_audit"})
    assert negative.status_code == 200
    assert negative.json()["status"] == "no_exact_match"

    on = client.patch(f"/documents/{ingested['doc_id']}/active", json={"active": True})
    assert on.status_code == 200
    restored = client.post("/ask", json={"question": question, "preset_id": "strict_audit"})
    assert restored.status_code == 200
    assert restored.json()["status"] == "ok"

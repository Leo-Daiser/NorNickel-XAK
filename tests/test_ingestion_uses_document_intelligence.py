from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.ingestion.document_models import DocumentIntelligenceResult, ParsedBlock, ParsedTable
from app.models.schemas import Chunk


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


class FakeParser:
    def __init__(self) -> None:
        self.called = False

    def parse_document_intelligence(self, file_path: str, doc_id: str | None = None, source_type: str = "file", source_url: str | None = None):
        self.called = True
        block = ParsedBlock(block_id="b1", doc_id=doc_id or "doc", block_type="table", text="table", ordinal=0)
        table = ParsedTable(table_id="t1", doc_id=doc_id or "doc", headers=["material"], rows=[["ВТ6"]], source_block_id="b1")
        chunk = Chunk(
            chunk_id="c1",
            doc_id=doc_id or "doc",
            text="Table columns: material\nmaterial: ВТ6",
            page_start=1,
            page_end=1,
            section_path="/",
            metadata={"chunk_kind": "table_row", "table_id": "t1", "row_id": "0", "parser_name": "fake"},
        )
        return DocumentIntelligenceResult(
            doc_id=doc_id or "doc",
            source_name="fake.csv",
            source_type=source_type,
            parser_name="fake",
            text="Технический отчет 2023: российская практика электроэкстракции никеля.",
            blocks=[block],
            tables=[table],
            chunks=[chunk],
            diagnostics={"parser_backend_requested": "fallback", "parser_backend_used": "fake"},
        )


def test_ingestion_uses_document_intelligence_parser(tmp_path: Path, monkeypatch) -> None:
    api = _reset_api(tmp_path)
    fake_parser = FakeParser()
    monkeypatch.setattr(api, "parser_router", fake_parser)
    client = TestClient(api.app)

    response = client.post("/ingest/documents", files=[("files", ("experiments.csv", b"material\nVT6\n", "text/csv"))])

    assert response.status_code == 200
    assert fake_parser.called is True
    doc_id = response.json()["ingested"][0]["doc_id"]
    chunks = api.catalog.list_chunks(doc_id)
    metadata = api.catalog.get_document_metadata(doc_id)
    assert chunks[0].metadata["chunk_kind"] == "table_row"
    assert chunks[0].metadata["table_id"] == "t1"
    assert metadata["parser_diagnostics"]["parser_backend_used"] == "fake"
    assert metadata["document_intelligence"]["tables_count"] == 1
    assert metadata["source_metadata"]["publication_year"] == 2023
    assert metadata["source_metadata"]["practice_scope"] == "domestic"
    assert chunks[0].metadata["publication_year"] == 2023
    assert chunks[0].metadata["reliability_level"] in {"medium", "unknown"}

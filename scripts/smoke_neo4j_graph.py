from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.extraction.extraction import EntityRelationExtractor  # noqa: E402
from app.graph.graph_db import GraphDB  # noqa: E402
from app.graph.graph_writer import sync_catalog_to_neo4j  # noqa: E402
from app.graph.neo4j_client import apply_schema  # noqa: E402
from app.graph.neo4j_repository import Neo4jGraphRepository  # noqa: E402
from app.storage.catalog import SQLiteCatalog  # noqa: E402
from app.storage.outbox import SQLiteOutbox  # noqa: E402
from app.retrieval.retrieval import RetrievalEngine  # noqa: E402


def _load_demo_into_temp_catalog(tmp: Path) -> SQLiteCatalog:
    import app.api as api

    api.graph_db = None
    api.catalog = SQLiteCatalog(tmp / "catalog.sqlite3")
    api.outbox = SQLiteOutbox(tmp / "outbox.sqlite3")
    api.retrieval_engine = RetrievalEngine()
    api.retrieval_engine.dense_retrieve = lambda question, top_k=20: []
    api.DOCUMENTS.clear()
    api.CHUNKS.clear()
    client = TestClient(api.app)
    allowed = {".csv", ".xlsx", ".txt", ".html", ".htm", ".docx", ".md"}
    files = [
        ("files", (path.name, path.read_bytes(), "application/octet-stream"))
        for path in sorted((ROOT / "demo_data").iterdir())
        if path.suffix.lower() in allowed
    ]
    response = client.post("/ingest/documents", files=files)
    if response.status_code != 200:
        raise RuntimeError(f"demo ingestion failed: {response.status_code} {response.text}")
    return api.catalog


def _load_smoke_catalog(tmp: Path) -> tuple[SQLiteCatalog, str]:
    demo_dir = ROOT / "demo_data"
    if demo_dir.exists():
        return _load_demo_into_temp_catalog(tmp), "demo"
    catalog = SQLiteCatalog(settings.catalog_db_path)
    if not catalog.list_documents():
        raise RuntimeError(f"catalog is empty and {demo_dir} is unavailable")
    return catalog, "current_catalog"


def main() -> int:
    try:
        graph_db = GraphDB()
    except Exception as exc:
        print(f"Neo4j unavailable: {exc}")
        return 2
    try:
        apply_schema(graph_db)
        with tempfile.TemporaryDirectory() as tmp:
            catalog, corpus_mode = _load_smoke_catalog(Path(tmp))
            sync_catalog_to_neo4j(graph_db=graph_db, catalog=catalog, extractor=EntityRelationExtractor(), document_getter=catalog.get_document)

        repository = Neo4jGraphRepository(graph_db)
        if corpus_mode == "demo":
            exact = repository.find_exact_material_regime_property("ВТ6", "отжиг", "прочность")
            if not exact:
                print("Neo4j graph smoke failed: exact ВТ6 + отжиг + прочность not found")
                return 1
            if not any(
                measurement.value_normalized is not None
                and measurement.unit_normalized
                and measurement.value_original is not None
                and measurement.unit_original
                and measurement.normalization_family
                for fact in exact
                for measurement in fact.measurements
            ):
                print("Neo4j graph smoke failed: normalized measurement fields missing after read")
                return 1
        else:
            chunk_rows = graph_db.run("MATCH (:Document)-[:HAS_CHUNK]->(:DocumentChunk) RETURN count(*) AS chunks")
            if not chunk_rows or int(chunk_rows[0]["chunks"] or 0) <= 0:
                print("Neo4j graph smoke failed: current catalog chunks were not projected")
                return 1
        normalized_rows = graph_db.run(
            """
            MATCH (meas:Measurement)
            WHERE meas.value IS NOT NULL
            RETURN
              count(meas) AS total,
              sum(CASE WHEN
                meas.value_original IS NULL OR
                meas.unit_original IS NULL OR
                meas.value_normalized IS NULL OR
                meas.unit_normalized IS NULL OR
                meas.normalization_family IS NULL
              THEN 1 ELSE 0 END) AS missing_normalized
            """
        )
        if normalized_rows:
            total = int(normalized_rows[0]["total"] or 0)
            missing = int(normalized_rows[0]["missing_normalized"] or 0)
            if total > 0 and missing > 0:
                print(f"Neo4j graph smoke failed: {missing}/{total} Measurement nodes lack normalized fields")
                return 1
        evidence_rows = graph_db.run(
            """
            MATCH (:Measurement)-[:SUPPORTED_BY]->(:DocumentChunk)
            RETURN count(*) AS evidence_edges
            """
        )
        if total > 0 and (not evidence_rows or int(evidence_rows[0]["evidence_edges"] or 0) <= 0):
            print("Neo4j graph smoke failed: measurement evidence links are missing")
            return 1
        if corpus_mode == "demo":
            missing = repository.find_exact_material_regime_property("ВТ6", "криообработка", "вязкость")
            if missing:
                print("Neo4j graph smoke failed: missing ВТ6 + криообработка + вязкость returned exact facts")
                return 1
        print("NEO4J GRAPH SMOKE TEST PASSED")
        return 0
    except Exception as exc:
        print(f"Neo4j graph smoke failed: {exc}")
        return 1
    finally:
        graph_db.close()


if __name__ == "__main__":
    raise SystemExit(main())

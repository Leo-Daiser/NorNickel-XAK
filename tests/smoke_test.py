"""End-to-end smoke test for the hackathon app.

Run from repository root:
    python -m tests.smoke_test

The test intentionally uses temporary SQLite files and does not require
Neo4j, Qdrant, Docling or embedding models.
"""

from __future__ import annotations

import os
import tempfile

from fastapi.testclient import TestClient


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["METADATA_DB_PATH"] = os.path.join(tmp, "outbox.sqlite3")
        os.environ["CATALOG_DB_PATH"] = os.path.join(tmp, "catalog.sqlite3")
        os.environ["DIRECT_QDRANT_PROJECTION"] = "false"

        try:
            from hackathon_project.app.api import app
        except ModuleNotFoundError:
            from app.api import app

        client = TestClient(app)
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["status"] == "ok"

        sample_text = (
            "Alloy X was annealed at 900 C. "
            "Strength increased to 1200 MPa after processing. "
            "Alloy X retained good hardness after aging. "
        ).encode("utf-8")

        ingest = client.post(
            "/ingest/documents",
            files=[("files", ("sample.txt", sample_text, "text/plain"))],
        )
        assert ingest.status_code == 200, ingest.text
        body = ingest.json()
        assert body["ingested"][0]["chunks"] >= 1
        doc_id = body["ingested"][0]["doc_id"]

        # Re-ingestion must be idempotent by document hash.
        ingest_again = client.post(
            "/ingest/documents",
            files=[("files", ("sample.txt", sample_text, "text/plain"))],
        )
        assert ingest_again.status_code == 200, ingest_again.text
        assert ingest_again.json()["ingested"][0]["doc_id"] == doc_id

        docs = client.get("/documents")
        assert docs.status_code == 200, docs.text
        matching_docs = [doc for doc in docs.json() if doc["doc_id"] == doc_id]
        assert len(matching_docs) == 1, docs.json()

        chunks = client.get(f"/documents/{doc_id}/chunks")
        assert chunks.status_code == 200, chunks.text
        assert chunks.json(), "Expected persisted chunks"

        query = client.post("/query", params={"question": "Alloy X strength", "top_k": 3})
        assert query.status_code == 200, query.text
        assert query.json(), "Expected at least one retrieved chunk"
        assert query.json()[0]["doc_id"] == doc_id

        debug = client.get("/debug/retrieval", params={"question": "Alloy X strength", "top_k": 3})
        assert debug.status_code == 200, debug.text
        assert debug.json(), "Expected lexical debug candidates"

        pending = client.get("/sync/outbox/pending", params={"limit": 10})
        assert pending.status_code == 200, pending.text
        # Since events are deduplicated by chunk hash, duplicate ingestion should not double count.
        assert len(pending.json()) == len(chunks.json()), pending.json()

        rebuild = client.post("/admin/rebuild-index")
        assert rebuild.status_code == 200, rebuild.text
        assert rebuild.json()["status"] == "rebuilt"

        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

from __future__ import annotations

from app.extraction.models import (
    EvidenceSpan,
    ExtractionBundle,
    ExtractionSource,
    ExtractedEntity,
    ExtractedExperiment,
    ExtractedMeasurement,
    ExtractedRegime,
    RejectedExtraction,
)
from app.graph.graph_writer import GraphWriter
from app.models.schemas import Chunk, Document
from app.storage.catalog import SQLiteCatalog


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params):
        self.calls.append((query, params))
        return []


class FakeGraphDB:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    def session(self):
        session = self._session

        class Context:
            def __enter__(self):
                return session

            def __exit__(self, exc_type, exc, tb):
                return False

        return Context()


class FakePipeline:
    def __init__(self) -> None:
        self.calls = 0

    def extract_from_chunk(self, chunk: Chunk) -> ExtractionBundle:
        self.calls += 1
        source = ExtractionSource(document_id=chunk.doc_id, chunk_id=chunk.chunk_id, source_name="sync-test.txt", page=1)
        evidence = [EvidenceSpan(source=source, quote=chunk.text)]
        return ExtractionBundle(
            document_id=chunk.doc_id,
            source_name="sync-test.txt",
            extractor_version="fake",
            entities=[
                ExtractedEntity(
                    entity_type="Material",
                    raw_name="ВТ6",
                    canonical_name="ВТ6",
                    confidence=0.9,
                    evidence=evidence,
                )
            ],
            experiments=[
                ExtractedExperiment(
                    experiment_id="EXP-ACCEPTED",
                    materials=[
                        ExtractedEntity(
                            entity_type="Material",
                            raw_name="ВТ6",
                            canonical_name="ВТ6",
                            confidence=0.9,
                            evidence=evidence,
                        )
                    ],
                    regimes=[ExtractedRegime(raw_name="отжиг", canonical_name="отжиг", confidence=0.8, evidence=evidence)],
                    measurements=[
                        ExtractedMeasurement(
                            property_raw="прочность",
                            property_canonical="прочность",
                            value=1120.0,
                            unit="MPa",
                            effect="increase",
                            confidence=0.9,
                            evidence=evidence,
                        )
                    ],
                    evidence=evidence,
                    confidence=0.95,
                )
            ],
            rejected_items=[
                RejectedExtraction(item_type="experiment", reason="missing_material", raw_payload={"experiment_id": "EXP-REJECTED"})
            ],
        )


def test_sync_uses_extraction_pipeline_and_skips_rejected(tmp_path) -> None:
    catalog = SQLiteCatalog(tmp_path / "catalog.sqlite3")
    document = Document(doc_id="doc", workspace_uid="test", title="sync-test.txt", parser="test", status="ready")
    chunk = Chunk(
        chunk_id="chunk",
        doc_id="doc",
        workspace_uid="test",
        text="Эксперимент EXP-ACCEPTED: ВТ6 после отжига показал прочность 1120 MPa.",
        page_start=1,
        page_end=1,
        section_path="test",
    )
    catalog.upsert_document(document)
    catalog.replace_chunks(document.doc_id, [chunk])
    session = FakeSession()
    pipeline = FakePipeline()

    stats = GraphWriter(FakeGraphDB(session), pipeline=pipeline).sync_catalog(catalog)  # type: ignore[arg-type]

    assert pipeline.calls == 1
    assert stats["accepted_experiments"] == 1
    assert stats["accepted_measurements"] == 1
    assert stats["rejected_items"] == 1
    assert stats["mean_confidence"] > 0
    written_values = {str(value) for _, params in session.calls for value in params.values()}
    assert "EXP-ACCEPTED" in written_values
    assert "EXP-REJECTED" not in written_values

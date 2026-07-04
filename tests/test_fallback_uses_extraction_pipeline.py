from __future__ import annotations

from pathlib import Path


def test_fallback_repository_uses_extraction_pipeline(tmp_path: Path) -> None:
    from app.extraction.models import (
        EvidenceSpan,
        ExtractedEntity,
        ExtractedExperiment,
        ExtractedMeasurement,
        ExtractedRegime,
        ExtractionBundle,
        ExtractionSource,
        RejectedExtraction,
    )
    from app.graph.graph_repository import CatalogGraphRepository
    from app.models.schemas import Chunk, Document
    from app.storage.catalog import SQLiteCatalog

    source = ExtractionSource(document_id="doc1", chunk_id="chunk1", source_name="sample.txt")
    evidence = EvidenceSpan(source=source, quote="ВТ6 отжиг прочность 1120 МПа")
    accepted_experiment = ExtractedExperiment(
        experiment_id="EXP-PIPELINE-1",
        materials=[ExtractedEntity(entity_type="Material", raw_name="ВТ6", canonical_name="ВТ6", confidence=0.9, evidence=[evidence])],
        regimes=[ExtractedRegime(raw_name="отжиг", canonical_name="отжиг", confidence=0.9, evidence=[evidence])],
        measurements=[
            ExtractedMeasurement(
                property_raw="прочность",
                property_canonical="прочность",
                value=1120,
                unit="MPa",
                confidence=0.9,
                evidence=[evidence],
            )
        ],
        evidence=[evidence],
        confidence=0.9,
    )

    class FakePipeline:
        calls = 0

        def extract_from_chunk(self, chunk):
            self.calls += 1
            return ExtractionBundle(
                document_id=chunk.doc_id,
                source_name="sample.txt",
                extractor_version="fake",
                experiments=[accepted_experiment],
                rejected_items=[RejectedExtraction(item_type="experiment", reason="missing_material", raw_payload={"bad": True})],
            )

    catalog = SQLiteCatalog(tmp_path / "catalog.sqlite3")
    doc = Document(doc_id="doc1", workspace_uid="w", title="sample.txt", parser="txt")
    chunk = Chunk(doc_id="doc1", chunk_id="chunk1", text="legacy noise", page_start=1, page_end=1, section_path="/")
    catalog.upsert_document(doc, metadata={"filename": "sample.txt"})
    catalog.replace_chunks("doc1", [chunk])

    pipeline = FakePipeline()
    repository = CatalogGraphRepository(catalog=catalog, extraction_pipeline=pipeline)
    facts = repository.find_exact_material_regime_property("ВТ6", "отжиг", "прочность")

    assert pipeline.calls == 1
    assert len(facts) == 1
    assert facts[0].experiment_id == "EXP-PIPELINE-1"
    assert facts[0].measurements[0].value == 1120


def test_fallback_repository_projects_structured_accepted_table_facts(tmp_path: Path) -> None:
    from app.graph.graph_repository import CatalogGraphRepository
    from app.models.schemas import Chunk, Document
    from app.storage.catalog import SQLiteCatalog

    catalog = SQLiteCatalog(tmp_path / "catalog.sqlite3")
    doc = Document(doc_id="doc-table", workspace_uid="w", title="assay.csv", parser="csv")
    chunk = Chunk(
        doc_id="doc-table",
        chunk_id="row-assay",
        workspace_uid="w",
        text=(
            "Материал: Пирротиновый концентрат | Технология: автоклавное выщелачивание | "
            "Ni, %: 0,5-1"
        ),
        page_start=1,
        page_end=1,
        section_path="table",
        metadata={"filename": "assay.csv", "chunk_kind": "table_row", "row_id": 3},
    )
    catalog.upsert_document(doc, metadata={"filename": "assay.csv"})
    catalog.replace_chunks("doc-table", [chunk])

    repository = CatalogGraphRepository(catalog=catalog)
    facts = repository.find_exact_material_regime_property("Пирротиновый концентрат", "выщелачивание", "содержание")

    assert len(facts) == 1
    measurement = facts[0].measurements[0]
    assert measurement.property_name == "содержание"
    assert measurement.value == 0.5
    assert measurement.value_min == 0.5
    assert measurement.value_max == 1.0
    assert measurement.analyte == "ni"
    assert measurement.fact_type == "ProcessParameterFact"
    assert measurement.source_adapter == "structured_table_adapter"
    assert measurement.evidence


def test_fallback_repository_finds_process_only_structured_facts(tmp_path: Path) -> None:
    from app.graph.graph_repository import CatalogGraphRepository
    from app.models.schemas import Chunk, Document
    from app.storage.catalog import SQLiteCatalog

    catalog = SQLiteCatalog(tmp_path / "catalog.sqlite3")
    doc = Document(doc_id="doc-process", workspace_uid="w", title="process.csv", parser="csv")
    chunk = Chunk(
        doc_id="doc-process",
        chunk_id="row-flow",
        workspace_uid="w",
        text="process: catholyte circulation | parameter: flow rate | value: 0.5 | unit: m/s",
        page_start=1,
        page_end=1,
        section_path="table",
        metadata={"filename": "process.csv", "chunk_kind": "table_row", "row_id": 4},
    )
    catalog.upsert_document(doc, metadata={"filename": "process.csv"})
    catalog.replace_chunks("doc-process", [chunk])

    repository = CatalogGraphRepository(catalog=catalog)
    facts = repository.find_experiments(regime="циркуляция католита", property_name="скорость потока")

    assert len(facts) == 1
    assert facts[0].materials == []
    assert facts[0].regimes == ["циркуляция католита"]
    measurement = facts[0].measurements[0]
    assert measurement.property_name == "скорость потока"
    assert measurement.value == 0.5
    assert measurement.unit == "m/s"
    assert measurement.source_adapter == "structured_table_adapter"


def test_fallback_repository_finds_material_content_facts_without_regime(tmp_path: Path) -> None:
    from app.graph.graph_repository import CatalogGraphRepository
    from app.models.schemas import Chunk, Document
    from app.storage.catalog import SQLiteCatalog

    catalog = SQLiteCatalog(tmp_path / "catalog.sqlite3")
    doc = Document(doc_id="doc-assay", workspace_uid="w", title="assay.csv", parser="csv")
    chunk = Chunk(
        doc_id="doc-assay",
        chunk_id="row-assay-no-process",
        workspace_uid="w",
        text="Материал: Пирротиновый концентрат | Ni, %: 0,5-1",
        page_start=1,
        page_end=1,
        section_path="table",
        metadata={"filename": "assay.csv", "chunk_kind": "table_row", "row_id": 5},
    )
    catalog.upsert_document(doc, metadata={"filename": "assay.csv"})
    catalog.replace_chunks("doc-assay", [chunk])

    repository = CatalogGraphRepository(catalog=catalog)
    facts = repository.find_experiments(material="Пирротиновый концентрат", property_name="содержание")

    assert len(facts) == 1
    assert facts[0].materials == ["Пирротиновый концентрат"]
    assert facts[0].regimes == []
    measurement = facts[0].measurements[0]
    assert measurement.property_name == "содержание"
    assert measurement.analyte == "ni"
    assert measurement.value_min == 0.5
    assert measurement.value_max == 1.0

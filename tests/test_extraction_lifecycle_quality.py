from __future__ import annotations

from pathlib import Path

from app.extraction.pipeline import ExtractionPipeline
from app.extraction.quality_report import build_extraction_quality_report
from app.models.schemas import Chunk, Document
from app.storage.catalog import SQLiteCatalog


def _chunk(text: str, chunk_id: str = "chunk") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="/",
        metadata={"filename": "fixture.txt", "source_name": "fixture.txt"},
    )


def test_pipeline_exposes_candidate_accepted_and_quarantine_lifecycle() -> None:
    pipeline = ExtractionPipeline(mode="deterministic", audit_enabled=False)
    bundle = pipeline.extract_from_chunk(_chunk("Эксперимент: сплав ВТ6 после отжига; относительное удлинение составило 14%."))

    assert bundle.candidate_facts
    assert bundle.accepted_facts
    assert all(item.evidence for item in bundle.accepted_facts)
    assert bundle.diagnostics["fact_lifecycle"]["accepted_facts_count"] == len(bundle.accepted_facts)


def test_uncertain_candidate_is_quarantined_not_materialized_as_accepted_fact() -> None:
    pipeline = ExtractionPipeline(mode="deterministic", audit_enabled=False)
    bundle = pipeline.extract_from_chunk(_chunk("Сплав ВТ6: значение 17% указано в таблице без маркера свойства."))

    accepted_measurements = [item for exp in bundle.experiments for item in exp.measurements]

    assert not accepted_measurements
    assert bundle.quarantined_items or bundle.rejected_items
    assert not any(item.normalized_fact.get("value") == 17.0 for item in bundle.accepted_facts)


def test_quality_report_counts_lifecycle_and_keeps_evidence(tmp_path: Path) -> None:
    catalog = SQLiteCatalog(tmp_path / "catalog.sqlite3")
    document = Document(
        doc_id="doc",
        workspace_uid="test",
        title="fixture.txt",
        parser="test",
        status="ok",
    )
    catalog.upsert_document(document, metadata={"active": True, "filename": "fixture.txt"})
    catalog.replace_chunks(
        "doc",
        [
            _chunk("Эксперимент: сплав ВТ6 после отжига; предел прочности составил 980 MPa.", "chunk_1"),
            _chunk("Directory of Copper Mines and Plants. Production capacity data: copper 35% summary country basis.", "chunk_2"),
        ],
    )

    report = build_extraction_quality_report(
        catalog,
        pipeline=ExtractionPipeline(mode="deterministic", audit_enabled=False),
    )

    assert report["documents_processed"] == 1
    assert report["chunks_processed"] == 2
    assert report["candidate_facts_count"] >= report["accepted_facts_count"]
    assert report["facts_without_evidence"] == 0
    assert report["accepted_by_fact_type"]
    assert "rejected_by_reason" in report
    assert "quarantine_by_reason" in report
    assert "rejected_by_extractor" in report
    assert "quarantine_by_extractor" in report
    assert "rejected_by_intended_fact_type" in report
    assert "quarantine_by_intended_fact_type" in report
    assert "missing_material_by_intended_fact_type" in report
    assert "unknown_property_schema_examples" in report
    assert "material_without_positive_validation_examples" in report


def test_quality_report_counts_structured_table_assay_properties(tmp_path: Path) -> None:
    catalog = SQLiteCatalog(tmp_path / "catalog.sqlite3")
    document = Document(
        doc_id="doc",
        workspace_uid="test",
        title="table.csv",
        parser="test",
        status="ok",
    )
    catalog.upsert_document(document, metadata={"active": True, "filename": "table.csv"})
    catalog.replace_chunks(
        "doc",
        [
            Chunk(
                chunk_id="table_row_1",
                doc_id="doc",
                workspace_uid="test",
                text=(
                    "Table: assay\n"
                    "Table columns: column_1 | Технология | Ni, % | Cu, %\n"
                    "column_1: Пирротиновый концентрат | Технология: выщелачивание | Ni, %: 0,5-1 | Cu, %: 0,1-0,2"
                ),
                page_start=1,
                page_end=1,
                section_path="/",
                metadata={"filename": "table.csv", "source_name": "table.csv", "chunk_kind": "table_row"},
            )
        ],
    )

    report = build_extraction_quality_report(
        catalog,
        pipeline=ExtractionPipeline(mode="deterministic", audit_enabled=False),
    )

    assert report["accepted_by_fact_type"]["ProcessParameterFact"] >= 2
    assert report["top_accepted_properties"]["содержание"] >= 2
    assert report["facts_without_evidence"] == 0

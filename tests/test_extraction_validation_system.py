from __future__ import annotations

from pathlib import Path

from app.extraction.models import EvidenceSpan, ExtractedEntity, ExtractedExperiment, ExtractedMeasurement, ExtractionSource
from app.extraction.pipeline import ExtractionPipeline
from app.extraction.validators import validate_entity, validate_experiment
from app.models.schemas import Chunk


def _evidence(text: str) -> EvidenceSpan:
    return EvidenceSpan(
        source=ExtractionSource(document_id="doc", chunk_id="chunk", source_name="fixture.txt"),
        quote=text,
        confidence=0.9,
    )


def _chunk(text: str, filename: str = "fixture.txt") -> Chunk:
    return Chunk(
        chunk_id="chunk",
        doc_id="doc",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="/",
        metadata={"filename": filename, "source_name": filename},
    )


def _bundle(text: str, filename: str = "fixture.txt"):
    return ExtractionPipeline(mode="deterministic", audit_enabled=False).extract_from_chunk(_chunk(text, filename))


def test_pdf_font_code_like_token_is_rejected_as_material_class() -> None:
    evidence = _evidence("/MT255 12 0 R /FontDescriptor /CIDFontType2 /Encoding Identity-H")
    entity = ExtractedEntity(entity_type="Material", raw_name="MT255", canonical_name="MT255", confidence=0.8, evidence=[evidence])

    accepted, rejected = validate_entity(entity)

    assert accepted is None
    assert rejected is not None
    assert rejected.reason == "pdf_font_code_without_domain_context"


def test_chemical_formula_is_valid_entity_but_not_mechanical_plasticity_subject() -> None:
    evidence = _evidence("SO2 concentration was measured; ductility was not a material-sample property here.")
    entity = ExtractedEntity(entity_type="Material", raw_name="SO2", canonical_name="SO2", confidence=0.8, evidence=[evidence])
    measurement = ExtractedMeasurement(
        property_raw="ductility",
        property_canonical="пластичность",
        value=12.0,
        unit="%",
        confidence=0.8,
        evidence=[evidence],
    )
    experiment = ExtractedExperiment(
        experiment_id="exp",
        materials=[entity],
        regimes=[],
        measurements=[measurement],
        evidence=[evidence],
        confidence=0.8,
    )

    accepted_entity, entity_rejection = validate_entity(entity)
    accepted_experiment, experiment_rejections = validate_experiment(experiment, min_confidence=0.55)

    assert accepted_entity is not None
    assert entity_rejection is None
    assert accepted_experiment is None
    assert any(item.reason == "chemical_substance_incompatible_with_mechanical_property" for item in experiment_rejections)


def test_bare_percent_does_not_infer_plasticity() -> None:
    bundle = _bundle("Directory of Copper Mines and Plants. Production capacity data: copper 35% summary country basis.")

    measurements = [item for experiment in bundle.experiments for item in experiment.measurements]

    assert all(item.property_canonical != "пластичность" for item in measurements)


def test_market_capacity_reference_does_not_create_mechanical_mrp_fact() -> None:
    bundle = _bundle(
        "Directory of Copper Mines and Plants. Facility-by-facility production capacity data. "
        "Copper capacity 35% by summary country basis; aging table number 12 is a reference code.",
        filename="capacity_reference.pdf",
    )

    measurements = [item for experiment in bundle.experiments for item in experiment.measurements]
    rejected_reasons = {item.reason for item in bundle.rejected_items}

    assert all(item.property_canonical not in {"пластичность", "прочность", "твёрдость"} for item in measurements)
    assert rejected_reasons & {
        "doc_type_incompatible_with_mechanical_property_fact",
        "doc_type_incompatible_with_heat_treatment_mrp",
        "all_measurements_rejected",
    }


def test_explicit_elongation_marker_can_create_plasticity_fact() -> None:
    bundle = _bundle("Эксперимент: сплав ВТ6 после отжига; относительное удлинение составило 14%.")

    assert any(
        item.property_canonical == "пластичность" and item.value == 14.0 and item.unit == "%"
        for experiment in bundle.experiments
        for item in experiment.measurements
    )


def test_schema_does_not_create_range_index_on_long_chunk_text() -> None:
    schema = Path("app/graph/schema.cypher").read_text(encoding="utf-8")

    assert "DROP INDEX chunk_text_index IF EXISTS" in schema
    assert "FOR (n:DocumentChunk) ON (n.text);" not in schema
    assert "chunk_text_hash_index" in schema
    assert "CREATE FULLTEXT INDEX chunk_fulltext" in schema


def test_code_like_material_without_reference_is_quarantined_by_feature_score() -> None:
    evidence = _evidence("В таблице указан служебный код MTS9841 рядом с процентом 12% и номером строки.")
    entity = ExtractedEntity(entity_type="Material", raw_name="MTS9841", canonical_name="MTS9841", confidence=0.8, evidence=[evidence])

    accepted, rejected = validate_entity(entity)

    assert accepted is None
    assert rejected is not None
    assert rejected.reason == "suspicious_code_like_entity_without_reference"


def test_known_material_grade_is_not_blocked_by_code_like_risk_score() -> None:
    evidence = _evidence("Сплав 7075-T6 после старения имел tensile strength 77 ksi.")
    entity = ExtractedEntity(entity_type="Material", raw_name="7075-T6", canonical_name="7075-T6", confidence=0.8, evidence=[evidence])

    accepted, rejected = validate_entity(entity)

    assert accepted is not None
    assert rejected is None

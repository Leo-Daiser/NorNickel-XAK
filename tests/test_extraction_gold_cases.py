from __future__ import annotations

from app.extraction.pipeline import ExtractionPipeline
from app.models.schemas import Chunk
from app.domain.unit_normalization import normalize_strength_to_mpa


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="chunk",
        doc_id="doc",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="test",
        metadata={"filename": "gold.txt"},
    )


def _bundle(text: str):
    return ExtractionPipeline(mode="deterministic", audit_enabled=False).extract_from_chunk(_chunk(text))


def test_english_7075_tensile_strength_ksi_is_extracted_and_normalized() -> None:
    bundle = _bundle("The 7075-T6 aluminum alloy showed tensile strength of 77 ksi after aging treatment.")
    measurements = [measurement for experiment in bundle.experiments for measurement in experiment.measurements]

    assert any(material.canonical_name == "7075-T6" for experiment in bundle.experiments for material in experiment.materials)
    assert any(regime.canonical_name == "старение" for experiment in bundle.experiments for regime in experiment.regimes)
    strength = next(item for item in measurements if item.property_canonical == "прочность" and item.unit == "ksi")
    converted, note = normalize_strength_to_mpa(strength.value, strength.unit)
    assert strength.value == 77.0
    assert converted is not None and abs(converted - 531.0) <= 1.0
    assert note == "77 ksi ≈ 531 MPa"
    assert strength.evidence and strength.evidence[0].source.chunk_id == "chunk"


def test_ti64_annealed_ultimate_tensile_strength_maps_to_vt6() -> None:
    bundle = _bundle("Ti-6Al-4V was annealed, resulting in ultimate tensile strength of 1120 MPa.")
    measurements = [measurement for experiment in bundle.experiments for measurement in experiment.measurements]

    assert any(material.canonical_name == "ВТ6" for experiment in bundle.experiments for material in experiment.materials)
    assert any(regime.canonical_name == "отжиг" for experiment in bundle.experiments for regime in experiment.regimes)
    assert any(item.property_canonical == "прочность" and item.value == 1120.0 and item.unit == "MPa" for item in measurements)


def test_corrosion_qualitative_sentence_does_not_hallucinate_numeric_value() -> None:
    bundle = _bundle("Коррозионная стойкость после обработки повысилась, но численные значения не приведены.")
    measurements = [measurement for experiment in bundle.experiments for measurement in experiment.measurements]
    properties = {entity.canonical_name for entity in bundle.entities if entity.entity_type == "Property"}

    assert "коррозионная стойкость" in properties
    assert all(measurement.value is None for measurement in measurements)

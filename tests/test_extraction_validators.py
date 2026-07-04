from __future__ import annotations

from app.extraction.models import EvidenceSpan, ExtractionSource, ExtractedExperiment, ExtractedMeasurement
from app.extraction.resolver import resolve_unit
from app.extraction.validators import validate_experiment, validate_measurement


def _evidence() -> list[EvidenceSpan]:
    return [EvidenceSpan(source=ExtractionSource(document_id="doc", chunk_id="chunk"), quote="ВТ6 отжиг прочность 1120 MPa")]


def test_experiment_without_material_rejected() -> None:
    experiment = ExtractedExperiment(
        experiment_id="EXP",
        materials=[],
        regimes=[],
        measurements=[],
        evidence=_evidence(),
        confidence=0.9,
    )

    accepted, rejected = validate_experiment(experiment, min_confidence=0.55)

    assert accepted is None
    assert rejected[0].reason == "missing_material"


def test_measurement_without_property_rejected() -> None:
    measurement = ExtractedMeasurement(property_raw="", property_canonical="", value=1.0, unit="MPa", confidence=0.9, evidence=_evidence())

    accepted, rejected = validate_measurement(measurement)

    assert accepted is None
    assert rejected is not None
    assert rejected.reason == "missing_property"


def test_measurement_without_evidence_rejected() -> None:
    measurement = ExtractedMeasurement(property_raw="прочность", property_canonical="прочность", value=1120.0, unit="MPa", confidence=0.9, evidence=[])

    accepted, rejected = validate_measurement(measurement)

    assert accepted is None
    assert rejected is not None
    assert rejected.reason == "missing_evidence"


def test_low_confidence_experiment_rejected() -> None:
    experiment = ExtractedExperiment(
        experiment_id="EXP",
        materials=[],
        regimes=[],
        measurements=[],
        evidence=_evidence(),
        confidence=0.1,
    )

    accepted, rejected = validate_experiment(experiment, min_confidence=0.55)

    assert accepted is None
    assert rejected


def test_units_normalize() -> None:
    assert resolve_unit("МПа") == "MPa"
    assert resolve_unit("ksi") == "ksi"
    assert resolve_unit("°С") == "C"
    assert resolve_unit("ч") == "h"
    assert resolve_unit("мин") == "min"

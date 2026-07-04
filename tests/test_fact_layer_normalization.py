from __future__ import annotations

from app.domain.fact_normalization import (
    build_conflict_summary,
    canonical_fact_key_from_row,
    dedupe_fact_rows,
    dedupe_measurements,
    measurement_normalization_fields,
)
from app.domain.ontology import Evidence, Measurement
from app.extraction.pipeline import ExtractionPipeline
from app.models.schemas import Chunk


def test_ksi_strength_gets_explicit_normalized_fields() -> None:
    fields = measurement_normalization_fields("tensile strength", 77, "ksi")

    assert fields["value_original"] == 77.0
    assert fields["unit_original"] == "ksi"
    assert abs(fields["value_normalized"] - 530.896289) < 0.001
    assert fields["unit_normalized"] == "MPa"
    assert fields["normalization_family"] == "strength"


def test_duplicate_measurements_merge_and_preserve_evidence() -> None:
    evidence_a = Evidence(document_id="doc-a", chunk_id="chunk-a", source_name="a.txt", quote="ВТ6 прочность 980 MPa")
    evidence_b = Evidence(document_id="doc-b", chunk_id="chunk-b", source_name="b.txt", quote="ВТ6 прочность 980 MPa")
    measurements = [
        Measurement(property_name="прочность", value=980.0, unit="MPa", effect="unknown", confidence=0.72, evidence=[evidence_a]),
        Measurement(property_name="tensile strength", value=980.0, unit="МПа", effect="unknown", confidence=0.9, evidence=[evidence_b]),
    ]

    deduped = dedupe_measurements(measurements, material="ВТ6", regime="отжиг")

    assert len(deduped) == 1
    assert deduped[0].confidence == 0.9
    assert {item.chunk_id for item in deduped[0].evidence} == {"chunk-a", "chunk-b"}


def test_fact_rows_dedupe_uses_canonical_key_and_merges_evidence() -> None:
    rows = [
        {
            "material": "Ti-6Al-4V",
            "regime": "annealing",
            "property": "ultimate tensile strength",
            "value": 1120.0,
            "unit": "MPa",
            "effect": "unknown",
            "evidence": [{"document_id": "doc-a", "chunk_id": "chunk-a", "quote": "q1"}],
        },
        {
            "material": "ВТ6",
            "regime": "отжиг",
            "property": "прочность",
            "value": 1120.0,
            "unit": "МПа",
            "effect": "unknown",
            "evidence": [{"document_id": "doc-b", "chunk_id": "chunk-b", "quote": "q2"}],
        },
    ]

    deduped = dedupe_fact_rows(rows)

    assert canonical_fact_key_from_row(rows[0]) == canonical_fact_key_from_row(rows[1])
    assert len(deduped) == 1
    assert deduped[0]["unit_normalized"] == "MPa"
    assert len(deduped[0]["evidence"]) == 2


def test_conflicting_values_are_reported_not_dropped() -> None:
    rows = [
        {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 980.0, "unit": "MPa", "evidence": [{"source_name": "a"}]},
        {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 1120.0, "unit": "MPa", "evidence": [{"source_name": "b"}]},
    ]

    conflicts = build_conflict_summary(rows)

    assert len(conflicts) == 1
    assert conflicts[0]["material"] == "ВТ6"
    assert conflicts[0]["property"] == "прочность"
    assert conflicts[0]["sources_count"] == 2
    assert {item["value"] for item in conflicts[0]["values"]} == {980.0, 1120.0}


def test_conflicting_effects_are_reported_even_with_same_value() -> None:
    rows = [
        {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 980.0, "unit": "MPa", "effect": "increase"},
        {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 980.0, "unit": "MPa", "effect": "decrease"},
    ]

    conflicts = build_conflict_summary(rows)

    assert len(conflicts) == 1
    assert conflicts[0]["possible_reason"] == "sources report different qualitative effects"


def test_far_away_material_value_binding_lowers_confidence() -> None:
    chunk = Chunk(
        chunk_id="chunk-far",
        doc_id="doc-far",
        workspace_uid="test",
        text=(
            "После отжига сплава ВТ6 описывали микроструктуру. "
            "В другой таблице без связи указана прочность 980 MPa."
        ),
        page_start=1,
        page_end=1,
        section_path="test",
        metadata={"filename": "far.txt"},
    )

    bundle = ExtractionPipeline(mode="deterministic", audit_enabled=False).extract_from_chunk(chunk)
    measurements = [measurement for experiment in bundle.experiments for measurement in experiment.measurements]

    assert measurements
    assert max(item.confidence for item in measurements) < 0.7

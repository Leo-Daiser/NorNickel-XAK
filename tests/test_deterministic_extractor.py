from __future__ import annotations

from app.extraction.pipeline import ExtractionPipeline
from app.models.schemas import Chunk


def _chunk(text: str, chunk_id: str = "chunk") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="test",
        metadata={"filename": "test.txt"},
    )


def test_vt6_annealing_strength_extracted_as_experiment() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _chunk("Эксперимент EXP-001: ВТ6 после отжига 900 C 2 h показал прочность 1120 MPa; прочность повысилась.")
    )

    assert len(bundle.experiments) == 1
    experiment = bundle.experiments[0]
    assert [item.canonical_name for item in experiment.materials] == ["ВТ6"]
    assert [item.canonical_name for item in experiment.regimes] == ["отжиг"]
    assert experiment.measurements[0].property_canonical == "прочность"
    assert experiment.measurements[0].value == 1120.0
    assert experiment.measurements[0].unit == "MPa"
    assert experiment.evidence[0].quote


def test_material_mention_only_is_not_experiment() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _chunk("Сплав ВТ6 относится к титановым альфа-бета материалам.")
    )

    assert bundle.experiments == []
    assert any("missing_regime_or_measurement" in item.reasons for item in bundle.quarantined_items)
    assert not bundle.accepted_facts


def test_property_without_value_or_effect_is_not_positive_measurement() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _chunk("Эксперимент EXP-CRYO: ВТ6 после криообработки. Вязкость упоминается как планируемое свойство.")
    )

    assert all(not experiment.measurements for experiment in bundle.experiments)
    assert not any(
        measurement.property_canonical == "вязкость"
        for experiment in bundle.experiments
        for measurement in experiment.measurements
    )


def test_multiple_experiment_ids_are_not_merged() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _chunk(
            "Эксперимент EXP-A: ВТ6 после отжига имеет прочность 1120 MPa. "
            "Эксперимент EXP-B: 7075-T6 после старения имеет прочность 540 MPa."
        )
    )

    assert len(bundle.experiments) == 2
    materials_per_experiment = [{item.canonical_name for item in exp.materials} for exp in bundle.experiments]
    assert {"ВТ6", "7075-T6"} not in materials_per_experiment


def test_gap_text_is_not_promoted_to_positive_experiment() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _chunk("Для 7075-T6 после старения нет данных по коррозионной стойкости; коррозионная стойкость не измерялась.")
    )

    assert bundle.experiments == []
    assert bundle.data_gaps
    assert any(gap.material == "7075-T6" for gap in bundle.data_gaps)

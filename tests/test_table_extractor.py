from __future__ import annotations

from app.extraction.pipeline import ExtractionPipeline
from app.models.schemas import Chunk


def _row(text: str, row_id: int = 3) -> Chunk:
    return Chunk(
        chunk_id=f"row-{row_id}",
        doc_id="doc-table",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="table",
        metadata={"filename": "experiments.csv", "chunk_kind": "table_row", "row_id": row_id},
    )


def test_csv_like_row_becomes_experiment() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row(
            "material: ВТ6 | regime: annealing | property: tensile strength | "
            "value: 1120 | unit: MPa | equipment: печь SNOL"
        )
    )

    assert len(bundle.experiments) == 1
    experiment = bundle.experiments[0]
    assert experiment.materials[0].canonical_name == "ВТ6"
    assert experiment.regimes[0].canonical_name == "отжиг"
    assert experiment.measurements[0].property_canonical == "прочность"
    assert experiment.measurements[0].unit == "MPa"


def test_russian_column_aliases_work() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row("Материал: 12Х18Н10Т | Режим: закалка | Свойство: твёрдость | Значение: 240 | Единица: HV")
    )

    experiment = bundle.experiments[0]
    assert experiment.materials[0].canonical_name == "12Х18Н10Т"
    assert experiment.regimes[0].canonical_name == "закалка"
    assert experiment.measurements[0].property_canonical == "твёрдость"


def test_empty_row_rejected() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(_row("material: | regime: | property: "))

    assert bundle.experiments == []
    assert any(item.reason in {"empty_row", "missing_material"} for item in bundle.rejected_items)


def test_row_index_is_preserved_in_evidence() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row("Материал: ВТ6 | Режим: отжиг | Свойство: прочность | Значение: 1120 | Единица: MPa", row_id=7)
    )

    assert bundle.experiments[0].evidence[0].source.row_index == 7


def test_multivalue_cells_are_handled_safely() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row("Материал: ВТ6; Ti-6Al-4V | Режим: отжиг | Свойство: прочность | Значение: 1120 | Единица: MPa")
    )

    materials = [item.canonical_name for item in bundle.experiments[0].materials]
    assert materials == ["ВТ6"]


def test_table_adapter_creates_typed_process_parameter_candidate_without_material() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row("process: catholyte circulation | parameter: flow rate | value: 0.5 | unit: m/s")
    )

    assert any(item.fact_type == "ProcessParameterFact" for item in bundle.candidate_facts)
    assert any(item.extractor_name.endswith("structured_table_adapter") for item in bundle.candidate_facts)
    assert any(
        item.fact_type == "ProcessParameterFact"
        and "accepted_structured_table_candidate" in item.validation_reasons
        for item in bundle.accepted_facts
    )


def test_table_adapter_creates_capacity_candidate_from_capacity_columns() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row("commodity: copper | facility: plant A | country: Chile | capacity: 1200 | unit: t/day | year: 2024")
    )

    capacity_candidates = [item for item in bundle.candidate_facts if item.fact_type == "FacilityCapacityFact"]

    assert capacity_candidates
    assert capacity_candidates[0].subject["facility"] == "plant A"
    assert capacity_candidates[0].subject["geography"] == "Chile"
    assert any(
        item.fact_type == "FacilityCapacityFact"
        and item.normalized_fact["source_adapter"] == "structured_table_adapter"
        for item in bundle.accepted_facts
    )


def test_ambiguous_table_candidate_goes_to_quarantine_not_accepted_fact() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row("parameter: unclear | value: 12 | unit: %")
    )

    assert not any(item.normalized_fact.get("value") == 12.0 for item in bundle.accepted_facts)
    assert bundle.quarantined_items


def test_assay_composition_columns_create_content_process_parameter_facts() -> None:
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(
        _row(
            "column_1: Пирротиновый концентрат | Технология: автоклавное выщелачивание | "
            "Ni, %: 0,5-1 | Cu, %: 0,1-0,2 | МПГ, г/т: 1-2 г/т"
        )
    )

    accepted = [
        item
        for item in bundle.accepted_facts
        if item.fact_type == "ProcessParameterFact"
        and item.normalized_fact["object"].get("property") == "содержание"
    ]

    assert len(accepted) == 3
    analytes = {item.normalized_fact["object"]["analyte"] for item in accepted}
    assert analytes == {"ni", "cu", "мпг"}
    ni_fact = next(item for item in accepted if item.normalized_fact["object"]["analyte"] == "ni")
    assert ni_fact.normalized_fact["unit"] == "%"
    assert ni_fact.normalized_fact["object"]["value_min"] == 0.5
    assert ni_fact.normalized_fact["object"]["value_max"] == 1.0

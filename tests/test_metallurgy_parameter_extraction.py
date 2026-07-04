from __future__ import annotations

from app.extraction.pipeline import ExtractionPipeline
from app.extraction.to_graph_models import bundle_to_experiment_facts
from app.models.schemas import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="chunk-met",
        doc_id="doc-met",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="test",
        metadata={"filename": "metallurgy.txt"},
    )


def _bundle(text: str):
    return ExtractionPipeline(mode="deterministic", audit_enabled=False).extract_from_chunk(_chunk(text))


def test_desalination_water_constraints_extract_process_measurements() -> None:
    bundle = _bundle(
        "Метод обессоливания шахтных вод: сульфаты 250 мг/л, хлориды 220 мг/л, "
        "Ca 210 мг/л, Mg 205 мг/л, Na 230 мг/л; сухой остаток ≤1000 мг/дм³."
    )

    assert bundle.experiments
    experiment = bundle.experiments[0]
    materials = {item.canonical_name for item in experiment.materials}
    regimes = {item.canonical_name for item in experiment.regimes}
    measurements = {
        (item.property_raw, item.property_canonical, item.value, item.unit)
        for item in experiment.measurements
    }

    assert "шахтные воды" in materials
    assert {"сульфаты", "хлориды", "Ca", "Mg", "Na"}.issubset(materials)
    assert "обессоливание" in regimes
    assert ("сульфаты", "концентрация", 250.0, "mg/L") in measurements
    assert ("хлориды", "концентрация", 220.0, "mg/L") in measurements
    assert ("сухой остаток", "сухой остаток", 1000.0, "mg/L") in measurements
    assert all(item.evidence for item in experiment.measurements)


def test_catholyte_circulation_flow_velocity_extracted() -> None:
    bundle = _bundle(
        "При электроэкстракции никеля циркуляция католита обеспечивала скорость потока 0.35 м/с."
    )

    assert bundle.experiments
    experiment = bundle.experiments[0]
    materials = {item.canonical_name for item in experiment.materials}
    regimes = {item.canonical_name for item in experiment.regimes}

    assert {"никель", "католит"}.issubset(materials)
    assert {"электроэкстракция", "циркуляция католита"}.issubset(regimes)
    assert any(
        item.property_canonical == "скорость потока" and item.value == 0.35 and item.unit == "m/s"
        for item in experiment.measurements
    )


def test_unrelated_number_without_process_property_is_rejected() -> None:
    bundle = _bundle(
        "Отчет содержит страницу 25 и архивный номер 300. Упоминается никель, но параметр процесса не указан."
    )

    assert bundle.experiments == []
    assert all(
        not (
            rejection.item_type == "measurement"
            and isinstance(rejection.raw_payload, dict)
            and rejection.raw_payload.get("value") in {25, 300}
        )
        for rejection in bundle.rejected_items
    )


def test_matte_slag_precious_metal_distribution_extracted() -> None:
    bundle = _bundle(
        "В пирометаллургическом эксперименте ПВП распределение Au между медным штейном и шлаком составило 92%."
    )

    assert bundle.experiments
    experiment = bundle.experiments[0]
    materials = {item.canonical_name for item in experiment.materials}
    regimes = {item.canonical_name for item in experiment.regimes}

    assert {"Au", "медь", "штейн", "шлак"}.issubset(materials)
    assert {"пирометаллургия", "ПВП"}.issubset(regimes)
    assert any(
        item.property_canonical == "распределение" and item.value == 92.0 and item.unit == "%"
        for item in experiment.measurements
    )
    assert all(item.evidence for item in experiment.measurements)


def test_economic_indicators_for_mine_water_injection_extracted() -> None:
    bundle = _bundle(
        "Для закачки шахтных вод CAPEX составил 1200 USD/t, OPEX составил 45 $/t."
    )

    assert bundle.experiments
    experiment = bundle.experiments[0]
    materials = {item.canonical_name for item in experiment.materials}
    regimes = {item.canonical_name for item in experiment.regimes}
    measurements = [
        item
        for item in experiment.measurements
        if item.property_canonical == "экономический показатель"
    ]

    assert "шахтные воды" in materials
    assert "закачка шахтных вод" in regimes
    assert {(item.property_raw.upper(), item.value, item.unit) for item in measurements} == {
        ("CAPEX", 1200.0, "USD/t"),
        ("OPEX", 45.0, "USD/t"),
    }


def test_economic_indicator_requires_economic_label_near_value() -> None:
    bundle = _bundle(
        "Отчет по шахтным водам содержит архивный номер 1200 USD/t в справочной строке, "
        "но капитальные или эксплуатационные затраты не указаны."
    )

    assert all(
        item.property_canonical != "экономический показатель"
        for experiment in bundle.experiments
        for item in experiment.measurements
    )


def test_equipment_team_and_topic_attach_to_experiment_without_llm() -> None:
    bundle = _bundle(
        "Команда Alpha выполнила электроэкстракцию никеля в лаборатории гидрометаллургии; "
        "оборудование: ванна электроэкстракции; тема: циркуляция католита. "
        "Скорость потока составила 0.42 м/с."
    )

    assert bundle.experiments
    experiment = bundle.experiments[0]

    assert any(item.canonical_name == "ванна электроэкстракции" for item in experiment.equipment)
    assert any(item.canonical_name == "Alpha" for item in experiment.teams)
    assert all(item.canonical_name != "Alpha" for item in experiment.laboratories)
    assert any(item.canonical_name == "циркуляция католита" for item in experiment.topic_tags)
    assert all(item.evidence for item in [*experiment.equipment, *experiment.teams, *experiment.topic_tags])

    facts = bundle_to_experiment_facts(bundle)
    assert facts[0].teams == ["Alpha"]
    assert facts[0].topic_tags == ["циркуляция католита"]


def test_unsupported_lab_equipment_probe_does_not_create_fact_without_material_or_measurement() -> None:
    bundle = _bundle(
        "Team Alpha in Laboratory L-3 used furnace F-900 and microscope M-12. "
        "No material, process result or measurement is reported."
    )

    entity_types = {(item.entity_type, item.canonical_name) for item in bundle.entities}

    assert bundle.experiments == []
    assert ("ResearchTeam", "Alpha") in entity_types
    assert any(item[0] == "Equipment" for item in entity_types)

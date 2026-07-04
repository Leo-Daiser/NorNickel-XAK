from __future__ import annotations

from app.domain.query_constraints import QueryIntent
from app.retrieval.query_planner import QueryPlanner


def test_material_regime_property_constraints() -> None:
    constraints = QueryPlanner().parse("Что уже делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?")
    assert constraints.intent == QueryIntent.MATERIAL_REGIME_PROPERTY_EFFECT
    assert constraints.materials == ["ВТ6"]
    assert constraints.regimes == ["отжиг"]
    assert constraints.properties == ["прочность"]
    assert constraints.require_exact_match is True


def test_cryo_toughness_constraints() -> None:
    constraints = QueryPlanner().parse("Что делали по ВТ6 при криообработке и как изменилась вязкость?")
    assert constraints.intent == QueryIntent.MATERIAL_REGIME_PROPERTY_EFFECT
    assert constraints.materials == ["ВТ6"]
    assert constraints.regimes == ["криообработка"]
    assert constraints.properties == ["вязкость"]
    assert constraints.require_exact_match is True


def test_decision_history_constraints() -> None:
    constraints = QueryPlanner().parse("Покажи историю решений по ВТ6.")
    assert constraints.intent == QueryIntent.DECISION_HISTORY
    assert constraints.materials == ["ВТ6"]
    assert constraints.require_exact_match is False


def test_gap_constraints() -> None:
    constraints = QueryPlanner().parse("Какие пробелы по 7075-T6 и коррозионной стойкости?")
    assert constraints.intent == QueryIntent.GAP_ANALYSIS
    assert constraints.materials == ["7075-T6"]
    assert constraints.properties == ["коррозионная стойкость"]


def test_tz_desalination_numeric_constraints() -> None:
    constraints = QueryPlanner().parse(
        "Какие методы обессоливания воды подходят для обогатительной фабрики, если исходная вода содержит "
        "сульфаты, хлориды, Ca, Mg, Na по 200–300 мг/л, а требуемый сухой остаток — ≤1000 мг/дм³?"
    )

    assert constraints.intent == QueryIntent.ENTITY_OVERVIEW
    assert constraints.materials == ["сульфаты", "хлориды", "Ca", "Mg", "Na"]
    assert constraints.regimes == ["обессоливание"]
    assert constraints.properties == ["сухой остаток"]
    assert constraints.topic_tags == ["обогатительная фабрика"]
    by_parameter = {item["parameter"]: item for item in constraints.numeric_constraints}
    assert by_parameter["сульфаты"]["value_min"] == 200.0
    assert by_parameter["сульфаты"]["value_max"] == 300.0
    assert by_parameter["сульфаты"]["unit"] == "mg/L"
    assert by_parameter["сухой остаток"]["operator"] == "<="
    assert by_parameter["сухой остаток"]["value"] == 1000.0


def test_tz_electrowinning_equipment_constraints() -> None:
    constraints = QueryPlanner().parse(
        "Какие схемы подачи электролита в ванны электроэкстракции никеля и диафрагменные ячейки описаны в мировой практике?"
    )

    assert constraints.intent == QueryIntent.EQUIPMENT_USAGE
    assert constraints.materials == ["никель", "электролит"]
    assert constraints.regimes == ["электроэкстракция"]
    assert constraints.equipment == ["ванна электроэкстракции", "диафрагменная ячейка"]
    assert constraints.geographies == ["мировая практика"]


def test_tz_flash_smelting_and_gas_cleaning_constraints() -> None:
    constraints = QueryPlanner().parse("Какие системы очистки газов и печи взвешенной плавки применяются для удаления SO2?")

    assert constraints.intent == QueryIntent.EQUIPMENT_USAGE
    assert "газоочистка" in constraints.regimes
    assert "удаление SO2" in constraints.regimes
    assert "ПВП" in constraints.regimes
    assert set(constraints.equipment) == {"система газоочистки", "ПВП"}

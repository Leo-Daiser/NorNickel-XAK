from __future__ import annotations

from app.domain.numeric_constraints import extract_numeric_constraints
from app.retrieval.query_planner import QueryPlanner


def _by_parameter(rows: list[dict], parameter: str) -> dict:
    return next(row for row in rows if row["parameter"] == parameter)


def test_desalination_query_extracts_multicomponent_water_constraints() -> None:
    question = (
        "Какие методы обессоливания воды подходят для обогатительной фабрики, если исходная вода содержит "
        "сульфаты, хлориды, Ca, Mg, Na по 200–300 мг/л, а требуемый сухой остаток — ≤1000 мг/дм³?"
    )

    constraints = QueryPlanner().parse(question)
    numeric = constraints.numeric_constraints

    assert constraints.intent == "entity_overview"
    assert constraints.require_exact_match is False
    assert "обессоливание" in constraints.regimes
    assert {"сульфаты", "хлориды", "Ca", "Mg", "Na"}.issubset(set(constraints.materials))
    assert "сухой остаток" in constraints.properties
    for parameter in ["сульфаты", "хлориды", "Ca", "Mg", "Na"]:
        row = _by_parameter(numeric, parameter)
        assert row["operator"] == "range"
        assert row["value_min"] == 200.0
        assert row["value_max"] == 300.0
        assert row["unit"] == "mg/L"
    dry_residue = _by_parameter(numeric, "сухой остаток")
    assert dry_residue["operator"] == "<="
    assert dry_residue["value"] == 1000.0
    assert dry_residue["unit"] == "mg/L"


def test_catholyte_query_extracts_process_material_geography_and_property() -> None:
    question = (
        "Какие технические решения организации циркуляции католита при электроэкстракции никеля описаны "
        "в мировой практике, и какая скорость потока считается оптимальной?"
    )

    constraints = QueryPlanner().parse(question)

    assert constraints.intent == "entity_overview"
    assert constraints.require_exact_match is False
    assert {"никель", "католит"}.issubset(set(constraints.materials))
    assert {"электроэкстракция", "циркуляция католита"}.issubset(set(constraints.regimes))
    assert "скорость потока" in constraints.properties
    assert constraints.geographies == ["мировая практика"]


def test_matte_slag_distribution_query_extracts_metals_and_relative_time_filter() -> None:
    question = "Покажите все эксперименты и публикации по распределению Au, Ag и МПГ между медным/никелевым штейном и шлаком за последние 5 лет."

    constraints = QueryPlanner().parse(question)

    assert constraints.intent == "entity_overview"
    assert {"Au", "Ag", "МПГ", "медь", "никель", "штейн", "шлак"}.issubset(set(constraints.materials))
    assert "распределение" in constraints.properties
    assert constraints.time_filters == [{"type": "relative_years", "years": 5}]


def test_mine_water_injection_query_extracts_domestic_and_foreign_practice() -> None:
    question = "Какие способы закачки шахтных вод в глубокие горизонты применялись в России и за рубежом, и каковы их технико-экономические показатели?"

    constraints = QueryPlanner().parse(question)

    assert constraints.intent == "entity_overview"
    assert constraints.require_exact_match is False
    assert "шахтные воды" in constraints.materials
    assert "закачка шахтных вод" in constraints.regimes
    assert "экономический показатель" in constraints.properties
    assert constraints.geographies == ["Россия", "зарубежная практика"]


def test_economic_numeric_constraint_extracts_cost_unit() -> None:
    question = "Найди способы закачки шахтных вод с CAPEX не более 1200 USD/t и OPEX < 50 $/t."

    constraints = QueryPlanner().parse(question)
    numeric = constraints.numeric_constraints

    assert "шахтные воды" in constraints.materials
    assert "закачка шахтных вод" in constraints.regimes
    assert "экономический показатель" in constraints.properties
    assert any(
        row["parameter"] == "экономический показатель"
        and row["operator"] == "<="
        and row["value"] == 1200.0
        and row["unit"] == "USD/t"
        for row in numeric
    )
    assert any(
        row["parameter"] == "экономический показатель"
        and row["operator"] == "<"
        and row["value"] == 50.0
        and row["unit"] == "USD/t"
        for row in numeric
    )


def test_numeric_constraints_do_not_emit_values_without_parameter_context() -> None:
    assert extract_numeric_constraints("В документе указаны страницы 10-20 и номер отчета 300.") == []

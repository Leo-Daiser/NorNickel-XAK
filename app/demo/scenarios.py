"""Curated demo scenarios exposed to the cockpit UI and evaluation."""

from __future__ import annotations

from pydantic import BaseModel


class DemoScenario(BaseModel):
    scenario_id: str
    title: str
    question: str
    expected_intent: str | None = None
    expected_status: str | None = None
    description: str


DEMO_SCENARIOS: list[DemoScenario] = [
    DemoScenario(
        scenario_id="strict_positive_vt6_annealing_strength",
        title="Strict positive: ВТ6 + отжиг + прочность",
        question="Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?",
        expected_intent="material_regime_property_effect",
        expected_status="ok",
        description="Показывает exact graph path Material + ProcessRegime + Property.",
    ),
    DemoScenario(
        scenario_id="strict_negative_vt6_cryo_toughness",
        title="Strict negative: ВТ6 + криообработка + вязкость",
        question="Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?",
        expected_intent="material_regime_property_effect",
        expected_status="no_exact_match",
        description="Демонстрирует no-hallucination behavior и inferred DataGap.",
    ),
    DemoScenario(
        scenario_id="material_overview_vt6",
        title="Material overview: ВТ6",
        question="Что уже делали по ВТ6?",
        expected_intent="material_overview",
        expected_status="ok",
        description="Обзор экспериментов, режимов, свойств, оборудования и источников.",
    ),
    DemoScenario(
        scenario_id="comparison_vt6_7075_strength",
        title="Comparison: ВТ6 vs 7075-T6 по прочности",
        question="Сравни ВТ6 и 7075-T6 по прочности.",
        expected_intent="material_comparison",
        expected_status="ok",
        description="Показывает сравнение и предупреждение о несопоставимых режимах/единицах.",
    ),
    DemoScenario(
        scenario_id="gaps_corrosion",
        title="Data gaps: коррозионная стойкость",
        question="Какие пробелы есть по коррозионной стойкости?",
        expected_intent="gap_analysis",
        expected_status="ok",
        description="Показывает явные gaps и evidence.",
    ),
    DemoScenario(
        scenario_id="similar_vt6_annealing",
        title="Similar experiments: ВТ6 при отжиге",
        question="Найди похожие эксперименты на ВТ6 при отжиге.",
        expected_intent="similar_experiments",
        expected_status="ok",
        description="Показывает score и объяснение graph-similarity.",
    ),
    DemoScenario(
        scenario_id="graph_neighborhood_vt6",
        title="Graph neighborhood: ВТ6",
        question="Покажи связанные сущности по ВТ6.",
        expected_intent="graph_neighborhood",
        expected_status="ok",
        description="Показывает compact subgraph вокруг материала.",
    ),
    DemoScenario(
        scenario_id="lab_steel_12x18n10t",
        title="Lab activity: 12Х18Н10Т",
        question="Какая лаборатория занималась 12Х18Н10Т?",
        expected_intent="lab_activity",
        expected_status="ok",
        description="Показывает лаборатории/команды и связанные эксперименты.",
    ),
]


def list_demo_scenarios() -> list[DemoScenario]:
    return list(DEMO_SCENARIOS)


def get_demo_scenario(scenario_id: str) -> DemoScenario | None:
    return next((item for item in DEMO_SCENARIOS if item.scenario_id == scenario_id), None)

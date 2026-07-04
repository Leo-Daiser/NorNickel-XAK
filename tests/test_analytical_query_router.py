from __future__ import annotations

from app.analytics.query_models import AnalyticalIntent
from app.analytics.router import AnalyticalQueryRouter
from app.retrieval.query_planner import QueryPlanner


def _intent(question: str) -> AnalyticalIntent:
    constraints = QueryPlanner().parse(question)
    return AnalyticalQueryRouter().build_plan(question, constraints).intent


def test_material_overview_intent() -> None:
    assert _intent("Что уже делали по ВТ6?") == AnalyticalIntent.MATERIAL_OVERVIEW


def test_material_comparison_intent() -> None:
    assert _intent("Сравни ВТ6 и 7075-T6 по прочности") == AnalyticalIntent.MATERIAL_COMPARISON


def test_similar_experiments_intent() -> None:
    assert _intent("Найди похожие эксперименты на ВТ6 при отжиге") == AnalyticalIntent.SIMILAR_EXPERIMENTS


def test_graph_neighborhood_intent() -> None:
    assert _intent("Покажи связанные сущности по ВТ6") == AnalyticalIntent.GRAPH_NEIGHBORHOOD


def test_strict_priority_over_analytical_intents() -> None:
    assert (
        _intent("Что делали по ВТ6 при отжиге и какой был эффект на прочность?")
        == AnalyticalIntent.STRICT_MATERIAL_REGIME_PROPERTY
    )

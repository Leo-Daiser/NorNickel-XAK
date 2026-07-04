from __future__ import annotations

from app.analytics.answer_synthesizer import AnswerSynthesizer
from app.analytics.query_models import GraphContext
from app.analytics.router import AnalyticalQueryRouter
from app.retrieval.query_planner import QueryPlanner


def _plan(question: str):
    constraints = QueryPlanner().parse(question)
    return AnalyticalQueryRouter().build_plan(question, constraints)


def test_overview_answer_includes_experiments_regimes_properties() -> None:
    plan = _plan("Что уже делали по ВТ6?")
    context = GraphContext(
        intent=plan.intent,
        constraints=plan.constraints,
        facts=[
            {
                "experiment_id": "EXP-1",
                "material": "ВТ6",
                "regime": "отжиг",
                "property": "прочность",
                "value": 1120,
                "unit": "MPa",
            }
        ],
    )
    answer = AnswerSynthesizer().synthesize(plan, context)
    assert "эксперимент" in answer.lower()
    assert "отжиг" in answer
    assert "прочность" in answer


def test_comparison_warns_when_units_or_regimes_differ() -> None:
    plan = _plan("Сравни ВТ6 и 7075-T6 по прочности")
    context = GraphContext(
        intent=plan.intent,
        constraints=plan.constraints,
        facts=[
            {"experiment_id": "EXP-1", "material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 1120, "unit": "MPa"},
            {"experiment_id": "EXP-2", "material": "7075-T6", "regime": "старение", "property": "прочность", "value": 77, "unit": "ksi"},
        ],
    )
    answer = AnswerSynthesizer().synthesize(plan, context)
    assert "предупреждение" in answer.lower()
    assert "различаются" in answer.lower()


def test_gaps_answer_only_lists_matching_gaps() -> None:
    plan = _plan("Какие пробелы по коррозионной стойкости 7075-T6?")
    context = GraphContext(
        intent=plan.intent,
        constraints=plan.constraints,
        gaps=[
            {
                "material": "7075-T6",
                "regime": "старение",
                "property": "коррозионная стойкость",
                "reason": "не измерялась",
            }
        ],
    )
    answer = AnswerSynthesizer().synthesize(plan, context)
    assert "не измерялась" in answer
    assert "7075-T6" in answer


def test_neighborhood_answer_lists_entity_types() -> None:
    plan = _plan("Покажи связанные сущности по ВТ6")
    context = GraphContext(
        intent=plan.intent,
        constraints=plan.constraints,
        entities=[
            {"type": "Material", "name": "ВТ6"},
            {"type": "ProcessRegime", "name": "отжиг"},
            {"type": "Property", "name": "прочность"},
        ],
    )
    answer = AnswerSynthesizer().synthesize(plan, context)
    assert "Material" in answer
    assert "ProcessRegime" in answer
    assert "Property" in answer


def test_template_and_hybrid_modes_work_without_llm() -> None:
    plan = _plan("Что уже делали по ВТ6?")
    context = GraphContext(intent=plan.intent, constraints=plan.constraints)
    assert AnswerSynthesizer(mode="template").synthesize(plan, context)
    assert AnswerSynthesizer(mode="hybrid").synthesize(plan, context)

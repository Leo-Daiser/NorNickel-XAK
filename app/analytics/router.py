"""Analytical intent router built on top of QueryConstraints."""

from __future__ import annotations

from ..domain.normalization import normalize_text
from ..domain.query_constraints import QueryIntent
from . import cypher_templates as templates
from .query_models import AnalyticalIntent, AnalyticalQueryPlan


class AnalyticalQueryRouter:
    """Convert strict QueryConstraints into an analytical query plan."""

    def build_plan(self, question: str, constraints) -> AnalyticalQueryPlan:
        q = normalize_text(question or "")
        if any(term in q for term in ["противореч", "неоднород", "расходятся", "разные значения", "conflict", "different values"]):
            return _plan(question, constraints, AnalyticalIntent.UNKNOWN, "search", None, retrieval_required=True, graph_expansion_required=False)
        if constraints.require_exact_match:
            return _plan(question, constraints, AnalyticalIntent.STRICT_MATERIAL_REGIME_PROPERTY, "strict", templates.template_for("strict_exact"))

        if constraints.intent == QueryIntent.DECISION_HISTORY:
            return _plan(question, constraints, AnalyticalIntent.DECISION_HISTORY, "history", templates.template_for("decision_history"))
        if constraints.intent == QueryIntent.GAP_ANALYSIS:
            return _plan(question, constraints, AnalyticalIntent.GAP_ANALYSIS, "gaps", templates.GAP_ANALYSIS)

        if any(term in q for term in ["сравн", "сопостав", "чем отличаются", "лучше"]):
            if len(constraints.materials) >= 2:
                return _plan(question, constraints, AnalyticalIntent.MATERIAL_COMPARISON, "comparison", templates.MATERIAL_COMPARISON)
            if len(constraints.regimes) >= 2:
                return _plan(question, constraints, AnalyticalIntent.REGIME_COMPARISON, "comparison", templates.REGIME_COMPARISON)

        if any(term in q for term in ["похож", "аналогич", "similar"]):
            return _plan(question, constraints, AnalyticalIntent.SIMILAR_EXPERIMENTS, "search", templates.SIMILAR_EXPERIMENTS, retrieval_required=True)

        if any(term in q for term in ["связан", "граф вокруг", "окружение", "neighborhood"]):
            return _plan(question, constraints, AnalyticalIntent.GRAPH_NEIGHBORHOOD, "neighborhood", templates.GRAPH_NEIGHBORHOOD)

        if any(term in q for term in ["оборудован", "установк", "печь", "прибор", "equipment"]):
            return _plan(question, constraints, AnalyticalIntent.EQUIPMENT_USAGE, "overview", templates.EQUIPMENT_USAGE)

        if any(term in q for term in ["лаборатор", "lab "]):
            return _plan(question, constraints, AnalyticalIntent.LAB_ACTIVITY, "overview", templates.LAB_ACTIVITY)

        if any(term in q for term in ["команд", "группа", "team"]):
            return _plan(question, constraints, AnalyticalIntent.TEAM_ACTIVITY, "overview", templates.LAB_ACTIVITY)

        if any(term in q for term in ["тема", "теме", "темы", "тематик", "документы по", "topic"]):
            return _plan(question, constraints, AnalyticalIntent.TOPIC_SEARCH, "search", templates.TOPIC_SEARCH, retrieval_required=True)

        if len(constraints.materials) == 1:
            return _plan(question, constraints, AnalyticalIntent.MATERIAL_OVERVIEW, "overview", templates.MATERIAL_OVERVIEW)
        if constraints.regimes:
            return _plan(question, constraints, AnalyticalIntent.REGIME_OVERVIEW, "overview", templates.REGIME_OVERVIEW)
        if constraints.properties:
            return _plan(question, constraints, AnalyticalIntent.PROPERTY_OVERVIEW, "overview", templates.PROPERTY_OVERVIEW)

        if any(term in q for term in ["что есть", "покажи", "найди", "что известно"]):
            return _plan(question, constraints, AnalyticalIntent.GENERAL_SEARCH, "search", None, retrieval_required=True)
        return _plan(question, constraints, AnalyticalIntent.UNKNOWN, "search", None, retrieval_required=True, graph_expansion_required=False)


def _plan(
    question: str,
    constraints,
    intent: AnalyticalIntent,
    mode,
    template: str | None,
    retrieval_required: bool = False,
    graph_expansion_required: bool = True,
) -> AnalyticalQueryPlan:
    return AnalyticalQueryPlan(
        raw_question=question,
        intent=intent,
        constraints=constraints,
        cypher_template=template,
        retrieval_required=retrieval_required,
        graph_expansion_required=graph_expansion_required,
        answer_mode=mode,
        diagnostics={
            "router": "AnalyticalQueryRouter",
            "selected_intent": intent.value,
            "cypher_template": template,
            "retrieval_required": retrieval_required,
        },
    )

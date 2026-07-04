"""Small helpers for analytical response diagnostics."""

from __future__ import annotations

from typing import Any

from .query_models import AnalyticalQueryPlan, GraphContext


def analytical_diagnostics(
    plan: AnalyticalQueryPlan,
    context: GraphContext,
    evidence_backend: str,
    synthesis_mode: str,
) -> dict[str, Any]:
    """Return compact diagnostics for API/UI without exposing internals."""
    return {
        "analytical_router": plan.diagnostics,
        "analytical_intent": plan.intent.value,
        "answer_mode": plan.answer_mode,
        "evidence_backend": evidence_backend,
        "answer_synthesis_mode": synthesis_mode,
        "graph_context": context.stats(),
        "context_builder": context.diagnostics,
    }

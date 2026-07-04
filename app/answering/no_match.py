"""No-match response helpers for strict constraint matching."""

from __future__ import annotations

from typing import Any

from ..domain.ontology import DataGap
from ..domain.query_constraints import QueryConstraints
from ..graph.graph_models import PartialMatches
from .gap_analyzer import inferred_gap_for_missing_exact


def build_no_match_payload(
    constraints: QueryConstraints,
    partial_matches: PartialMatches,
    existing_gaps: list[DataGap] | None = None,
) -> dict[str, Any]:
    """Build a grounded no-exact-match payload without promoting partial facts."""
    inferred_gap = inferred_gap_for_missing_exact(constraints)
    gaps = _dedupe_gaps([*(existing_gaps or []), inferred_gap])
    partial_payload = partial_matches.to_response()
    partial_text = _partial_match_text(partial_payload)
    constraint_text = _constraint_text(constraints)
    answer = (
        f"Точных данных не найдено по сочетанию {constraint_text}. "
        "В загруженном корпусе нет одного эксперимента, который одновременно связывает указанные материал, режим и свойство. "
        f"{partial_text} Источники для несуществующего exact-факта отсутствуют."
    ).strip()
    return {
        "answer": answer,
        "status": "no_exact_match",
        "constraints": constraints.model_dump(),
        "partial_matches": partial_payload,
        "data_gaps": [gap.model_dump() for gap in gaps],
        "gaps": [gap.model_dump() for gap in gaps],
        "facts": [],
        "sources": [],
        "subgraph": {"nodes": [], "edges": []},
    }


def _constraint_text(constraints: QueryConstraints) -> str:
    parts: list[str] = []
    if constraints.materials:
        parts.append(f"материал: {', '.join(constraints.materials)}")
    if constraints.regimes:
        parts.append(f"режим: {', '.join(constraints.regimes)}")
    if constraints.properties:
        parts.append(f"свойство: {', '.join(constraints.properties)}")
    return "; ".join(parts) if parts else "без распознанных ограничений"


def _partial_match_text(partial_payload: dict[str, list[dict[str, Any]]]) -> str:
    counts = {key: len(value) for key, value in partial_payload.items()}
    total = sum(counts.values())
    if total == 0:
        return "Частичных совпадений по этим ограничениям также не найдено."
    readable = {
        "same_material": "по тому же материалу",
        "same_material_and_regime": "по тому же материалу и режиму",
        "same_material_and_property": "по тому же материалу и свойству",
        "same_regime_and_property": "по тому же режиму и свойству",
    }
    fragments = [f"{readable.get(key, key)}: {count}" for key, count in counts.items() if count]
    return "Найдены только частичные совпадения, они не являются ответом на исходный вопрос: " + "; ".join(fragments) + "."


def _dedupe_gaps(gaps: list[DataGap]) -> list[DataGap]:
    seen = set()
    result: list[DataGap] = []
    for gap in gaps:
        key = (gap.material, gap.regime, gap.property, gap.reason)
        if key in seen:
            continue
        seen.add(key)
        result.append(gap)
    return result


"""Grounded analytical answer synthesis.

The synthesizer is template-first.  LLM usage can be added as a polish layer,
but facts always come from GraphContext rather than unrestricted retrieval.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..config import settings
from .query_models import AnalyticalIntent, AnalyticalQueryPlan, GraphContext


class AnswerSynthesizer:
    """Build human-readable analytical answers from graph context."""

    def __init__(self, mode: str | None = None) -> None:
        configured = mode or getattr(settings, "answer_synthesis_mode", "template")
        self.mode = configured if configured in {"template", "hybrid", "llm"} else "template"

    def synthesize(self, plan: AnalyticalQueryPlan, context: GraphContext) -> str:
        """Return a grounded answer for the selected analytical mode."""
        if plan.answer_mode == "comparison":
            return self._comparison(plan, context)
        if plan.answer_mode == "history":
            return self._history(plan, context)
        if plan.answer_mode == "gaps":
            return self._gaps(plan, context)
        if plan.answer_mode == "neighborhood":
            return self._neighborhood(plan, context)
        if plan.intent == AnalyticalIntent.SIMILAR_EXPERIMENTS:
            return self._similar(plan, context)
        if plan.answer_mode == "search":
            return self._search(plan, context)
        return self._overview(plan, context)

    def _overview(self, plan: AnalyticalQueryPlan, context: GraphContext) -> str:
        material = _first(plan.constraints.materials)
        regime = _first(plan.constraints.regimes)
        prop = _first(plan.constraints.properties)
        subject = material or regime or prop or "запросу"
        experiments = sorted({str(row.get("experiment_id")) for row in context.facts if row.get("experiment_id")})
        regimes = sorted({str(row.get("regime")) for row in context.facts if row.get("regime")})
        properties = sorted({str(row.get("property")) for row in context.facts if row.get("property")})
        equipment = sorted({item["name"] for item in context.entities if item.get("type") == "Equipment"})
        labs = sorted({item["name"] for item in context.entities if item.get("type") == "Laboratory"})

        if not context.facts:
            if context.evidence:
                return (
                    f"По {subject} найдены релевантные фрагменты, но структурированных "
                    "экспериментальных фактов в графе недостаточно. Используйте источники "
                    "ниже как evidence, но это не считается подтверждённым graph fact."
                )
            return f"По {subject} структурированных фактов в графе не найдено."

        parts = [f"По {subject} найдено экспериментов: {len(experiments) or len(context.grouped_facts)}."]
        if regimes:
            parts.append(f"Режимы: {', '.join(regimes)}.")
        if properties:
            parts.append(f"Измерялись свойства: {', '.join(properties)}.")
        if equipment:
            parts.append(f"Оборудование: {', '.join(equipment)}.")
        if labs:
            parts.append(f"Лаборатории/команды: {', '.join(labs)}.")
        if context.gaps:
            parts.append(f"Пробелы в данных: {len(context.gaps)}.")
        if context.sources:
            source_names = sorted({str(src.get("title") or src.get("filename")) for src in context.sources if src.get("title") or src.get("filename")})
            if source_names:
                parts.append(f"Источники: {', '.join(source_names[:5])}.")
        return " ".join(parts)

    def _comparison(self, plan: AnalyticalQueryPlan, context: GraphContext) -> str:
        if not context.facts:
            return "Для сравнения в графе не найдено структурированных фактов по указанным ограничениям."

        by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        use_materials = len(plan.constraints.materials) >= 2
        key_name = "material" if use_materials else "regime"
        requested = plan.constraints.materials if use_materials else plan.constraints.regimes
        for row in context.facts:
            key = str(row.get(key_name) or "")
            if key:
                by_entity[key].append(row)

        lines = [f"Сравнение по графу для: {', '.join(requested) if requested else 'найденных сущностей'}."]
        regimes = {str(row.get("regime")) for row in context.facts if row.get("regime")}
        units = {str(row.get("unit")) for row in context.facts if row.get("unit")}
        for entity, rows in sorted(by_entity.items()):
            measurements = _measurement_summaries(rows)
            lines.append(f"- {entity}: {measurements or 'нет численных измерений'}")
        if len(regimes) > 1 or len(units) > 1:
            lines.append(
                "Предупреждение: прямое сравнение ограничено, потому что режимы "
                "или единицы измерения различаются между найденными фактами."
            )
        else:
            lines.append("Найденные факты сопоставимы по режиму и единицам в пределах загруженного корпуса.")
        return " ".join(lines)

    def _history(self, plan: AnalyticalQueryPlan, context: GraphContext) -> str:
        material = _first(plan.constraints.materials) or "материалу"
        if not context.decision_history:
            return f"История решений по {material} в графе не найдена."
        parts = [f"История решений по {material}: найдено записей: {len(context.decision_history)}."]
        for item in context.decision_history[:5]:
            measurements = _measurement_summaries([m if isinstance(m, dict) else m.model_dump() for m in item.get("measurements", [])])
            regime = item.get("regime") or "режим не указан"
            parts.append(f"{item.get('experiment_id')}: {regime}; измерения: {measurements or 'нет'}")
        return " ".join(parts)

    def _gaps(self, plan: AnalyticalQueryPlan, context: GraphContext) -> str:
        if not context.gaps:
            return "Релевантных пробелов в данных по заданным ограничениям в графе не найдено."
        lines = ["Найдены пробелы в данных:"]
        for index, gap in enumerate(context.gaps[:10], start=1):
            subject = " + ".join(str(gap.get(key)) for key in ["material", "regime", "property"] if gap.get(key))
            lines.append(f"{index}. {subject or 'область не уточнена'}: {gap.get('reason')}")
        return " ".join(lines)

    def _search(self, plan: AnalyticalQueryPlan, context: GraphContext) -> str:
        if context.facts:
            return self._overview(plan, context)
        if context.evidence:
            return (
                "Найдены релевантные документы и фрагменты, но структурированных "
                "graph facts недостаточно для утверждения. Смотрите evidence и источники."
            )
        return "По запросу не найдено релевантных структурированных фактов или evidence."

    def _neighborhood(self, plan: AnalyticalQueryPlan, context: GraphContext) -> str:
        target = _first(plan.constraints.materials) or _first(plan.constraints.regimes) or _first(plan.constraints.properties) or "запроса"
        grouped: dict[str, set[str]] = defaultdict(set)
        for item in context.entities:
            grouped[str(item.get("type") or "Entity")].add(str(item.get("name")))
        if not grouped:
            return f"Связанные сущности вокруг {target} в графе не найдены."
        parts = [f"Связанные сущности вокруг {target}:"]
        for entity_type, names in sorted(grouped.items()):
            parts.append(f"{entity_type}: {', '.join(sorted(names)[:8])}.")
        return " ".join(parts)

    def _similar(self, plan: AnalyticalQueryPlan, context: GraphContext) -> str:
        if not context.facts:
            return "Похожих экспериментов в графе не найдено."
        lines = ["Похожие эксперименты найдены по совпадению материала, режима, свойства, оборудования и лаборатории."]
        for row in context.facts[:8]:
            score = row.get("similarity_score")
            score_text = f"; score={score:.2f}" if isinstance(score, float) else ""
            lines.append(
                f"- {row.get('experiment_id')}: {row.get('material')}, "
                f"{row.get('regime')}, {row.get('property')}{score_text}"
            )
        return " ".join(lines)


def _first(values: list[str]) -> str | None:
    return values[0] if values else None


def _measurement_summaries(rows: list[dict[str, Any]]) -> str:
    values = []
    for row in rows:
        prop = row.get("property") or row.get("property_name")
        value = row.get("value")
        raw_value = row.get("raw_value")
        unit = row.get("unit")
        effect = row.get("effect")
        if prop:
            measurement = str(prop)
            if value is not None or raw_value:
                measurement += f"={value if value is not None else raw_value}"
            if unit:
                measurement += f" {unit}"
            if effect:
                measurement += f" ({effect})"
            values.append(measurement)
    return "; ".join(dict.fromkeys(values))

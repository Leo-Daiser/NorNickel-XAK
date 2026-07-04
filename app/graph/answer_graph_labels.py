"""Human-readable label enrichment for the answer graph."""

from __future__ import annotations

import re
from typing import Any


FORBIDDEN_DISPLAY_RE = re.compile(
    r"\b(?:doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|EXP-[A-Za-z0-9_-]+|SCI-[A-Za-z0-9_-]+|"
    r"Experiment|PropertyValue|SourceChunk|FACT_SUPPORTED_BY_CHUNK|OF_PROPERTY|STUDIES|MEASURES|USES_REGIME|"
    r"increase|decrease|unknown)\b"
)


KNOWN_MATERIALS: dict[str, tuple[str, ...]] = {
    "вт6": ("титановый сплав", "аналог Ti-6Al-4V"),
    "ti-6al-4v": ("титановый сплав", "аналог ВТ6"),
    "7075-t6": ("алюминиевый сплав", "состояние T6"),
    "7075": ("алюминиевый сплав", "серия 7xxx"),
    "12х18н10т": ("нержавеющая сталь", "аналог AISI 321"),
    "aisi 321": ("нержавеющая сталь", "аналог 12Х18Н10Т"),
    "09г2с": ("низколегированная сталь",),
    "aisi 304": ("нержавеющая сталь",),
}

KNOWN_REGIMES: dict[str, tuple[str, ...]] = {
    "отжиг": ("термообработка",),
    "старение": ("термообработка",),
    "закалка": ("термообработка",),
    "криообработка": ("низкотемпературная обработка",),
    "термообработка": ("режим обработки",),
}

KNOWN_PROPERTIES: dict[str, tuple[str, ...]] = {
    "прочность": ("диапазон/значения",),
    "твердость": ("измеряемое свойство",),
    "твёрдость": ("измеряемое свойство",),
    "вязкость": ("измеряемое свойство",),
    "пластичность": ("измеряемое свойство",),
    "коррозионная стойкость": ("измеряемое свойство",),
    "удлинение": ("измеряемое свойство",),
}


def enrich_answer_graph_labels(graph: Any, payload: dict[str, Any]) -> Any:
    """Add compact explanatory labels and tooltips to answer graph nodes."""

    facts = _facts(payload)
    evidence = payload.get("evidence") or payload.get("sources") or []
    for node in graph.nodes:
        display_label, tooltip = enrich_node_label(
            node_type=str(node.type),
            label=str(node.label),
            facts=facts,
            evidence=evidence,
            details=getattr(node, "details", None) or {},
            existing_tooltip=getattr(node, "tooltip", None),
        )
        node.label = display_label
        node.tooltip = tooltip
    return graph


def enrich_node_label(
    *,
    node_type: str,
    label: str,
    facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    details: dict[str, Any] | None = None,
    existing_tooltip: str | None = None,
) -> tuple[str, str]:
    """Return a short display label and a safe tooltip for one answer graph node."""

    clean = _clean(label)
    if node_type == "material":
        lines = _material_lines(clean, facts)
    elif node_type == "regime":
        lines = _known_or_generic(clean, KNOWN_REGIMES, "режим обработки", title_case=True)
    elif node_type == "property":
        unit = _dominant_unit(facts, clean)
        lines = _known_or_generic(clean, KNOWN_PROPERTIES, "измеряемое свойство", title_case=True)
        if unit and unit not in " ".join(lines):
            lines.append(unit)
    elif node_type == "source_summary":
        lines = [clean or "источники", "подтверждают данные"] if evidence else [clean or "источники", "источники не указаны"]
    elif node_type == "measurement_summary":
        lines = [clean, "сводка измерений"] if clean else ["сводка измерений"]
    elif node_type == "fact":
        lines = [clean, "подтверждённый факт"] if clean else ["подтверждённый факт"]
    elif node_type == "gap":
        lines = [clean, "нет exact-факта"] if clean else ["нет exact-факта"]
    elif node_type == "conclusion":
        lines = [clean, "вывод по данным"] if clean else ["вывод по данным"]
    else:
        lines = [clean or "узел", "данные из корпуса"]

    display = "\n".join(_limit_lines(lines, max_lines=3, max_chars=24))
    tooltip_parts = [clean]
    tooltip_parts.extend(part for part in lines[1:] if part)
    if existing_tooltip:
        tooltip_parts.append(_clean(existing_tooltip, preserve_newlines=True))
    if details:
        tooltip_parts.extend(_details_tooltip(details))
    tooltip = "\n".join(_dedupe(_clean(part, preserve_newlines=True) for part in tooltip_parts if part))
    return display, tooltip or display


def _material_lines(label: str, facts: list[dict[str, Any]]) -> list[str]:
    known = _lookup_known(label, KNOWN_MATERIALS)
    if known:
        return [label, *known]
    corpus_hint = _material_hint_from_facts(label, facts)
    if corpus_hint:
        return [label, corpus_hint]
    return [label or "материал", "материал из корпуса"]


def _material_hint_from_facts(label: str, facts: list[dict[str, Any]]) -> str | None:
    text = " ".join(
        str(value)
        for fact in facts
        if _same_text(str(fact.get("material") or ""), label)
        for value in [fact.get("material"), fact.get("description"), fact.get("quote")]
        if value
    ).lower()
    if "алюмини" in text:
        return "алюминиевый сплав"
    if "титан" in text or "ti-" in text:
        return "титановый сплав"
    if "сталь" in text or "steel" in text:
        return "сталь"
    return None


def _known_or_generic(label: str, known: dict[str, tuple[str, ...]], generic: str, *, title_case: bool = False) -> list[str]:
    display = _title_first(label) if title_case else label
    hints = _lookup_known(label, known)
    return [display or generic, *(hints or (generic,))]


def _lookup_known(label: str, known: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    key = _norm(label)
    if key in known:
        return known[key]
    for candidate, value in known.items():
        if candidate and (candidate in key or key in candidate):
            return value
    return None


def _dominant_unit(facts: list[dict[str, Any]], property_label: str) -> str | None:
    units: dict[str, int] = {}
    for fact in facts:
        if property_label and not _same_text(str(fact.get("property") or ""), property_label):
            continue
        unit = _clean(str(fact.get("unit") or ""))
        if unit:
            units[unit] = units.get(unit, 0) + 1
    if units:
        return max(units.items(), key=lambda item: item[1])[0]
    return None


def _facts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("primary_facts") or payload.get("facts") or []
    return [row for row in rows if isinstance(row, dict)]


def _details_tooltip(details: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key in ("material", "regime", "property", "unit"):
        value = details.get(key)
        if value:
            result.append(f"{key}: {value}")
    return result


def _limit_lines(lines: list[str], *, max_lines: int, max_chars: int) -> list[str]:
    result: list[str] = []
    for line in lines:
        clean = _shorten(_clean(line), max_chars)
        if clean and clean not in result:
            result.append(clean)
        if len(result) >= max_lines:
            break
    return result or ["узел"]


def _clean(value: str, *, preserve_newlines: bool = False) -> str:
    text = str(value or "")
    text = text.replace("\\n", "\n")
    text = FORBIDDEN_DISPLAY_RE.sub("", text)
    if preserve_newlines:
        text = "\n".join(re.sub(r"\s+", " ", line).strip(" :-") for line in text.splitlines())
        return "\n".join(line for line in text.splitlines() if line)
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip(" :-")


def _shorten(value: str, limit: int) -> str:
    return value[: limit - 1] + "…" if len(value) > limit else value


def _title_first(value: str) -> str:
    if not value:
        return value
    return value[:1].upper() + value[1:]


def _norm(value: str) -> str:
    return _clean(value).lower().replace("ё", "е")


def _same_text(left: str, right: str) -> bool:
    left_norm = _norm(left)
    right_norm = _norm(right)
    return bool(left_norm and right_norm and (left_norm == right_norm or left_norm in right_norm or right_norm in left_norm))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result

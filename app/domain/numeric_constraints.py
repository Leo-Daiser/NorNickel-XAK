"""Deterministic numeric constraint parsing for technical R&D queries."""

from __future__ import annotations

import re
from typing import Any

from .aliases import MATERIAL_ALIASES, PROPERTY_ALIASES
from .normalization import canonical_material, canonical_property, normalize_text
from .unit_normalization import normalize_unit_label


UNIT_RE = (
    r"мг\s*/\s*(?:л|дм\s*[³3])|mg\s*/\s*(?:l|dm\s*3)|"
    r"г\s*/\s*л|g\s*/\s*l|"
    r"м\s*/\s*с|m\s*/\s*s|"
    r"м3\s*/\s*ч|m3\s*/\s*h|"
    r"т\s*/\s*сут|t\s*/\s*day|t\s*/\s*d|"
    r"т\s*/\s*год|т\s*/\s*г|t\s*/\s*y|t\s*/\s*year|tpa|"
    r"кт\s*/\s*год|тыс\.?\s*т\s*/\s*год|kt\s*/\s*y|kt\s*/\s*year|ktpa|"
    r"млн\.?\s*т\s*/\s*год|мт\s*/\s*год|Mt\s*/\s*y|Mt\s*/\s*year|Mtpa|"
    r"USD\s*/\s*t|\$\s*/\s*t|руб\.?\s*/\s*т|RUB\s*/\s*t|"
    r"USD\s*/\s*m3|руб\.?\s*/\s*м[³3]|RUB\s*/\s*m3|"
    r"млн\s*руб(?:\s*/\s*год)?|"
    r"°?\s*[CС]|%|ppm|MPa|МПа|ksi|HV|HRC"
)
NUMBER_RE = r"\d+(?:[.,]\d+)?"

OPERATOR_ALIASES = {
    "≤": "<=",
    "<=": "<=",
    "не более": "<=",
    "не выше": "<=",
    "до": "<=",
    "less than": "<",
    "<": "<",
    "≥": ">=",
    ">=": ">=",
    "не менее": ">=",
    "от": ">=",
    "at least": ">=",
    ">": ">",
    "=": "=",
}

PARAMETER_ALIASES = {
    **{alias: canonical for alias, canonical in MATERIAL_ALIASES.items()},
    **{alias: canonical for alias, canonical in PROPERTY_ALIASES.items()},
    "сухой остаток": "сухой остаток",
    "tds": "сухой остаток",
    "total dissolved solids": "сухой остаток",
    "сульфаты": "сульфаты",
    "хлориды": "хлориды",
    "ca": "Ca",
    "mg": "Mg",
    "na": "Na",
    "скорость": "скорость потока",
    "скорость потока": "скорость потока",
    "скорость циркуляции": "скорость потока",
    "flow velocity": "скорость потока",
    "расход": "расход",
    "расход раствора": "расход",
    "расход католита": "расход",
    "flow rate": "расход",
    "температура": "температура",
    "temperature": "температура",
    "производительность": "производительность",
    "throughput": "производительность",
    "capacity": "производительность",
}


def extract_numeric_constraints(text: str) -> list[dict[str, Any]]:
    """Extract numeric ranges/operators without using LLMs.

    The parser is intentionally conservative: it only emits a constraint when a
    nearby parameter/substance label and a unit are present in the same local
    phrase.
    """

    raw = str(text or "")
    constraints: list[dict[str, Any]] = []
    constraints.extend(_extract_list_range_constraints(raw))
    constraints.extend(_extract_single_range_constraints(raw))
    constraints.extend(_extract_operator_constraints(raw))
    constraints.extend(_extract_exact_value_constraints(raw))
    return _dedupe_constraints(constraints)


def _extract_list_range_constraints(text: str) -> list[dict[str, Any]]:
    aliases = sorted(PARAMETER_ALIASES, key=len, reverse=True)
    alias_group = "|".join(re.escape(alias) for alias in aliases)
    pattern = re.compile(
        rf"(?P<params>(?:{alias_group})(?:\s*(?:,|и|and)\s*(?:{alias_group})){{1,8}})"
        rf"\s+по\s+(?P<left>{NUMBER_RE})\s*[-–—]\s*(?P<right>{NUMBER_RE})\s*(?P<unit>{UNIT_RE})",
        re.IGNORECASE,
    )
    rows: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        for parameter in _split_parameter_list(match.group("params")):
            canonical = _canonical_parameter(parameter)
            if canonical:
                rows.append(
                    _range_constraint(
                        canonical,
                        match.group("left"),
                        match.group("right"),
                        match.group("unit"),
                        match.group(0),
                    )
                )
    return rows


def _extract_single_range_constraints(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        rf"(?P<context>[A-Za-zА-Яа-я0-9₂²³/ ._-]{{0,70}}?)"
        rf"(?P<left>{NUMBER_RE})\s*(?:[-–—]|до)\s*(?P<right>{NUMBER_RE})\s*(?P<unit>{UNIT_RE})",
        re.IGNORECASE,
    )
    rows: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        parameter = _parameter_from_context(match.group("context"))
        if parameter:
            rows.append(_range_constraint(parameter, match.group("left"), match.group("right"), match.group("unit"), match.group(0)))
    return rows


def _extract_operator_constraints(text: str) -> list[dict[str, Any]]:
    operator_group = "|".join(re.escape(item) for item in sorted(OPERATOR_ALIASES, key=len, reverse=True))
    pattern = re.compile(
        rf"(?P<context>[A-Za-zА-Яа-я0-9₂²³/ ._—–-]{{0,80}}?)"
        rf"\s*(?:—|–|-|:)?\s*"
        rf"(?P<operator>{operator_group})\s*(?P<value>{NUMBER_RE})\s*(?P<unit>{UNIT_RE})",
        re.IGNORECASE,
    )
    rows: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        parameter = _parameter_from_context(match.group("context"))
        if not parameter:
            continue
        value = _float(match.group("value"))
        unit = normalize_unit_label(match.group("unit"))
        rows.append(
            {
                "parameter": _parameter_for_unit(parameter, unit),
                "operator": OPERATOR_ALIASES.get(normalize_text(match.group("operator")), match.group("operator")),
                "value": value,
                "value_min": None,
                "value_max": None,
                "unit": unit,
                "raw_text": _clean(match.group(0)),
            }
        )
    return rows


def _extract_exact_value_constraints(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        rf"(?P<context>[A-Za-zА-Яа-я0-9₂²³/ ._—–-]{{0,80}}?)"
        rf"(?<![-–—<>=])(?P<value>{NUMBER_RE})\s*(?P<unit>{UNIT_RE})",
        re.IGNORECASE,
    )
    rows: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        raw_match = match.group(0)
        if re.search(r"\d+(?:[.,]\d+)?\s*[-–—]\s*\d", raw_match):
            continue
        context_norm = normalize_text(match.group("context"))
        if any(context_norm.endswith(normalize_text(operator)) for operator in OPERATOR_ALIASES if operator not in {"=", "<", ">", "<=", ">=", "≤", "≥"}):
            continue
        parameter = _parameter_from_context(match.group("context"))
        if not parameter:
            continue
        rows.append(
            {
                "parameter": _parameter_for_unit(parameter, normalize_unit_label(match.group("unit"))),
                "operator": "=",
                "value": _float(match.group("value")),
                "value_min": None,
                "value_max": None,
                "unit": normalize_unit_label(match.group("unit")),
                "raw_text": _clean(match.group(0)),
            }
        )
    return rows


def _range_constraint(parameter: str, left: str, right: str, unit: str, raw_text: str) -> dict[str, Any]:
    normalized_unit = normalize_unit_label(unit)
    return {
        "parameter": _parameter_for_unit(parameter, normalized_unit),
        "operator": "range",
        "value": None,
        "value_min": _float(left),
        "value_max": _float(right),
        "unit": normalized_unit,
        "raw_text": _clean(raw_text),
    }


def _parameter_from_context(context: str) -> str:
    normalized_context = normalize_text(context)
    best: tuple[int, str] | None = None
    for alias, canonical in PARAMETER_ALIASES.items():
        alias_norm = normalize_text(alias)
        position = _alias_position(normalized_context, alias_norm)
        if not alias_norm or position < 0:
            continue
        if best is None or position > best[0]:
            best = (position, canonical)
    if best is None:
        return ""
    return _canonical_parameter(best[1])


def _alias_position(context: str, alias: str) -> int:
    if not alias:
        return -1
    if alias in {"ca", "mg", "na", "au", "ag"}:
        matches = list(re.finditer(rf"(?<![a-zа-я0-9]){re.escape(alias)}(?![a-zа-я0-9])", context, flags=re.IGNORECASE))
        return matches[-1].start() if matches else -1
    return context.rfind(alias)


def _split_parameter_list(value: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"\s*(?:,|/|\||\bи\b|\band\b)\s*", value, flags=re.IGNORECASE)
        if item.strip()
    ]


def _canonical_parameter(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = normalize_text(raw)
    mapped = PARAMETER_ALIASES.get(normalized, raw)
    if normalize_text(mapped) in {normalize_text(item) for item in PROPERTY_ALIASES.values()}:
        return canonical_property(mapped)
    if normalize_text(mapped) in {normalize_text(item) for item in MATERIAL_ALIASES.values()}:
        return canonical_material(mapped)
    return str(mapped).strip()


def _parameter_for_unit(parameter: str, unit: str) -> str:
    if parameter == "расход" and unit == "m/s":
        return "скорость потока"
    if parameter == "скорость потока" and unit in {"m3/h", "t/day"}:
        return "расход"
    return parameter


def _dedupe_constraints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("parameter"), row.get("operator"), row.get("value"), row.get("value_min"), row.get("value_max"), row.get("unit"))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _float(value: str) -> float:
    return float(str(value).replace(",", "."))


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" .;,\n\t"))

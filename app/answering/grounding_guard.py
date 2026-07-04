"""Grounding guard for optional LLM-polished answers.

The guard is intentionally conservative.  It does not try to prove every
sentence.  It blocks the risky demo failures: unsupported measurements,
known material/regime/property claims outside the grounded fact set and
strong positive claims when there are no facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..domain.aliases import MATERIAL_ALIASES, PROPERTY_ALIASES, REGIME_ALIASES
from ..domain.fact_normalization import measurement_normalization_fields
from ..domain.unit_normalization import normalize_unit_label


INTERNAL_ID_RE = re.compile(r"\b(?:doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|EXP-[A-Za-z0-9_-]+|SCI-[A-Za-z0-9_-]+)\b")
MEASUREMENT_RE = re.compile(
    r"(?<![A-Za-zА-Яа-я0-9])"
    r"(?P<left>\d+(?:[.,]\d+)?)"
    r"(?:\s*[-–—]\s*(?P<right>\d+(?:[.,]\d+)?))?"
    r"\s*(?P<unit>MPa|МПа|ksi|HV|HRC|%|°\s*[CС]|[CС]\b|ч\b|h\b)",
    re.IGNORECASE,
)
STRONG_CLAIM_RE = re.compile(
    r"\b(?:подтвержден[аоы]?|подтвержд[её]нн\w*|доказан\w*|лучше|выше|прочнее|"
    r"составил[ао]?|достиг\w*|показал\w*|имеет)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumericClaim:
    value: float
    unit: str
    text: str


@dataclass
class GroundingContext:
    allowed_materials: set[str] = field(default_factory=set)
    allowed_regimes: set[str] = field(default_factory=set)
    allowed_properties: set[str] = field(default_factory=set)
    allowed_numeric_values_original: list[NumericClaim] = field(default_factory=list)
    allowed_numeric_values_normalized: list[NumericClaim] = field(default_factory=list)
    allowed_units: set[str] = field(default_factory=set)
    allowed_source_names: set[str] = field(default_factory=set)
    allowed_effects: set[str] = field(default_factory=set)
    allowed_conflict_groups: list[dict[str, Any]] = field(default_factory=list)
    no_facts_mode: bool = False
    source_grounded_mode: bool = False

    def diagnostics(self) -> dict[str, Any]:
        return {
            "allowed_materials": sorted(self.allowed_materials),
            "allowed_regimes": sorted(self.allowed_regimes),
            "allowed_properties": sorted(self.allowed_properties),
            "allowed_numeric_values_original": [_claim_diag(item) for item in self.allowed_numeric_values_original],
            "allowed_numeric_values_normalized": [_claim_diag(item) for item in self.allowed_numeric_values_normalized],
            "allowed_units": sorted(self.allowed_units),
            "allowed_source_names": sorted(self.allowed_source_names),
            "allowed_effects": sorted(self.allowed_effects),
            "allowed_conflict_groups_count": len(self.allowed_conflict_groups),
            "no_facts_mode": self.no_facts_mode,
            "source_grounded_mode": self.source_grounded_mode,
        }


@dataclass
class GroundingGuardResult:
    status: str
    violations: list[dict[str, str]]
    grounding_context: GroundingContext
    fallback_reason: str = ""
    first_pass: bool = False
    repair_attempted: bool = False
    repair_passed: bool = False
    repaired_violations: list[dict[str, str]] = field(default_factory=list)
    unsafe_answer_blocked: bool = False

    @property
    def passed(self) -> bool:
        return self.status in {"pass", "repaired"}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "first_pass": self.first_pass,
            "repair_attempted": self.repair_attempted,
            "repair_passed": self.repair_passed,
            "violations_count": len(self.violations),
            "violations": self.violations[:10],
            "repaired_violations_count": len(self.repaired_violations),
            "repaired_violations": self.repaired_violations[:10],
            "fallback_reason": self.fallback_reason,
            "unsafe_answer_blocked": self.unsafe_answer_blocked,
            "grounding_context": self.grounding_context.diagnostics(),
        }


def build_grounding_context(payload: dict[str, Any]) -> GroundingContext:
    """Build the set of claims an LLM-polished answer may use."""

    facts = [row for row in payload.get("facts") or [] if isinstance(row, dict)]
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    source_grounded_mode = bool(payload.get("source_grounded_answer_used")) or str(payload.get("answer_mode") or "") == "source_grounded_answer"
    context = GroundingContext(no_facts_mode=not bool(facts), source_grounded_mode=source_grounded_mode)
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}

    for value in constraints.get("materials") or []:
        _add_material(context, value)
    for value in constraints.get("regimes") or []:
        _add_regime(context, value)
    for value in constraints.get("properties") or []:
        _add_property(context, value)

    for row in payload.get("materials") or []:
        _add_material(context, _name_value(row))
    for row in payload.get("laboratories") or []:
        _add_source(context, _name_value(row))

    for fact in facts:
        _add_material(context, fact.get("material"))
        _add_regime(context, fact.get("regime"))
        _add_property(context, fact.get("property"))
        _add_effect(context, fact.get("effect"))
        _add_numeric_claims(context, fact)
        for evidence in fact.get("evidence") or []:
            if isinstance(evidence, dict):
                _add_source(context, evidence.get("source_name") or evidence.get("title") or evidence.get("filename"))

    for row in [*(payload.get("sources") or []), *(payload.get("evidence") or [])]:
        if isinstance(row, dict):
            _add_source(context, row.get("source_name") or row.get("title") or row.get("filename"))

    for gap in [*(payload.get("data_gaps") or []), *(payload.get("gaps") or [])]:
        if not isinstance(gap, dict):
            continue
        _add_material(context, gap.get("material"))
        _add_regime(context, gap.get("regime"))
        _add_property(context, gap.get("property"))
        for evidence in gap.get("evidence") or []:
            if isinstance(evidence, dict):
                _add_source(context, evidence.get("source_name") or evidence.get("title") or evidence.get("filename"))

    conflicts = diagnostics.get("fact_conflicts") or []
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            continue
        context.allowed_conflict_groups.append(_safe_conflict_group(conflict))
        _add_material(context, conflict.get("material"))
        _add_regime(context, conflict.get("regime"))
        _add_property(context, conflict.get("property"))
        for item in conflict.get("values") or []:
            if isinstance(item, dict):
                _add_numeric_pair(context.allowed_numeric_values_normalized, item.get("value"), item.get("unit"))
                _add_numeric_pair(context.allowed_numeric_values_original, item.get("value_original"), item.get("unit_original"))

    if any(item in context.allowed_regimes for item in {"отжиг", "старение", "закалка", "криообработка"}):
        context.allowed_regimes.add("термообработка")
    return context


def guard_llm_polished_answer(answer: str, payload: dict[str, Any]) -> GroundingGuardResult:
    """Validate a candidate LLM-polished answer against grounded claims."""

    context = build_grounding_context(payload)
    result = validate_text_against_grounding(answer, context, enforce_no_facts_policy=True)
    result.first_pass = result.status == "pass"
    if result.violations:
        result.status = "fallback"
        result.fallback_reason = "unsupported_llm_claims"
        result.unsafe_answer_blocked = True
    elif context.no_facts_mode and not context.source_grounded_mode:
        result.status = "fallback"
        result.fallback_reason = "no_facts_mode_deterministic_policy"
        result.first_pass = False
        result.unsafe_answer_blocked = bool(str(answer or "").strip())
    return result


def validate_text_against_payload(answer: str, payload: dict[str, Any]) -> GroundingGuardResult:
    """Validate final answer text without forcing the no-facts deterministic policy."""

    return validate_text_against_grounding(answer, build_grounding_context(payload), enforce_no_facts_policy=False)


def validate_text_against_grounding(
    answer: str,
    context: GroundingContext,
    *,
    enforce_no_facts_policy: bool,
) -> GroundingGuardResult:
    text = str(answer or "")
    violations: list[dict[str, str]] = []
    if INTERNAL_ID_RE.search(text):
        violations.append(_violation("raw_id", "internal id", "LLM answer contains an internal provenance id"))

    for claim in extract_measurement_claims(text):
        if context.source_grounded_mode:
            continue
        if context.no_facts_mode and not context.source_grounded_mode:
            violations.append(_violation("unsupported_numeric", claim.text, "no_facts_mode blocks numeric measurements"))
            continue
        if not _is_supported_numeric_claim(claim, context):
            violations.append(_violation("unsupported_numeric", claim.text, "numeric value/unit is absent from grounded facts"))

    if not context.source_grounded_mode:
        for entity_type, mentioned, allowed in [
            ("unsupported_material", _mentioned_aliases(text, MATERIAL_ALIASES), context.allowed_materials),
            ("unsupported_regime", _mentioned_aliases(text, REGIME_ALIASES), context.allowed_regimes),
            ("unsupported_property", _mentioned_aliases(text, PROPERTY_ALIASES), context.allowed_properties),
        ]:
            for canonical, matched in mentioned:
                if canonical not in allowed:
                    violations.append(_violation(entity_type, matched, f"{canonical} is absent from grounded claims"))

    if context.no_facts_mode and not context.source_grounded_mode:
        for match in STRONG_CLAIM_RE.finditer(text):
            if not _has_negation_nearby(text, match.start(), match.end()):
                violations.append(_violation("unsupported_strong_claim", match.group(0), "no_facts_mode blocks positive conclusions"))

    status = "fallback" if violations else "pass"
    if enforce_no_facts_policy and context.no_facts_mode and not context.source_grounded_mode and not violations:
        status = "fallback"
    return GroundingGuardResult(status=status, violations=violations, grounding_context=context)


def skipped_guard_diagnostics(reason: str = "llm_polish_not_used") -> dict[str, Any]:
    return {
        "status": "skipped",
        "first_pass": False,
        "repair_attempted": False,
        "repair_passed": False,
        "violations_count": 0,
        "violations": [],
        "repaired_violations_count": 0,
        "repaired_violations": [],
        "fallback_reason": reason,
        "unsafe_answer_blocked": False,
        "grounding_context": {},
    }


def build_repair_request(
    *,
    question: str,
    unsafe_answer: str,
    deterministic_answer: str,
    first_result: GroundingGuardResult,
) -> dict[str, Any]:
    context = first_result.grounding_context.diagnostics()
    return {
        "question": _safe_text(question),
        "unsafe_answer": _safe_text(unsafe_answer),
        "deterministic_answer": _safe_text(deterministic_answer),
        "allowed_materials": context["allowed_materials"],
        "allowed_regimes": context["allowed_regimes"],
        "allowed_properties": context["allowed_properties"],
        "allowed_numeric_values_original": context["allowed_numeric_values_original"],
        "allowed_numeric_values_normalized": context["allowed_numeric_values_normalized"],
        "allowed_units": context["allowed_units"],
        "allowed_source_names": context["allowed_source_names"],
        "allowed_effects": context["allowed_effects"],
        "allowed_conflict_groups_count": context["allowed_conflict_groups_count"],
        "no_facts_mode": context["no_facts_mode"],
        "violations": first_result.violations[:10],
        "instructions": [
            "Rewrite the answer in Russian, in a readable product style.",
            "Do not add numbers outside allowed_numeric_values_original or allowed_numeric_values_normalized.",
            "Do not add materials outside allowed_materials.",
            "Do not add regimes outside allowed_regimes.",
            "Do not add properties outside allowed_properties.",
            "Do not make conclusions not supported by the grounded context.",
            "If no_facts_mode is true, say that grounded facts are missing and do not include measurements.",
            "Do not include raw document ids, chunk ids, tracebacks or technical graph labels.",
        ],
    }


def diagnostics_after_repair(
    first_result: GroundingGuardResult,
    repair_result: GroundingGuardResult | None,
    *,
    fallback_reason: str = "",
) -> dict[str, Any]:
    diagnostics = first_result.diagnostics()
    diagnostics["repair_attempted"] = True
    diagnostics["unsafe_answer_blocked"] = True
    if repair_result is not None and repair_result.status == "pass":
        diagnostics["status"] = "repaired"
        diagnostics["repair_passed"] = True
        diagnostics["repaired_violations_count"] = 0
        diagnostics["repaired_violations"] = []
        diagnostics["fallback_reason"] = ""
        return diagnostics
    diagnostics["status"] = "fallback"
    diagnostics["repair_passed"] = False
    diagnostics["fallback_reason"] = fallback_reason or "repair_failed_grounding_guard"
    repaired_violations = repair_result.violations if repair_result is not None else []
    diagnostics["repaired_violations_count"] = len(repaired_violations)
    diagnostics["repaired_violations"] = repaired_violations[:10]
    return diagnostics


def extract_measurement_claims(answer: str) -> list[NumericClaim]:
    claims: list[NumericClaim] = []
    for match in MEASUREMENT_RE.finditer(str(answer or "")):
        unit = _normalize_guard_unit(match.group("unit"))
        for group_name in ["left", "right"]:
            raw_value = match.group(group_name)
            if raw_value is None:
                continue
            value = _float_or_none(raw_value)
            if value is None:
                continue
            text = f"{raw_value} {unit}".strip()
            claims.append(NumericClaim(value=value, unit=unit, text=text))
    return claims


def _add_numeric_claims(context: GroundingContext, fact: dict[str, Any]) -> None:
    value = fact.get("value") if fact.get("value") is not None else fact.get("raw_value")
    unit = fact.get("unit")
    _add_numeric_pair(context.allowed_numeric_values_original, fact.get("value_original"), fact.get("unit_original"))
    _add_numeric_pair(context.allowed_numeric_values_original, value, unit)
    _add_numeric_pair(context.allowed_numeric_values_normalized, fact.get("value_normalized"), fact.get("unit_normalized"))
    normalized = measurement_normalization_fields(fact.get("property"), value, unit)
    _add_numeric_pair(context.allowed_numeric_values_original, normalized.get("value_original"), normalized.get("unit_original"))
    _add_numeric_pair(context.allowed_numeric_values_normalized, normalized.get("value_normalized"), normalized.get("unit_normalized"))
    for claim in [*context.allowed_numeric_values_original, *context.allowed_numeric_values_normalized]:
        context.allowed_units.add(claim.unit)


def _add_numeric_pair(target: list[NumericClaim], value: Any, unit: Any) -> None:
    numeric = _float_or_none(value)
    normalized_unit = _normalize_guard_unit(unit)
    if numeric is None or not normalized_unit:
        return
    claim = NumericClaim(value=numeric, unit=normalized_unit, text=f"{numeric:g} {normalized_unit}")
    if not any(_same_claim(claim, existing) for existing in target):
        target.append(claim)


def _is_supported_numeric_claim(claim: NumericClaim, context: GroundingContext) -> bool:
    if claim.unit not in context.allowed_units:
        return False
    return any(
        _same_claim(claim, allowed)
        for allowed in [*context.allowed_numeric_values_original, *context.allowed_numeric_values_normalized]
    )


def _same_claim(left: NumericClaim, right: NumericClaim) -> bool:
    if left.unit != right.unit:
        return False
    tolerance = max(1.0, abs(right.value) * 0.005)
    return abs(left.value - right.value) <= tolerance


def _mentioned_aliases(text: str, aliases: dict[str, str]) -> list[tuple[str, str]]:
    normalized = str(text or "").lower().replace("ё", "е")
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for alias, canonical in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        alias_norm = alias.lower().replace("ё", "е")
        pattern = r"(?<![A-Za-zА-Яа-я0-9])" + re.escape(alias_norm) + r"(?![A-Za-zА-Яа-я0-9])"
        if re.search(pattern, normalized, re.IGNORECASE):
            canonical_norm = _canonical_value(canonical)
            key = (canonical_norm, alias)
            if key not in seen:
                seen.add(key)
                result.append((canonical_norm, alias))
    return result


def _add_material(context: GroundingContext, value: Any) -> None:
    for item in _iter_claim_values(value):
        normalized = _canonical_value(_alias_lookup(item, MATERIAL_ALIASES))
        if normalized:
            context.allowed_materials.add(normalized)


def _add_regime(context: GroundingContext, value: Any) -> None:
    for item in _iter_claim_values(value):
        normalized = _canonical_value(_alias_lookup(item, REGIME_ALIASES))
        if normalized:
            context.allowed_regimes.add(normalized)


def _add_property(context: GroundingContext, value: Any) -> None:
    for item in _iter_claim_values(value):
        normalized = _canonical_value(_alias_lookup(item, PROPERTY_ALIASES))
        if normalized:
            context.allowed_properties.add(normalized)


def _add_source(context: GroundingContext, value: Any) -> None:
    safe = _safe_source_name(value)
    if safe:
        context.allowed_source_names.add(safe)


def _add_effect(context: GroundingContext, value: Any) -> None:
    effect = str(value or "").strip().lower()
    if effect and effect not in {"unknown", "none", "null"}:
        context.allowed_effects.add(effect)


def _alias_lookup(value: Any, aliases: dict[str, str]) -> str:
    raw = str(value or "").strip()
    return aliases.get(raw.lower().replace("ё", "е"), raw)


def _iter_claim_values(value: Any) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    parts = [
        item.strip()
        for item in re.split(r"\s*(?:[,;|/]|(?:\s+и\s+)|(?:\s+and\s+))\s*", raw, flags=re.IGNORECASE)
        if item.strip()
    ]
    return parts or [raw]


def _canonical_value(value: Any) -> str:
    return str(value or "").strip().lower().replace("ё", "е")


def _normalize_guard_unit(unit: Any) -> str:
    raw = str(unit or "").strip()
    if not raw:
        return ""
    lowered = raw.lower().replace(" ", "")
    if lowered in {"°c", "°с", "c", "с"}:
        return "°C"
    if lowered in {"ч", "h"}:
        return "h"
    if lowered == "%":
        return "%"
    normalized = normalize_unit_label(raw)
    return normalized or raw


def _has_negation_nearby(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 32) : start].lower()
    after = text[end : min(len(text), end + 32)].lower()
    return bool(re.search(r"(нет|не|без|отсутств|не\s+найден)", before + " " + after))


def _safe_source_name(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    raw = re.sub(r"^doc_[0-9a-fA-F]{8,64}_", "", raw)
    raw = INTERNAL_ID_RE.sub("", raw)
    raw = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", raw)
    raw = raw.replace("_", " ").replace("-", " ")
    raw = re.sub(r"\b(?:synthetic|demo|test|source|doc|chunk)\b", "", raw, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", raw).strip()[:80]


def _safe_text(value: Any) -> str:
    text = INTERNAL_ID_RE.sub("", str(value or ""))
    return re.sub(r"[ \t]{2,}", " ", text).strip()[:6000]


def _safe_conflict_group(conflict: dict[str, Any]) -> dict[str, Any]:
    return {
        "material": conflict.get("material"),
        "regime": conflict.get("regime"),
        "property": conflict.get("property"),
        "values_count": len(conflict.get("values") or []),
        "sources_count": conflict.get("sources_count"),
    }


def _name_value(row: Any) -> Any:
    if isinstance(row, dict):
        return row.get("canonical_name") or row.get("name") or row.get("label")
    return row


def _claim_diag(item: NumericClaim) -> dict[str, Any]:
    return {"value": round(item.value, 3), "unit": item.unit}


def _violation(kind: str, claim: str, reason: str) -> dict[str, str]:
    safe_claim = INTERNAL_ID_RE.sub("", str(claim or "")).strip()
    return {"kind": kind, "claim": safe_claim[:80], "reason": reason}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None

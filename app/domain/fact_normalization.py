"""Canonical fact keys, measurement normalization and conflict reporting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .normalization import canonical_material, canonical_property, canonical_regime
from .ontology import Evidence, Measurement
from .unit_normalization import normalize_strength_to_mpa, normalize_unit_label


NORMALIZED_VALUE_PRECISION = 3


def measurement_normalization_fields(
    property_name: str | None,
    value: Any,
    unit: str | None,
) -> dict[str, Any]:
    """Return explicit normalized numeric fields for a measurement.

    The conversion is deliberately conservative: only strength values are
    converted across units. Other numeric values keep their original unit.
    """

    numeric = _float_or_none(value)
    canonical_prop = canonical_property(property_name)
    unit_original = str(unit).strip() if unit is not None and str(unit).strip() else None
    unit_label = normalize_unit_label(unit_original)
    unit_normalized = unit_label or None
    value_normalized = numeric
    family = "raw_numeric" if numeric is not None else None

    if canonical_prop == "прочность":
        family = "strength"
        converted, _ = normalize_strength_to_mpa(numeric, unit_original)
        if converted is not None:
            value_normalized = converted
            unit_normalized = "MPa"
        elif unit_label:
            unit_normalized = unit_label

    return {
        "value_original": numeric,
        "unit_original": unit_original,
        "value_normalized": value_normalized,
        "unit_normalized": unit_normalized,
        "normalization_family": family,
    }


def with_normalized_measurement_fields(measurement: Measurement) -> Measurement:
    """Return a measurement with normalized numeric fields populated."""

    fields = measurement_normalization_fields(
        measurement.property_name,
        measurement.value if measurement.value is not None else measurement.raw_value,
        measurement.unit,
    )
    updates = {
        key: getattr(measurement, key, None) if getattr(measurement, key, None) is not None else value
        for key, value in fields.items()
    }
    return _model_copy(measurement, updates)


def canonical_fact_key(
    *,
    material: str | None = None,
    regime: str | None = None,
    property_name: str | None = None,
    value: Any = None,
    unit: str | None = None,
    effect: str | None = None,
    value_normalized: Any = None,
    unit_normalized: str | None = None,
    evidence: Iterable[Any] | None = None,
    include_source: bool = False,
) -> str:
    """Build a stable key for one canonical fact.

    Source identity is optional. For answer/report deduplication it is omitted
    so duplicate evidence can be merged into one canonical fact.
    """

    normalized = measurement_normalization_fields(property_name, value, unit)
    normalized_value = value_normalized if value_normalized is not None else normalized["value_normalized"]
    normalized_unit = unit_normalized or normalized["unit_normalized"]
    value_part = _rounded_value(normalized_value)
    effect_part = _effect_part(effect)
    parts: list[str] = [
        canonical_material(material),
        canonical_regime(regime),
        canonical_property(property_name),
        value_part,
        normalized_unit or "",
        effect_part,
    ]
    if include_source:
        parts.append(_evidence_identity(evidence or []))
    return "|".join(_key_part(part) for part in parts)


def canonical_fact_key_from_measurement(
    measurement: Measurement,
    *,
    material: str | None = None,
    regime: str | None = None,
    include_source: bool = False,
) -> str:
    return canonical_fact_key(
        material=material,
        regime=regime,
        property_name=measurement.property_name,
        value=measurement.value if measurement.value is not None else measurement.raw_value,
        unit=measurement.unit,
        effect=measurement.effect,
        value_normalized=measurement.value_normalized,
        unit_normalized=measurement.unit_normalized,
        evidence=measurement.evidence,
        include_source=include_source,
    )


def canonical_fact_key_from_row(row: dict[str, Any], *, include_source: bool = False) -> str:
    evidence = row.get("evidence") if isinstance(row.get("evidence"), list) else []
    return canonical_fact_key(
        material=row.get("material"),
        regime=row.get("regime"),
        property_name=row.get("property"),
        value=row.get("value") if row.get("value") is not None else row.get("raw_value"),
        unit=row.get("unit"),
        effect=row.get("effect"),
        value_normalized=row.get("value_normalized"),
        unit_normalized=row.get("unit_normalized"),
        evidence=evidence,
        include_source=include_source,
    )


def dedupe_measurements(
    measurements: list[Measurement],
    *,
    material: str | None = None,
    regime: str | None = None,
) -> list[Measurement]:
    """Merge duplicate measurements and preserve all evidence spans."""

    by_key: dict[str, Measurement] = {}
    for item in measurements:
        normalized = with_normalized_measurement_fields(item)
        key = canonical_fact_key_from_measurement(normalized, material=material, regime=regime)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = normalized
            continue
        by_key[key] = _merge_measurements(existing, normalized)
    return list(by_key.values())


def dedupe_fact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate API fact rows by canonical key while preserving evidence."""

    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("predicate"):
            normalized = dict(row)
            key = _legacy_fact_key(normalized)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = normalized
                continue
            merged = dict(existing)
            merged["evidence"] = _merge_evidence_payloads(existing.get("evidence") or [], normalized.get("evidence") or [])
            by_key[key] = merged
            continue
        normalized = _row_with_normalized_fields(row)
        key = canonical_fact_key_from_row(normalized)
        normalized["canonical_fact_key"] = key
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = normalized
            continue
        merged = dict(existing)
        merged["evidence"] = _merge_evidence_payloads(existing.get("evidence") or [], normalized.get("evidence") or [])
        merged["evidence_count"] = len(merged["evidence"])
        for field in ["source_chunk_id", "doc_id", "experiment_id"]:
            if not merged.get(field) and normalized.get(field):
                merged[field] = normalized.get(field)
        by_key[key] = merged
    return list(by_key.values())


def _legacy_fact_key(row: dict[str, Any]) -> str:
    parts = [
        "legacy",
        row.get("subject") or row.get("source") or row.get("object") or "",
        row.get("predicate") or "",
        row.get("object") or row.get("target") or row.get("value") or "",
        row.get("source_chunk_id") or "",
    ]
    return "|".join(_key_part(part) for part in parts)


def build_conflict_summary(rows: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    """Return report-only conflict groups for same material/regime/property."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = _row_with_normalized_fields(row)
        material = canonical_material(normalized.get("material"))
        regime = canonical_regime(normalized.get("regime"))
        prop = canonical_property(normalized.get("property"))
        if not (material and regime and prop):
            continue
        groups[(material, regime, prop)].append(normalized)

    conflicts: list[dict[str, Any]] = []
    for (material, regime, prop), items in groups.items():
        identities = {_value_identity(item) for item in items}
        identities.discard(("", ""))
        if len(identities) < 2:
            continue
        values = [_value_summary(item) for item in items]
        units = sorted({str(item.get("unit_normalized") or item.get("unit") or "") for item in items if item.get("unit_normalized") or item.get("unit")})
        conflicts.append(
            {
                "material": material,
                "regime": regime,
                "property": prop,
                "values": _dedupe_value_summaries(values),
                "units": units,
                "sources_count": _sources_count(items),
                "possible_reason": _possible_conflict_reason(items),
            }
        )
        if len(conflicts) >= limit:
            break
    return conflicts


def fact_rows_from_experiments(experiments: Iterable[Any]) -> list[dict[str, Any]]:
    """Flatten ExperimentFact-like objects into normalized fact rows."""

    rows: list[dict[str, Any]] = []
    for exp in experiments:
        materials = list(getattr(exp, "materials", None) or [""])
        regimes = list(getattr(exp, "regimes", None) or [""])
        measurements = list(getattr(exp, "measurements", None) or [])
        for material in materials:
            for regime in regimes:
                for measurement in measurements:
                    if not isinstance(measurement, Measurement):
                        continue
                    normalized = with_normalized_measurement_fields(measurement)
                    rows.append(
                        {
                            "experiment_id": getattr(exp, "experiment_id", None),
                            "material": material,
                            "regime": regime,
                            "property": normalized.property_name,
                            "value": normalized.value,
                            "value_min": normalized.value_min,
                            "value_max": normalized.value_max,
                            "raw_value": normalized.raw_value,
                            "unit": normalized.unit,
                            "effect": normalized.effect,
                            "analyte": normalized.analyte,
                            "fact_type": normalized.fact_type,
                            "source_adapter": normalized.source_adapter,
                            "value_original": normalized.value_original,
                            "unit_original": normalized.unit_original,
                            "value_normalized": normalized.value_normalized,
                            "unit_normalized": normalized.unit_normalized,
                            "normalization_family": normalized.normalization_family,
                            "evidence": [item.model_dump() for item in normalized.evidence or getattr(exp, "evidence", [])],
                        }
                    )
    return rows


def _merge_measurements(left: Measurement, right: Measurement) -> Measurement:
    evidence = _merge_evidence_objects(left.evidence, right.evidence)
    confidence_values = [value for value in [left.confidence, right.confidence] if value is not None]
    updates = {
        "evidence": evidence,
        "confidence": max(confidence_values) if confidence_values else left.confidence,
        "raw_value": left.raw_value if left.raw_value not in {None, ""} else right.raw_value,
    }
    for field in ["value_original", "unit_original", "value_normalized", "unit_normalized", "normalization_family"]:
        if getattr(left, field, None) is None and getattr(right, field, None) is not None:
            updates[field] = getattr(right, field)
    return _model_copy(left, updates)


def _row_with_normalized_fields(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    fields = measurement_normalization_fields(
        result.get("property"),
        result.get("value") if result.get("value") is not None else result.get("raw_value"),
        result.get("unit"),
    )
    for key, value in fields.items():
        result.setdefault(key, value)
        if result.get(key) is None:
            result[key] = value
    return result


def _merge_evidence_objects(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    seen = set()
    result: list[Evidence] = []
    for item in [*left, *right]:
        key = (item.document_id, item.chunk_id, item.quote)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _merge_evidence_payloads(left: list[Any], right: list[Any]) -> list[dict[str, Any]]:
    seen = set()
    result: list[dict[str, Any]] = []
    for item in [*left, *right]:
        if not isinstance(item, dict):
            continue
        key = (item.get("document_id") or item.get("doc_id"), item.get("chunk_id"), item.get("quote"))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _value_identity(row: dict[str, Any]) -> tuple[str, str]:
    value = _rounded_value(row.get("value_normalized"))
    unit = str(row.get("unit_normalized") or "")
    effect = _effect_part(row.get("effect"))
    if value:
        return value, f"{unit}|{effect}"
    return effect, ""


def _value_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": row.get("value_normalized"),
        "unit": row.get("unit_normalized"),
        "value_original": row.get("value_original"),
        "unit_original": row.get("unit_original"),
        "effect": None if _effect_part(row.get("effect")) == "" else row.get("effect"),
    }


def _dedupe_value_summaries(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result: list[dict[str, Any]] = []
    for item in values:
        key = tuple(sorted((key, str(value)) for key, value in item.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result[:10]


def _sources_count(rows: list[dict[str, Any]]) -> int:
    seen = set()
    for row in rows:
        for item in row.get("evidence") or []:
            if not isinstance(item, dict):
                continue
            key = (item.get("document_id") or item.get("doc_id") or item.get("source_name"), item.get("chunk_id"), item.get("quote"))
            if any(key):
                seen.add(key)
    return len(seen)


def _possible_conflict_reason(rows: list[dict[str, Any]]) -> str:
    units = {str(row.get("unit_original") or row.get("unit") or "") for row in rows if row.get("unit_original") or row.get("unit")}
    if len(units) > 1:
        return "values reported in different source units; normalized values are shown for comparison"
    effects = {_effect_part(row.get("effect")) for row in rows}
    effects.discard("")
    if len(effects) > 1:
        return "sources report different qualitative effects"
    return "sources report different numeric values for the same material/regime/property; check source conditions"


def _evidence_identity(items: Iterable[Any]) -> str:
    parts = []
    for item in items:
        if isinstance(item, Evidence):
            parts.append("|".join(str(value or "") for value in [item.document_id, item.chunk_id, item.quote]))
        elif isinstance(item, dict):
            parts.append("|".join(str(value or "") for value in [item.get("document_id") or item.get("doc_id"), item.get("chunk_id"), item.get("quote")]))
    return ";".join(sorted(parts))


def _effect_part(effect: Any) -> str:
    value = str(effect or "").strip().lower()
    return "" if value in {"", "unknown", "none", "null"} else value


def _rounded_value(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return ""
    return f"{round(numeric, NORMALIZED_VALUE_PRECISION):.{NORMALIZED_VALUE_PRECISION}f}".rstrip("0").rstrip(".")


def _key_part(value: Any) -> str:
    return str(value or "").strip().lower().replace("ё", "е")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_copy(model: Any, updates: dict[str, Any]) -> Any:
    if hasattr(model, "model_copy"):
        return model.model_copy(update=updates)
    return model.copy(update=updates)

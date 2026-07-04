"""Validation and rejection rules for extraction bundles."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..domain.aliases import MATERIAL_ALIASES
from ..domain.property_schema import PROPERTY_SCHEMA, has_required_property_marker
from ..domain.reference_loader import reference_entity_type
from .document_profile import DocumentProfile
from .models import (
    EvidenceSpan,
    ExtractedDataGap,
    ExtractedEntity,
    ExtractedExperiment,
    ExtractedMeasurement,
    RejectedExtraction,
)
from .resolver import resolve_unit


VALID_UNITS = {
    "MPa",
    "GPa",
    "ksi",
    "HV",
    "HRC",
    "%",
    "C",
    "h",
    "min",
    "mg/L",
    "g/L",
    "m/s",
    "m3/h",
    "t/day",
    "t/y",
    "kt/y",
    "Mt/y",
    "ppm",
    "г/т",
    "g/t",
    "mg/kg",
    "pH",
    "bar",
    "USD/t",
    "EUR/t",
    "RUB/t",
    "USD/m3",
    "RUB/m3",
    "mln RUB",
    "mln RUB/year",
}


@dataclass
class ValidationResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    experiments: list[ExtractedExperiment] = field(default_factory=list)
    data_gaps: list[ExtractedDataGap] = field(default_factory=list)
    rejected: list[RejectedExtraction] = field(default_factory=list)


def validate_entity(entity: ExtractedEntity) -> tuple[ExtractedEntity | None, RejectedExtraction | None]:
    if not entity.canonical_name:
        return None, _reject("entity", "missing_canonical_name", entity)
    if not 0 <= entity.confidence <= 1:
        return None, _reject("entity", "invalid_confidence", entity)
    if not entity.evidence:
        return None, _reject("entity", "missing_evidence", entity)
    if entity.entity_type == "Material":
        accepted, reason = _validate_material_entity(entity)
        if not accepted:
            return None, _reject("entity", reason, entity)
    return entity, None


def validate_measurement(
    measurement: ExtractedMeasurement,
    *,
    subject_types: set[str] | None = None,
    profile: DocumentProfile | None = None,
) -> tuple[ExtractedMeasurement | None, RejectedExtraction | None]:
    if not measurement.property_canonical:
        return None, _reject("measurement", "missing_property", measurement)
    if not measurement.evidence:
        return None, _reject("measurement", "missing_evidence", measurement)
    if measurement.value is None and measurement.effect == "unknown" and measurement.delta_abs is None and measurement.delta_rel_percent is None:
        return None, _reject("measurement", "measurement_without_value_or_effect", measurement)
    evidence_text = " ".join(item.quote.lower().replace("ё", "е") for item in measurement.evidence if item.quote)
    if measurement.property_canonical == "коррозионная стойкость" and "не измер" in evidence_text:
        return None, _reject("measurement", "gap_phrase_not_measurement", measurement)
    unit = resolve_unit(measurement.unit)
    if measurement.property_canonical == "прочность" and unit == "%":
        return None, _reject("measurement", "property_unit_mismatch", measurement)
    if measurement.property_canonical == "пластичность" and unit in {"MPa", "GPa"}:
        return None, _reject("measurement", "property_unit_mismatch", measurement)
    schema_rejection = _validate_property_schema(measurement, unit, subject_types or set(), profile)
    if schema_rejection:
        return None, _reject("measurement", schema_rejection, measurement)
    if measurement.value is not None and not _property_near_value(measurement):
        return None, _reject("measurement", "value_without_property_window", measurement)
    if unit and unit in VALID_UNITS:
        measurement = measurement.model_copy(update={"unit": unit}) if hasattr(measurement, "model_copy") else measurement.copy(update={"unit": unit})
    return measurement, None


def validate_experiment(
    experiment: ExtractedExperiment,
    min_confidence: float,
    profile: DocumentProfile | None = None,
) -> tuple[ExtractedExperiment | None, list[RejectedExtraction]]:
    rejected: list[RejectedExtraction] = []
    materials: list[ExtractedEntity] = []
    for material in experiment.materials:
        accepted_material, material_rejection = validate_entity(material)
        if accepted_material is not None:
            materials.append(accepted_material)
        if material_rejection is not None:
            rejected.append(material_rejection)
    equipment = _validated_support_entities(experiment.equipment, rejected)
    laboratories = _validated_support_entities(experiment.laboratories, rejected)
    teams = _validated_support_entities(experiment.teams, rejected)
    employees = _validated_support_entities(experiment.employees, rejected)
    topic_tags = _validated_support_entities(experiment.topic_tags, rejected)
    experiment = (
        experiment.model_copy(
            update={
                "materials": materials,
                "equipment": equipment,
                "laboratories": laboratories,
                "teams": teams,
                "employees": employees,
                "topic_tags": topic_tags,
            }
        )
        if hasattr(experiment, "model_copy")
        else experiment.copy(
            update={
                "materials": materials,
                "equipment": equipment,
                "laboratories": laboratories,
                "teams": teams,
                "employees": employees,
                "topic_tags": topic_tags,
            }
        )
    )
    if not experiment.materials:
        return None, [*rejected, _reject("experiment", "missing_material", experiment)]
    if not experiment.regimes and not experiment.measurements:
        return None, [_reject("experiment", "missing_regime_or_measurement", experiment)]
    if not experiment.evidence:
        return None, [_reject("experiment", "missing_evidence", experiment)]
    if experiment.regimes and not experiment.measurements and _gap_signal(experiment.evidence):
        return None, [_reject("experiment", "gap_only_not_experiment", experiment)]
    doc_rejection = _validate_experiment_document_policy(experiment, profile)
    if doc_rejection:
        return None, [_reject("experiment", doc_rejection, experiment)]
    measurements: list[ExtractedMeasurement] = []
    subject_types = _experiment_subject_types(experiment)
    for measurement in experiment.measurements:
        accepted, rejection = validate_measurement(measurement, subject_types=subject_types, profile=profile)
        if accepted is not None:
            measurements.append(accepted)
        if rejection is not None:
            rejected.append(rejection)
    updated = experiment.model_copy(update={"measurements": measurements}) if hasattr(experiment, "model_copy") else experiment.copy(update={"measurements": measurements})
    if experiment.measurements and not measurements:
        return None, [*rejected, _reject("experiment", "all_measurements_rejected", updated)]
    if updated.confidence < min_confidence:
        return None, [*rejected, _reject("experiment", "low_confidence", updated)]
    return updated, rejected


def validate_gap(gap: ExtractedDataGap) -> tuple[ExtractedDataGap | None, RejectedExtraction | None]:
    if not gap.reason:
        return None, _reject("data_gap", "missing_reason", gap)
    if not gap.evidence:
        return None, _reject("data_gap", "missing_evidence", gap)
    return gap, None


def validate_items(
    entities: list[ExtractedEntity],
    experiments: list[ExtractedExperiment],
    data_gaps: list[ExtractedDataGap],
    min_confidence: float,
    profile: DocumentProfile | None = None,
) -> ValidationResult:
    result = ValidationResult()
    for entity in entities:
        accepted, rejection = validate_entity(entity)
        if accepted is not None:
            result.entities.append(accepted)
        if rejection is not None:
            result.rejected.append(rejection)
    for experiment in experiments:
        accepted, rejections = validate_experiment(experiment, min_confidence=min_confidence, profile=profile)
        if accepted is not None:
            result.experiments.append(accepted)
        result.rejected.extend(rejections)
    for gap in data_gaps:
        accepted, rejection = validate_gap(gap)
        if accepted is not None:
            result.data_gaps.append(accepted)
        if rejection is not None:
            result.rejected.append(rejection)
    return result


def has_evidence_quotes(evidence: list[EvidenceSpan]) -> bool:
    return all(bool(item.quote.strip()) for item in evidence)


def _reject(item_type: str, reason: str, item) -> RejectedExtraction:
    payload = item.model_dump() if hasattr(item, "model_dump") else item.dict()
    evidence = payload.get("evidence") if isinstance(payload, dict) else []
    return RejectedExtraction(item_type=item_type, reason=reason, raw_payload=payload, evidence=evidence or [])


def _gap_signal(evidence: list[EvidenceSpan]) -> bool:
    text = " ".join(item.quote for item in evidence).lower().replace("ё", "е")
    return any(marker in text for marker in ["нет данных", "не измер", "отсутств", "missing data", "not measured"])


def _validate_material_entity(entity: ExtractedEntity) -> tuple[bool, str]:
    text = str(entity.canonical_name or entity.raw_name or "").strip()
    context = " ".join(item.quote for item in entity.evidence if item.quote)
    if not text:
        return False, "material_missing_name"
    if _looks_like_unit(text):
        return False, "unit_like_material"
    known_alias = _is_known_material_alias(text)
    chemical_formula = _looks_like_chemical_formula(text)
    material_grade = _looks_like_material_grade(text)
    context_support = _has_material_context(text, context)
    if _looks_like_pdf_font_code(text, context) and not (known_alias or chemical_formula):
        return False, "pdf_font_code_without_domain_context"
    if _looks_like_relation_fragment(text):
        return False, "relation_fragment_not_material"
    if _suspicious_code_like_entity(text) and not (known_alias or chemical_formula or material_grade):
        if not _has_strong_material_grade_context(text, context):
            return False, "suspicious_code_like_entity_without_reference"
    if known_alias or chemical_formula or material_grade or context_support:
        return True, ""
    return False, "material_without_positive_validation"


def _validate_experiment_document_policy(experiment: ExtractedExperiment, profile: DocumentProfile | None) -> str | None:
    if not profile:
        return None
    doc_type = profile.detected_type
    if doc_type in {"market_capacity_reference", "directory_or_catalog"}:
        mechanical = {"прочность", "твёрдость", "пластичность", "вязкость", "коррозионная стойкость"}
        if any(item.property_canonical in mechanical for item in experiment.measurements):
            return "doc_type_incompatible_with_mechanical_property_fact"
        heat_treatments = {"отжиг", "старение", "закалка", "криообработка", "термообработка"}
        if any(item.canonical_name in heat_treatments for item in experiment.regimes):
            return "doc_type_incompatible_with_heat_treatment_mrp"
    return None


def _validate_property_schema(
    measurement: ExtractedMeasurement,
    unit: str | None,
    subject_types: set[str],
    profile: DocumentProfile | None,
) -> str | None:
    schema = PROPERTY_SCHEMA.get(measurement.property_canonical)
    if not schema:
        return "unknown_property_schema"
    if measurement.value is not None:
        if unit not in schema.allowed_units:
            return "unit_incompatible_with_property"
        context = " ".join(item.quote for item in measurement.evidence if item.quote)
        if not has_required_property_marker(measurement.property_canonical, context):
            return "missing_required_property_marker"
    if profile and schema.compatible_doc_types and profile.detected_type not in schema.compatible_doc_types:
        return "doc_type_incompatible_with_property"
    if "ChemicalSubstance" in subject_types and measurement.property_canonical in {"пластичность", "прочность", "твёрдость", "вязкость"}:
        if subject_types.isdisjoint({"Alloy", "Metal", "MaterialSample"}):
            return "chemical_substance_incompatible_with_mechanical_property"
    if subject_types and schema.compatible_subject_types and subject_types.isdisjoint(schema.compatible_subject_types):
        return "subject_type_incompatible_with_property"
    return None


def _experiment_subject_types(experiment: ExtractedExperiment) -> set[str]:
    types: set[str] = set()
    for material in experiment.materials:
        types.add(_classify_material_subject(material.canonical_name, material.evidence))
    for regime in experiment.regimes:
        if regime.canonical_name in {"электроэкстракция", "циркуляция католита", "обессоливание", "закачка шахтных вод", "ПВП", "газоочистка"}:
            types.add("Process")
    return types


def _classify_material_subject(value: str | None, evidence: list[EvidenceSpan] | None = None) -> str:
    text = str(value or "")
    norm = text.lower().replace("ё", "е")
    ref_type = reference_entity_type("materials", text)
    if ref_type:
        return ref_type
    if _looks_like_chemical_formula(text) or norm in {"ca", "mg", "na", "au", "ag", "so2", "sio2", "al2o3", "tio2", "h2so4"}:
        return "ChemicalSubstance"
    if norm in {"сульфаты", "хлориды"}:
        return "ChemicalSubstance"
    if norm in {"шахтные воды"}:
        return "Water"
    if norm in {"католит", "электролит"}:
        return "Catholyte" if norm == "католит" else "Electrolyte"
    if norm in {"штейн"}:
        return "Matte"
    if norm in {"шлак"}:
        return "Slag"
    if norm in {"медь", "никель", "вт6", "7075-t6", "12х18н10т", "09г2с"} or _looks_like_material_grade(text):
        return "Alloy" if norm not in {"медь", "никель"} else "Metal"
    context = " ".join(item.quote for item in evidence or [] if item.quote).lower().replace("ё", "е")
    if any(marker in context for marker in ["руда", "ore"]):
        return "Ore"
    return "MaterialSample"


def _is_known_material_alias(value: str) -> bool:
    norm = value.lower().replace("ё", "е").strip()
    aliases = {key.lower().replace("ё", "е") for key in MATERIAL_ALIASES}
    canonicals = {val.lower().replace("ё", "е") for val in MATERIAL_ALIASES.values()}
    return norm in aliases or norm in canonicals


def _looks_like_chemical_formula(value: str) -> bool:
    text = value.strip()
    if not re.fullmatch(r"(?:[A-ZА-Я][a-zа-я]?\d*){1,6}(?:[+-]\d*)?", text):
        return False
    known_elements = {
        "H", "C", "N", "O", "S", "P", "Cl", "Ca", "Mg", "Na", "K", "Fe", "Cu", "Ni", "Co", "Zn",
        "Al", "Si", "Ti", "V", "Cr", "Mn", "Mo", "W", "Au", "Ag", "Pt", "Pd", "Rh", "Ir", "Ru",
    }
    tokens = re.findall(r"[A-Z][a-z]?", text)
    return bool(tokens) and all(token in known_elements for token in tokens)


def _looks_like_material_grade(value: str) -> bool:
    text = value.strip()
    patterns = [
        r"(?i)^vt-?6$",
        r"^ВТ-?6$",
        r"(?i)^ti-?6al-?4v$",
        r"(?i)^7075(?:-t6)?$",
        r"(?i)^aisi\s*(?:304|321|316)$",
        r"^09Г2С$",
        r"^12[ХX]18[НH]10[ТT]$",
    ]
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def _suspicious_code_like_entity(value: str) -> bool:
    text = value.strip()
    if not text or " " in text or len(text) > 18:
        return False
    alpha = len(re.findall(r"[A-Za-zА-Яа-яЁё]", text))
    digits = len(re.findall(r"\d", text))
    if digits < 2 or alpha < 1:
        return False
    digit_ratio = digits / max(1, len(re.sub(r"[^A-Za-zА-Яа-яЁё0-9]", "", text)))
    if digit_ratio < 0.35:
        return False
    if re.fullmatch(r"\d+(?:[.,]\d+)?", text):
        return False
    return bool(re.fullmatch(r"[A-Za-zА-Яа-яЁё]{1,6}[-_/]?\d{2,8}[A-Za-zА-Яа-яЁё0-9_-]*", text))


def _has_strong_material_grade_context(value: str, context: str) -> bool:
    text = context.lower().replace("ё", "е")
    token = re.escape(value.lower().replace("ё", "е"))
    strong_markers = [
        "марка",
        "grade",
        "сплав",
        "alloy",
        "сталь",
        "steel",
        "образец",
        "sample",
        "material grade",
    ]
    for marker in strong_markers:
        for match in re.finditer(re.escape(marker), text):
            window = text[max(0, match.start() - 80): match.end() + 80]
            if re.search(token, window):
                return True
    return False


def _has_material_context(value: str, context: str) -> bool:
    norm = value.lower().replace("ё", "е")
    if len(norm) < 3:
        return False
    if _looks_like_relation_fragment(value):
        return False
    window = context.lower().replace("ё", "е")
    return any(
        marker in window
        for marker in [
            "сплав",
            "сталь",
            "alloy",
            "steel",
            "material",
            "материал",
            "руда",
            "ore",
            "slag",
            "шлак",
            "matte",
            "штейн",
            "solution",
            "раствор",
        ]
    ) and bool(re.search(r"[A-Za-zА-Яа-яЁё]{3,}", norm))


def _looks_like_pdf_font_code(value: str, context: str) -> bool:
    text = value.strip()
    if not re.fullmatch(r"[A-ZА-Я]{1,4}\d{2,5}", text):
        return False
    ctx = context.lower()
    if any(marker in ctx for marker in ["/font", "fontdescriptor", "/mt", "glyph", "cidfont", "encoding"]):
        return True
    return bool(re.search(rf"/{re.escape(text)}\b", context))


def _looks_like_unit(value: str) -> bool:
    return bool(re.fullmatch(r"(?i)(mpa|gpa|hv|hrc|ppm|mg/l|g/l|m/s|m3/h|t/day|t/d|t/y|kt/y|mt/y|tpa|ktpa|mtpa|%|c|h|min|bar)", value.strip()))


def _looks_like_relation_fragment(value: str) -> bool:
    norm = value.lower().replace("ё", "е").strip()
    return norm in {"сплав из", "сплава из", "alloy of", "material of", "из"} or norm.endswith(" из")


def _validated_support_entities(items: list[ExtractedEntity], rejected: list[RejectedExtraction]) -> list[ExtractedEntity]:
    result: list[ExtractedEntity] = []
    for item in items:
        accepted, rejection = validate_entity(item)
        if accepted is not None:
            result.append(accepted)
        if rejection is not None:
            rejected.append(rejection)
    return result


def _property_near_value(measurement: ExtractedMeasurement) -> bool:
    text = " ".join(item.quote for item in measurement.evidence if item.quote).lower().replace("ё", "е")
    if not text:
        return False
    value = f"{float(measurement.value):g}" if measurement.value is not None else ""
    value_match = re.search(re.escape(value), text) if value else None
    if not value_match:
        return True
    terms = [measurement.property_raw, measurement.property_canonical]
    if measurement.property_canonical == "прочность":
        terms.extend(["прочност", "tensile strength", "ultimate tensile strength", "strength"])
    if measurement.property_canonical == "коррозионная стойкость":
        terms.extend(["коррозион", "corrosion resistance"])
    if measurement.property_canonical == "концентрация":
        terms.extend(["концентрац", "concentration", "сульфат", "хлорид", "sulfate", "sulphate", "chloride", "ca", "mg", "na"])
    if measurement.property_canonical == "сухой остаток":
        terms.extend(["сухой остаток", "tds", "total dissolved solids"])
    if measurement.property_canonical == "скорость потока":
        terms.extend(["скорость", "скорость потока", "циркуляц", "flow velocity", "velocity"])
    if measurement.property_canonical == "расход":
        terms.extend(["расход", "flow rate", "подача", "flow"])
    if measurement.property_canonical == "производительность":
        terms.extend(["производительность", "capacity", "throughput"])
    if measurement.property_canonical == "извлечение":
        terms.extend(["извлечение", "recovery"])
    if measurement.property_canonical == "выход металла":
        terms.extend(["выход", "metal yield", "yield"])
    if measurement.property_canonical == "распределение":
        terms.extend(["распределен", "коэффициент распределен", "distribution", "distribution coefficient"])
    if measurement.property_canonical == "экономический показатель":
        terms.extend(["capex", "opex", "затрат", "стоимост", "эконом", "cost"])
    window = text[max(0, value_match.start() - 100): value_match.end() + 100]
    return any(str(term or "").lower().replace("ё", "е") in window for term in terms if term)

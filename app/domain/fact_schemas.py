"""Typed fact schemas for validation/reporting.

These schemas are deliberately lightweight: they define accepted fact families
without making the system LLM-heavy or overfitting to one source file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FactTypeSchema:
    required_fields: set[str]
    compatible_doc_types: set[str] = field(default_factory=set)
    compatible_subject_types: set[str] = field(default_factory=set)
    compatible_properties: set[str] = field(default_factory=set)
    allowed_units: set[str] = field(default_factory=set)
    required_markers: set[str] = field(default_factory=set)


FACT_TYPE_SCHEMAS: dict[str, FactTypeSchema] = {
    "MechanicalPropertyFact": FactTypeSchema(
        required_fields={"material", "property", "value_or_effect", "evidence"},
        compatible_doc_types={"experiment_report", "materials_article", "unknown"},
        compatible_subject_types={"Alloy", "Metal", "MaterialSample"},
        compatible_properties={"прочность", "твёрдость", "пластичность", "вязкость"},
        allowed_units={"MPa", "GPa", "ksi", "HV", "HRC", "%"},
        required_markers={"прочность", "твёрдость", "твердость", "удлинение", "ductility", "elongation", "strength", "hardness"},
    ),
    "ProcessParameterFact": FactTypeSchema(
        required_fields={"process", "parameter", "value_or_effect", "evidence"},
        compatible_doc_types={"experiment_report", "process_report", "review_article", "unknown"},
        compatible_subject_types={"Process", "Solution", "Water", "Electrolyte", "Catholyte", "ChemicalSubstance"},
        compatible_properties={
            "концентрация",
            "сухой остаток",
            "скорость потока",
            "расход",
            "извлечение",
            "выход металла",
            "распределение",
            "содержание",
            "температура",
            "pH",
            "давление",
        },
        allowed_units={"mg/L", "g/L", "ppm", "m/s", "m3/h", "%", "г/т", "g/t", "mg/kg", "C", "pH", "bar", "MPa", "t/day"},
    ),
    "ExperimentResultFact": FactTypeSchema(
        required_fields={"experiment", "subject", "result", "evidence"},
        compatible_doc_types={"experiment_report", "materials_article", "process_report", "unknown"},
    ),
    "TechnologySolutionFact": FactTypeSchema(
        required_fields={"technology_or_process", "target_problem_or_domain", "evidence"},
        compatible_doc_types={"review_article", "patent", "presentation", "process_report", "report", "standard_or_normative", "unknown"},
        required_markers={
            "метод",
            "способ",
            "технология",
            "схема",
            "система",
            "решение",
            "применяется",
            "используется",
            "позволяет",
            "предназначен",
            "рекомендуется",
            "method",
            "technology",
            "approach",
            "system",
            "solution",
            "used for",
            "applied for",
            "designed for",
            "recommended",
        },
    ),
    "FacilityCapacityFact": FactTypeSchema(
        required_fields={"commodity", "capacity_or_scope", "evidence"},
        compatible_doc_types={"market_capacity_reference", "directory_or_catalog", "report", "unknown"},
        compatible_properties={"производительность"},
        allowed_units={"t/day", "t/y", "kt/y", "Mt/y", "m3/h", "%"},
        required_markers={"capacity", "production capacity", "производительность", "throughput", "мощность"},
    ),
    "EconomicIndicatorFact": FactTypeSchema(
        required_fields={"technology_or_process", "indicator", "value", "unit", "evidence"},
        compatible_doc_types={"experiment_report", "process_report", "review_article", "report", "unknown"},
        compatible_properties={"экономический показатель"},
        allowed_units={"USD/t", "EUR/t", "RUB/t", "USD/m3", "RUB/m3", "mln RUB", "mln RUB/year"},
    ),
    "PublicationClaimFact": FactTypeSchema(
        required_fields={"claim", "source", "evidence"},
        compatible_doc_types={"review_article", "publication", "patent", "presentation", "process_report", "experiment_report", "report", "unknown"},
    ),
    "ExpertiseFact": FactTypeSchema(
        required_fields={"expert_or_team", "topic", "evidence"},
        compatible_doc_types={"directory_or_catalog", "presentation", "publication", "review_article", "report", "unknown"},
    ),
    "DataGapFact": FactTypeSchema(
        required_fields={"missing_subject", "reason", "evidence"},
    ),
}


MECHANICAL_PROPERTIES = FACT_TYPE_SCHEMAS["MechanicalPropertyFact"].compatible_properties
PROCESS_PARAMETER_PROPERTIES = FACT_TYPE_SCHEMAS["ProcessParameterFact"].compatible_properties


def classify_measurement_fact_type(property_name: str | None, unit: str | None = None) -> str:
    prop = str(property_name or "")
    if prop in MECHANICAL_PROPERTIES:
        return "MechanicalPropertyFact"
    if prop in PROCESS_PARAMETER_PROPERTIES:
        return "ProcessParameterFact"
    if prop == "производительность":
        return "FacilityCapacityFact"
    if prop == "экономический показатель":
        return "EconomicIndicatorFact"
    if unit in {"USD/t", "EUR/t", "RUB/t", "USD/m3", "RUB/m3", "mln RUB", "mln RUB/year"}:
        return "EconomicIndicatorFact"
    return "ExperimentResultFact"


def schema_summary() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "required_fields": sorted(schema.required_fields),
            "compatible_doc_types": sorted(schema.compatible_doc_types),
            "compatible_subject_types": sorted(schema.compatible_subject_types),
            "compatible_properties": sorted(schema.compatible_properties),
            "allowed_units": sorted(schema.allowed_units),
            "required_markers": sorted(schema.required_markers),
        }
        for name, schema in FACT_TYPE_SCHEMAS.items()
    }

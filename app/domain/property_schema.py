"""Positive validation schema for extracted technical properties."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PropertySchema:
    allowed_units: set[str]
    required_markers: set[str]
    compatible_subject_types: set[str]
    compatible_doc_types: set[str] | None = None


MECHANICAL_DOC_TYPES = {"experiment_report", "materials_article", "unknown"}
PROCESS_DOC_TYPES = {"experiment_report", "process_report", "review_article", "unknown"}


PROPERTY_SCHEMA: dict[str, PropertySchema] = {
    "прочность": PropertySchema(
        allowed_units={"MPa", "GPa", "ksi"},
        required_markers={"прочность", "предел прочности", "tensile strength", "ultimate tensile strength", "strength", "σв"},
        compatible_subject_types={"Alloy", "Metal", "MaterialSample"},
        compatible_doc_types=MECHANICAL_DOC_TYPES,
    ),
    "твёрдость": PropertySchema(
        allowed_units={"HV", "HRC"},
        required_markers={"твёрдость", "твердость", "hardness", "hv", "hrc"},
        compatible_subject_types={"Alloy", "Metal", "MaterialSample"},
        compatible_doc_types=MECHANICAL_DOC_TYPES,
    ),
    "пластичность": PropertySchema(
        allowed_units={"%"},
        required_markers={
            "пластичность",
            "удлинение",
            "относительное удлинение",
            "ductility",
            "elongation",
            "plasticity",
            "δ",
        },
        compatible_subject_types={"Alloy", "Metal", "MaterialSample"},
        compatible_doc_types=MECHANICAL_DOC_TYPES,
    ),
    "извлечение": PropertySchema(
        allowed_units={"%"},
        required_markers={"извлечение", "степень извлечения", "recovery", "extraction yield"},
        compatible_subject_types={"Ore", "Solution", "Concentrate", "Slag", "Matte", "MaterialSample"},
        compatible_doc_types=PROCESS_DOC_TYPES,
    ),
    "содержание": PropertySchema(
        allowed_units={"%", "ppm", "mg/kg", "г/т", "g/t"},
        required_markers={"содержание", "content", "grade", "assay"},
        compatible_subject_types={"Ore", "Concentrate", "Slag", "Matte", "Solution", "MaterialSample"},
        compatible_doc_types=PROCESS_DOC_TYPES,
    ),
    "концентрация": PropertySchema(
        allowed_units={"mg/L", "g/L", "ppm"},
        required_markers={
            "концентрация",
            "concentration",
            "содержание в растворе",
            "сульфат",
            "хлорид",
            "sulfate",
            "sulphate",
            "chloride",
            "ca",
            "mg",
            "na",
        },
        compatible_subject_types={"Solution", "Water", "Electrolyte", "Catholyte", "ChemicalSubstance"},
        compatible_doc_types=PROCESS_DOC_TYPES,
    ),
    "сухой остаток": PropertySchema(
        allowed_units={"mg/L", "g/L", "ppm"},
        required_markers={"сухой остаток", "total dissolved solids", "tds"},
        compatible_subject_types={"Solution", "Water"},
        compatible_doc_types=PROCESS_DOC_TYPES,
    ),
    "скорость потока": PropertySchema(
        allowed_units={"m/s"},
        required_markers={"скорость", "скорость потока", "циркуляция", "flow velocity", "velocity", "circulation velocity"},
        compatible_subject_types={"Process", "Solution", "Electrolyte", "Catholyte"},
        compatible_doc_types=PROCESS_DOC_TYPES,
    ),
    "расход": PropertySchema(
        allowed_units={"m3/h", "t/day"},
        required_markers={"расход", "flow rate", "подача", "flow"},
        compatible_subject_types={"Process", "Solution", "Water", "Electrolyte", "Catholyte", "Facility"},
        compatible_doc_types=PROCESS_DOC_TYPES,
    ),
    "производительность": PropertySchema(
        allowed_units={"t/day", "t/y", "kt/y", "Mt/y", "m3/h", "%"},
        required_markers={"производительность", "capacity", "throughput"},
        compatible_subject_types={"Process", "Facility", "Ore", "Concentrate", "Commodity"},
        compatible_doc_types={"experiment_report", "process_report", "review_article", "market_capacity_reference", "directory_or_catalog", "unknown"},
    ),
    "выход металла": PropertySchema(
        allowed_units={"%"},
        required_markers={"выход", "metal yield", "yield"},
        compatible_subject_types={"Ore", "Concentrate", "Slag", "Matte", "Metal", "MaterialSample"},
        compatible_doc_types=PROCESS_DOC_TYPES,
    ),
    "распределение": PropertySchema(
        allowed_units={"%"},
        required_markers={"распределение", "коэффициент распределения", "distribution", "distribution coefficient"},
        compatible_subject_types={"Slag", "Matte", "Metal", "ChemicalSubstance", "MaterialSample"},
        compatible_doc_types=PROCESS_DOC_TYPES,
    ),
    "экономический показатель": PropertySchema(
        allowed_units={"USD/t", "EUR/t", "RUB/t", "USD/m3", "RUB/m3", "mln RUB", "mln RUB/year"},
        required_markers={"capex", "opex", "затрат", "стоимост", "эконом", "cost"},
        compatible_subject_types={"Process", "Technology", "Facility"},
        compatible_doc_types={"experiment_report", "process_report", "review_article", "unknown"},
    ),
    "температура": PropertySchema(
        allowed_units={"C"},
        required_markers={"температура", "temperature", "нагрев", "heating"},
        compatible_subject_types={"Process", "Solution", "Water", "Electrolyte", "Catholyte", "MaterialSample", "Facility"},
        compatible_doc_types=PROCESS_DOC_TYPES,
    ),
    "pH": PropertySchema(
        allowed_units={"pH"},
        required_markers={"ph", "pH", "кислотность"},
        compatible_subject_types={"Solution", "Water", "Electrolyte", "Catholyte"},
        compatible_doc_types=PROCESS_DOC_TYPES,
    ),
    "давление": PropertySchema(
        allowed_units={"bar", "MPa"},
        required_markers={"давление", "pressure", "бар", "bar"},
        compatible_subject_types={"Process", "Solution", "Water", "Electrolyte", "Catholyte", "Facility"},
        compatible_doc_types=PROCESS_DOC_TYPES,
    ),
}


def has_required_property_marker(property_name: str, context: str) -> bool:
    schema = PROPERTY_SCHEMA.get(property_name)
    if not schema:
        return False
    normalized = str(context or "").lower().replace("ё", "е")
    return any(marker.lower().replace("ё", "е") in normalized for marker in schema.required_markers)

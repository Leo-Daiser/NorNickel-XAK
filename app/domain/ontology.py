"""Typed ontology models for graph-grounded technical QA."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    document_id: str | None = None
    chunk_id: str | None = None
    source_name: str | None = None
    page: int | None = None
    quote: str | None = None
    confidence: float | None = None


class Material(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    family: str | None = None


class MaterialFamily(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)


class ProcessRegime(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    temperature: float | None = None
    temperature_unit: str | None = None
    duration: float | None = None
    duration_unit: str | None = None
    medium: str | None = None


class ProcessStep(BaseModel):
    canonical_name: str
    order: int | None = None
    regime: str | None = None


class Property(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    unit_family: str | None = None


class Measurement(BaseModel):
    property_name: str
    value: float | None = None
    value_min: float | None = None
    value_max: float | None = None
    raw_value: str | None = None
    unit: str | None = None
    value_original: float | None = None
    unit_original: str | None = None
    value_normalized: float | None = None
    unit_normalized: str | None = None
    normalization_family: str | None = None
    effect: str | None = None
    baseline_value: float | None = None
    delta_abs: float | None = None
    delta_rel_percent: float | None = None
    confidence: float | None = None
    analyte: str | None = None
    fact_type: str | None = None
    source_adapter: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class Equipment(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)


class Laboratory(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)


class ResearchTeam(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)


class Employee(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)


class Conclusion(BaseModel):
    text: str
    evidence: list[Evidence] = Field(default_factory=list)


class DataGap(BaseModel):
    gap_id: str
    material: str | None = None
    regime: str | None = None
    property: str | None = None
    reason: str
    evidence: list[Evidence] = Field(default_factory=list)


class TopicTag(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)


class Experiment(BaseModel):
    experiment_id: str
    materials: list[str]
    regimes: list[str]
    measurements: list[Measurement]
    equipment: list[str] = Field(default_factory=list)
    laboratories: list[str] = Field(default_factory=list)
    teams: list[str] = Field(default_factory=list)
    employees: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    def to_fact_dicts(self) -> list[dict[str, Any]]:
        """Represent experiment measurements as API-compatible fact rows."""
        rows: list[dict[str, Any]] = []
        for material in self.materials:
            for regime in self.regimes or [""]:
                for measurement in self.measurements:
                    rows.append(
                        {
                            "experiment_id": self.experiment_id,
                            "material": material,
                            "regime": regime,
                            "property": measurement.property_name,
                            "value": measurement.value if measurement.value is not None else measurement.raw_value,
                            "value_min": measurement.value_min,
                            "value_max": measurement.value_max,
                            "unit": measurement.unit,
                            "value_original": measurement.value_original,
                            "unit_original": measurement.unit_original,
                            "value_normalized": measurement.value_normalized,
                            "unit_normalized": measurement.unit_normalized,
                            "normalization_family": measurement.normalization_family,
                            "effect": measurement.effect,
                            "analyte": measurement.analyte,
                            "fact_type": measurement.fact_type,
                            "source_adapter": measurement.source_adapter,
                            "evidence": [item.model_dump() for item in measurement.evidence],
                        }
                    )
        return rows

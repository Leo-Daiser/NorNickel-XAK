"""Typed extraction contract used before graph materialization."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExtractionSource(BaseModel):
    document_id: str | None = None
    chunk_id: str | None = None
    source_name: str | None = None
    page: int | None = None
    section_path: str | None = None
    block_type: str | None = None
    row_index: int | None = None
    column_name: str | None = None


class EvidenceSpan(BaseModel):
    source: ExtractionSource
    quote: str = Field(min_length=1)
    start_char: int | None = None
    end_char: int | None = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class ExtractedEntity(BaseModel):
    entity_type: Literal[
        "Material",
        "ProcessRegime",
        "Property",
        "Equipment",
        "Laboratory",
        "ResearchTeam",
        "Employee",
        "TopicTag",
    ]
    raw_name: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceSpan]


class ExtractedMeasurement(BaseModel):
    property_raw: str
    property_canonical: str
    value: float | None = None
    unit: str | None = None
    effect: Literal["increase", "decrease", "no_change", "mixed", "unknown"] = "unknown"
    baseline_value: float | None = None
    delta_abs: float | None = None
    delta_rel_percent: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceSpan]


class ExtractedRegime(BaseModel):
    raw_name: str
    canonical_name: str
    temperature: float | None = None
    temperature_unit: str | None = None
    duration: float | None = None
    duration_unit: str | None = None
    medium: str | None = None
    pressure: float | None = None
    pressure_unit: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceSpan]


class ExtractedExperiment(BaseModel):
    experiment_id: str
    materials: list[ExtractedEntity]
    regimes: list[ExtractedRegime]
    measurements: list[ExtractedMeasurement]
    equipment: list[ExtractedEntity] = Field(default_factory=list)
    laboratories: list[ExtractedEntity] = Field(default_factory=list)
    teams: list[ExtractedEntity] = Field(default_factory=list)
    employees: list[ExtractedEntity] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
    topic_tags: list[ExtractedEntity] = Field(default_factory=list)
    evidence: list[EvidenceSpan]
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedDataGap(BaseModel):
    gap_id: str
    material: str | None = None
    regime: str | None = None
    property: str | None = None
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceSpan]


class RejectedExtraction(BaseModel):
    item_type: str
    reason: str
    raw_payload: dict[str, Any] | str
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class CandidateFact(BaseModel):
    """A validator input candidate. It is not allowed to reach graph/answers directly."""

    candidate_id: str
    fact_type: str = "UnknownFact"
    extractor_name: str
    document_id: str | None = None
    chunk_id: str | None = None
    source_name: str | None = None
    subject: dict[str, Any] = Field(default_factory=dict)
    predicate: str = ""
    object: dict[str, Any] = Field(default_factory=dict)
    value: float | None = None
    unit: str | None = None
    evidence_quote: str = ""
    raw_span: str = ""
    context_window: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    document_type: str = "unknown"


class AcceptedFact(BaseModel):
    """A validated fact that may be used by graph materialization and answers."""

    candidate_id: str
    fact_type: str
    normalized_fact: dict[str, Any]
    evidence: list[EvidenceSpan]
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    validation_reasons: list[str] = Field(default_factory=list)


class RejectedCandidate(BaseModel):
    candidate: CandidateFact
    reasons: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class QuarantineCandidate(BaseModel):
    candidate: CandidateFact
    reasons: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    suggested_fact_type: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractionBundle(BaseModel):
    document_id: str | None = None
    source_name: str | None = None
    extractor_version: str
    entities: list[ExtractedEntity] = Field(default_factory=list)
    experiments: list[ExtractedExperiment] = Field(default_factory=list)
    data_gaps: list[ExtractedDataGap] = Field(default_factory=list)
    rejected_items: list[RejectedExtraction] = Field(default_factory=list)
    candidate_facts: list[CandidateFact] = Field(default_factory=list)
    accepted_facts: list[AcceptedFact] = Field(default_factory=list)
    quarantined_items: list[QuarantineCandidate] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

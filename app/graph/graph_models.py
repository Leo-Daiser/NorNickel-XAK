"""Structured graph QA result models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..domain.ontology import DataGap, Evidence, Experiment, Measurement


class ExperimentFact(Experiment):
    """Experiment fact used by strict graph QA."""

    source_chunk_ids: list[str] = Field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "materials": self.materials,
            "regimes": self.regimes,
            "measurements": [item.model_dump() for item in self.measurements],
            "equipment": self.equipment,
            "laboratories": self.laboratories,
            "teams": self.teams,
            "employees": self.employees,
            "topic_tags": self.topic_tags,
            "conclusions": self.conclusions,
            "evidence": [item.model_dump() for item in self.evidence],
        }


class PartialMatches(BaseModel):
    same_material: list[ExperimentFact] = Field(default_factory=list)
    same_material_and_regime: list[ExperimentFact] = Field(default_factory=list)
    same_material_and_property: list[ExperimentFact] = Field(default_factory=list)
    same_regime_and_property: list[ExperimentFact] = Field(default_factory=list)

    def to_response(self, limit: int = 5) -> dict[str, list[dict[str, Any]]]:
        return {
            "same_material": [item.summary() for item in self.same_material[:limit]],
            "same_material_and_regime": [item.summary() for item in self.same_material_and_regime[:limit]],
            "same_material_and_property": [item.summary() for item in self.same_material_and_property[:limit]],
            "same_regime_and_property": [item.summary() for item in self.same_regime_and_property[:limit]],
        }


class DecisionHistoryItem(BaseModel):
    experiment_id: str
    material: str
    regime: str | None = None
    equipment: list[str] = Field(default_factory=list)
    laboratory: str | None = None
    measurements: list[Measurement] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class GraphQueryResult(BaseModel):
    exact: list[ExperimentFact] = Field(default_factory=list)
    partial_matches: PartialMatches = Field(default_factory=PartialMatches)
    gaps: list[DataGap] = Field(default_factory=list)


class EntitySummary(BaseModel):
    id: str
    type: str
    label: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    degree: int = 0
    facts_count: int = 0


class EntityCard(BaseModel):
    entity: dict[str, Any]
    related: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    subgraph: dict[str, list[dict[str, Any]]] = Field(default_factory=lambda: {"nodes": [], "edges": []})
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class GraphStats(BaseModel):
    documents: int = 0
    chunks: int = 0
    experiments: int = 0
    materials: int = 0
    regimes: int = 0
    properties: int = 0
    measurements: int = 0
    equipment: int = 0
    laboratories: int = 0
    teams: int = 0
    employees: int = 0
    data_gaps: int = 0
    relationships: int = 0
    kg_backend_active: str = "unknown"
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class SimilarExperiment(BaseModel):
    experiment_id: str
    score: float
    explanation: str
    material: str | None = None
    regime: str | None = None
    property: str | None = None
    source: str | None = None
    experiment: dict[str, Any] = Field(default_factory=dict)

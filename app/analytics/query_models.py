"""Typed models for analytical GraphRAG query planning and context."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..domain.query_constraints import QueryConstraints


class AnalyticalIntent(str, Enum):
    STRICT_MATERIAL_REGIME_PROPERTY = "strict_material_regime_property"
    MATERIAL_OVERVIEW = "material_overview"
    REGIME_OVERVIEW = "regime_overview"
    PROPERTY_OVERVIEW = "property_overview"
    DECISION_HISTORY = "decision_history"
    GAP_ANALYSIS = "gap_analysis"
    MATERIAL_COMPARISON = "material_comparison"
    REGIME_COMPARISON = "regime_comparison"
    SIMILAR_EXPERIMENTS = "similar_experiments"
    EQUIPMENT_USAGE = "equipment_usage"
    LAB_ACTIVITY = "lab_activity"
    TEAM_ACTIVITY = "team_activity"
    TOPIC_SEARCH = "topic_search"
    GRAPH_NEIGHBORHOOD = "graph_neighborhood"
    GENERAL_SEARCH = "general_search"
    UNKNOWN = "unknown"


AnswerMode = Literal["strict", "overview", "comparison", "history", "gaps", "search", "neighborhood"]


class AnalyticalQueryPlan(BaseModel):
    raw_question: str
    intent: AnalyticalIntent
    constraints: QueryConstraints
    cypher_template: str | None = None
    retrieval_required: bool = False
    graph_expansion_required: bool = True
    evidence_required: bool = True
    answer_mode: AnswerMode
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    source_name: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    section_path: str | None = None
    quote: str = Field(min_length=1)
    score: float
    retrieval_backend: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphContext(BaseModel):
    intent: AnalyticalIntent
    constraints: QueryConstraints
    facts: list[dict[str, Any]] = Field(default_factory=list)
    grouped_facts: list[dict[str, Any]] = Field(default_factory=list)
    decision_history: list[dict[str, Any]] = Field(default_factory=list)
    gaps: list[dict[str, Any]] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    subgraph: dict[str, Any] = Field(default_factory=lambda: {"nodes": [], "edges": []})
    partial_matches: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    def stats(self) -> dict[str, int]:
        return {
            "facts_count": len(self.facts),
            "sources_count": len(self.sources),
            "evidence_count": len(self.evidence),
            "subgraph_nodes": len(self.subgraph.get("nodes") or []),
            "subgraph_edges": len(self.subgraph.get("edges") or []),
        }

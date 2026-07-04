"""Query intent and canonical constraints used by graph QA."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    MATERIAL_REGIME_PROPERTY_EFFECT = "material_regime_property_effect"
    DECISION_HISTORY = "decision_history"
    GAP_ANALYSIS = "gap_analysis"
    ENTITY_OVERVIEW = "entity_overview"
    EQUIPMENT_USAGE = "equipment_usage"
    TEAM_ACTIVITY = "team_activity"
    UNKNOWN = "unknown"


class QueryConstraints(BaseModel):
    intent: QueryIntent
    raw_question: str
    materials: list[str] = Field(default_factory=list)
    regimes: list[str] = Field(default_factory=list)
    properties: list[str] = Field(default_factory=list)
    equipment: list[str] = Field(default_factory=list)
    laboratories: list[str] = Field(default_factory=list)
    teams: list[str] = Field(default_factory=list)
    employees: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    numeric_constraints: list[dict] = Field(default_factory=list)
    geographies: list[str] = Field(default_factory=list)
    time_filters: list[dict] = Field(default_factory=list)
    target_fact_types: list[str] = Field(default_factory=list)
    answer_mode: str | None = None
    require_exact_match: bool = False

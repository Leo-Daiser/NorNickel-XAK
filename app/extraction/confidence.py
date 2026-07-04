"""Confidence scoring heuristics for structured extraction."""

from __future__ import annotations

from .models import ExtractedExperiment


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def experiment_confidence(experiment: ExtractedExperiment, ambiguous: bool = False) -> float:
    """Compute a conservative confidence score for an extracted experiment."""
    score = 0.15
    if experiment.materials:
        score += 0.25
    if experiment.regimes:
        score += 0.20
    if any(measurement.property_canonical for measurement in experiment.measurements):
        score += 0.20
    if any(measurement.value is not None and measurement.unit for measurement in experiment.measurements):
        score += 0.15
    if experiment.equipment or experiment.laboratories or experiment.teams:
        score += 0.10
    if experiment.conclusions or any(measurement.effect != "unknown" for measurement in experiment.measurements):
        score += 0.10
    if not experiment.evidence:
        score -= 0.20
    if ambiguous:
        score -= 0.20
    return clamp(score)


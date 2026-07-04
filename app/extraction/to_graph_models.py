"""Adapter from ExtractionBundle to graph ontology models used by GraphWriter."""

from __future__ import annotations

import hashlib
from typing import Any

from ..domain.fact_normalization import measurement_normalization_fields
from ..domain.ontology import DataGap, Evidence, Measurement
from ..graph.graph_models import ExperimentFact
from .models import AcceptedFact, EvidenceSpan, ExtractionBundle


def bundle_to_experiment_facts(bundle: ExtractionBundle) -> list[ExperimentFact]:
    """Convert accepted extracted experiments to GraphWriter-compatible facts."""
    facts: list[ExperimentFact] = []
    for experiment in bundle.experiments:
        facts.append(
            ExperimentFact(
                experiment_id=experiment.experiment_id,
                materials=[item.canonical_name for item in experiment.materials],
                regimes=[item.canonical_name for item in experiment.regimes],
                measurements=[
                    _to_measurement(measurement)
                    for measurement in experiment.measurements
                ],
                equipment=[item.canonical_name for item in experiment.equipment],
                laboratories=[item.canonical_name for item in experiment.laboratories],
                teams=[item.canonical_name for item in experiment.teams],
                employees=[item.canonical_name for item in experiment.employees],
                topic_tags=[item.canonical_name for item in experiment.topic_tags],
                conclusions=experiment.conclusions,
                evidence=[_to_evidence(item) for item in experiment.evidence],
                source_chunk_ids=list(dict.fromkeys(item.source.chunk_id for item in experiment.evidence if item.source.chunk_id)),
            )
        )
    return facts


def bundle_to_structured_accepted_experiment_facts(bundle: ExtractionBundle) -> list[ExperimentFact]:
    """Convert validated source-adapter facts into the existing graph ontology.

    The current graph reader is intentionally entity/property centric, with
    material and process as optional anchors.  Direct accepted table facts are
    therefore projected only when they have enough anchors to be useful in
    that graph: property plus at least one material or process anchor.  No
    synthetic Material or ProcessRegime is created.
    """

    facts: list[ExperimentFact] = []
    for accepted in bundle.accepted_facts:
        fact = _structured_accepted_fact_to_experiment(accepted)
        if fact is not None:
            facts.append(fact)
    return facts


def _structured_accepted_fact_to_experiment(accepted: AcceptedFact) -> ExperimentFact | None:
    normalized = accepted.normalized_fact or {}
    if not str(normalized.get("source_adapter") or "").strip():
        return None
    subject = _as_dict(normalized.get("subject"))
    obj = _as_dict(normalized.get("object"))

    material = _first_text(subject, "material", "material_raw")
    regime = _first_text(subject, "process", "process_raw") or _first_text(obj, "process")
    property_name = _first_text(obj, "property", "parameter") or _default_property_for_fact_type(accepted.fact_type)
    if not property_name or not (material or regime):
        return None

    evidence = [_to_evidence(item) for item in accepted.evidence]
    if not evidence:
        return None
    value = _float_or_none(normalized.get("value"))
    value_min = _float_or_none(obj.get("value_min"))
    value_max = _float_or_none(obj.get("value_max"))
    raw_value = obj.get("raw_value")
    if raw_value is None and (value_min is not None or value_max is not None):
        raw_value = _range_raw_value(value_min, value_max)
    elif raw_value is None and value is not None:
        raw_value = str(value)

    fields = measurement_normalization_fields(property_name, value, normalized.get("unit"))
    measurement = Measurement(
        property_name=property_name,
        value=value,
        value_min=value_min,
        value_max=value_max,
        raw_value=str(raw_value) if raw_value not in {None, ""} else None,
        unit=normalized.get("unit"),
        confidence=accepted.score,
        evidence=evidence,
        analyte=_first_text(obj, "analyte"),
        fact_type=accepted.fact_type,
        source_adapter=normalized.get("source_adapter"),
        **fields,
    )
    experiment_id = _stable_id("accepted_fact", accepted.candidate_id)
    return ExperimentFact(
        experiment_id=experiment_id,
        materials=[material] if material else [],
        regimes=[regime] if regime else [],
        measurements=[measurement],
        evidence=evidence,
        source_chunk_ids=list(dict.fromkeys(item.chunk_id for item in evidence if item.chunk_id)),
    )


def _to_measurement(measurement) -> Measurement:
    normalized = measurement_normalization_fields(measurement.property_canonical, measurement.value, measurement.unit)
    return Measurement(
        property_name=measurement.property_canonical,
        value=measurement.value,
        raw_value=None if measurement.value is not None else "",
        unit=measurement.unit,
        effect=measurement.effect,
        baseline_value=measurement.baseline_value,
        delta_abs=measurement.delta_abs,
        delta_rel_percent=measurement.delta_rel_percent,
        confidence=measurement.confidence,
        evidence=[_to_evidence(item) for item in measurement.evidence],
        **normalized,
    )


def bundle_to_data_gaps(bundle: ExtractionBundle) -> list[DataGap]:
    """Convert accepted extracted gaps to graph data gaps."""
    return [
        DataGap(
            gap_id=gap.gap_id,
            material=gap.material,
            regime=gap.regime,
            property=gap.property,
            reason=gap.reason,
            evidence=[_to_evidence(item) for item in gap.evidence],
        )
        for gap in bundle.data_gaps
    ]


def _to_evidence(span: EvidenceSpan) -> Evidence:
    return Evidence(
        document_id=span.source.document_id,
        chunk_id=span.source.chunk_id,
        source_name=span.source.source_name,
        page=span.source.page,
        quote=span.quote,
        confidence=span.confidence,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_text(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return None


def _default_property_for_fact_type(fact_type: str) -> str | None:
    return {
        "FacilityCapacityFact": "производительность",
        "EconomicIndicatorFact": "экономический показатель",
    }.get(fact_type)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _range_raw_value(value_min: float | None, value_max: float | None) -> str:
    if value_min is None:
        return str(value_max)
    if value_max is None or value_max == value_min:
        return str(value_min)
    return f"{value_min}-{value_max}"


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:32]}"

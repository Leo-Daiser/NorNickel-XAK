"""Explicit candidate/accepted/quarantine fact lifecycle helpers."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from ..domain.fact_schemas import FACT_TYPE_SCHEMAS, classify_measurement_fact_type
from .models import (
    AcceptedFact,
    CandidateFact,
    EvidenceSpan,
    ExtractedExperiment,
    ExtractionBundle,
    QuarantineCandidate,
    RejectedExtraction,
)


QUARANTINE_REASONS = {
    "unknown_property_schema",
    "missing_required_property_marker",
    "value_without_property_window",
    "subject_type_incompatible_with_property",
    "missing_regime_or_measurement",
    "all_measurements_rejected",
    "suspicious_code_like_entity_without_reference",
}

DIRECT_SOURCE_ADAPTERS = {
    "structured_table_adapter",
    "extractive_circulation_solution_adapter",
    "extractive_technology_solution_adapter",
    "extractive_claim_adapter",
    "extractive_expertise_adapter",
    "extractive_economic_adapter",
}


def candidate_facts_from_bundle(bundle: ExtractionBundle, document_type: str = "unknown") -> list[CandidateFact]:
    candidates: list[CandidateFact] = []
    for entity in bundle.entities:
        evidence = entity.evidence[0] if entity.evidence else None
        candidates.append(
            CandidateFact(
                candidate_id=_candidate_id("entity", entity.entity_type, entity.canonical_name, evidence),
                fact_type=f"{entity.entity_type}MentionFact",
                extractor_name=bundle.extractor_version,
                document_id=bundle.document_id,
                chunk_id=evidence.source.chunk_id if evidence else None,
                source_name=bundle.source_name,
                subject={"type": entity.entity_type, "name": entity.canonical_name, "raw_name": entity.raw_name},
                predicate="MENTIONS_ENTITY",
                object={},
                evidence_quote=evidence.quote if evidence else "",
                raw_span=entity.raw_name,
                context_window=evidence.quote if evidence else "",
                confidence=entity.confidence,
                document_type=document_type,
            )
        )
    for experiment in bundle.experiments:
        candidates.extend(_experiment_candidates(experiment, bundle, document_type))
    for gap in bundle.data_gaps:
        evidence = gap.evidence[0] if gap.evidence else None
        candidates.append(
            CandidateFact(
                candidate_id=_candidate_id("gap", gap.gap_id, gap.reason, evidence),
                fact_type="DataGapFact",
                extractor_name=bundle.extractor_version,
                document_id=bundle.document_id,
                chunk_id=evidence.source.chunk_id if evidence else None,
                source_name=bundle.source_name,
                subject={"material": gap.material, "regime": gap.regime, "property": gap.property},
                predicate="DATA_GAP",
                object={"reason": gap.reason},
                evidence_quote=evidence.quote if evidence else "",
                raw_span=gap.reason,
                context_window=evidence.quote if evidence else "",
                confidence=gap.confidence,
                document_type=document_type,
            )
        )
    return candidates


def accepted_facts_from_bundle(bundle: ExtractionBundle, document_type: str = "unknown") -> list[AcceptedFact]:
    accepted: list[AcceptedFact] = []
    for candidate in candidate_facts_from_bundle(bundle, document_type=document_type):
        if candidate.fact_type.endswith("MentionFact"):
            continue
        evidence = _candidate_evidence(candidate)
        accepted.append(
            AcceptedFact(
                candidate_id=candidate.candidate_id,
                fact_type=candidate.fact_type,
                normalized_fact={
                    "subject": candidate.subject,
                    "predicate": candidate.predicate,
                    "object": candidate.object,
                    "value": candidate.value,
                    "unit": candidate.unit,
                    "document_type": candidate.document_type,
                },
                evidence=evidence,
                score=candidate.confidence,
                validation_reasons=["accepted_by_validation_pipeline"],
            )
        )
    return accepted


def validate_direct_candidate_facts(
    candidates: list[CandidateFact],
    *,
    document_type: str = "unknown",
) -> tuple[list[AcceptedFact], list[QuarantineCandidate]]:
    """Validate candidates produced directly by source-specific adapters.

    This is intentionally stricter than raw extraction: only whitelisted adapter
    candidates with enough positive evidence become AcceptedFact.  They do not
    bypass legacy entity/property validators.
    """

    accepted: list[AcceptedFact] = []
    quarantined: list[QuarantineCandidate] = []
    for candidate in candidates:
        source_adapter = _source_adapter(candidate)
        if source_adapter not in DIRECT_SOURCE_ADAPTERS:
            continue
        reasons = _validate_structured_candidate(candidate, document_type=document_type)
        if reasons:
            quarantined.append(
                QuarantineCandidate(
                    candidate=candidate,
                    reasons=reasons,
                    missing_requirements=_candidate_missing_requirements(reasons),
                    suggested_fact_type=candidate.fact_type if candidate.fact_type != "UnknownFact" else None,
                    score=candidate.confidence,
                )
            )
            continue
        evidence = _candidate_evidence(candidate)
        validation_reasons = [f"accepted_{source_adapter}_candidate"]
        if source_adapter == "structured_table_adapter":
            validation_reasons.insert(0, "accepted_structured_table_candidate")
        accepted.append(
            AcceptedFact(
                candidate_id=candidate.candidate_id,
                fact_type=candidate.fact_type,
                normalized_fact={
                    "subject": candidate.subject,
                    "predicate": candidate.predicate,
                    "object": candidate.object,
                    "value": candidate.value,
                    "unit": candidate.unit,
                    "document_type": document_type,
                    "source_adapter": source_adapter,
                },
                evidence=evidence,
                score=candidate.confidence,
                validation_reasons=validation_reasons,
            )
        )
    return accepted, quarantined


def _source_adapter(candidate: CandidateFact) -> str:
    extractor_name = str(candidate.extractor_name or "")
    for adapter in DIRECT_SOURCE_ADAPTERS:
        if adapter in extractor_name:
            return adapter
    return ""


def quarantine_from_rejection(item: RejectedExtraction, document_type: str = "unknown") -> QuarantineCandidate | None:
    if item.reason not in QUARANTINE_REASONS:
        return None
    candidate = _candidate_from_rejected(item, document_type=document_type)
    return QuarantineCandidate(
        candidate=candidate,
        reasons=[item.reason],
        missing_requirements=_missing_requirements(item.reason),
        suggested_fact_type=candidate.fact_type if candidate.fact_type != "UnknownFact" else None,
        score=candidate.confidence,
    )


def split_rejected_and_quarantine(items: list[RejectedExtraction], document_type: str = "unknown") -> tuple[list[RejectedExtraction], list[QuarantineCandidate]]:
    rejected: list[RejectedExtraction] = []
    quarantine: list[QuarantineCandidate] = []
    for item in items:
        quarantined = quarantine_from_rejection(item, document_type=document_type)
        if quarantined is not None:
            quarantine.append(quarantined)
        else:
            rejected.append(item)
    return rejected, quarantine


def lifecycle_summary(bundle: ExtractionBundle) -> dict[str, Any]:
    accepted_by_type = Counter(item.fact_type for item in bundle.accepted_facts)
    quarantine_by_reason = Counter(reason for item in bundle.quarantined_items for reason in item.reasons)
    rejected_by_reason = Counter(item.reason for item in bundle.rejected_items)
    return {
        "candidate_facts_count": len(bundle.candidate_facts),
        "accepted_facts_count": len(bundle.accepted_facts),
        "rejected_candidates_count": len(bundle.rejected_items),
        "quarantine_candidates_count": len(bundle.quarantined_items),
        "accepted_by_fact_type": dict(accepted_by_type),
        "rejected_by_reason": dict(rejected_by_reason),
        "quarantine_by_reason": dict(quarantine_by_reason),
    }


def _experiment_candidates(experiment: ExtractedExperiment, bundle: ExtractionBundle, document_type: str) -> list[CandidateFact]:
    result: list[CandidateFact] = []
    exp_evidence = experiment.evidence[0] if experiment.evidence else None
    result.append(
        CandidateFact(
            candidate_id=_candidate_id("experiment", experiment.experiment_id, bundle.document_id, exp_evidence),
            fact_type="ExperimentResultFact",
            extractor_name=bundle.extractor_version,
            document_id=bundle.document_id,
            chunk_id=exp_evidence.source.chunk_id if exp_evidence else None,
            source_name=bundle.source_name,
            subject={"experiment_id": experiment.experiment_id},
            predicate="EXPERIMENT_RESULT",
            object={
                "materials": [item.canonical_name for item in experiment.materials],
                "regimes": [item.canonical_name for item in experiment.regimes],
            },
            evidence_quote=exp_evidence.quote if exp_evidence else "",
            raw_span=experiment.experiment_id,
            context_window=exp_evidence.quote if exp_evidence else "",
            confidence=experiment.confidence,
            document_type=document_type,
        )
    )
    material_names = [item.canonical_name for item in experiment.materials]
    regime_names = [item.canonical_name for item in experiment.regimes]
    for measurement in experiment.measurements:
        evidence = measurement.evidence[0] if measurement.evidence else exp_evidence
        result.append(
            CandidateFact(
                candidate_id=_candidate_id(
                    "measurement",
                    experiment.experiment_id,
                    ",".join(material_names),
                    ",".join(regime_names),
                    measurement.property_canonical,
                    measurement.value,
                    measurement.unit,
                    evidence,
                ),
                fact_type=classify_measurement_fact_type(measurement.property_canonical, measurement.unit),
                extractor_name=bundle.extractor_version,
                document_id=bundle.document_id,
                chunk_id=evidence.source.chunk_id if evidence else None,
                source_name=bundle.source_name,
                subject={"materials": material_names, "regimes": regime_names},
                predicate="MEASUREMENT",
                object={"property": measurement.property_canonical, "effect": measurement.effect},
                value=measurement.value,
                unit=measurement.unit,
                evidence_quote=evidence.quote if evidence else "",
                raw_span=measurement.property_raw,
                context_window=evidence.quote if evidence else "",
                confidence=measurement.confidence,
                document_type=document_type,
            )
        )
    return result


def _candidate_from_rejected(item: RejectedExtraction, document_type: str) -> CandidateFact:
    payload = item.raw_payload if isinstance(item.raw_payload, dict) else {"raw": item.raw_payload}
    evidence = item.evidence[0] if item.evidence else None
    fact_type = _suggest_fact_type_from_payload(payload)
    return CandidateFact(
        candidate_id=_candidate_id("rejected", item.item_type, item.reason, payload, evidence),
        fact_type=fact_type,
        extractor_name="validation_pipeline",
        document_id=evidence.source.document_id if evidence else None,
        chunk_id=evidence.source.chunk_id if evidence else None,
        source_name=evidence.source.source_name if evidence else None,
        subject=_safe_dict(payload.get("materials") or payload.get("subject") or payload.get("material") or {}),
        predicate=item.item_type,
        object=_safe_dict(payload.get("measurements") or payload.get("object") or payload),
        value=_payload_value(payload),
        unit=_payload_unit(payload),
        evidence_quote=evidence.quote if evidence else "",
        raw_span=str(payload.get("property_raw") or payload.get("raw_name") or item.item_type),
        context_window=evidence.quote if evidence else "",
        confidence=float(payload.get("confidence") or 0.0) if isinstance(payload, dict) else 0.0,
        document_type=document_type,
    )


def _candidate_evidence(candidate: CandidateFact) -> list[EvidenceSpan]:
    if not candidate.evidence_quote:
        return []
    return [
        EvidenceSpan(
            source={
                "document_id": candidate.document_id,
                "chunk_id": candidate.chunk_id,
                "source_name": candidate.source_name,
            },
            quote=candidate.evidence_quote,
            confidence=candidate.confidence,
        )
    ]


def _validate_structured_candidate(candidate: CandidateFact, *, document_type: str) -> list[str]:
    reasons: list[str] = []
    schema = FACT_TYPE_SCHEMAS.get(candidate.fact_type)
    if schema is None:
        return ["unknown_fact_schema"]
    if not candidate.evidence_quote.strip():
        reasons.append("missing_evidence")
    if candidate.confidence < 0.75:
        reasons.append("low_structured_candidate_confidence")
    effective_doc_type = document_type or candidate.document_type or "unknown"
    if schema.compatible_doc_types and effective_doc_type not in schema.compatible_doc_types:
        reasons.append("doc_type_incompatible_with_fact_schema")

    if candidate.fact_type == "ProcessParameterFact":
        prop = str(candidate.object.get("property") or "")
        if prop not in schema.compatible_properties:
            reasons.append("property_incompatible_with_process_parameter_schema")
        if candidate.value is None:
            reasons.append("missing_numeric_value")
        if candidate.unit not in schema.allowed_units:
            reasons.append("unit_incompatible_with_fact_schema")
        if not (candidate.subject.get("process") or candidate.subject.get("material") or candidate.subject.get("equipment")):
            reasons.append("missing_process_or_subject")

    elif candidate.fact_type == "FacilityCapacityFact":
        if candidate.value is None:
            reasons.append("missing_capacity_value")
        if not candidate.unit:
            reasons.append("missing_capacity_unit")
        elif candidate.unit not in schema.allowed_units:
            reasons.append("unit_incompatible_with_fact_schema")
        if not (candidate.subject.get("material") or candidate.subject.get("material_raw") or candidate.subject.get("facility") or candidate.subject.get("geography")):
            reasons.append("missing_commodity_or_facility")

    elif candidate.fact_type == "EconomicIndicatorFact":
        if candidate.value is None:
            reasons.append("missing_numeric_value")
        if candidate.unit not in schema.allowed_units:
            reasons.append("unit_incompatible_with_fact_schema")
        if not (
            candidate.subject.get("process")
            or candidate.subject.get("equipment")
            or candidate.subject.get("facility")
            or candidate.subject.get("material")
            or candidate.object.get("technology")
            or candidate.object.get("process")
        ):
            reasons.append("missing_technology_or_process")

    elif candidate.fact_type == "ExperimentResultFact":
        if not (candidate.subject.get("material") or candidate.subject.get("process")):
            reasons.append("missing_experiment_subject")
        if not (candidate.object.get("property") or candidate.object.get("source_note")):
            reasons.append("missing_result_property_or_note")
        if candidate.value is None and not candidate.object.get("source_note"):
            reasons.append("missing_result_value_or_note")

    elif candidate.fact_type == "ExpertiseFact":
        if not (candidate.subject.get("expert") or candidate.subject.get("laboratory") or candidate.subject.get("team") or candidate.subject.get("expert_or_lab")):
            reasons.append("missing_expert_or_team")
        if not (candidate.object.get("topic") or candidate.object.get("process") or candidate.object.get("material")):
            reasons.append("missing_expertise_topic")

    elif candidate.fact_type == "TechnologySolutionFact":
        text = " ".join(
            str(value or "")
            for value in [
                candidate.context_window,
                candidate.evidence_quote,
                candidate.object.get("claim"),
                candidate.object.get("solution_name"),
                candidate.object.get("target_problem"),
            ]
        ).lower().replace("ё", "е")
        markers = {marker.lower().replace("ё", "е") for marker in schema.required_markers}
        if not any(marker in text for marker in markers):
            reasons.append("missing_technology_solution_marker")
        if not (
            candidate.subject.get("process")
            or candidate.subject.get("material")
            or candidate.subject.get("media")
            or candidate.subject.get("equipment")
            or candidate.object.get("process_context")
            or candidate.object.get("target_problem")
            or candidate.object.get("technology")
            or candidate.object.get("solution_name")
        ):
            reasons.append("missing_technology_domain_entity")
        if not (candidate.object.get("claim") or candidate.object.get("source_note")):
            reasons.append("missing_technology_solution_claim")
        if _source_adapter(candidate) == "extractive_circulation_solution_adapter":
            if not _has_any_text(text, ["электроэкстракц", "electrowinning", "никел", "nickel"]):
                reasons.append("missing_electrowinning_or_nickel_context")
            if not _has_any_text(text, ["католит", "catholyte", "электролит", "electrolyte"]):
                reasons.append("missing_catholyte_or_electrolyte_media")
            if not _has_any_text(text, ["циркуляц", "рециркуляц", "поток", "расход", "скорость", "flow", "circulation", "recirculation"]):
                reasons.append("missing_circulation_or_flow_marker")

    elif candidate.fact_type == "PublicationClaimFact":
        if not (candidate.object.get("claim") or candidate.object.get("source_note")):
            reasons.append("missing_claim")
        if not (candidate.object.get("topic") or candidate.subject.get("process") or candidate.subject.get("material")):
            reasons.append("missing_claim_topic")

    else:
        reasons.append("direct_candidate_fact_type_not_supported")
    return list(dict.fromkeys(reasons))


def _candidate_missing_requirements(reasons: list[str]) -> list[str]:
    mapping = {
        "unknown_fact_schema": "known fact schema",
        "missing_evidence": "evidence quote",
        "low_structured_candidate_confidence": "candidate confidence >= 0.75",
        "doc_type_incompatible_with_fact_schema": "compatible document type",
        "property_incompatible_with_process_parameter_schema": "compatible process parameter property",
        "missing_numeric_value": "numeric value",
        "unit_incompatible_with_fact_schema": "schema-compatible unit",
        "missing_process_or_subject": "process or subject",
        "missing_capacity_value": "capacity value",
        "missing_capacity_unit": "capacity unit",
        "missing_commodity_or_facility": "commodity or facility/geography",
        "missing_technology_or_process": "technology/process/facility subject",
        "missing_experiment_subject": "experiment subject",
        "missing_result_property_or_note": "result property or note",
        "missing_result_value_or_note": "result value or note",
        "missing_expert_or_team": "expert, laboratory, or team",
        "missing_expertise_topic": "expertise topic/process/material",
        "missing_technology_solution_marker": "method/technology/solution marker",
        "missing_technology_domain_entity": "domain process/material/equipment/problem",
        "missing_technology_solution_claim": "extractive technology solution claim",
        "missing_electrowinning_or_nickel_context": "electrowinning or nickel electrolyte context",
        "missing_catholyte_or_electrolyte_media": "catholyte/electrolyte media",
        "missing_circulation_or_flow_marker": "circulation/flow marker",
        "missing_claim": "publication claim or source note",
        "missing_claim_topic": "claim topic/process/material",
        "direct_candidate_fact_type_not_supported": "supported direct candidate fact type",
    }
    return [mapping.get(reason, reason) for reason in reasons]


def _has_any_text(text: str, markers: list[str]) -> bool:
    return any(marker.lower().replace("ё", "е") in text for marker in markers)


def _payload_value(payload: dict[str, Any]) -> float | None:
    value = payload.get("value")
    if value is None and isinstance(payload.get("measurements"), list) and payload["measurements"]:
        value = payload["measurements"][0].get("value")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _payload_unit(payload: dict[str, Any]) -> str | None:
    unit = payload.get("unit")
    if unit is None and isinstance(payload.get("measurements"), list) and payload["measurements"]:
        unit = payload["measurements"][0].get("unit")
    return str(unit) if unit else None


def _safe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"items": value}
    if value:
        return {"value": value}
    return {}


def _suggest_fact_type_from_payload(payload: dict[str, Any]) -> str:
    if "property_canonical" in payload:
        return classify_measurement_fact_type(payload.get("property_canonical"), payload.get("unit"))
    if isinstance(payload.get("measurements"), list) and payload["measurements"]:
        first = payload["measurements"][0]
        if isinstance(first, dict):
            return classify_measurement_fact_type(first.get("property_canonical"), first.get("unit"))
    if "gap_id" in payload:
        return "DataGapFact"
    return "UnknownFact"


def _missing_requirements(reason: str) -> list[str]:
    mapping = {
        "unknown_property_schema": ["known property schema"],
        "missing_required_property_marker": ["local property marker near value"],
        "value_without_property_window": ["property mention near numeric value"],
        "subject_type_incompatible_with_property": ["compatible subject type"],
        "missing_regime_or_measurement": ["process regime or measurement"],
        "all_measurements_rejected": ["at least one validated measurement"],
        "suspicious_code_like_entity_without_reference": ["known alias, valid formula, material grade, or strong material-grade context"],
    }
    return mapping.get(reason, ["additional validation evidence"])


def _candidate_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return "cand_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]

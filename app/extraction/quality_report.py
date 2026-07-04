"""Quality observability for accepted/rejected/quarantined extraction facts."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from ..storage.catalog import SQLiteCatalog
from ..domain.fact_schemas import classify_measurement_fact_type
from .pipeline import ExtractionPipeline


def build_extraction_quality_report(
    catalog: SQLiteCatalog,
    *,
    pipeline: ExtractionPipeline | None = None,
    active_only: bool = True,
    max_chunks: int | None = None,
    quarantine_sample_limit: int = 25,
) -> dict[str, Any]:
    """Run extraction over catalog chunks and return lifecycle diagnostics.

    This report is intentionally independent from LLM/embeddings and is safe
    for economy_core: it observes parser/extractor/validator quality only.
    """

    active_pipeline = pipeline or ExtractionPipeline(audit_enabled=False)
    documents = catalog.list_documents()
    report: dict[str, Any] = {
        "documents_processed": 0,
        "chunks_processed": 0,
        "tables_processed": 0,
        "candidate_facts_count": 0,
        "accepted_facts_count": 0,
        "rejected_candidates_count": 0,
        "quarantine_candidates_count": 0,
        "acceptance_rate": 0.0,
        "facts_without_evidence": 0,
        "accepted_by_fact_type": {},
        "rejected_by_reason": {},
        "quarantine_by_reason": {},
        "accepted_by_doc_type": {},
        "rejected_by_doc_type": {},
        "quarantine_by_doc_type": {},
        "facts_by_extractor": {},
        "rejected_by_extractor": {},
        "quarantine_by_extractor": {},
        "rejected_by_intended_fact_type": {},
        "rejected_by_fact_type": {},
        "quarantine_by_intended_fact_type": {},
        "quarantine_by_fact_type": {},
        "missing_material_by_intended_fact_type": {},
        "missing_fields_by_fact_type": {},
        "top_unknown_properties": {},
        "top_unknown_entities": {},
        "top_table_column_patterns": {},
        "accepted_from_tables_count": 0,
        "accepted_from_text_count": 0,
        "economic_candidates_count": 0,
        "economic_quarantine_by_reason": {},
        "top_economic_markers_seen": {},
        "top_economic_units_seen": {},
        "accepted_by_answer_mode": {},
        "typed_retrieval_coverage": {},
        "accepted_facts_used_in_answers_count": 0,
        "chunks_only_questions_count": 0,
        "no_structured_facts_by_intent": {},
        "top_suspicious_entities": {},
        "top_accepted_entities": {},
        "top_accepted_properties": {},
        "unknown_property_schema_examples": [],
        "material_without_positive_validation_examples": [],
        "quarantine_samples": [],
    }

    counters = {
        "accepted_by_fact_type": Counter(),
        "rejected_by_reason": Counter(),
        "quarantine_by_reason": Counter(),
        "accepted_by_doc_type": Counter(),
        "rejected_by_doc_type": Counter(),
        "quarantine_by_doc_type": Counter(),
        "facts_by_extractor": Counter(),
        "rejected_by_extractor": Counter(),
        "quarantine_by_extractor": Counter(),
        "rejected_by_intended_fact_type": Counter(),
        "rejected_by_fact_type": Counter(),
        "quarantine_by_intended_fact_type": Counter(),
        "quarantine_by_fact_type": Counter(),
        "missing_material_by_intended_fact_type": Counter(),
        "missing_fields_by_fact_type": Counter(),
        "top_unknown_properties": Counter(),
        "top_unknown_entities": Counter(),
        "top_table_column_patterns": Counter(),
        "economic_quarantine_by_reason": Counter(),
        "top_economic_markers_seen": Counter(),
        "top_economic_units_seen": Counter(),
        "accepted_by_answer_mode": Counter(),
        "typed_retrieval_coverage": Counter(),
        "top_suspicious_entities": Counter(),
        "top_accepted_entities": Counter(),
        "top_accepted_properties": Counter(),
    }
    processed_chunks = 0

    for document in documents:
        if active_only and hasattr(catalog, "is_document_active") and not catalog.is_document_active(document.doc_id):
            continue
        report["documents_processed"] += 1
        for chunk in catalog.list_chunks(document.doc_id, active_only=active_only):
            if max_chunks is not None and processed_chunks >= max_chunks:
                break
            processed_chunks += 1
            report["chunks_processed"] += 1
            if chunk.metadata.get("chunk_kind") == "table_row":
                report["tables_processed"] += 1
            bundle = active_pipeline.extract_from_chunk(chunk)
            document_type = (bundle.diagnostics.get("document_profile") or {}).get("detected_type") or "unknown"
            report["candidate_facts_count"] += len(bundle.candidate_facts)
            report["accepted_facts_count"] += len(bundle.accepted_facts)
            report["rejected_candidates_count"] += len(bundle.rejected_items)
            report["quarantine_candidates_count"] += len(bundle.quarantined_items)
            for candidate in bundle.candidate_facts:
                counters["facts_by_extractor"][candidate.extractor_name] += 1
                if "structured_table_adapter" in candidate.extractor_name:
                    pattern = _table_column_pattern(candidate.raw_span)
                    if pattern:
                        counters["top_table_column_patterns"][pattern] += 1
                if _is_economic_candidate(candidate):
                    report["economic_candidates_count"] += 1
                    for marker in _economic_markers_from_text(_candidate_text(candidate)):
                        counters["top_economic_markers_seen"][marker] += 1
                    for unit in _economic_units_from_text(_candidate_text(candidate)):
                        counters["top_economic_units_seen"][unit] += 1
            for accepted in bundle.accepted_facts:
                counters["accepted_by_fact_type"][accepted.fact_type] += 1
                counters["accepted_by_doc_type"][document_type] += 1
                if _accepted_from_table(accepted):
                    report["accepted_from_tables_count"] += 1
                else:
                    report["accepted_from_text_count"] += 1
                answer_mode = _answer_mode_for_fact_type(accepted.fact_type)
                counters["accepted_by_answer_mode"][answer_mode] += 1
                counters["typed_retrieval_coverage"][accepted.fact_type] += 1
                _count_accepted_fact_observability(accepted, counters)
                if not accepted.evidence:
                    report["facts_without_evidence"] += 1
            for entity in bundle.entities:
                counters["top_accepted_entities"][f"{entity.entity_type}:{entity.canonical_name}"] += 1
            for experiment in bundle.experiments:
                for measurement in experiment.measurements:
                    counters["top_accepted_properties"][measurement.property_canonical] += 1
            for rejected in bundle.rejected_items:
                counters["rejected_by_reason"][rejected.reason] += 1
                counters["rejected_by_doc_type"][document_type] += 1
                counters["rejected_by_extractor"][_extractor_name_from_rejection(rejected)] += 1
                intended_type = _intended_fact_type_from_payload(rejected.raw_payload)
                counters["rejected_by_intended_fact_type"][intended_type] += 1
                counters["rejected_by_fact_type"][intended_type] += 1
                if rejected.reason == "missing_material":
                    counters["missing_material_by_intended_fact_type"][intended_type] += 1
                if rejected.reason.startswith("missing_"):
                    counters["missing_fields_by_fact_type"][f"{intended_type}:{rejected.reason}"] += 1
                suspicious = _candidate_name_from_payload(rejected.raw_payload)
                if suspicious:
                    counters["top_suspicious_entities"][suspicious] += 1
                    if rejected.reason in {"unknown_property_schema", "missing_property"}:
                        counters["top_unknown_properties"][suspicious] += 1
                    if rejected.reason in {"material_without_positive_validation", "missing_canonical_name"}:
                        counters["top_unknown_entities"][suspicious] += 1
                if rejected.reason == "unknown_property_schema":
                    _append_example(report["unknown_property_schema_examples"], rejected, intended_type)
                if rejected.reason == "material_without_positive_validation":
                    _append_example(report["material_without_positive_validation_examples"], rejected, intended_type)
            for quarantined in bundle.quarantined_items:
                for reason in quarantined.reasons:
                    counters["quarantine_by_reason"][reason] += 1
                    if (quarantined.suggested_fact_type or quarantined.candidate.fact_type) == "EconomicIndicatorFact":
                        counters["economic_quarantine_by_reason"][reason] += 1
                counters["quarantine_by_doc_type"][quarantined.candidate.document_type or document_type] += 1
                counters["quarantine_by_extractor"][quarantined.candidate.extractor_name] += 1
                counters["quarantine_by_intended_fact_type"][quarantined.suggested_fact_type or quarantined.candidate.fact_type] += 1
                counters["quarantine_by_fact_type"][quarantined.suggested_fact_type or quarantined.candidate.fact_type] += 1
                for requirement in quarantined.missing_requirements:
                    counters["missing_fields_by_fact_type"][f"{quarantined.suggested_fact_type or quarantined.candidate.fact_type}:{requirement}"] += 1
                if "missing_material" in quarantined.reasons:
                    counters["missing_material_by_intended_fact_type"][quarantined.suggested_fact_type or quarantined.candidate.fact_type] += 1
                suspicious = _candidate_name_from_payload(quarantined.candidate.subject) or quarantined.candidate.raw_span
                if suspicious:
                    counters["top_suspicious_entities"][str(suspicious)] += 1
                    if "unknown_property_schema" in quarantined.reasons:
                        counters["top_unknown_properties"][str(suspicious)] += 1
                    if "material_without_positive_validation" in quarantined.reasons:
                        counters["top_unknown_entities"][str(suspicious)] += 1
                if "unknown_property_schema" in quarantined.reasons:
                    _append_quarantine_example(report["unknown_property_schema_examples"], quarantined)
                if "material_without_positive_validation" in quarantined.reasons:
                    _append_quarantine_example(report["material_without_positive_validation_examples"], quarantined)
                if len(report["quarantine_samples"]) < quarantine_sample_limit:
                    report["quarantine_samples"].append(
                        {
                            "candidate": _dump_model(quarantined.candidate),
                            "evidence_quote": quarantined.candidate.evidence_quote,
                            "source": {
                                "document_id": quarantined.candidate.document_id,
                                "chunk_id": quarantined.candidate.chunk_id,
                                "source_name": quarantined.candidate.source_name,
                            },
                            "validator_reasons": quarantined.reasons,
                            "missing_requirements": quarantined.missing_requirements,
                            "suggested_fact_type": quarantined.suggested_fact_type,
                            "score": quarantined.score,
                        }
                    )
        if max_chunks is not None and processed_chunks >= max_chunks:
            break

    if report["candidate_facts_count"]:
        report["acceptance_rate"] = round(report["accepted_facts_count"] / report["candidate_facts_count"], 6)
    for key, counter in counters.items():
        report[key] = _counter_dict(counter)
    return report


def _counter_dict(counter: Counter, limit: int = 50) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common(limit)}


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _append_example(target: list[dict[str, Any]], rejected: Any, intended_fact_type: str, limit: int = 20) -> None:
    if len(target) >= limit:
        return
    evidence = rejected.evidence[0] if getattr(rejected, "evidence", None) else None
    source = getattr(evidence, "source", None)
    target.append(
        {
            "reason": rejected.reason,
            "intended_fact_type": intended_fact_type,
            "candidate": _candidate_name_from_payload(rejected.raw_payload),
            "quote": getattr(evidence, "quote", "") if evidence else "",
            "document_id": getattr(source, "document_id", None) if source else None,
            "chunk_id": getattr(source, "chunk_id", None) if source else None,
            "source_name": getattr(source, "source_name", None) if source else None,
        }
    )


def _append_quarantine_example(target: list[dict[str, Any]], quarantined: Any, limit: int = 20) -> None:
    if len(target) >= limit:
        return
    candidate = quarantined.candidate
    target.append(
        {
            "reason": ", ".join(quarantined.reasons),
            "intended_fact_type": quarantined.suggested_fact_type or candidate.fact_type,
            "candidate": _candidate_name_from_payload(candidate.subject) or candidate.raw_span,
            "quote": candidate.evidence_quote,
            "document_id": candidate.document_id,
            "chunk_id": candidate.chunk_id,
            "source_name": candidate.source_name,
        }
    )


def _extractor_name_from_rejection(rejected: Any) -> str:
    payload = rejected.raw_payload if isinstance(rejected.raw_payload, dict) else {}
    return str(payload.get("extractor_name") or payload.get("extractor") or "validation_pipeline")


def _intended_fact_type_from_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "UnknownFact"
    if "property_canonical" in payload:
        return classify_measurement_fact_type(payload.get("property_canonical"), payload.get("unit"))
    measurements = payload.get("measurements")
    if isinstance(measurements, list) and measurements:
        first = measurements[0]
        if isinstance(first, dict):
            return classify_measurement_fact_type(first.get("property_canonical"), first.get("unit"))
    if payload.get("gap_id"):
        return "DataGapFact"
    if payload.get("materials") or payload.get("regimes"):
        return "ExperimentResultFact"
    return "UnknownFact"


def _candidate_name_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return str(payload)[:120] if payload else None
    for key in ("canonical_name", "raw_name", "property_canonical", "property_raw", "name", "material"):
        value = payload.get(key)
        if value:
            return str(value)[:120]
    for key in ("materials", "measurements", "entities"):
        values = payload.get(key)
        if isinstance(values, list) and values:
            first = values[0]
            found = _candidate_name_from_payload(first)
            if found:
                return found
    subject = payload.get("subject")
    if isinstance(subject, dict):
        return _candidate_name_from_payload(subject)
    return None


def _count_accepted_fact_observability(accepted: Any, counters: dict[str, Counter]) -> None:
    normalized = getattr(accepted, "normalized_fact", {}) or {}
    if not isinstance(normalized, dict):
        return
    subject = normalized.get("subject") or {}
    obj = normalized.get("object") or {}
    if isinstance(subject, dict):
        for key in ("material", "material_raw", "process", "facility", "geography"):
            value = subject.get(key)
            if value:
                counters["top_accepted_entities"][f"{key}:{value}"] += 1
    if isinstance(obj, dict):
        prop = obj.get("property") or obj.get("parameter")
        if prop:
            counters["top_accepted_properties"][str(prop)] += 1
        analyte = obj.get("analyte")
        if analyte:
            counters["top_accepted_entities"][f"analyte:{analyte}"] += 1


def _accepted_from_table(accepted: Any) -> bool:
    normalized = getattr(accepted, "normalized_fact", {}) or {}
    return isinstance(normalized, dict) and normalized.get("source_adapter") == "structured_table_adapter"


def _table_column_pattern(raw_span: str | None) -> str:
    text = str(raw_span or "")
    keys = []
    for part in text.split(" | "):
        if ":" not in part:
            continue
        key, _ = part.split(":", 1)
        key = key.strip().lower().replace("ё", "е")
        if key:
            keys.append(key)
    return " | ".join(keys[:12])


def _answer_mode_for_fact_type(fact_type: str) -> str:
    return {
        "ProcessParameterFact": "process_parameter_search",
        "ExperimentResultFact": "experiment_catalog_search",
        "FacilityCapacityFact": "technology_comparison",
        "EconomicIndicatorFact": "domestic_vs_foreign_practice",
        "PublicationClaimFact": "literature_review",
        "ExpertiseFact": "expert_search",
        "DataGapFact": "knowledge_gap_search",
    }.get(str(fact_type or ""), "generic_typed_fact_summary")


def _is_economic_candidate(candidate: Any) -> bool:
    if getattr(candidate, "fact_type", "") == "EconomicIndicatorFact":
        return True
    return bool(_economic_markers_from_text(_candidate_text(candidate)) or _economic_units_from_text(_candidate_text(candidate)))


def _candidate_text(candidate: Any) -> str:
    parts = [
        getattr(candidate, "raw_span", ""),
        getattr(candidate, "context_window", ""),
        getattr(candidate, "evidence_quote", ""),
        str(getattr(candidate, "subject", "") or ""),
        str(getattr(candidate, "object", "") or ""),
        str(getattr(candidate, "unit", "") or ""),
    ]
    return " ".join(str(part) for part in parts if part)


def _economic_markers_from_text(text: str) -> list[str]:
    norm = str(text or "").lower().replace("ё", "е")
    markers = {
        "capex": ["capex", "capital cost", "capital expenditure", "капитальные затраты"],
        "opex": ["opex", "operating cost", "operational cost", "операционные затраты"],
        "cost": ["cost", "стоимость", "затраты", "экономика", "tariff", "тариф"],
    }
    result: list[str] = []
    for label, terms in markers.items():
        if any(term in norm for term in terms):
            result.append(label)
    return result


def _economic_units_from_text(text: str) -> list[str]:
    units = []
    for pattern, label in [
        (r"\bUSD\s*/\s*t\b|\$\s*/\s*t", "USD/t"),
        (r"\bEUR\s*/\s*t\b|€\s*/\s*t", "EUR/t"),
        (r"\bRUB\s*/\s*t\b|руб\.?\s*/\s*т", "RUB/t"),
        (r"\bUSD\s*/\s*m3\b|\$\s*/\s*м3", "USD/m3"),
        (r"\bRUB\s*/\s*m3\b|руб\.?\s*/\s*м3", "RUB/m3"),
    ]:
        if re.search(pattern, text or "", flags=re.IGNORECASE):
            units.append(label)
    return units

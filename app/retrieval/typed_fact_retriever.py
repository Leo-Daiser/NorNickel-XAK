"""AcceptedFact-first retrieval for typed scientific KG answers."""

from __future__ import annotations

import re
import os
from typing import Any

from pydantic import BaseModel, Field

from ..domain.normalization import material_matches, normalize_text, property_matches, regime_matches
from ..domain.query_constraints import QueryConstraints
from ..domain.unit_normalization import normalize_unit_label
from ..extraction.models import AcceptedFact, EvidenceSpan
from ..models.schemas import Chunk


class TypedFactQuery(BaseModel):
    question: str
    target_fact_types: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    processes: list[str] = Field(default_factory=list)
    properties: list[str] = Field(default_factory=list)
    equipment: list[str] = Field(default_factory=list)
    geography: list[str] = Field(default_factory=list)
    time_filters: list[dict[str, Any]] = Field(default_factory=list)
    numeric_constraints: list[dict[str, Any]] = Field(default_factory=list)
    answer_mode: str = "generic_typed_fact_summary"

    @classmethod
    def from_constraints(cls, question: str, constraints: QueryConstraints) -> "TypedFactQuery":
        return cls(
            question=question,
            target_fact_types=list(constraints.target_fact_types or []),
            materials=list(constraints.materials or []),
            processes=list(constraints.regimes or []),
            properties=list(constraints.properties or []),
            equipment=list(constraints.equipment or []),
            geography=list(constraints.geographies or []),
            time_filters=list(constraints.time_filters or []),
            numeric_constraints=list(constraints.numeric_constraints or []),
            answer_mode=constraints.answer_mode or "generic_typed_fact_summary",
        )


class TypedFactSearchResult(BaseModel):
    query: TypedFactQuery
    accepted_facts: list[AcceptedFact] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    relaxed_matches: list[AcceptedFact] = Field(default_factory=list)
    fallback_chunks: list[Chunk] = Field(default_factory=list)
    missing_filters: list[str] = Field(default_factory=list)
    retrieval_status: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class TypedFactRetriever:
    """Search verified typed facts before falling back to text chunks."""

    def __init__(self, repository: Any, retrieval_engine: Any | None = None) -> None:
        self.repository = repository
        self.retrieval_engine = retrieval_engine

    def search(self, query: TypedFactQuery, *, top_k: int = 12) -> TypedFactSearchResult:
        candidates = self._accepted_facts(query.target_fact_types, limit=max(300, top_k * 20))
        exact, missing, exact_reports = self._filter_facts(candidates, query, strict=True)
        base_diagnostics = {
            "candidate_facts_seen": len(candidates),
            "normalized_query_terms": _normalized_query_terms(query),
            "numeric_constraints_parsed": query.numeric_constraints,
            "match_reports": exact_reports[:20],
        }
        if exact:
            numeric_matched = _numeric_constraints_match_count(exact, query)
            return TypedFactSearchResult(
                query=query,
                accepted_facts=exact[:top_k],
                evidence=_dedupe_evidence(exact),
                missing_filters=[],
                retrieval_status="exact_facts_found",
                diagnostics={
                    **base_diagnostics,
                    "matched_facts": len(exact),
                    "numeric_constraints_matched": numeric_matched,
                    "matched_required_anchors": _aggregate_report_values(exact_reports, "matched_required_anchors"),
                    "missing_required_anchors": [],
                    "matched_optional_anchors": _aggregate_report_values(exact_reports, "matched_optional_anchors"),
                    "coverage_score": max((float(item.get("coverage_score", 0.0)) for item in exact_reports), default=1.0),
                    "exact_match": True,
                    "relaxed_match": False,
                    "missing_structured_fact_types": [],
                    "partial_reason": "",
                    "chunk_retrieval_executed": False,
                    "chunk_retrieval_skip_reason": "accepted_fact_path_satisfied",
                    "chunks_found_bm25": 0,
                    "chunks_found_dense": 0,
                    "chunks_after_fusion": 0,
                    **self._retrieval_stats_for_short_circuit(),
                },
            )

        relaxed, relaxed_missing, relaxed_reports = self._filter_facts(candidates, query, strict=False)
        if relaxed:
            numeric_matched = _numeric_constraints_match_count(relaxed, query)
            missing_types = _missing_fact_types(query.target_fact_types, relaxed)
            partial_reason = _relaxed_partial_reason(query, relaxed_reports)
            return TypedFactSearchResult(
                query=query,
                accepted_facts=relaxed[:top_k],
                relaxed_matches=relaxed[:top_k],
                evidence=_dedupe_evidence(relaxed),
                missing_filters=missing or relaxed_missing,
                retrieval_status="relaxed_facts_found",
                diagnostics={
                    **base_diagnostics,
                    "match_reports": relaxed_reports[:20],
                    "matched_facts": len(relaxed),
                    "relaxed": True,
                    "numeric_constraints_matched": numeric_matched,
                    "matched_required_anchors": _aggregate_report_values(relaxed_reports, "matched_required_anchors"),
                    "missing_required_anchors": _aggregate_report_values(relaxed_reports, "missing_required_anchors"),
                    "matched_optional_anchors": _aggregate_report_values(relaxed_reports, "matched_optional_anchors"),
                    "coverage_score": max((float(item.get("coverage_score", 0.0)) for item in relaxed_reports), default=0.0),
                    "exact_match": False,
                    "relaxed_match": True,
                    "missing_structured_fact_types": missing_types,
                    "partial_reason": partial_reason,
                    "chunk_retrieval_executed": False,
                    "chunk_retrieval_skip_reason": "relaxed_accepted_fact_path_satisfied",
                    "chunks_found_bm25": 0,
                    "chunks_found_dense": 0,
                    "chunks_after_fusion": 0,
                    **self._retrieval_stats_for_short_circuit(),
                },
            )

        chunks, evidence_diagnostics = self._fallback_chunks(query, top_k=top_k)
        retrieval_stats = self._retrieval_stats()
        return TypedFactSearchResult(
            query=query,
            accepted_facts=[],
            evidence=[],
            fallback_chunks=chunks,
            missing_filters=[] if chunks else (missing or relaxed_missing),
            retrieval_status="chunks_only_no_structured_facts" if chunks else "no_relevant_sources",
            diagnostics={
                **base_diagnostics,
                "fallback_chunks": len(chunks),
                "chunks_found_bm25": retrieval_stats.get("chunks_found_bm25", 0),
                "chunks_found_dense": retrieval_stats.get("chunks_found_dense", 0),
                "chunks_after_fusion": retrieval_stats.get("chunks_after_fusion", 0),
                **evidence_diagnostics,
                "effective_retrieval_mode": retrieval_stats.get("effective_retrieval_mode"),
                "degraded_reason": retrieval_stats.get("degraded_reason") or retrieval_stats.get("hybrid_degraded_reason", ""),
                "embedding_status": retrieval_stats.get("embedding_status", {}),
                "top_fused_chunks": retrieval_stats.get("top_fused_chunks", []),
                "missing_structured_fact_types": query.target_fact_types,
                "numeric_constraints_matched": 0,
                "matched_required_anchors": [],
                "missing_required_anchors": [] if chunks else (_required_anchor_labels(query) or missing or relaxed_missing),
                "matched_optional_anchors": [],
                "coverage_score": None if chunks else 0.0,
                "exact_match": False,
                "relaxed_match": False,
                "chunk_retrieval_executed": True,
                "chunk_retrieval_skip_reason": "",
                "partial_reason": (
                    "relevant chunks found but no accepted structured facts matched"
                    if chunks
                    else "no accepted structured facts and no relevant chunks found"
                ),
            },
        )

    def _accepted_facts(self, fact_types: list[str], *, limit: int) -> list[AcceptedFact]:
        finder = getattr(self.repository, "find_accepted_facts", None)
        if not callable(finder):
            return []
        try:
            return list(finder(fact_types=fact_types or None, limit=limit))
        except TypeError:
            return list(finder(limit=limit))

    def _fallback_chunks(self, query: TypedFactQuery, *, top_k: int) -> tuple[list[Chunk], dict[str, Any]]:
        if self.retrieval_engine is None or not hasattr(self.retrieval_engine, "query"):
            return [], source_grounded_filter_diagnostics(query, [], [])
        retrieval_query = _expanded_typed_query(query)
        chunks = list(self.retrieval_engine.query(retrieval_query, top_k=max(top_k * 4, 24)))
        return filter_source_grounded_chunks(chunks, query, top_k=top_k)

    def _retrieval_stats(self) -> dict[str, Any]:
        stats = getattr(self.retrieval_engine, "stats", None)
        if not callable(stats):
            return {}
        try:
            value = stats()
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _retrieval_stats_for_short_circuit(self) -> dict[str, Any]:
        stats = self._retrieval_stats()
        embedding_status = dict(stats.get("embedding_status") or {})
        return {
            "effective_retrieval_mode": stats.get("effective_retrieval_mode"),
            "degraded_reason": stats.get("degraded_reason") or stats.get("hybrid_degraded_reason", ""),
            "embedding_status": embedding_status,
        }

    def _filter_facts(
        self,
        facts: list[AcceptedFact],
        query: TypedFactQuery,
        *,
        strict: bool,
    ) -> tuple[list[AcceptedFact], list[str], list[dict[str, Any]]]:
        result: list[AcceptedFact] = []
        missing_totals: set[str] = set()
        matched_reports: list[dict[str, Any]] = []
        for fact in facts:
            ok, missing = _fact_matches_query(fact, query, strict=strict)
            if ok:
                result.append(fact)
                matched_reports.append(_fact_match_report(fact, query))
            else:
                missing_totals.update(missing)
        result.sort(key=_fact_rank, reverse=True)
        deduped = _dedupe_facts(result)
        wanted = {fact.candidate_id for fact in deduped}
        return deduped, sorted(missing_totals), [item for item in matched_reports if item.get("fact_id") in wanted]


def _fact_matches_query(fact: AcceptedFact, query: TypedFactQuery, *, strict: bool) -> tuple[bool, list[str]]:
    missing: list[str] = []
    if query.target_fact_types and fact.fact_type not in query.target_fact_types:
        return False, ["fact_type"]
    report = _fact_match_report(fact, query)
    if strict and not bool(report["exact_match"]):
        return False, list(report["missing_required_anchors"] or ["required_anchors"])
    normalized = fact.normalized_fact or {}
    subject = normalized.get("subject") if isinstance(normalized.get("subject"), dict) else {}
    obj = normalized.get("object") if isinstance(normalized.get("object"), dict) else {}

    dimensions_checked = 0
    dimensions_matched = 0
    matched_names: set[str] = set()
    checks = [
        ("material", query.materials, _fact_materials(subject, obj), _matches_material),
        ("process", query.processes, _fact_processes(subject, obj), _matches_process),
        ("property", query.properties, _fact_properties(subject, obj), _matches_property),
        ("equipment", query.equipment, _fact_equipment(subject, obj), _matches_text),
        ("geography", query.geography, _fact_geography(subject, obj), _matches_text),
    ]
    for name, requested, actual, matcher in checks:
        if not requested:
            continue
        dimensions_checked += 1
        if _any_match(requested, actual, matcher):
            dimensions_matched += 1
            matched_names.add(name)
        else:
            missing.append(name)

    if query.time_filters:
        dimensions_checked += 1
        if _matches_time_filters(query.time_filters, _fact_years(subject, obj)):
            dimensions_matched += 1
            matched_names.add("year")
        else:
            missing.append("year")

    if query.numeric_constraints:
        dimensions_checked += 1
        if _matches_numeric_constraints(query.numeric_constraints, normalized, obj):
            dimensions_matched += 1
            matched_names.add("numeric_constraints")
        else:
            missing.append("numeric_constraints")

    if strict:
        return not missing, missing
    if not bool(report["relaxed_match"]):
        missing.extend(list(report["missing_required_anchors"] or []))
        return False, list(dict.fromkeys(missing or ["anchor_coverage"]))
    if dimensions_checked == 0:
        return True, missing
    if matched_names == {"material"} and (query.processes or query.properties or query.equipment):
        return False, missing
    return dimensions_matched > 0, missing


def _fact_match_report(fact: AcceptedFact, query: TypedFactQuery) -> dict[str, Any]:
    normalized = fact.normalized_fact or {}
    subject = normalized.get("subject") if isinstance(normalized.get("subject"), dict) else {}
    obj = normalized.get("object") if isinstance(normalized.get("object"), dict) else {}
    fact_text = _fact_search_text(fact, subject, obj)
    required = _required_anchors(query)
    optional = _optional_anchors(query, required)
    matched_required = [anchor["label"] for anchor in required if _anchor_matches(anchor, fact_text)]
    missing_required = [anchor["label"] for anchor in required if anchor["label"] not in matched_required]
    matched_optional = [anchor["label"] for anchor in optional if _anchor_matches(anchor, fact_text)]
    required_count = len(required)
    optional_count = len(optional)
    if required_count == 0 and optional_count == 0:
        return {
            "fact_id": fact.candidate_id,
            "fact_type": fact.fact_type,
            "matched_required_anchors": [],
            "missing_required_anchors": ["query_anchor"],
            "matched_optional_anchors": [],
            "coverage_score": 0.0,
            "exact_match": False,
            "relaxed_match": False,
        }
    coverage = (
        (len(matched_required) + 0.35 * len(matched_optional))
        / max(1.0, float(required_count + 0.35 * optional_count))
    )
    exact = not missing_required
    relaxed = exact or bool(matched_required or matched_optional)
    return {
        "fact_id": fact.candidate_id,
        "fact_type": fact.fact_type,
        "matched_required_anchors": matched_required,
        "missing_required_anchors": missing_required,
        "matched_optional_anchors": matched_optional,
        "coverage_score": round(float(coverage), 4),
        "exact_match": exact,
        "relaxed_match": relaxed,
    }


def _required_anchors(query: TypedFactQuery) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    q_text = normalize_text(query.question)
    for process in query.processes:
        norm = normalize_text(process)
        if _is_specific_process_anchor(norm, q_text):
            anchors.append({"kind": "process", "label": process, "terms": _anchor_terms(process)})
    for material in query.materials:
        norm = normalize_text(material)
        if _is_specific_media_anchor(norm, q_text):
            anchors.append({"kind": "media", "label": material, "terms": _anchor_terms(material)})
    for prop in query.properties:
        norm = normalize_text(prop)
        if norm and query.answer_mode in {"technology_solution_search", "process_parameter_search", "experiment_catalog_search"}:
            anchors.append({"kind": "property", "label": prop, "terms": _anchor_terms(prop)})
    for constraint in query.numeric_constraints:
        parameter = str(constraint.get("parameter") or "").strip()
        if parameter:
            anchors.append({"kind": "numeric_parameter", "label": parameter, "terms": _anchor_terms(parameter)})
    return _dedupe_anchors(anchors)


def _optional_anchors(query: TypedFactQuery, required: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_labels = {item["label"] for item in required}
    anchors: list[dict[str, Any]] = []
    for kind, values in [
        ("material", query.materials),
        ("process", query.processes),
        ("property", query.properties),
        ("equipment", query.equipment),
        ("geography", query.geography),
    ]:
        for value in values:
            if value not in required_labels:
                anchors.append({"kind": kind, "label": value, "terms": _anchor_terms(value)})
    return _dedupe_anchors(anchors)


def _required_anchor_labels(query: TypedFactQuery) -> list[str]:
    return [item["label"] for item in _required_anchors(query)]


def _is_specific_process_anchor(norm: str, question_norm: str) -> bool:
    if not norm:
        return False
    specific_markers = [
        "циркуляц",
        "католит",
        "схем",
        "подач",
        "организац",
        "закач",
        "обессол",
        "газоочист",
        "удален",
        "кучн",
        "выщелач",
        "desalination",
        "circulation",
        "catholyte",
        "injection",
        "gas cleaning",
        "leaching",
    ]
    return any(marker in norm or marker in question_norm for marker in specific_markers) and norm not in {"электроэкстракция", "electrowinning"}


def _is_specific_media_anchor(norm: str, question_norm: str) -> bool:
    if not norm:
        return False
    media_terms = {
        "католит",
        "catholyte",
        "электролит",
        "electrolyte",
        "шахтные воды",
        "mine water",
        "сульфаты",
        "хлориды",
        "раствор",
        "solution",
    }
    return norm in media_terms or any(term in question_norm for term in media_terms if term in norm or norm in term)


def _anchor_terms(value: str) -> list[str]:
    norm = normalize_text(value)
    aliases = {
        "католит": ["католит", "catholyte", "электролит", "electrolyte"],
        "циркуляция католита": ["циркуляция католита", "циркуляц", "catholyte circulation", "electrolyte circulation"],
        "электроэкстракция": ["электроэкстракция", "electrowinning"],
        "сульфаты": ["сульфаты", "сульфат", "sulfate", "sulphate", "so4"],
        "хлориды": ["хлориды", "хлорид", "chloride"],
        "обессоливание": ["обессол", "desalination"],
    }
    terms = [norm, value]
    for key, vals in aliases.items():
        if norm == normalize_text(key) or norm in [normalize_text(item) for item in vals]:
            terms.extend(vals)
    return list(dict.fromkeys(normalize_text(item) for item in terms if normalize_text(item)))


def _anchor_matches(anchor: dict[str, Any], fact_text: str) -> bool:
    terms = anchor.get("terms") or []
    return any(term and term in fact_text for term in terms)


def _fact_search_text(fact: AcceptedFact, subject: dict[str, Any], obj: dict[str, Any]) -> str:
    parts: list[str] = [fact.fact_type]
    for mapping in [subject, obj, fact.normalized_fact or {}]:
        for value in mapping.values():
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif isinstance(value, dict):
                parts.extend(str(item) for item in value.values())
            else:
                parts.append(str(value))
    for evidence in fact.evidence:
        parts.extend([evidence.quote or "", evidence.source.source_name or "", evidence.source.section_path or ""])
    return normalize_text(" ".join(parts))


def _dedupe_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for anchor in anchors:
        label = str(anchor.get("label") or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        result.append(anchor)
    return result


def _aggregate_report_values(reports: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for report in reports:
        items = report.get(key) or []
        if isinstance(items, list):
            values.extend(str(item) for item in items if str(item or "").strip())
    return list(dict.fromkeys(values))


def _missing_fact_types(target_fact_types: list[str], facts: list[AcceptedFact]) -> list[str]:
    present = {fact.fact_type for fact in facts}
    return [fact_type for fact_type in target_fact_types if fact_type not in present]


def _relaxed_partial_reason(query: TypedFactQuery, reports: list[dict[str, Any]]) -> str:
    missing = _aggregate_report_values(reports, "missing_required_anchors")
    if missing:
        return "accepted facts matched broad context but missed required anchors: " + ", ".join(missing[:6])
    return "accepted facts matched only part of requested constraints"


def _fact_materials(subject: dict[str, Any], obj: dict[str, Any]) -> list[str]:
    return _values(subject, "material", "material_raw", "materials", "media", "commodity") + _values(obj, "material", "materials", "media", "commodity")


def _fact_processes(subject: dict[str, Any], obj: dict[str, Any]) -> list[str]:
    return _values(subject, "process", "process_raw", "regime", "regimes", "subprocess") + _values(obj, "process", "regime", "regimes", "subprocess", "process_context", "technology", "solution_name")


def _fact_properties(subject: dict[str, Any], obj: dict[str, Any]) -> list[str]:
    return _values(obj, "property", "properties", "parameter", "parameters", "indicator", "metric", "analyte", "target_problem", "claim") + _values(subject, "parameter", "property", "properties")


def _fact_equipment(subject: dict[str, Any], obj: dict[str, Any]) -> list[str]:
    return _values(subject, "equipment") + _values(obj, "equipment", "technology", "solution_name")


def _fact_geography(subject: dict[str, Any], obj: dict[str, Any]) -> list[str]:
    return _values(subject, "geography", "country", "region") + _values(obj, "geography", "country", "region")


def _fact_years(subject: dict[str, Any], obj: dict[str, Any]) -> list[int]:
    years: list[int] = []
    for value in [*_values(subject, "year", "date", "period"), *_values(obj, "year", "date", "period")]:
        years.extend(int(item) for item in re.findall(r"\b(?:19|20)\d{2}\b", str(value)))
    return list(dict.fromkeys(years))


def _values(mapping: dict[str, Any], *keys: str) -> list[str]:
    result: list[str] = []
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list):
            result.extend(str(item) for item in value if str(item or "").strip())
        elif value is not None and str(value).strip():
            result.append(str(value).strip())
    return list(dict.fromkeys(result))


def _any_match(requested: list[str], actual: list[str], matcher) -> bool:
    return bool(requested and actual and any(matcher(left, right) for left in actual for right in requested))


def _matches_material(actual: str, requested: str) -> bool:
    return material_matches(actual, requested) or normalize_text(actual) == normalize_text(requested)


def _matches_process(actual: str, requested: str) -> bool:
    return regime_matches(actual, requested) or _matches_text(actual, requested)


def _matches_property(actual: str, requested: str) -> bool:
    return property_matches(actual, requested) or _matches_text(actual, requested)


def _matches_text(actual: str, requested: str) -> bool:
    left = normalize_text(actual)
    right = normalize_text(requested)
    return bool(left and right and (left == right or left in right or right in left))


def _matches_time_filters(filters: list[dict[str, Any]], years: list[int]) -> bool:
    if not years:
        return False
    for item in filters:
        if item.get("type") == "year_range":
            start = int(item.get("start_year") or min(years))
            end = int(item.get("end_year") or max(years))
            if any(start <= year <= end for year in years):
                return True
        if item.get("type") == "relative_years":
            return True
    return False


def _matches_numeric_constraints(constraints: list[dict[str, Any]], normalized: dict[str, Any], obj: dict[str, Any]) -> bool:
    return any(_numeric_constraint_matches_fact(constraint, normalized, obj) for constraint in constraints)


def _numeric_constraint_matches_fact(constraint: dict[str, Any], normalized: dict[str, Any], obj: dict[str, Any]) -> bool:
    fact_unit = normalize_unit_label(str(normalized.get("unit") or obj.get("unit") or ""))
    constraint_unit = normalize_unit_label(str(constraint.get("unit") or ""))
    if constraint_unit and fact_unit and constraint_unit != fact_unit:
        return False
    parameter = str(constraint.get("parameter") or "")
    if parameter and not _constraint_parameter_matches(parameter, normalized, obj):
        return False
    value = normalized.get("value")
    value_min = obj.get("value_min")
    value_max = obj.get("value_max")
    values = [float(item) for item in [value, value_min, value_max] if isinstance(item, (int, float))]
    if not values:
        return False
    expected = constraint.get("value")
    expected_min = constraint.get("value_min")
    expected_max = constraint.get("value_max")
    operator = str(constraint.get("operator") or "").strip()
    fact_min = min(values)
    fact_max = max(values)
    if isinstance(expected, (int, float)):
        target = float(expected)
        if operator in {"<=", "<"}:
            return fact_min <= target if operator == "<=" else fact_min < target
        if operator in {">=", ">"}:
            return fact_max >= target if operator == ">=" else fact_max > target
        return _any_numeric_close(values, target)
    if isinstance(expected_min, (int, float)) and isinstance(expected_max, (int, float)):
        left = float(expected_min)
        right = float(expected_max)
        return fact_min <= right and fact_max >= left
    return False


def _constraint_parameter_matches(parameter: str, normalized: dict[str, Any], obj: dict[str, Any]) -> bool:
    values = [
        str(obj.get("property") or ""),
        str(obj.get("parameter") or ""),
        str(obj.get("indicator") or ""),
        str(obj.get("metric") or ""),
        str(obj.get("analyte") or ""),
        str((normalized.get("subject") or {}).get("material") if isinstance(normalized.get("subject"), dict) else ""),
    ]
    return any(_matches_property(value, parameter) or _matches_material(value, parameter) or _matches_text(value, parameter) for value in values if value)


def _any_numeric_close(values: list[Any], expected: float) -> bool:
    return any(abs(float(item) - expected) <= max(1e-6, abs(expected) * 0.02) for item in values)


def _dedupe_facts(facts: list[AcceptedFact]) -> list[AcceptedFact]:
    seen: set[str] = set()
    result: list[AcceptedFact] = []
    for fact in facts:
        key = fact.candidate_id
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result


def _dedupe_evidence(facts: list[AcceptedFact]) -> list[EvidenceSpan]:
    seen: set[tuple[str | None, str | None, str]] = set()
    result: list[EvidenceSpan] = []
    for fact in facts:
        for evidence in fact.evidence:
            key = (evidence.source.document_id, evidence.source.chunk_id, evidence.quote)
            if key in seen:
                continue
            seen.add(key)
            result.append(evidence)
    return result


def _fact_rank(fact: AcceptedFact) -> float:
    score = float(fact.score or 0.0)
    if fact.evidence:
        score += 0.25
    normalized = fact.normalized_fact or {}
    if normalized.get("value") is not None:
        score += 0.1
    if normalized.get("source_adapter"):
        score += 0.05
    return score


def _numeric_constraints_match_count(facts: list[AcceptedFact], query: TypedFactQuery) -> int:
    count = 0
    for fact in facts:
        normalized = fact.normalized_fact or {}
        obj = normalized.get("object") if isinstance(normalized.get("object"), dict) else {}
        if query.numeric_constraints and _matches_numeric_constraints(query.numeric_constraints, normalized, obj):
            count += 1
    return count


def _normalized_query_terms(query: TypedFactQuery) -> dict[str, list[str]]:
    return {
        "materials": [item for item in query.materials],
        "processes": [item for item in query.processes],
        "properties": [item for item in query.properties],
        "equipment": [item for item in query.equipment],
        "geography": [item for item in query.geography],
    }


def _expanded_typed_query(query: TypedFactQuery) -> str:
    parts = [query.question, *query.materials, *query.processes, *query.properties, *query.equipment, *query.geography]
    for constraint in query.numeric_constraints:
        parts.extend(str(constraint.get(key) or "") for key in ("parameter", "value", "value_min", "value_max", "unit"))
    return " ".join(part for part in parts if str(part or "").strip())


def _rerank_chunks(chunks: list[Chunk], query: TypedFactQuery) -> list[Chunk]:
    if not chunks:
        return []
    return [
        chunk
        for _, chunk in sorted(
            ((_chunk_relevance_boost(chunk, query), chunk) for chunk in chunks),
            key=lambda item: item[0],
            reverse=True,
        )
    ]


TEST_SOURCE_PATTERNS = [
    "kg ui smoke",
    "kg_ui_smoke",
    "ui smoke",
    "smoke",
    "test fixture",
    "тестовый технический документ",
]

DOMAIN_ANCHOR_GROUPS: dict[str, list[str]] = {
    "mine": ["рудник", "рудник", "шахт", "подзем", "глубок", "mine", "mining", "underground", "kidd", "mponeng"],
    "cooling": ["охлажд", "холодиль", "чиллер", "лед", "холодный забой", "bac", "mwr", "cool", "cooling", "chiller", "refrigeration", "ice", "cold stope"],
    "heat": ["тепл", "самосжат", "геотерм", "heat", "thermal", "autocompression", "geothermal"],
    "ventilation": ["вентиляц", "воздух", "ventilation", "airflow"],
    "process_solution": ["метод", "способ", "технолог", "схем", "систем", "решен", "method", "technology", "system", "solution"],
}

QUERY_STOPWORDS = {
    "какие", "какой", "какая", "какое", "как", "почему", "что", "для", "при", "или", "это", "этот",
    "этого", "описаны", "применяются", "используется", "работает", "основные", "возникают", "документе",
    "каковы", "какими", "where", "what", "which", "how", "why", "the", "and", "with", "from", "that",
}


def filter_source_grounded_chunks(chunks: list[Chunk], query: TypedFactQuery, *, top_k: int) -> tuple[list[Chunk], dict[str, Any]]:
    """Deduplicate and production-filter chunks used as source-grounded evidence."""

    before = len(chunks)
    production_filter = not _bool_env("RETRIEVAL_INCLUDE_TEST_SOURCES", False)
    deduped: list[Chunk] = []
    seen_chunks: set[tuple[str, str]] = set()
    deduplicated_count = 0
    excluded_test_count = 0
    excluded_sources: set[str] = set()
    for chunk in chunks:
        key = (str(chunk.doc_id or ""), str(chunk.chunk_id or ""))
        if key in seen_chunks:
            deduplicated_count += 1
            continue
        seen_chunks.add(key)
        if production_filter and _is_test_or_smoke_chunk(chunk):
            excluded_test_count += 1
            excluded_sources.add(_chunk_source_name(chunk))
            continue
        deduped.append(chunk)

    profile = _evidence_anchor_profile(query)
    scored: list[tuple[float, Chunk, dict[str, Any]]] = []
    dropped_low = 0
    matched_anchors: set[str] = set()
    for chunk in deduped:
        score, report = _evidence_chunk_score(chunk, query, profile)
        if not report["passes"]:
            dropped_low += 1
            continue
        matched_anchors.update(report["matched_anchors"])
        scored.append((score, chunk, report))
    scored.sort(key=lambda item: item[0], reverse=True)

    if scored:
        source_totals: dict[str, float] = {}
        source_max: dict[str, float] = {}
        source_reports: dict[str, list[dict[str, Any]]] = {}
        for score, chunk, report in scored:
            source_name = _chunk_source_name(chunk)
            source_totals[source_name] = source_totals.get(source_name, 0.0) + score
            source_max[source_name] = max(source_max.get(source_name, 0.0), score)
            source_reports.setdefault(source_name, []).append(report)
        best_source = max(source_totals, key=source_totals.get)
        best_total = source_totals.get(best_source, 0.0)
        best_max = max(source_max.values(), default=0.0)
        required_labels = set(profile.get("required_anchor_labels") or [])
        best_source_anchors = {
            anchor
            for report in source_reports.get(best_source, [])
            for anchor in report.get("matched_anchors", [])
        }
        if len(source_reports.get(best_source, [])) >= 2 and (not required_labels or required_labels.issubset(best_source_anchors)):
            eligible_sources = {best_source}
        else:
            eligible_sources = {
                source
                for source, total in source_totals.items()
                if total >= max(1.0, best_total * 0.55) or source_max.get(source, 0.0) >= max(1.0, best_max * 0.9)
            }
        eligible_scored = [item for item in scored if _chunk_source_name(item[1]) in eligible_sources]
        dropped_low += max(0, len(scored) - len(eligible_scored))
        scored = eligible_scored

    selected: list[Chunk] = []
    per_source_counts: dict[str, int] = {}
    max_per_source = max(3, max(1, top_k) // 2)
    for _, chunk, _ in scored:
        source_name = _chunk_source_name(chunk)
        if per_source_counts.get(source_name, 0) >= max_per_source:
            continue
        selected.append(chunk)
        per_source_counts[source_name] = per_source_counts.get(source_name, 0) + 1
        if len(selected) >= top_k:
            break

    diagnostics = source_grounded_filter_diagnostics(query, chunks, selected)
    diagnostics.update(
        {
            "evidence_candidates_before_filter": before,
            "evidence_candidates_after_filter": len(selected),
            "evidence_deduplicated_count": deduplicated_count,
            "evidence_dropped_low_relevance_count": dropped_low,
            "excluded_test_chunks_count": excluded_test_count,
            "excluded_test_sources": sorted(item for item in excluded_sources if item),
            "retrieval_filtered_for_production": production_filter,
            "evidence_source_groups_count": len(per_source_counts),
        }
    )
    all_required = set(profile["required_anchor_labels"])
    matched_required = {item for item in matched_anchors if item in all_required}
    diagnostics["matched_evidence_anchors"] = sorted(matched_required)
    diagnostics["missing_evidence_anchors"] = sorted(all_required - matched_required)
    diagnostics["evidence_coverage_score"] = round(len(matched_required) / max(1, len(all_required)), 4) if all_required else 1.0
    return selected, diagnostics


def source_grounded_filter_diagnostics(query: TypedFactQuery, chunks_before: list[Chunk], chunks_after: list[Chunk]) -> dict[str, Any]:
    profile = _evidence_anchor_profile(query)
    matched: set[str] = set()
    for chunk in chunks_after:
        _, report = _evidence_chunk_score(chunk, query, profile)
        matched.update(report["matched_anchors"])
    required = set(profile["required_anchor_labels"])
    matched_required = {item for item in matched if item in required}
    return {
        "evidence_candidates_before_filter": len(chunks_before),
        "evidence_candidates_after_filter": len(chunks_after),
        "evidence_deduplicated_count": 0,
        "evidence_dropped_low_relevance_count": 0,
        "excluded_test_chunks_count": 0,
        "excluded_test_sources": [],
        "retrieval_filtered_for_production": not _bool_env("RETRIEVAL_INCLUDE_TEST_SOURCES", False),
        "matched_evidence_anchors": sorted(matched_required),
        "missing_evidence_anchors": sorted(required - matched_required),
        "evidence_coverage_score": round(len(matched_required) / max(1, len(required)), 4) if required else 1.0,
    }


def _chunk_relevance_boost(chunk: Chunk, query: TypedFactQuery) -> float:
    text = normalize_text(" ".join([chunk.text or "", str((chunk.metadata or {}).get("source_name") or ""), chunk.section_path or ""]))
    score = 0.0
    for weight, terms in [
        (3.0, query.materials),
        (3.0, query.processes),
        (2.5, query.properties),
        (2.0, query.equipment),
        (1.5, query.geography),
    ]:
        for term in terms:
            norm = normalize_text(term)
            if norm and norm in text:
                score += weight
    for constraint in query.numeric_constraints:
        parameter = normalize_text(str(constraint.get("parameter") or ""))
        unit = normalize_text(str(constraint.get("unit") or ""))
        if parameter and parameter in text:
            score += 2.0
        if unit and unit in text:
            score += 1.0
    return score


def _evidence_anchor_profile(query: TypedFactQuery) -> dict[str, Any]:
    query_text = _loose_norm(_expanded_typed_query(query))
    active_groups: dict[str, list[str]] = {}
    for group, terms in DOMAIN_ANCHOR_GROUPS.items():
        if any(_loose_norm(term) in query_text for term in terms):
            active_groups[group] = terms
    words = [
        item
        for item in re.findall(r"[a-zа-яё0-9]{4,}", query_text, flags=re.IGNORECASE)
        if item not in QUERY_STOPWORDS and not item.isdigit()
    ]
    required_labels: list[str] = []
    required_labels.extend(active_groups.keys())
    for term in query.materials + query.processes + query.properties + query.equipment + query.geography:
        norm = _loose_norm(term)
        if norm and norm not in required_labels:
            required_labels.append(norm)
    if not required_labels:
        required_labels.extend(words[:5])
    return {
        "query_text": query_text,
        "active_groups": active_groups,
        "query_words": list(dict.fromkeys(words)),
        "required_anchor_labels": list(dict.fromkeys(required_labels)),
    }


def _evidence_chunk_score(chunk: Chunk, query: TypedFactQuery, profile: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    text = _loose_norm(" ".join([chunk.text or "", chunk.section_path or "", _chunk_source_name(chunk)]))
    matched_groups: set[str] = set()
    for group, terms in (profile.get("active_groups") or {}).items():
        if any(_loose_norm(term) in text for term in terms):
            matched_groups.add(group)
    query_words = list(profile.get("query_words") or [])
    matched_words = {word for word in query_words if len(word) >= 4 and word in text}
    active_groups = dict(profile.get("active_groups") or {})
    active_group_count = len(active_groups)
    # Generic solution words ("method", "system", "technology") should boost
    # already relevant evidence, but must not make an unrelated technical chunk
    # pass for a specific domain question such as deep-mine cooling.
    essential_groups = [group for group in active_groups if group != "process_solution"]
    if len(essential_groups) >= 2:
        missing_essential = [group for group in essential_groups if group not in matched_groups]
        if missing_essential:
            return 0.0, {"passes": False, "matched_anchors": sorted(matched_groups), "matched_words": sorted(matched_words)}
    required_group_matches = min(2, active_group_count) if active_group_count >= 2 else active_group_count
    if active_group_count and len(matched_groups) < required_group_matches:
        return 0.0, {"passes": False, "matched_anchors": sorted(matched_groups), "matched_words": sorted(matched_words)}
    if not active_group_count and query_words and not matched_words:
        return 0.0, {"passes": False, "matched_anchors": [], "matched_words": []}
    score = 5.0 * len(matched_groups) + 0.35 * len(matched_words) + _chunk_relevance_boost(chunk, query)
    source = _loose_norm(_chunk_source_name(chunk))
    if "глубокие рудники" in source:
        score += 1.0
    return score, {
        "passes": True,
        "matched_anchors": sorted(matched_groups | {word for word in matched_words if word in profile.get("required_anchor_labels", [])}),
        "matched_words": sorted(matched_words),
    }


def _is_test_or_smoke_chunk(chunk: Chunk) -> bool:
    metadata = chunk.metadata or {}
    haystack = _loose_norm(
        " ".join(
            str(value or "")
            for value in [
                metadata.get("source_name"),
                metadata.get("source_title"),
                metadata.get("filename"),
                chunk.text[:500],
            ]
        )
    )
    return any(_loose_norm(pattern) in haystack for pattern in TEST_SOURCE_PATTERNS)


def _chunk_source_name(chunk: Chunk) -> str:
    metadata = chunk.metadata or {}
    return str(metadata.get("source_name") or metadata.get("source_title") or metadata.get("filename") or chunk.doc_id or "")


def _loose_norm(value: Any) -> str:
    text = normalize_text(str(value or "")).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

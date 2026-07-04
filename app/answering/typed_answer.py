"""Native answer payloads for verified typed AcceptedFact results."""

from __future__ import annotations

import re
from typing import Any

from ..extraction.models import AcceptedFact, EvidenceSpan
from ..models.schemas import Chunk
from ..retrieval.typed_fact_retriever import TypedFactSearchResult


TYPED_ANSWER_MODES = {
    "process_parameter_search",
    "technology_solution_search",
    "experiment_catalog_search",
    "technology_comparison",
    "domestic_vs_foreign_practice",
    "expert_search",
    "knowledge_gap_search",
    "literature_review",
    "generic_typed_fact_summary",
}


def build_typed_answer_payload(
    *,
    question: str,
    result: TypedFactSearchResult,
    retrieval: dict[str, Any],
    kg_diagnostics: dict[str, Any],
    llm: dict[str, Any],
) -> dict[str, Any]:
    match_quality = "exact" if result.retrieval_status == "exact_facts_found" else "relaxed" if result.retrieval_status == "relaxed_facts_found" else "none"
    facts = [_fact_row(item, match_quality=match_quality) for item in result.accepted_facts]
    evidence = [_evidence_row(item) for item in result.evidence]
    fallback_sources = [_chunk_source_row(item) for item in result.fallback_chunks]
    subgraph = _typed_subgraph(facts, fallback_sources)
    status = "ok" if result.retrieval_status == "exact_facts_found" else "partial" if facts or fallback_sources else "no_exact_match"
    answer = _draft_answer(result, facts, fallback_sources)
    sources = _source_rows(facts, evidence, fallback_sources)
    retrieval_status = result.retrieval_status
    source_grounded = bool(fallback_sources and not facts)
    payload_answer_mode = "source_grounded_answer" if source_grounded else result.query.answer_mode
    typed_diag = result.diagnostics or {}
    embedding_status = retrieval.get("embedding_status") or typed_diag.get("embedding_status") or {}
    effective_retrieval_mode = (
        typed_diag.get("effective_retrieval_mode")
        or retrieval.get("effective_retrieval_mode")
        or embedding_status.get("effective_retrieval_mode")
    )
    degraded_reason = typed_diag.get("degraded_reason") or retrieval.get("degraded_reason") or embedding_status.get("degraded_reason", "")
    partial_reason = typed_diag.get("partial_reason") or (
        "" if facts else "relevant chunks found but no accepted structured facts matched" if fallback_sources else "no relevant sources found"
    )
    partial_details = _partial_navigation_details(result, fallback_sources, partial_reason)
    chunks_found_bm25 = typed_diag.get("chunks_found_bm25", retrieval.get("chunks_found_bm25", 0))
    chunks_found_dense = typed_diag.get("chunks_found_dense", retrieval.get("chunks_found_dense", 0))
    chunks_after_fusion = typed_diag.get("chunks_after_fusion", retrieval.get("chunks_after_fusion", 0))
    return {
        "answer": answer,
        "status": status,
        "answer_mode": payload_answer_mode,
        "analytical_intent": result.query.answer_mode,
        "intent": payload_answer_mode if source_grounded else result.query.answer_mode,
        "answer_is_verified": result.retrieval_status == "exact_facts_found",
        "source_grounded_answer_used": source_grounded,
        "semantic_fallback_executed": source_grounded,
        "fallback_reason": "typed_facts_absent_relevant_chunks_found" if source_grounded else "",
        "evidence_chunks_used_count": len(fallback_sources) if source_grounded else 0,
        "accepted_facts_used_count": len(facts),
        "constraints": {
            "raw_question": question,
            "materials": result.query.materials,
            "regimes": result.query.processes,
            "properties": result.query.properties,
            "equipment": result.query.equipment,
            "geographies": result.query.geography,
            "time_filters": result.query.time_filters,
            "numeric_constraints": result.query.numeric_constraints,
            "target_fact_types": result.query.target_fact_types,
            "answer_mode": result.query.answer_mode,
        },
        "facts": facts,
        "typed_facts": facts,
        "experiments": [],
        "technical_objects": [],
        "parts": [],
        "parameters": _entity_rows(facts, "parameter"),
        "standards": [],
        "materials": _entity_rows(facts, "material"),
        "requirements": [],
        "equipment": _entity_rows(facts, "equipment"),
        "laboratories": _entity_rows(facts, "laboratory"),
        "sources": sources,
        "evidence": evidence or fallback_sources,
        "gaps": [],
        "data_gaps": [],
        "partial_matches": {
            "retrieval_status": retrieval_status,
            "relaxed_matches_count": len(result.relaxed_matches),
            "missing_filters": result.missing_filters,
            **partial_details,
        },
        "decision_history": [],
        "subgraph": subgraph,
        "graph_context": {
            "facts_count": len(facts),
            "sources_count": len(sources),
            "evidence_count": len(evidence or fallback_sources),
            "subgraph_nodes": len(subgraph.get("nodes") or []),
            "subgraph_edges": len(subgraph.get("edges") or []),
        },
        "retrieval": {
            **retrieval,
            "typed_fact_retrieval_status": retrieval_status,
            "retrieval_status": retrieval_status,
            "answer_mode": result.query.answer_mode,
            "target_fact_types": result.query.target_fact_types,
            "typed_facts_found": len(facts),
            "answer_is_verified": result.retrieval_status == "exact_facts_found",
            "source_grounded_answer_used": source_grounded,
            "semantic_fallback_executed": source_grounded,
            "fallback_reason": "typed_facts_absent_relevant_chunks_found" if source_grounded else "",
            "evidence_chunks_used_count": len(fallback_sources) if source_grounded else 0,
            "accepted_facts_used_count": len(facts),
            "chunks_found_bm25": chunks_found_bm25,
            "chunks_found_dense": chunks_found_dense,
            "chunks_after_fusion": chunks_after_fusion,
            "chunk_retrieval_executed": typed_diag.get("chunk_retrieval_executed", bool(fallback_sources)),
            "chunk_retrieval_skip_reason": typed_diag.get("chunk_retrieval_skip_reason", ""),
            "effective_retrieval_mode": effective_retrieval_mode,
            "degraded_reason": degraded_reason,
            "embedding_status": embedding_status,
            "top_fused_chunks": typed_diag.get("top_fused_chunks", retrieval.get("top_fused_chunks", [])),
            "relevant_sources": partial_details["relevant_sources"],
            "matched_normalized_terms": partial_details["matched_normalized_terms"],
            "matched_required_anchors": typed_diag.get("matched_required_anchors", []),
            "missing_required_anchors": typed_diag.get("missing_required_anchors", []),
            "matched_optional_anchors": typed_diag.get("matched_optional_anchors", []),
            "coverage_score": typed_diag.get("coverage_score"),
            "exact_match": typed_diag.get("exact_match", retrieval_status == "exact_facts_found"),
            "relaxed_match": typed_diag.get("relaxed_match", retrieval_status == "relaxed_facts_found"),
            "why_not_verified": partial_details["why_not_verified"],
            "suggested_next_extraction_target": partial_details["suggested_next_extraction_target"],
            "partial_reason": partial_reason,
        },
        "llm": llm,
        "diagnostics": {
            **kg_diagnostics,
            "selected_answer_mode": result.query.answer_mode,
            "retrieval_status": retrieval_status,
            "typed_facts_found": len(facts),
            "answer_is_verified": result.retrieval_status == "exact_facts_found",
            "source_grounded_answer_used": source_grounded,
            "semantic_fallback_executed": source_grounded,
            "fallback_reason": "typed_facts_absent_relevant_chunks_found" if source_grounded else "",
            "evidence_chunks_used_count": len(fallback_sources) if source_grounded else 0,
            "accepted_facts_used_count": len(facts),
            "chunks_found_bm25": chunks_found_bm25,
            "chunks_found_dense": chunks_found_dense,
            "chunks_after_fusion": chunks_after_fusion,
            "chunk_retrieval_executed": typed_diag.get("chunk_retrieval_executed", bool(fallback_sources)),
            "chunk_retrieval_skip_reason": typed_diag.get("chunk_retrieval_skip_reason", ""),
            "effective_retrieval_mode": effective_retrieval_mode,
            "degraded_reason": degraded_reason,
            "embedding_status": embedding_status,
            "top_fused_chunks": typed_diag.get("top_fused_chunks", retrieval.get("top_fused_chunks", [])),
            "normalized_query_terms": typed_diag.get("normalized_query_terms", {}),
            "matched_required_anchors": typed_diag.get("matched_required_anchors", []),
            "missing_required_anchors": typed_diag.get("missing_required_anchors", []),
            "matched_optional_anchors": typed_diag.get("matched_optional_anchors", []),
            "coverage_score": typed_diag.get("coverage_score"),
            "exact_match": typed_diag.get("exact_match", retrieval_status == "exact_facts_found"),
            "relaxed_match": typed_diag.get("relaxed_match", retrieval_status == "relaxed_facts_found"),
            "missing_structured_fact_types": typed_diag.get("missing_structured_fact_types", []),
            "numeric_constraints_parsed": typed_diag.get("numeric_constraints_parsed", result.query.numeric_constraints),
            "numeric_constraints_matched": typed_diag.get("numeric_constraints_matched", 0),
            "partial_reason": partial_reason,
            "relevant_sources": partial_details["relevant_sources"],
            "matched_normalized_terms": partial_details["matched_normalized_terms"],
            "why_not_verified": partial_details["why_not_verified"],
            "suggested_next_extraction_target": partial_details["suggested_next_extraction_target"],
            "typed_fact_retrieval": {
                **result.diagnostics,
                "status": retrieval_status,
                "missing_filters": result.missing_filters,
                "accepted_facts_used_in_answers_count": len(facts),
                "chunks_only_questions_count": 1 if retrieval_status == "chunks_only_no_structured_facts" else 0,
                "no_structured_facts_by_intent": {result.query.answer_mode: 1} if not facts else {},
            },
        },
    }


def _draft_answer(result: TypedFactSearchResult, facts: list[dict[str, Any]], fallback_sources: list[dict[str, Any]]) -> str:
    if facts:
        if result.retrieval_status == "relaxed_facts_found":
            missing = result.diagnostics.get("missing_required_anchors") if result.diagnostics else []
            missing_text = ", ".join(str(item) for item in (missing or [])[:4])
            lines = [
                "Найдено связанное подтверждённое утверждение, но точного подтверждённого ответа по всем условиям в structured graph пока нет.",
            ]
            if missing_text:
                lines.append(f"Не покрыты обязательные элементы запроса: {missing_text}.")
        elif result.query.answer_mode == "technology_solution_search":
            lines = ["В источниках описано следующее решение:"]
        else:
            mode_title = _mode_title(result.query.answer_mode)
            lines = [mode_title, "Подтверждённые факты:"]
        for row in facts[:8]:
            lines.append(f"- {_fact_line(row)}")
        if result.retrieval_status == "relaxed_facts_found":
            lines.append("Часть фильтров не совпала полностью; результат показан как близкие подтверждённые факты.")
        return "\n".join(lines)
    if fallback_sources:
        lines = ["Найдены релевантные источники, но структурированных подтверждённых фактов недостаточно."]
        lines.append("Нормализованные темы и фильтры использованы только для поиска; verified-факты по ним не сформированы.")
        if result.query.target_fact_types:
            lines.append("Не удалось структурно подтвердить типы фактов: " + ", ".join(result.query.target_fact_types) + ".")
        source_names = list(dict.fromkeys(_safe_source_label(row.get("source_name")) for row in fallback_sources if row.get("source_name")))
        if source_names:
            lines.append("Релевантные источники для проверки: " + "; ".join(source_names[:5]) + ".")
        lines.append("Эти источники можно использовать как навигацию, но не как verified factual conclusion.")
        return "\n".join(lines)
    return "В активном корпусе релевантные источники не найдены."


def _partial_navigation_details(
    result: TypedFactSearchResult,
    fallback_sources: list[dict[str, Any]],
    partial_reason: str,
) -> dict[str, Any]:
    normalized_terms = result.diagnostics.get("normalized_query_terms", {}) if result.diagnostics else {}
    missing_types = list(result.diagnostics.get("missing_structured_fact_types") or result.query.target_fact_types or [])
    why: list[str] = []
    if result.retrieval_status == "exact_facts_found":
        return {
            "relevant_sources": [],
            "matched_normalized_terms": normalized_terms,
            "missing_structured_fact_types": [],
            "why_not_verified": [],
            "suggested_next_extraction_target": "",
        }
    if not result.accepted_facts:
        why.append("no accepted fact")
    elif result.retrieval_status == "relaxed_facts_found":
        missing_required = result.diagnostics.get("missing_required_anchors") if result.diagnostics else []
        if missing_required:
            why.append("missing required anchors: " + ", ".join(str(item) for item in missing_required[:6]))
    if partial_reason:
        why.append(partial_reason)
    if missing_types:
        why.append("missing structured fact types: " + ", ".join(missing_types[:6]))
    if result.query.numeric_constraints:
        why.append("numeric constraints require accepted typed values with compatible units")
    return {
        "relevant_sources": [
            {
                "source_name": _safe_source_label(row.get("source_name")),
                "page": row.get("page"),
                "section_path": row.get("section_path"),
                "retrieval_backend": row.get("retrieval_backend"),
                "quote": row.get("quote"),
            }
            for row in fallback_sources[:8]
        ],
        "matched_normalized_terms": normalized_terms,
        "missing_structured_fact_types": missing_types,
        "why_not_verified": list(dict.fromkeys(item for item in why if item)),
        "suggested_next_extraction_target": _suggest_next_extraction_target(result.query.target_fact_types, normalized_terms),
    }


def _suggest_next_extraction_target(target_fact_types: list[str], normalized_terms: dict[str, Any]) -> str:
    targets = set(target_fact_types or [])
    if "ExpertiseFact" in targets:
        return "expertise topic adapter"
    if "TechnologySolutionFact" in targets:
        return "technology solution adapter"
    if "PublicationClaimFact" in targets:
        return "publication claim adapter"
    if "EconomicIndicatorFact" in targets:
        return "economic indicator adapter"
    if "FacilityCapacityFact" in targets:
        return "capacity unit/table adapter"
    if normalized_terms.get("numeric_constraints"):
        return "numeric typed fact extraction"
    return "typed fact schema/source adapter"


def _fact_row(fact: AcceptedFact, *, match_quality: str = "exact") -> dict[str, Any]:
    normalized = fact.normalized_fact or {}
    subject = normalized.get("subject") if isinstance(normalized.get("subject"), dict) else {}
    obj = normalized.get("object") if isinstance(normalized.get("object"), dict) else {}
    evidence = [_evidence_row(item) for item in fact.evidence]
    material = _first_value(subject, "material", "material_raw", "media", "commodity") or _first_value(obj, "material", "commodity")
    process = _first_value(subject, "process", "process_raw", "subprocess") or _first_value(obj, "process", "subprocess")
    prop = _first_value(obj, "property", "parameter", "indicator", "metric", "analyte", "topic") or _first_value(subject, "property", "parameter")
    technology = _first_value(obj, "technology", "solution_name") or process
    return {
        "fact_id": fact.candidate_id,
        "fact_type": fact.fact_type,
        "material": material,
        "media": material,
        "process": process,
        "regime": process,
        "property": prop,
        "parameter": prop,
        "technology": technology,
        "target_problem": _first_value(obj, "target_problem"),
        "conditions": _first_value(obj, "applicable_conditions", "condition"),
        "value": normalized.get("value"),
        "value_min": obj.get("value_min"),
        "value_max": obj.get("value_max"),
        "raw_value": obj.get("raw_value"),
        "unit": normalized.get("unit"),
        "equipment": _first_value(subject, "equipment") or _first_value(obj, "equipment"),
        "facility": _first_value(subject, "facility") or _first_value(obj, "facility"),
        "geography": _first_value(subject, "geography", "country", "region") or _first_value(obj, "geography", "country", "region"),
        "year": _first_value(subject, "year", "date", "period") or _first_value(obj, "year", "date", "period"),
        "expert": _first_value(subject, "expert", "expert_or_lab"),
        "laboratory": _first_value(subject, "laboratory"),
        "team": _first_value(subject, "team"),
        "claim": _first_value(obj, "claim", "source_note"),
        "source_note": _first_value(obj, "source_note"),
        "confidence": fact.score,
        "match_quality": match_quality,
        "evidence": evidence,
    }


def _evidence_row(item: EvidenceSpan) -> dict[str, Any]:
    source_name = _safe_source_label(item.source.source_name)
    return {
        "source_name": source_name,
        "document_id": item.source.document_id,
        "chunk_id": item.source.chunk_id,
        "page": item.source.page,
        "section_path": item.source.section_path,
        "quote": item.quote,
        "score": item.confidence,
        "retrieval_backend": "accepted_fact",
    }


def _chunk_source_row(chunk: Chunk) -> dict[str, Any]:
    source_name = _safe_source_label((chunk.metadata or {}).get("source_name") or (chunk.metadata or {}).get("filename") or "Источник корпуса")
    quote = (chunk.text or "").strip()
    return {
        "source_name": source_name,
        "document_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "page": chunk.page_start,
        "section_path": chunk.section_path,
        "quote": quote[:700] + ("..." if len(quote) > 700 else ""),
        "score": None,
        "retrieval_backend": "chunk_fallback",
    }


def _source_rows(facts: list[dict[str, Any]], evidence: list[dict[str, Any]], fallback_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = evidence or fallback_sources
    seen: set[tuple[Any, Any, Any]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("document_id"), row.get("chunk_id"), row.get("quote"))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    if result:
        return result[:12]
    for fact in facts:
        for row in fact.get("evidence") or []:
            key = (row.get("document_id"), row.get("chunk_id"), row.get("quote"))
            if key not in seen:
                seen.add(key)
                result.append(row)
    return result[:12]


def _typed_subgraph(facts: list[dict[str, Any]], fallback_sources: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()

    def add_node(node_id: str, label: str, node_type: str) -> None:
        if not label or node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append({"id": node_id, "label": label, "type": node_type, "properties": {}})

    for index, fact in enumerate(facts[:12], start=1):
        fact_id = f"Fact:{index}"
        add_node(fact_id, _fact_node_label(fact), "AcceptedFact")
        for key, node_type in [("material", "Material"), ("process", "ProcessRegime"), ("property", "Property"), ("equipment", "Equipment")]:
            value = str(fact.get(key) or "").strip()
            if not value:
                continue
            node_id = f"{node_type}:{value}"
            add_node(node_id, value, node_type)
            edges.append({"id": f"{fact_id}:{key}:{node_id}", "source": fact_id, "target": node_id, "label": key, "type": key, "properties": {}})
    if not nodes:
        add_node("TypedQuery:focus", "Запрос\nstructured facts", "Query")
        for index, source in enumerate((fallback_sources or [])[:8], start=1):
            label = _safe_source_label(source.get("source_name") or source.get("title") or "Источник корпуса")
            node_id = f"Source:{index}"
            add_node(node_id, label, "Source")
            edges.append(
                {
                    "id": f"TypedQuery:focus:source:{index}",
                    "source": "TypedQuery:focus",
                    "target": node_id,
                    "label": "релевантный источник",
                    "type": "RELEVANT_SOURCE",
                    "properties": {},
                }
            )
    return {"nodes": nodes[:24], "edges": edges[:32]}


def _fact_node_label(fact: dict[str, Any]) -> str:
    prop = str(fact.get("property") or fact.get("parameter") or fact.get("fact_type") or "Факт")
    value = _value_text(fact)
    return f"{prop}\n{value}" if value else prop


def _mode_title(mode: str) -> str:
    return {
        "technology_solution_search": "Технологические решения",
        "process_parameter_search": "Параметры процесса",
        "experiment_catalog_search": "Экспериментальные факты",
        "technology_comparison": "Сравнение технологических решений",
        "domestic_vs_foreign_practice": "Отечественная и зарубежная практика",
        "expert_search": "Эксперты и команды",
        "knowledge_gap_search": "Пробелы знаний",
        "literature_review": "Обзор источников",
    }.get(mode, "Подтверждённые typed facts")


def _fact_line(row: dict[str, Any]) -> str:
    fact_type = str(row.get("fact_type") or "")
    source = _source_label(row)
    if fact_type == "PublicationClaimFact":
        parts = [str(item) for item in [row.get("material"), row.get("process"), row.get("property")] if item]
        claim = str(row.get("claim") or row.get("source_note") or "").strip()
        if claim:
            parts.append(claim)
        if source:
            parts.append(f"источник: {source}")
        return " | ".join(parts) or "подтверждённое утверждение из источника"
    if fact_type == "TechnologySolutionFact":
        parts = [
            str(item)
            for item in [
                row.get("technology") or row.get("process"),
                row.get("target_problem") or row.get("property"),
                row.get("material"),
                row.get("equipment"),
                row.get("conditions"),
            ]
            if item
        ]
        claim = str(row.get("claim") or row.get("source_note") or "").strip()
        if claim and claim not in parts:
            parts.append(claim)
        if source:
            parts.append(f"источник: {source}")
        return " | ".join(parts) or "подтверждённое технологическое решение"
    if fact_type == "ExpertiseFact":
        who = row.get("expert") or row.get("laboratory") or row.get("team")
        topic = row.get("property") or row.get("process") or row.get("material")
        parts = [str(item) for item in [who, topic] if item]
        if source:
            parts.append(f"источник: {source}")
        return " | ".join(parts) or "подтверждённая экспертиза/команда из источника"
    if fact_type == "FacilityCapacityFact":
        parts = [str(item) for item in [row.get("material"), row.get("facility"), row.get("geography"), row.get("year")] if item]
        value = _value_text(row)
        if value:
            parts.append(value)
        if source:
            parts.append(f"источник: {source}")
        return " | ".join(parts) or "подтверждённая мощность/производительность"
    if fact_type == "EconomicIndicatorFact":
        parts = [str(item) for item in [row.get("process"), row.get("material"), row.get("geography"), row.get("year")] if item]
        value = _value_text(row)
        if value:
            parts.append(value)
        if source:
            parts.append(f"источник: {source}")
        return " | ".join(parts) or "подтверждённый экономический показатель"
    parts = []
    for key in ["material", "process", "property", "equipment", "geography", "year"]:
        value = row.get(key)
        if value:
            parts.append(str(value))
    value = _value_text(row)
    if value:
        parts.append(value)
    if source:
        parts.append(f"источник: {source}")
    return " | ".join(parts) or str(row.get("fact_type") or "typed fact")


def _value_text(row: dict[str, Any]) -> str:
    unit = str(row.get("unit") or "").strip()
    if row.get("value_min") is not None and row.get("value_max") is not None:
        return f"{row['value_min']}–{row['value_max']} {unit}".strip()
    if row.get("value") is not None:
        return f"{row['value']} {unit}".strip()
    if row.get("raw_value"):
        return f"{row['raw_value']} {unit}".strip()
    if row.get("claim"):
        return str(row["claim"])
    return ""


def _source_label(row: dict[str, Any]) -> str:
    for evidence in row.get("evidence") or []:
        value = evidence.get("source_name")
        if value:
            return _safe_source_label(value)
    return ""


def _entity_rows(facts: list[dict[str, Any]], key: str) -> list[dict[str, str]]:
    values = []
    for fact in facts:
        value = fact.get(key)
        if value:
            values.append(str(value))
    return [{"name": item} for item in dict.fromkeys(values)]


def _first_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _safe_source_label(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\bdoc_[0-9a-fA-F]{8,64}_?", "", text)
    text = re.sub(r"\bchunk_[A-Za-z0-9_:-]+", "", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip(" -_")
    return text or "источник корпуса"

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.extraction.extraction import EntityRelationExtractor  # noqa: E402
from app.extraction.models import CandidateFact, ExtractionBundle  # noqa: E402
from app.extraction.pipeline import ExtractionPipeline  # noqa: E402
from app.graph.graph_repository import GraphRepositoryFactory  # noqa: E402
from app.models.schemas import Chunk  # noqa: E402
from app.retrieval.query_planner import QueryPlanner  # noqa: E402
from app.retrieval.retrieval import RetrievalEngine  # noqa: E402
from app.retrieval.typed_fact_retriever import TypedFactQuery, TypedFactRetriever  # noqa: E402
from app.storage.catalog import SQLiteCatalog  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Explain why retrieved chunks do or do not produce accepted typed facts.")
    parser.add_argument("question", help="Natural-language question to debug.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of fused chunks to inspect.")
    parser.add_argument("--catalog", default=str(settings.catalog_db_path), help="SQLite catalog path.")
    parser.add_argument("--json-only", action="store_true", help="Print only JSON output.")
    args = parser.parse_args()

    catalog = SQLiteCatalog(args.catalog)
    chunks = catalog.list_chunks(active_only=True)
    retrieval_engine = RetrievalEngine()
    retrieval_engine.index_chunks(chunks)
    pipeline = ExtractionPipeline(audit_enabled=False)
    repository = GraphRepositoryFactory.create(
        catalog=catalog,
        extractor=EntityRelationExtractor(),
        configured_backend="fallback",
        extraction_pipeline=pipeline,
    )

    constraints = QueryPlanner().parse(args.question)
    typed_query = TypedFactQuery.from_constraints(args.question, constraints)
    typed_result = TypedFactRetriever(repository, retrieval_engine=retrieval_engine).search(typed_query, top_k=max(args.top_k, 12))
    retrieval_chunks = typed_result.fallback_chunks or retrieval_engine.query(args.question, top_k=max(args.top_k, 12))
    selected_chunks = retrieval_chunks[: args.top_k]
    chunk_reports = [_inspect_chunk(chunk, pipeline, typed_query) for chunk in selected_chunks]
    stats = retrieval_engine.stats()
    anchor_diagnostics = _anchor_diagnostics(
        typed_result.diagnostics.get("missing_required_anchors") or [],
        selected_chunks,
        chunk_reports,
        typed_query,
    )

    report = {
        "question": args.question,
        "planned_constraints": _dump_model(constraints),
        "typed_query": _dump_model(typed_query),
        "retrieval": {
            "effective_retrieval_mode": stats.get("effective_retrieval_mode"),
            "degraded_reason": stats.get("degraded_reason") or stats.get("hybrid_degraded_reason"),
            "chunks_found_bm25": stats.get("chunks_found_bm25", 0),
            "chunks_found_dense": stats.get("chunks_found_dense", 0),
            "chunks_after_fusion": stats.get("chunks_after_fusion", 0),
            "embedding_status": stats.get("embedding_status", {}),
        },
        "typed_fact_search": {
            "retrieval_status": typed_result.retrieval_status,
            "accepted_typed_facts_found": len(typed_result.accepted_facts),
            "target_fact_types": typed_query.target_fact_types,
            "answer_mode": typed_query.answer_mode,
            "missing_filters": typed_result.missing_filters,
            "diagnostics": typed_result.diagnostics,
        },
        "chunk_diagnostics": chunk_reports,
        "missing_anchor_diagnostics": anchor_diagnostics,
        "summary": _summary(chunk_reports),
    }
    if not args.json_only:
        _print_summary(report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


def _inspect_chunk(chunk: Chunk, pipeline: ExtractionPipeline, typed_query: TypedFactQuery) -> dict[str, Any]:
    bundle = pipeline.extract_from_chunk(chunk)
    candidate_rows = [_candidate_row(candidate, bundle) for candidate in bundle.candidate_facts]
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source": (chunk.metadata or {}).get("source_name") or (chunk.metadata or {}).get("filename") or chunk.doc_id,
        "page": chunk.page_start,
        "section_path": chunk.section_path,
        "matched_terms": _matched_terms(chunk, typed_query),
        "circulation_terms": _circulation_terms_in_text(chunk.text or ""),
        "candidate_facts_count": len(bundle.candidate_facts),
        "accepted_facts_count": len(bundle.accepted_facts),
        "rejected_count": len(bundle.rejected_items),
        "quarantine_count": len(bundle.quarantined_items),
        "accepted_fact_types": _counts(item.fact_type for item in bundle.accepted_facts),
        "candidate_facts": candidate_rows,
        "accepted_facts": [
            {
                "candidate_id": item.candidate_id,
                "fact_type": item.fact_type,
                "score": item.score,
                "validation_reasons": item.validation_reasons,
            }
            for item in bundle.accepted_facts
        ],
        "top_rejection_reasons": (bundle.diagnostics or {}).get("top_rejection_reasons", {}),
        "top_quarantine_reasons": (bundle.diagnostics or {}).get("top_quarantine_reasons", {}),
        "suggested_adapter_or_fact_type": _suggest_adapter(bundle, chunk, typed_query),
        "text_preview": " ".join((chunk.text or "").split())[:500],
    }


def _candidate_row(candidate: CandidateFact, bundle: ExtractionBundle) -> dict[str, Any]:
    accepted_ids = {item.candidate_id for item in bundle.accepted_facts}
    quarantine = {item.candidate.candidate_id: item for item in bundle.quarantined_items}
    status = "accepted" if candidate.candidate_id in accepted_ids else "quarantine" if candidate.candidate_id in quarantine else "candidate_only"
    item = quarantine.get(candidate.candidate_id)
    return {
        "candidate_id": candidate.candidate_id,
        "fact_type": candidate.fact_type,
        "extractor_name": candidate.extractor_name,
        "status": status,
        "reasons": item.reasons if item else [],
        "missing_fields": item.missing_requirements if item else [],
        "subject": candidate.subject,
        "object": candidate.object,
        "value": candidate.value,
        "unit": candidate.unit,
        "confidence": candidate.confidence,
        "evidence_quote": candidate.evidence_quote[:500],
    }


def _matched_terms(chunk: Chunk, query: TypedFactQuery) -> dict[str, list[str]]:
    text = " ".join([chunk.text or "", chunk.section_path or "", str((chunk.metadata or {}).get("source_name") or "")]).lower().replace("ё", "е")
    result: dict[str, list[str]] = {}
    for key, terms in {
        "materials": query.materials,
        "processes": query.processes,
        "properties": query.properties,
        "equipment": query.equipment,
        "geography": query.geography,
    }.items():
        matched = [term for term in terms if str(term or "").lower().replace("ё", "е") in text]
        if matched:
            result[key] = list(dict.fromkeys(matched))
    return result


def _suggest_adapter(bundle: ExtractionBundle, chunk: Chunk, query: TypedFactQuery) -> str:
    if bundle.accepted_facts:
        return "accepted facts already produced"
    text = (chunk.text or "").lower().replace("ё", "е")
    targets = set(query.target_fact_types or [])
    if "TechnologySolutionFact" in targets and any(marker in text for marker in ["циркуляц", "католит", "электролит", "flow", "circulation"]):
        return "catholyte/electrolyte circulation adapter: check media+flow+electrowinning validation"
    if "TechnologySolutionFact" in targets and any(marker in text for marker in ["метод", "способ", "технолог", "система", "решение", "method", "technology", "system", "solution"]):
        return "TechnologySolutionFact adapter: check missing domain marker/evidence in candidate diagnostics"
    if "ProcessParameterFact" in targets and any(marker in text for marker in ["мг/л", "mg/l", "м/с", "m/s", "температур", "скорость", "расход", "concentration"]):
        return "ProcessParameterFact adapter: check value/unit/property schema"
    if "ExperimentResultFact" in targets and any(marker in text for marker in ["эксперимент", "experiment", "опыт", "результат", "reported"]):
        return "ExperimentResultFact adapter: check subject/result fields"
    if not bundle.candidate_facts:
        return "no candidate generated from retrieved chunk"
    return "candidate generated but validation did not accept it"


def _summary(chunk_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "chunks_inspected": len(chunk_reports),
        "candidate_facts_generated": sum(int(item["candidate_facts_count"]) for item in chunk_reports),
        "accepted_facts_generated": sum(int(item["accepted_facts_count"]) for item in chunk_reports),
        "quarantine_candidates": sum(int(item["quarantine_count"]) for item in chunk_reports),
        "rejected_items": sum(int(item["rejected_count"]) for item in chunk_reports),
        "accepted_by_fact_type": _counts(
            fact_type
            for item in chunk_reports
            for fact_type, count in (item.get("accepted_fact_types") or {}).items()
            for _ in range(int(count))
        ),
    }


def _anchor_diagnostics(
    missing_anchors: list[Any],
    chunks: list[Chunk],
    chunk_reports: list[dict[str, Any]],
    query: TypedFactQuery,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for anchor in [str(item) for item in missing_anchors if str(item or "").strip()]:
        terms = _anchor_terms(anchor)
        chunks_with_anchor = []
        candidates_with_anchor = []
        accepted_with_anchor = []
        quarantine_with_anchor = []
        for chunk, chunk_report in zip(chunks, chunk_reports):
            text = _norm(" ".join([chunk.text or "", chunk.section_path or "", str((chunk.metadata or {}).get("source_name") or "")]))
            if any(term in text for term in terms):
                chunks_with_anchor.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk_report.get("source"),
                        "page": chunk_report.get("page"),
                        "terms_found": [term for term in terms if term in text],
                    }
                )
            for candidate in chunk_report.get("candidate_facts") or []:
                candidate_text = _norm(
                    " ".join(
                        [
                            candidate.get("fact_type") or "",
                            json.dumps(candidate.get("subject") or {}, ensure_ascii=False),
                            json.dumps(candidate.get("object") or {}, ensure_ascii=False),
                            candidate.get("evidence_quote") or "",
                        ]
                    )
                )
                if not any(term in candidate_text for term in terms):
                    continue
                row = {
                    "chunk_id": chunk.chunk_id,
                    "candidate_id": candidate.get("candidate_id"),
                    "fact_type": candidate.get("fact_type"),
                    "status": candidate.get("status"),
                    "reasons": candidate.get("reasons") or [],
                    "missing_fields": candidate.get("missing_fields") or [],
                }
                candidates_with_anchor.append(row)
                if candidate.get("status") == "accepted":
                    accepted_with_anchor.append(row)
                if candidate.get("status") == "quarantine":
                    quarantine_with_anchor.append(row)
        reports.append(
            {
                "anchor": anchor,
                "terms": terms,
                "query_planning": "present" if anchor in [*query.materials, *query.processes, *query.properties, *query.equipment] else "not_explicit_constraint",
                "retrieval": "found_in_top_chunks" if chunks_with_anchor else "not_found_in_top_chunks",
                "candidate_extraction": "candidate_generated" if candidates_with_anchor else "no_candidate_with_anchor",
                "validation": "accepted" if accepted_with_anchor else "quarantine" if quarantine_with_anchor else "not_accepted",
                "graph_projection": "needs_sync_or_typed_matching_check" if accepted_with_anchor else "not_reached",
                "typed_matching": "missing_required_anchor",
                "stage_lost": _stage_lost(chunks_with_anchor, candidates_with_anchor, accepted_with_anchor, quarantine_with_anchor),
                "chunks": chunks_with_anchor[:8],
                "candidates": candidates_with_anchor[:12],
            }
        )
    return reports


def _stage_lost(chunks: list[dict[str, Any]], candidates: list[dict[str, Any]], accepted: list[dict[str, Any]], quarantine: list[dict[str, Any]]) -> str:
    if not chunks:
        return "retrieval"
    if not candidates:
        return "candidate_extraction"
    if not accepted:
        return "validation" if quarantine else "validation_or_legacy_candidate_not_direct_fact"
    return "graph_projection_or_typed_matching"


def _anchor_terms(anchor: str) -> list[str]:
    norm = _norm(anchor)
    aliases = {
        "католит": ["католит", "catholyte", "электролит", "electrolyte"],
        "циркуляция католита": ["циркуляция католита", "циркуляц", "католит", "catholyte circulation", "electrolyte circulation", "recirculation"],
        "электролит": ["электролит", "electrolyte", "католит", "catholyte"],
        "скорость потока": ["скорость потока", "flow velocity"],
        "расход": ["расход", "flow rate"],
    }
    terms = [norm]
    for key, values in aliases.items():
        values_norm = [_norm(value) for value in values]
        if norm == _norm(key) or norm in values_norm:
            terms.extend(values_norm)
    return list(dict.fromkeys(term for term in terms if term))


def _circulation_terms_in_text(text: str) -> list[str]:
    norm = _norm(text)
    terms = ["католит", "электролит", "циркуляц", "рециркуляц", "расход", "скорость", "catholyte", "electrolyte", "flow", "circulation"]
    return [term for term in terms if term in norm]


def _norm(value: Any) -> str:
    return str(value or "").lower().replace("ё", "е")


def _print_summary(report: dict[str, Any]) -> None:
    retrieval = report["retrieval"]
    typed = report["typed_fact_search"]
    summary = report["summary"]
    print(f"Question: {report['question']}")
    print(f"Mode: {typed['answer_mode']} | targets: {', '.join(typed['target_fact_types']) or 'none'}")
    print(
        "Retrieval: "
        f"{retrieval.get('effective_retrieval_mode')} "
        f"bm25={retrieval.get('chunks_found_bm25')} "
        f"dense={retrieval.get('chunks_found_dense')} "
        f"fused={retrieval.get('chunks_after_fusion')}"
    )
    print(
        "Extraction: "
        f"chunks={summary['chunks_inspected']} "
        f"candidates={summary['candidate_facts_generated']} "
        f"accepted={summary['accepted_facts_generated']} "
        f"quarantine={summary['quarantine_candidates']} "
        f"rejected={summary['rejected_items']}"
    )
    for index, item in enumerate(report["chunk_diagnostics"], start=1):
        print(
            f"{index}. {item['source']} p.{item['page']} "
            f"candidates={item['candidate_facts_count']} accepted={item['accepted_facts_count']} "
            f"quarantine={item['quarantine_count']} suggestion={item['suggested_adapter_or_fact_type']}"
        )
    for item in report.get("missing_anchor_diagnostics") or []:
        print(
            f"Anchor {item['anchor']}: retrieval={item['retrieval']} "
            f"candidate_extraction={item['candidate_extraction']} validation={item['validation']} "
            f"stage_lost={item['stage_lost']}"
        )


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value or "")
        if not key:
            continue
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items(), key=lambda item: (-item[1], item[0])))


def _dump_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        try:
            return model.model_dump(mode="json")
        except TypeError:
            return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


if __name__ == "__main__":
    raise SystemExit(main())

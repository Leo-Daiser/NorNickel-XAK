"""FastAPI application for ingestion, retrieval and graph browsing."""

from __future__ import annotations

import hashlib
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi import Query
from pydantic import BaseModel, Field
import requests

from .analytics.answer_synthesizer import AnswerSynthesizer
from .analytics.diagnostics import analytical_diagnostics
from .analytics.evidence_reranker import EvidenceReranker
from .analytics.evidence_search import EvidenceSearch
from .analytics.graph_context import GraphContextBuilder
from .analytics.query_models import AnalyticalIntent, AnalyticalQueryPlan
from .analytics.router import AnalyticalQueryRouter
from .config import settings
from .answering.answer_builder import AnswerBuilder
from .answering.human_answer import enhance_answer_payload
from .answering.typed_answer import build_typed_answer_payload
from .demo.scenarios import get_demo_scenario, list_demo_scenarios
from .domain.query_constraints import QueryIntent
from .extraction.extraction import EntityRelationExtractor, canonical_material_name, is_false_material, normalise, stable_entity_uid
from .extraction.extraction import MATERIAL_PATTERNS, PROPERTY_TERMS, PROCESS_TERMS
from .extraction.pipeline import ExtractionPipeline
from .graph.graph_db import GraphDB
from .graph.neo4j_connection import check_neo4j_connection
from .graph.graph_repository import GraphRepositoryFactory, repository_backend_name
from .graph.graph_writer import sync_catalog_to_neo4j
from .ingestion.parser_router import ParserRouter
from .knowledge.expansion import KnowledgeExpansionEngine, build_knowledge_expansion_report
from .llm.structured_llm import StructuredLLM
from .models.schemas import Chunk, Document
from .parsing.source_metadata import infer_source_metadata
from .retrieval.graph_retriever import GraphRetriever
from .retrieval.metadata_filters import rerank_chunks_by_source_metadata
from .retrieval.query_planner import QueryPlanner
from .retrieval.retrieval import RetrievalEngine
from .retrieval.typed_fact_retriever import TypedFactQuery, TypedFactRetriever
from .runtime.profiles import runtime_profile_summary
from .runtime.presets import RuntimePresetId, get_runtime_preset, list_runtime_presets, preset_diagnostics
from .security.url_safety import UnsafeUrlError, fetch_url_safely
from .storage.catalog import SQLiteCatalog
from .storage.outbox import SQLiteOutbox


app = FastAPI(title="Scientific Knowledge Extraction API", version="0.4.0")


class AskRequest(BaseModel):
    """JSON body for /ask. Query params remain supported for compatibility."""

    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)
    preset_id: RuntimePresetId | None = None


class RuntimePresetRequest(BaseModel):
    """Request body for runtime preset validation/checks."""

    preset_id: RuntimePresetId = RuntimePresetId.EXPERT_MAX


class DocumentActiveRequest(BaseModel):
    """Request body for toggling document participation in QA/retrieval."""

    active: bool = True


class DocumentsActiveBatchRequest(BaseModel):
    """Batch document active flag update from the UI data editor."""

    changes: Dict[str, bool] = Field(default_factory=dict)
    sync_neo4j: bool = False


DOCUMENTS: Dict[str, Document] = {}
CHUNKS: Dict[str, List[Chunk]] = {}

parser_router = ParserRouter()
retrieval_engine = RetrievalEngine()
extractor = EntityRelationExtractor()
answer_extraction_pipeline = ExtractionPipeline(audit_enabled=False)
llm_client = StructuredLLM()
query_planner = QueryPlanner()
answer_builder = AnswerBuilder()
analytical_router = AnalyticalQueryRouter()
graph_context_builder = GraphContextBuilder()
evidence_reranker = EvidenceReranker()
answer_synthesizer = AnswerSynthesizer()
outbox = SQLiteOutbox(settings.metadata_db_path)
catalog = SQLiteCatalog(settings.catalog_db_path)

graph_db: GraphDB | None = None
graph_db_error: str | None = None
graph_db_last_failure_at: float = 0.0
NEO4J_RETRY_TTL_SECONDS = 5.0


def init_graph_db() -> GraphDB | None:
    global graph_db_error, graph_db_last_failure_at
    try:
        status = check_neo4j_connection(
            getattr(settings, "neo4j_uri", ""),
            getattr(settings, "neo4j_user", ""),
            getattr(settings, "neo4j_password", ""),
            getattr(settings, "neo4j_database", "neo4j"),
        )
        if not status.available:
            graph_db_error = status.error
            graph_db_last_failure_at = time.monotonic()
            return None
        db = GraphDB()
        db.create_constraints()
        graph_db_error = None
        graph_db_last_failure_at = 0.0
        return db
    except Exception as exc:
        graph_db_error = str(exc)
        graph_db_last_failure_at = time.monotonic()
        return None


def get_graph_db(*, force_retry: bool = False) -> GraphDB | None:
    """Return active GraphDB, retrying lazy connection once if needed."""
    global graph_db
    if graph_db is None:
        if not force_retry and graph_db_last_failure_at:
            elapsed = time.monotonic() - graph_db_last_failure_at
            if elapsed < NEO4J_RETRY_TTL_SECONDS:
                return None
        graph_db = init_graph_db()
    return graph_db

def _configured_kg_backend(override: str | None = None) -> str:
    mode = str(override or getattr(settings, "kg_backend", "auto") or "auto").lower()
    return mode if mode in {"auto", "neo4j", "fallback"} else "auto"


graph_db = None if _configured_kg_backend() == "fallback" else init_graph_db()


def _graph_db_for_repository(configured_backend: str | None = None, *, force_retry: bool = False) -> GraphDB | None:
    mode = _configured_kg_backend(configured_backend)
    if mode == "fallback":
        return None
    if mode == "neo4j":
        return get_graph_db(force_retry=force_retry)
    return get_graph_db(force_retry=force_retry)


def _kg_backend_diagnostics(
    active_graph: GraphDB | None = None,
    active_backend: str | None = None,
    configured_backend: str | None = None,
) -> Dict[str, Any]:
    configured = _configured_kg_backend(configured_backend)
    neo4j_available = active_graph is not None
    neo4j_error = "" if neo4j_available else (graph_db_error or "")
    if active_graph is not None and hasattr(active_graph, "run"):
        try:
            active_graph.run("RETURN 1 AS ok")
            neo4j_available = True
            neo4j_error = ""
        except Exception as exc:
            neo4j_available = False
            neo4j_error = str(exc)
    if active_backend is None:
        if configured == "fallback":
            active_backend = "fallback"
        elif configured == "neo4j":
            active_backend = "neo4j"
        else:
            active_backend = "neo4j" if neo4j_available else "fallback"
    if configured == "fallback":
        decision_reason = "KG_BACKEND=fallback requested validated fallback graph"
    elif neo4j_available:
        decision_reason = "Neo4j connection check succeeded with RETURN 1"
    else:
        decision_reason = f"Neo4j connection check failed: {neo4j_error}" if neo4j_error else "Neo4j unavailable"
    return {
        "kg_backend_configured": configured,
        "kg_backend_active": active_backend,
        "neo4j_available": neo4j_available,
        "neo4j_uri": getattr(settings, "neo4j_uri", ""),
        "neo4j_user": getattr(settings, "neo4j_user", ""),
        "neo4j_password_configured": bool(getattr(settings, "neo4j_password", "")),
        "neo4j_error": neo4j_error,
        "kg_backend_decision": {
            "configured": configured,
            "selected": active_backend,
            "reason": decision_reason,
        },
    }


def _create_graph_repository_or_503():
    """Create active graph repository for API handlers."""
    repository_graph = _graph_db_for_repository()
    try:
        repository = GraphRepositoryFactory.create(
            catalog=catalog,
            extractor=extractor,
            graph_db=repository_graph,
            document_getter=_get_document_meta,
            configured_backend=_configured_kg_backend(),
        )
        return repository, repository_graph
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={**_kg_backend_diagnostics(repository_graph), **_extraction_diagnostics(), "error": str(exc)},
        ) from exc


def _extraction_diagnostics() -> Dict[str, Any]:
    """Return structured extraction configuration for health/API diagnostics."""
    llm_status = llm_client.status()
    return {
        "extraction_mode": getattr(settings, "extraction_mode", "deterministic"),
        "extraction_min_confidence": getattr(settings, "extraction_min_confidence", 0.55),
        "extraction_on_ingest": getattr(settings, "extraction_on_ingest", False),
        "llm_extraction_available": bool(
            getattr(settings, "extraction_enable_llm", False)
            and llm_status.get("enabled")
            and not llm_status.get("last_error")
        ),
        "audit_enabled": bool(getattr(settings, "extraction_audit_dir", None)),
    }


def _parser_diagnostics() -> Dict[str, Any]:
    """Return parser backend and OCR configuration."""
    return {
        "parser_backend": getattr(settings, "parser_backend", "auto"),
        "docling_available": getattr(parser_router, "_docling_converter", None) is not None,
        "markitdown_available": getattr(parser_router, "_markdown_converter", None) is not None,
        "ocr_enabled": getattr(settings, "enable_ocr", False),
        "ocr_backend": getattr(settings, "ocr_backend", "none"),
        "parser_audit_enabled": bool(getattr(settings, "parser_audit_dir", None)),
    }


def _answer_synthesis_diagnostics() -> Dict[str, Any]:
    return {
        "answer_synthesis_mode": getattr(settings, "answer_synthesis_mode", "template"),
    }


def _llm_status_with_effective_runtime(raw_status: Dict[str, Any]) -> Dict[str, Any]:
    """Keep configured provider diagnostics and add profile-enforced LLM use."""

    effective_enabled = bool(getattr(settings, "enable_llm", False))
    effective_ready = bool(effective_enabled and raw_status.get("ready"))
    provider = getattr(settings, "llm_provider", raw_status.get("provider") or "offline")
    return {
        **raw_status,
        "effective_enabled": effective_enabled,
        "effective_ready": effective_ready,
        "effective_provider": provider,
        "effective_provider_active": provider if effective_enabled else "offline",
        "provider_available": bool(raw_status.get("ready")),
    }


def _sync_strict_graph_to_neo4j(active_graph: GraphDB | None) -> Dict[str, Any]:
    if active_graph is None or _configured_kg_backend() == "fallback":
        return {"status": "skipped", "reason": "neo4j_unavailable_or_fallback"}
    try:
        stats = sync_catalog_to_neo4j(
            graph_db=active_graph,
            catalog=catalog,
            extractor=extractor,
            document_getter=_get_document_meta,
        )
        return {"status": "synced", **stats}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _bootstrap_from_catalog() -> None:
    """Load persisted local catalog into in-memory structures and BM25."""
    global retrieval_engine
    docs = catalog.list_documents()
    all_chunks: List[Chunk] = []
    for doc in docs:
        DOCUMENTS[doc.doc_id] = doc
        chunks = catalog.list_chunks(doc.doc_id)
        CHUNKS[doc.doc_id] = chunks
        all_chunks.extend(chunks)
    if all_chunks:
        retrieval_engine.index_chunks(all_chunks)


_bootstrap_from_catalog()


def _rebuild_runtime_indexes_from_active_catalog() -> Dict[str, int]:
    """Rebuild in-memory document/chunk maps and retrieval index from active catalog rows."""

    global retrieval_engine
    DOCUMENTS.clear()
    CHUNKS.clear()
    retrieval_engine = RetrievalEngine()
    active_chunks: List[Chunk] = []
    active_docs = 0
    for doc in catalog.list_documents():
        DOCUMENTS[doc.doc_id] = doc
        chunks = catalog.list_chunks(doc.doc_id)
        CHUNKS[doc.doc_id] = chunks
        if chunks or catalog.is_document_active(doc.doc_id):
            active_docs += 1 if catalog.is_document_active(doc.doc_id) else 0
        active_chunks.extend(chunks)
    if active_chunks:
        retrieval_engine.index_chunks(active_chunks)
    return {"active_documents": active_docs, "active_chunks": len(active_chunks)}


def _reset_runtime_corpus(*, clear_neo4j: bool = True) -> Dict[str, Any]:
    """Clear local catalog, in-memory indexes and optionally Neo4j projection."""

    global retrieval_engine
    before_catalog = catalog.clear()
    DOCUMENTS.clear()
    CHUNKS.clear()
    retrieval_engine = RetrievalEngine()
    graph_result: Dict[str, Any] = {"status": "skipped", "reason": "clear_neo4j_false"}
    if clear_neo4j:
        active_graph = _graph_db_for_repository(force_retry=True)
        if active_graph is None:
            graph_result = {"status": "skipped", "reason": "neo4j_unavailable"}
        else:
            try:
                before_graph = active_graph.clear_all()
                graph_result = {"status": "cleared", "before": before_graph}
            except Exception as exc:
                graph_result = {"status": "error", "error": str(exc)}
    return {
        "catalog_before": before_catalog,
        "catalog_after": catalog.counts(),
        "neo4j": graph_result,
        "runtime": {"documents": len(DOCUMENTS), "chunks": sum(len(items) for items in CHUNKS.values())},
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_doc_id(content_hash: str) -> str:
    return f"doc_{content_hash[:24]}"


def _stable_source_id(content_hash: str) -> str:
    return f"source_{content_hash[:24]}"


def _document_version_for(source_name: str, content_hash: str) -> int:
    """Return stable version for same content and next version for changed file content."""

    versions: list[int] = []
    for item in catalog.list_document_records():
        metadata = item.get("metadata") or {}
        title = item.get("filename") or item.get("title")
        if title != source_name:
            continue
        version = int(item.get("version") or metadata.get("document_version") or 1)
        if metadata.get("content_hash") == content_hash:
            return version
        versions.append(version)
    return (max(versions) + 1) if versions else 1


def _knowledge_report(active_only: bool = True, document_id: str | None = None) -> Dict[str, Any]:
    return build_knowledge_expansion_report(catalog, document_id=document_id, active_only=active_only)


def _allowed_upload_extensions() -> set[str]:
    raw = str(getattr(settings, "allowed_upload_extensions", ".pdf,.docx,.pptx,.xlsx,.csv,.html,.htm,.txt,.md"))
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _validate_upload_name(filename: str) -> str:
    safe_name = Path(filename or "uploaded.bin").name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _allowed_upload_extensions():
        allowed = ", ".join(sorted(_allowed_upload_extensions()))
        raise HTTPException(status_code=400, detail=f"Unsupported file extension {suffix or '<none>'}; allowed: {allowed}")
    return safe_name


def _validate_upload_batch(files: List[UploadFile]) -> None:
    max_files = int(getattr(settings, "max_upload_files", 20))
    if len(files) > max_files:
        raise HTTPException(status_code=400, detail=f"Too many files: {len(files)} > {max_files}")


def _validate_upload_size(content: bytes, filename: str) -> None:
    max_bytes = int(getattr(settings, "max_upload_mb", 25)) * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File {filename} exceeds MAX_UPLOAD_MB={getattr(settings, 'max_upload_mb', 25)}")


def _node_id(entity_type: str, name: str, workspace_uid: str = "default-workspace") -> str:
    return stable_entity_uid(workspace_uid, entity_type, name)


def _edge_id(source: str, predicate: str, target: str) -> str:
    return hashlib.sha256(f"{source}|{predicate}|{target}".encode("utf-8")).hexdigest()[:24]




def _get_document_meta(doc_id: str):
    doc_meta = DOCUMENTS.get(doc_id)
    if doc_meta is not None:
        return doc_meta
    try:
        return catalog.get_document(doc_id)
    except Exception:
        return None

def _source_for_chunk(chunk: Chunk) -> Dict[str, Any]:
    doc_meta = _get_document_meta(chunk.doc_id)
    source_name = chunk.metadata.get("source_name") or chunk.metadata.get("filename") or (doc_meta.title if doc_meta else chunk.doc_id)
    return {
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "title": source_name,
        "filename": chunk.metadata.get("filename") or (doc_meta.title if doc_meta else None),
        "source_name": source_name,
        "source_type": chunk.metadata.get("source_type", "file"),
        "source_url": chunk.metadata.get("source_url"),
        "source_metadata": chunk.metadata.get("source_metadata") or {},
        "publication_year": chunk.metadata.get("publication_year"),
        "geographies": chunk.metadata.get("geographies") or [],
        "practice_scope": chunk.metadata.get("practice_scope"),
        "reliability_level": chunk.metadata.get("reliability_level"),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_path": chunk.section_path,
        "table_id": chunk.metadata.get("table_id"),
        "row_id": chunk.metadata.get("row_id"),
        "image_refs": chunk.metadata.get("image_refs", []),
        "quote": chunk.text[:700] + ("..." if len(chunk.text) > 700 else ""),
    }


def _parsed_metadata(parsed: Any, source_url: str | None = None, source_name: str | None = None) -> Dict[str, Any]:
    """Build backward-compatible metadata from DocumentIntelligenceResult."""
    diagnostics = dict(getattr(parsed, "diagnostics", {}) or {})
    images = getattr(parsed, "images", []) or []
    source_profile = infer_source_metadata(
        source_name=source_name or getattr(parsed, "source_name", None),
        source_url=source_url,
        source_type=getattr(parsed, "source_type", diagnostics.get("source_type", "file")),
        parser_name=getattr(parsed, "parser_name", diagnostics.get("parser_name", "unknown")),
        text=getattr(parsed, "text", None) or "\n".join(chunk.text for chunk in (getattr(parsed, "chunks", []) or [])),
        diagnostics=diagnostics,
    )
    return {
        **diagnostics,
        **source_profile,
        "parser": getattr(parsed, "parser_name", diagnostics.get("parser", "unknown")),
        "parser_name": getattr(parsed, "parser_name", diagnostics.get("parser_name", "unknown")),
        "parser_version": getattr(parsed, "parser_version", diagnostics.get("parser_version")),
        "source_type": getattr(parsed, "source_type", diagnostics.get("source_type", "file")),
        "source_url": source_url,
        "blocks_count": len(getattr(parsed, "blocks", []) or []),
        "tables_count": len(getattr(parsed, "tables", []) or []),
        "images_count": len(images),
        "chunks_count": len(getattr(parsed, "chunks", []) or []),
        "image_refs": [
            {
                "url": image.source_path_or_url,
                "alt": image.alt_text,
                "caption": image.caption,
                "section_path": image.section_path,
            }
            for image in images
        ],
        "document_intelligence": {
            "blocks_count": len(getattr(parsed, "blocks", []) or []),
            "tables_count": len(getattr(parsed, "tables", []) or []),
            "images_count": len(images),
            "parser_diagnostics": diagnostics,
        },
    }


def _document_parse_status(parsed: Any) -> str:
    diagnostics = getattr(parsed, "diagnostics", {}) or {}
    if diagnostics.get("scanned_pdf_detected") and not getattr(settings, "enable_ocr", False):
        return "ocr_required"
    if getattr(parsed, "chunks", None):
        return "ingested"
    if getattr(parsed, "text", ""):
        return "partial"
    return "empty_or_parse_failed"


def _qdrant_outbox_enabled() -> bool:
    """Whether new chunks should be queued for optional Qdrant projection."""
    return bool(settings.direct_qdrant_projection)


def _enqueue_qdrant_projection(chunks: List[Chunk]) -> int:
    if not _qdrant_outbox_enabled():
        return 0
    queued = 0
    for chunk in chunks:
        outbox.enqueue(
            aggregate_type="Chunk",
            aggregate_uid=chunk.chunk_id,
            op="upsert",
            version=1,
            payload={
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "workspace_uid": chunk.workspace_uid,
                "text_hash": chunk.text_hash,
                "embedding_version": chunk.embedding_version,
            },
            dedupe_key=f"qdrant:{chunk.chunk_id}:{chunk.text_hash}",
        )
        queued += 1
    return queued


def _project_document_to_graph(active_graph: GraphDB | None, doc_meta: Document, chunks: List[Chunk], source_uri: str | None = None) -> Dict[str, int]:
    """Project document, chunks, extracted entities and fact relations to Neo4j.

    The local JSON subgraph remains the fallback for `/ask`; this projection is
    for the optional Neo4j layer and diagnostics. It is intentionally best
    effort at the call sites so an unavailable graph never breaks fallback.
    """
    if active_graph is None:
        return {"documents": 0, "chunks": 0, "entities": 0, "relations": 0}

    workspace_uid = doc_meta.workspace_uid or "default-workspace"
    source_uid = doc_meta.source_uid or _stable_source_id(doc_meta.external_id or doc_meta.doc_id)
    active_graph.upsert_workspace(uid=workspace_uid, slug="default", name="Default Workspace")
    active_graph.upsert_source(
        uid=source_uid,
        type_="url" if source_uri and source_uri.startswith(("http://", "https://")) else "file",
        uri=source_uri or doc_meta.title,
        checksum=doc_meta.external_id,
        imported_at=None,
    )
    active_graph.link_workspace_source(workspace_uid=workspace_uid, source_uid=source_uid)
    active_graph.upsert_document(
        doc_id=doc_meta.doc_id,
        workspace_uid=workspace_uid,
        title=doc_meta.title,
        source_uid=source_uid,
        external_id=doc_meta.external_id,
        parser=doc_meta.parser,
        language=doc_meta.language,
        status=doc_meta.status,
        created_at=doc_meta.created_at,
        updated_at=doc_meta.updated_at,
        version=doc_meta.version,
    )
    active_graph.link_document_source(document_uid=doc_meta.doc_id, source_uid=source_uid)

    entity_type_by_name: Dict[str, str] = {}
    projected_entities = 0
    projected_relations = 0
    for chunk in chunks:
        active_graph.upsert_chunk_node(
            {
                "uid": chunk.chunk_id,
                "document_uid": doc_meta.doc_id,
                "workspace_uid": workspace_uid,
                "ordinal": chunk.ordinal,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_path": chunk.section_path,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_count": chunk.token_count,
                "text_hash": chunk.text_hash,
                "preview": chunk.preview or chunk.text[:200],
                "embedding_version": chunk.embedding_version,
                "updated_at": chunk.updated_at,
            }
        )
        extraction = answer_extraction_pipeline.extract_from_chunk(chunk)
        for entity in extraction.entities:
            uid = stable_entity_uid(workspace_uid, entity.entity_type, entity.canonical_name)
            entity_type_by_name[entity.canonical_name] = entity.entity_type
            entity_type_by_name[normalise(entity.canonical_name)] = entity.entity_type
            norm_name = normalise(entity.canonical_name)
            active_graph.upsert_entity(
                entity_id=uid,
                label=entity.canonical_name,
                properties={"type": entity.entity_type, "norm_name": norm_name},
            )
            active_graph.link_chunk_entity(
                chunk_uid=chunk.chunk_id,
                entity_uid=uid,
                canonical_name=entity.canonical_name,
                entity_type=entity.entity_type,
                norm_name=norm_name,
                confidence=0.75,
                count=1,
            )
            projected_entities += 1

        for relation in _accepted_relation_facts_from_bundle(chunk, extraction):
            subject = str(relation.get("subject") or "")
            obj = str(relation.get("object") or "")
            predicate = str(relation.get("predicate") or "")
            subject_type = entity_type_by_name.get(subject) or entity_type_by_name.get(normalise(subject))
            object_type = entity_type_by_name.get(obj) or entity_type_by_name.get(normalise(obj))
            if not subject_type or not object_type:
                continue
            subject_uid = stable_entity_uid(workspace_uid, subject_type, subject)
            object_uid = stable_entity_uid(workspace_uid, object_type, obj)
            active_graph.upsert_relation(
                subject_id=subject_uid,
                predicate=predicate,
                object_id=object_uid,
                qualifiers=relation.get("qualifiers") or {},
                confidence=float(relation.get("confidence") or 0.0),
                evidence_chunk_ids=[str(relation.get("source_chunk_id") or chunk.chunk_id)],
            )
            projected_relations += 1

    active_graph.link_chunk_sequence(doc_meta.doc_id)
    return {"documents": 1, "chunks": len(chunks), "entities": projected_entities, "relations": projected_relations}


def _relations_by_subject(extraction, predicate: str) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for rel in extraction.relations:
        if rel.predicate == predicate:
            grouped.setdefault(rel.subject, []).append(rel)
    return grouped


def _first_relation_object(relations: Dict[str, List[Any]], subject: str) -> str | None:
    values = relations.get(subject) or []
    return values[0].object if values else None


def _unique(values: List[Any]) -> List[Any]:
    return list(dict.fromkeys(value for value in values if value))


def _unique_norm(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if not value:
            continue
        key = normalise(str(value))
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


LEGACY_RELATION_INTENTS = {
    "object_overview",
    "parameter_lookup",
    "part_article_lookup",
    "standard_lookup",
    "requirement_lookup",
    "image_lookup",
}


def _extract_answer_bundle(chunk: Chunk, *, intent: str | None = None) -> Any:
    """Return extraction for answer fallback.

    Scientific/material answers must use the validated pipeline. Legacy
    equipment/object lookup still depends on the older relation model, so it
    is kept behind explicit technical intents only.
    """

    if intent in LEGACY_RELATION_INTENTS:
        return extractor.extract_from_chunk(chunk)
    return answer_extraction_pipeline.extract_from_chunk(chunk)


def _build_local_subgraph(extractions: List[tuple[Chunk, Any]], limit: int = 180) -> Dict[str, List[Dict[str, Any]]]:
    """Build a typed, evidence-aware local knowledge graph.

    This is the core fallback graph used when Neo4j is absent.  It is not a
    visualization shortcut: every fact edge is grounded in a SourceChunk and
    entity nodes are typed according to the extractor ontology.
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Dict[str, Any]] = {}
    entity_type_by_name: Dict[str, str] = {}

    for _, extraction in extractions:
        for entity in extraction.entities:
            entity_type_by_name[entity.canonical_name] = entity.entity_type
            entity_type_by_name[normalise(entity.canonical_name)] = entity.entity_type

    predicate_type_hints = {
        "OBJECT_HAS_PARAMETER": ("TechnicalObject", "Parameter"),
        "OBJECT_HAS_PART": ("TechnicalObject", "Part"),
        "PART_HAS_ARTICLE_NUMBER": ("Part", "ArticleNumber"),
        "OBJECT_MADE_OF_MATERIAL": ("TechnicalObject", "Material"),
        "OBJECT_COMPLIES_WITH_STANDARD": ("TechnicalObject", "Standard"),
        "REQUIREMENT_APPLIES_TO_OBJECT": ("Requirement", "TechnicalObject"),
        "IMAGE_LINKED_TO_SECTION": ("ImageArtifact", "Section"),
        "STUDIES": ("Experiment", "Material"),
        "USES_REGIME": ("Experiment", "ProcessRegime"),
        "USES_EQUIPMENT": ("Experiment", "Equipment"),
        "PERFORMED_BY": ("Experiment", "Laboratory"),
        "MEASURES": ("Experiment", "PropertyValue"),
        "OF_PROPERTY": ("PropertyValue", "Property"),
        "HAS_CHANGE": ("PropertyValue", "PropertyChange"),
        "HAS_MEASUREMENT": ("Material", "PropertyValue"),
        "MISSING_FOR": ("DataGap", "Entity"),
        "SUPPORTED_BY": ("Conclusion", "SourceChunk"),
    }

    def add_node(node_id: str, label: str, type_: str, **props: Any) -> None:
        if len(nodes) >= limit and node_id not in nodes:
            return
        if node_id in nodes:
            nodes[node_id].setdefault("properties", {}).update({k: v for k, v in props.items() if v is not None})
            return
        nodes[node_id] = {
            "id": node_id,
            "label": label,
            "type": type_,
            "properties": {k: v for k, v in props.items() if v is not None},
        }

    def add_edge(source_id: str, predicate: str, target_id: str, **props: Any) -> None:
        if source_id not in nodes or target_id not in nodes:
            return
        eid = _edge_id(source_id, predicate, target_id)
        if eid in edges:
            edges[eid].setdefault("properties", {}).update({k: v for k, v in props.items() if v is not None})
            return
        edges[eid] = {
            "id": eid,
            "source": source_id,
            "target": target_id,
            "label": predicate,
            "properties": {k: v for k, v in props.items() if v is not None},
        }

    def infer_type(name: str, fallback: str = "Entity") -> str:
        return entity_type_by_name.get(name) or entity_type_by_name.get(normalise(name)) or fallback

    def entity_node_id(entity_type: str, name: str, workspace_uid: str) -> str:
        return _node_id(entity_type, name, workspace_uid)

    for chunk, extraction in extractions:
        workspace_uid = chunk.workspace_uid or "default-workspace"
        doc_meta = _get_document_meta(chunk.doc_id)
        doc_node_id = f"doc:{chunk.doc_id}"
        section_node_id = f"section:{chunk.doc_id}:{chunk.section_path or '/'}"
        source_node_id = f"chunk:{chunk.chunk_id}"

        add_node(doc_node_id, doc_meta.title if doc_meta else chunk.doc_id, "Document", doc_id=chunk.doc_id)
        add_node(
            section_node_id,
            chunk.section_path or "/",
            "Section",
            doc_id=chunk.doc_id,
            section_path=chunk.section_path,
        )
        add_node(
            source_node_id,
            f"Chunk {chunk.ordinal if chunk.ordinal is not None else ''}".strip(),
            "SourceChunk",
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            section_path=chunk.section_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            table_id=chunk.metadata.get("table_id"),
            row_id=chunk.metadata.get("row_id"),
            quote=chunk.text[:600],
        )
        add_edge(doc_node_id, "DOCUMENT_HAS_SECTION", section_node_id)
        add_edge(section_node_id, "SECTION_HAS_CHUNK", source_node_id)

        entity_lookup: Dict[str, str] = {}
        for entity in extraction.entities:
            eid = entity_node_id(entity.entity_type, entity.canonical_name, workspace_uid)
            entity_lookup[entity.canonical_name] = eid
            add_node(
                eid,
                entity.canonical_name,
                entity.entity_type,
                norm_name=getattr(entity, "norm_name", None),
                source_chunk_id=chunk.chunk_id,
            )
            add_edge(source_node_id, "CHUNK_MENTIONS_ENTITY", eid, source_chunk_id=chunk.chunk_id)

        for rel in getattr(extraction, "relations", []):
            src_hint, dst_hint = predicate_type_hints.get(rel.predicate, ("Entity", "Entity"))
            source_type = infer_type(rel.subject, src_hint)
            target_type = infer_type(rel.object, dst_hint)

            if rel.predicate in {"SUPPORTED_BY"} and rel.object == chunk.chunk_id:
                source_id = entity_lookup.get(rel.subject) or entity_node_id(source_type, rel.subject, workspace_uid)
                target_id = source_node_id
            elif rel.predicate == "IMAGE_LINKED_TO_SECTION":
                source_id = entity_lookup.get(rel.subject) or entity_node_id("ImageArtifact", rel.subject, workspace_uid)
                target_id = section_node_id
            elif rel.subject == chunk.chunk_id:
                source_id = source_node_id
                target_id = entity_lookup.get(rel.object) or entity_node_id(target_type, rel.object, workspace_uid)
            else:
                source_id = entity_lookup.get(rel.subject) or entity_node_id(source_type, rel.subject, workspace_uid)
                target_id = entity_lookup.get(rel.object) or entity_node_id(target_type, rel.object, workspace_uid)

            add_node(source_id, rel.subject if not source_id.startswith("chunk:") else "Source chunk", source_type)
            if target_id == section_node_id:
                add_node(target_id, chunk.section_path or "/", "Section", doc_id=chunk.doc_id, section_path=chunk.section_path)
            elif target_id == source_node_id:
                add_node(target_id, "Source chunk", "SourceChunk", doc_id=chunk.doc_id, chunk_id=chunk.chunk_id)
            else:
                add_node(target_id, rel.object, target_type)

            add_edge(
                source_id,
                rel.predicate,
                target_id,
                confidence=rel.confidence,
                qualifiers=rel.qualifiers,
                source_chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
            )
            if source_id != source_node_id and target_id != source_node_id:
                add_edge(source_id, "FACT_SUPPORTED_BY_CHUNK", source_node_id, source_chunk_id=chunk.chunk_id)

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _query_terms(question: str) -> Dict[str, List[str]]:
    q_lower = (question or "").lower()
    materials: List[str] = []
    for pattern in MATERIAL_PATTERNS:
        for match in pattern.finditer(question or ""):
            if is_false_material(match.group(0)):
                continue
            name = canonical_material_name(match.group(0))
            if name and name not in materials:
                materials.append(name)
    properties: List[str] = []
    for term, canonical in PROPERTY_TERMS.items():
        if term in q_lower and canonical not in properties:
            properties.append(canonical)
    processes: List[str] = []
    for term, canonical in PROCESS_TERMS.items():
        if term in q_lower and canonical not in processes:
            processes.append(canonical)
    return {"materials": materials, "properties": properties, "processes": processes}


DOMAIN_QUESTION_MARKERS = [
    "сплав", "материал", "сталь", "алюмин", "титан", "вт6", "7075", "12х18н10т", "09г2с",
    "клапан", "насос", "dn", "pn", "артикул", "деталь", "корпус", "узел",
    "прочность", "твёрд", "тверд", "пластич", "корроз", "давлен", "температур",
    "отжиг", "закал", "старен", "режим", "эксперимент", "измер", "свойств",
    "параметр", "числен", "источник", "вывод", "подтвержд", "гидрометаллург",
    "противореч", "неоднород",
    "пробел", "не хватает", "нет данных",
    "оборуд", "установк", "лаборатор", "команд", "стандарт", "гост", "iso", "astm",
    "изображ", "схем", "монтаж", "чертеж", "рисунок",
    "рудник", "шахт", "подземн", "глубок", "сверхглубок", "охлажден", "охлажд",
    "вентиляц", "теплов", "источники тепла", "источник тепла", "компрессион",
    "абсорбцион", "эжектор", "холодильн", "хладагент", "холодный забой", "лед",
    "bac", "mwr",
    "valve", "pump", "alloy", "steel", "material", "strength", "hardness", "corrosion",
    "anneal", "quench", "aging", "aged", "experiment", "property", "parameter", "source", "evidence",
    "standard", "gap", "image", "diagram", "mine", "underground", "cooling",
    "refrigeration", "ventilation", "heat source", "heat load", "chiller", "compressor",
]


def _question_understanding(question: str) -> Dict[str, Any]:
    text = str(question or "").strip()
    q = normalise(text)
    tokens = re.findall(r"[A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9\-]*", q)
    meaningful_tokens = [token for token in tokens if len(token) >= 2]
    terms = _query_terms(text)
    constraints = _question_constraints(text)
    focus = _focus_terms(text)
    planned = query_planner.parse(text)
    planner_known = bool(
        planned.materials
        or planned.regimes
        or planned.properties
        or planned.equipment
        or planned.topic_tags
        or planned.numeric_constraints
        or planned.geographies
        or planned.time_filters
    )
    has_domain_marker = any(marker in q for marker in DOMAIN_QUESTION_MARKERS)
    has_identifier = bool(focus or re.search(r"\b(?:dn|pn)\s*\d+\b|\b\d{2,4}[a-zа-я0-9\-]*\b", q, re.IGNORECASE))
    known_terms = bool(any(terms.values()) or any(constraints.values()) or planner_known)
    repeated_noise = bool(re.search(r"([a-zа-я])\1{2,}", q, re.IGNORECASE)) and not has_domain_marker
    too_short = len(meaningful_tokens) < 2 and not has_identifier
    unsupported = not (has_domain_marker or has_identifier or known_terms)
    needs_clarification = too_short or unsupported or repeated_noise
    return {
        "tokens": meaningful_tokens,
        "has_domain_marker": has_domain_marker,
        "has_identifier": has_identifier,
        "known_terms": known_terms,
        "repeated_noise": repeated_noise,
        "needs_clarification": needs_clarification,
        "reason": "no_domain_terms" if unsupported else "too_short" if too_short else "noise" if repeated_noise else "ok",
    }


def _clarification_response(question: str, reason: str, retrieval_meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    examples = [
        "Что делали с ВТ6 при отжиге и как изменилась прочность?",
        "Какие параметры указаны для клапана DN50 PN16?",
        "Какие артикулы относятся к насосу NPK-200?",
        "Где не хватает данных по коррозионной стойкости 7075-T6?",
    ]
    answer = (
        "Я не могу корректно ответить на этот запрос: в нём нет распознаваемого технического объекта, "
        "материала, свойства, режима, стандарта или артикула. "
        "Чтобы не подставлять случайные фрагменты из базы, уточните вопрос. "
        "Примеры: " + " | ".join(examples)
    )
    gap = {
        "gap": "Запрос не содержит достаточных доменных ограничений для grounded retrieval.",
        "missing_for": question,
        "source_chunk_id": None,
        "doc_id": None,
    }
    return {
        "answer": answer,
        "status": "needs_clarification",
        "answer_mode": "needs_clarification",
        "intent": "clarification",
        "constraints": {},
        "facts": [],
        "experiments": [],
        "technical_objects": [],
        "parts": [],
        "parameters": [],
        "standards": [],
        "materials": [],
        "requirements": [],
        "equipment": [],
        "laboratories": [],
        "sources": [],
        "gaps": [gap],
        "data_gaps": [gap],
        "partial_matches": {},
        "decision_history": [],
        "subgraph": {"nodes": [], "edges": []},
        "retrieval": {**retrieval_engine.stats(), **_kg_backend_diagnostics(_graph_db_for_repository()), "query_understanding": {"reason": reason}, **(retrieval_meta or {})},
        "llm": llm_client.status(),
        "diagnostics": _kg_backend_diagnostics(_graph_db_for_repository()),
    }


def _can_try_source_grounded_for_unclear(question: str, understanding: Dict[str, Any]) -> bool:
    if understanding.get("reason") != "no_domain_terms":
        return False
    tokens = understanding.get("tokens") or []
    if len(tokens) < 3:
        return False
    q = normalise(question)
    return bool(
        re.search(r"\b(что|какие|какой|как|почему|где|покажи|перечисли|what|which|how|why|where|show|list)\b", q)
        or len(tokens) >= 5
    )


def _source_grounded_findings(sources: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    findings: List[str] = []
    seen: set[str] = set()
    for source in sources[:limit]:
        quote = re.sub(r"\s+", " ", str(source.get("quote") or "")).strip()
        if not quote:
            continue
        chunks = [
            item.strip(" -•\t")
            for item in re.split(r"(?<=[.!?])\s+|(?:\n|\\n)+|[;•]\s+", quote)
            if item and len(item.strip()) >= 30
        ]
        if not chunks:
            chunks = [quote]
        label = str(source.get("source_name") or source.get("filename") or source.get("title") or "источник")
        page = source.get("page_start") or source.get("page")
        suffix = f" ({label}" + (f", стр. {page}" if page else "") + ")"
        for sentence in chunks[:1]:
            sentence = sentence[:280].rstrip(" ,;")
            if len(sentence) >= 277:
                sentence += "..."
            identity = re.sub(r"\s+", " ", sentence.lower()).strip()
            if not sentence or identity in seen:
                continue
            seen.add(identity)
            findings.append(sentence + suffix)
            break
    return findings[:limit]


def _source_grounded_answer_text(question: str, sources: List[Dict[str, Any]]) -> str:
    findings = _source_grounded_findings(sources)
    lines = [
        "Структурированных AcceptedFact недостаточно, поэтому ответ дан как навигационный по найденным фрагментам источников.",
    ]
    if findings:
        lines.append("По найденным evidence-фрагментам:")
        lines.extend(f"- {item}" for item in findings)
    else:
        lines.append("Найдены релевантные источники, но в них не удалось выделить компактные цитаты для verified ответа.")
    lines.append("Это не verified KG-вывод: используйте источники ниже для проверки формулировок.")
    return "\n".join(lines)


def _source_grounded_payload(
    *,
    question: str,
    sources: List[Dict[str, Any]],
    planned_constraints: Any,
    retrieval: Dict[str, Any],
    kg_diagnostics: Dict[str, Any],
    llm: Dict[str, Any],
    ask_intent: str,
    fallback_reason: str,
) -> Dict[str, Any]:
    evidence_sources = sources[: min(len(sources), 8)]
    diagnostics = {
        **kg_diagnostics,
        "selected_answer_mode": "source_grounded_answer",
        "retrieval_status": "chunks_only_no_structured_facts",
        "answer_is_verified": False,
        "source_grounded_answer_used": True,
        "semantic_fallback_executed": True,
        "fallback_reason": fallback_reason,
        "evidence_chunks_used_count": len(evidence_sources),
        "accepted_facts_used_count": 0,
        "typed_facts_found": 0,
        "chunks_found_bm25": retrieval.get("chunks_found_bm25", 0),
        "chunks_found_dense": retrieval.get("chunks_found_dense", 0),
        "chunks_after_fusion": retrieval.get("chunks_after_fusion", 0),
        "top_fused_chunks": retrieval.get("top_fused_chunks", []),
        "partial_reason": "accepted facts отсутствуют; ответ ограничен найденными evidence chunks",
    }
    return {
        "answer": _source_grounded_answer_text(question, evidence_sources),
        "status": "partial",
        "answer_mode": "source_grounded_answer",
        "analytical_intent": "source_grounded_answer",
        "intent": ask_intent,
        "answer_is_verified": False,
        "source_grounded_answer_used": True,
        "semantic_fallback_executed": True,
        "fallback_reason": fallback_reason,
        "evidence_chunks_used_count": len(evidence_sources),
        "accepted_facts_used_count": 0,
        "constraints": planned_constraints.model_dump() if hasattr(planned_constraints, "model_dump") else {"raw_question": question},
        "facts": [],
        "typed_facts": [],
        "experiments": [],
        "technical_objects": [],
        "parts": [],
        "parameters": [],
        "standards": [],
        "materials": [],
        "requirements": [],
        "equipment": [],
        "laboratories": [],
        "sources": evidence_sources,
        "evidence": evidence_sources,
        "gaps": [],
        "data_gaps": [],
        "partial_matches": {
            "retrieval_status": "chunks_only_no_structured_facts",
            "why_not_verified": ["no accepted fact"],
            "missing_structured_fact_types": list(getattr(planned_constraints, "target_fact_types", []) or []),
        },
        "decision_history": [],
        "subgraph": {"nodes": [], "edges": []},
        "retrieval": {
            **retrieval,
            "retrieval_status": "chunks_only_no_structured_facts",
            "answer_mode": "source_grounded_answer",
            "answer_is_verified": False,
            "source_grounded_answer_used": True,
            "semantic_fallback_executed": True,
            "fallback_reason": fallback_reason,
            "evidence_chunks_used_count": len(evidence_sources),
            "accepted_facts_used_count": 0,
            "typed_facts_found": 0,
            "partial_reason": diagnostics["partial_reason"],
        },
        "llm": llm,
        "diagnostics": diagnostics,
    }


def _source_grounded_fallback_from_query(
    *,
    question: str,
    top_k: int,
    planned_constraints: Any,
    kg_diagnostics: Dict[str, Any],
    ask_intent: str,
    fallback_reason: str,
) -> Dict[str, Any] | None:
    chunks = retrieval_engine.query(question, top_k=max(top_k, 8))
    if not chunks:
        return None
    sources = [_source_for_chunk(chunk) for chunk in chunks[: max(1, min(len(chunks), top_k, 8))]]
    if not sources:
        return None
    return _source_grounded_payload(
        question=question,
        sources=sources,
        planned_constraints=planned_constraints,
        retrieval={**retrieval_engine.stats(), **kg_diagnostics},
        kg_diagnostics=kg_diagnostics,
        llm=llm_client.status(),
        ask_intent=ask_intent,
        fallback_reason=fallback_reason,
    )


def _fact_matches_query(fact: Dict[str, Any], terms: Dict[str, List[str]]) -> bool:
    if terms["materials"]:
        material = normalise(str(fact.get("material") or ""))
        if not any(normalise(m) in material or material in normalise(m) for m in terms["materials"]):
            return False
    if terms["processes"]:
        regime = normalise(str(fact.get("process_regime") or fact.get("process") or ""))
        process_aliases = {
            "отжиг": {"отжиг", "annealing"},
            "annealing": {"отжиг", "annealing"},
            "закалка": {"закалка", "quenching"},
            "quenching": {"закалка", "quenching"},
            "старение": {"старение", "aging"},
            "aging": {"старение", "aging"},
        }
        accepted = set()
        for process in terms["processes"]:
            accepted.update(process_aliases.get(normalise(process), {normalise(process)}))
        if not any(alias in regime or regime in alias for alias in accepted):
            return False
    if terms["properties"]:
        prop = normalise(str(fact.get("property") or ""))
        if not any(normalise(p) in prop or prop in normalise(p) for p in terms["properties"]):
            return False
    return True


@app.get("/health")
async def health():
    """Return service status and active storage backends."""
    active_graph = _graph_db_for_repository(force_retry=True)
    kg_status = _kg_backend_diagnostics(active_graph)
    extraction_status = _extraction_diagnostics()
    parser_status = _parser_diagnostics()
    answer_status = _answer_synthesis_diagnostics()
    profile_summary = runtime_profile_summary(settings)
    llm_status = _llm_status_with_effective_runtime(llm_client.status())
    return {
        "status": "ok",
        "runtime_profile": getattr(settings, "runtime_profile", "economy_core"),
        "runtime_profile_summary": profile_summary,
        "graph": "neo4j" if active_graph is not None else "disabled",
        **kg_status,
        **extraction_status,
        **parser_status,
        **answer_status,
        "llm_enabled": profile_summary.get("llm_enabled"),
        "llm_provider_configured": llm_status.get("llm_provider_configured"),
        "llm_provider_active": llm_status.get("llm_provider_active"),
        "llm_effective_provider_active": llm_status.get("effective_provider_active"),
        "mistral_base_url": llm_status.get("mistral_base_url"),
        "mistral_model": llm_status.get("mistral_model"),
        "mistral_api_key_configured": llm_status.get("mistral_api_key_configured"),
        "openrouter_api_key_configured": llm_status.get("openrouter_api_key_configured"),
        "llm_ready": llm_status.get("effective_ready"),
        "llm_last_error": llm_status.get("llm_last_error"),
        "llm_fallback_reason": llm_status.get("fallback_reason"),
        "catalog": catalog.counts(),
        "retrieval": retrieval_engine.stats(),
        "extraction": extraction_status,
        "parser": parser_status,
        "answering": answer_status,
        "llm": llm_status,
        "qdrant_projection_enabled": _qdrant_outbox_enabled(),
        "qdrant_outbox_pending": len(outbox.pending(limit=1000)),
        # Backward-compatible alias for older smoke/eval tooling.
        "outbox_pending": len(outbox.pending(limit=1000)),
    }


@app.get("/system/capabilities")
async def system_capabilities():
    """Return feature flags and configured/active system capabilities."""
    active_graph = _graph_db_for_repository(force_retry=True)
    kg_status = _kg_backend_diagnostics(active_graph)
    parser_status = _parser_diagnostics()
    extraction_status = _extraction_diagnostics()
    retrieval_status = retrieval_engine.stats()
    llm_status = _llm_status_with_effective_runtime(llm_client.status())
    return {
        "kg_backend": {
            "configured": kg_status.get("kg_backend_configured"),
            "active": kg_status.get("kg_backend_active"),
            "neo4j_available": kg_status.get("neo4j_available"),
            "neo4j_uri": kg_status.get("neo4j_uri"),
            "neo4j_user": kg_status.get("neo4j_user"),
            "neo4j_password_configured": kg_status.get("neo4j_password_configured"),
            "neo4j_error": kg_status.get("neo4j_error"),
            "decision": kg_status.get("kg_backend_decision"),
        },
        "parser": parser_status,
        "extraction": extraction_status,
        "analytics": {
            "answer_synthesis_mode": getattr(settings, "answer_synthesis_mode", "template"),
            "supported_intents": [item.value for item in AnalyticalIntent],
        },
        "runtime_presets": [item.model_dump() for item in list_runtime_presets()],
        "llm": llm_status,
        "retrieval": retrieval_status,
        "optional_features": {
            "llm_available": bool(llm_status.get("ready")),
            "qdrant_available": bool(retrieval_status.get("qdrant_ready")),
            "docling_available": bool(parser_status.get("docling_available")),
            "markitdown_available": bool(parser_status.get("markitdown_available")),
            "ocr_enabled": bool(parser_status.get("ocr_enabled")),
            "local_embeddings_available": bool(retrieval_status.get("local_embeddings_ready")),
        },
    }


@app.post("/system/test-llm")
async def system_test_llm():
    """Run a minimal configured LLM/OpenRouter connectivity check."""

    return llm_client.test_connection()


@app.get("/runtime/presets")
async def runtime_presets():
    """Return the three curated user-facing runtime modes."""
    return {"items": [item.model_dump() for item in list_runtime_presets()]}


@app.post("/runtime/validate-preset")
async def runtime_validate_preset(request: RuntimePresetRequest):
    """Validate one runtime preset against current optional backend availability."""
    preset = get_runtime_preset(request.preset_id)
    active_graph = _graph_db_for_repository(preset.kg_backend)
    active_backend = "fallback" if preset.kg_backend == "fallback" else "neo4j" if active_graph else "fallback"
    diagnostics = {
        **_kg_backend_diagnostics(active_graph, active_backend, configured_backend=preset.kg_backend),
        **preset_diagnostics(
            preset,
            active_backend=active_backend,
            neo4j_available=active_graph is not None,
            input_source="runtime_validate_preset",
        ),
    }
    return {
        "preset": preset.model_dump(),
        "valid": True,
        "diagnostics": diagnostics,
    }


@app.post("/runtime/run-preset-check")
async def runtime_run_preset_check(request: RuntimePresetRequest):
    """Run lightweight checks for one runtime preset without mutating global settings."""
    preset = get_runtime_preset(request.preset_id)
    checks: dict[str, Any] = {}
    active_graph = _graph_db_for_repository(preset.kg_backend)
    try:
        repository = GraphRepositoryFactory.create(
            catalog=catalog,
            extractor=extractor,
            graph_db=active_graph,
            document_getter=_get_document_meta,
            configured_backend=preset.kg_backend,
        )
        checks["graph_stats"] = repository.get_graph_stats().model_dump()
    except Exception as exc:
        checks["graph_stats_error"] = str(exc)

    questions = {
        "strict_positive": "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?",
        "strict_negative": "Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?",
        "material_overview": "Что уже делали по ВТ6?",
    }
    for key, question in questions.items():
        try:
            response = await _ask_impl(
                question=question,
                top_k=8,
                preset_id=preset.preset_id,
                input_source="runtime_preset_check",
                query_params_ignored=False,
            )
            checks[key] = {
                "status": response.get("status"),
                "analytical_intent": response.get("analytical_intent"),
                "facts_count": len(response.get("facts") or []),
                "sources_count": len(response.get("sources") or response.get("evidence") or []),
                "diagnostics_present": bool(response.get("diagnostics")),
            }
        except Exception as exc:
            checks[key] = {"error": str(exc)}

    passed = bool(checks.get("graph_stats")) and all("error" not in checks.get(key, {}) for key in questions)
    return {
        "preset": preset.model_dump(),
        "passed": passed,
        "checks": checks,
        "diagnostics": preset_diagnostics(
            preset,
            active_backend=(checks.get("graph_stats") or {}).get("kg_backend_active"),
            neo4j_available=active_graph is not None,
            input_source="runtime_preset_check",
        ),
    }


@app.get("/graph/stats")
async def graph_stats():
    """Return KG counts for cockpit readiness and diagnostics."""
    repository, repository_graph = _create_graph_repository_or_503()
    stats = repository.get_graph_stats().model_dump()
    stats["kg_backend_active"] = repository_backend_name(repository)
    stats["diagnostics"] = {
        **(stats.get("diagnostics") or {}),
        **_kg_backend_diagnostics(repository_graph, repository_backend_name(repository)),
    }
    return stats


@app.get("/graph/entities")
async def graph_entities(
    entity_type: str | None = Query(default=None),
    query: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """List graph entities through a whitelisted explorer interface."""
    repository, repository_graph = _create_graph_repository_or_503()
    try:
        items = repository.list_entities(entity_type=entity_type, query=query, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "items": [item.model_dump() for item in items],
        "diagnostics": _kg_backend_diagnostics(repository_graph, repository_backend_name(repository)),
    }


@app.get("/graph/entity/{entity_type}/{entity_id}")
async def graph_entity_card(entity_type: str, entity_id: str):
    """Return one entity card with related facts, sources and subgraph."""
    repository, repository_graph = _create_graph_repository_or_503()
    try:
        card = repository.get_entity_card(entity_type=entity_type, entity_id=entity_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = card.model_dump()
    payload["diagnostics"] = {
        **payload.get("diagnostics", {}),
        **_kg_backend_diagnostics(repository_graph, repository_backend_name(repository)),
    }
    return payload


@app.get("/graph/neighborhood")
async def graph_neighborhood(
    entity_type: str = Query(...),
    entity_id: str = Query(...),
    depth: int = Query(default=1, ge=1, le=2),
    limit_nodes: int = Query(default=50, ge=1, le=300),
    limit_edges: int = Query(default=80, ge=1, le=500),
):
    """Return a compact subgraph around one entity."""
    repository, repository_graph = _create_graph_repository_or_503()
    try:
        subgraph = repository.get_neighborhood(
            entity_type=entity_type,
            entity_id=entity_id,
            depth=depth,
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "subgraph": subgraph,
        "diagnostics": _kg_backend_diagnostics(repository_graph, repository_backend_name(repository)),
    }


@app.get("/graph/gaps")
async def graph_gaps(
    material: str | None = Query(default=None),
    regime: str | None = Query(default=None),
    property: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return constraint-filtered DataGap rows."""
    repository, repository_graph = _create_graph_repository_or_503()
    gaps = repository.get_gaps(material=material, regime=regime, property_name=property, limit=limit)
    return {
        "items": [gap.model_dump() for gap in gaps],
        "diagnostics": _kg_backend_diagnostics(repository_graph, repository_backend_name(repository)),
    }


@app.get("/graph/decision-history")
async def graph_decision_history(
    material: str | None = Query(default=None),
    regime: str | None = Query(default=None),
    property: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return decision history timeline rows with optional constraints."""
    repository, repository_graph = _create_graph_repository_or_503()
    history = repository.get_decision_history_filtered(
        material=material,
        regime=regime,
        property_name=property,
        limit=limit,
    )
    return {
        "items": [item.model_dump() for item in history],
        "diagnostics": _kg_backend_diagnostics(repository_graph, repository_backend_name(repository)),
    }


@app.get("/graph/similar-experiments")
async def graph_similar_experiments(
    material: str | None = Query(default=None),
    regime: str | None = Query(default=None),
    property: str | None = Query(default=None),
    experiment_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=100),
):
    """Return graph-similar experiments with transparent score/explanation."""
    repository, repository_graph = _create_graph_repository_or_503()
    rows = repository.get_similar_experiments(
        material=material,
        regime=regime,
        property_name=property,
        experiment_id=experiment_id,
        limit=limit,
    )
    return {
        "items": [item.model_dump() for item in rows],
        "diagnostics": _kg_backend_diagnostics(repository_graph, repository_backend_name(repository)),
    }


@app.get("/demo/scenarios")
async def demo_scenarios():
    """Return curated demo scenarios for the Streamlit cockpit."""
    return {"items": [item.model_dump() for item in list_demo_scenarios()]}


@app.post("/demo/run-scenario")
async def demo_run_scenario(
    scenario_id: str = Query(...),
    top_k: int = Query(default=12, ge=1, le=50),
    preset_id: RuntimePresetId | None = Query(default=None),
):
    """Run one curated scenario through the normal /ask handler."""
    scenario = get_demo_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Unknown scenario_id={scenario_id!r}")
    payload = await _ask_impl(
        question=scenario.question,
        top_k=top_k,
        preset_id=preset_id,
        input_source="demo_run_scenario",
        query_params_ignored=False,
    )
    if isinstance(payload, dict):
        payload = {
            **payload,
            "scenario": scenario.model_dump(),
        }
    return payload


@app.get("/debug/graph/stats")
async def debug_graph_stats():
    active_graph = _graph_db_for_repository()
    if active_graph is None:
        return {"graph": "disabled", "stats": {}, **_kg_backend_diagnostics(active_graph)}
    try:
        return {"graph": "neo4j", "stats": active_graph.stats(), **_kg_backend_diagnostics(active_graph)}
    except Exception as exc:
        return {"graph": "neo4j", "error": str(exc), "stats": {}, **_kg_backend_diagnostics(active_graph)}


@app.get("/knowledge/summary")
async def knowledge_summary():
    report = _knowledge_report(active_only=True)
    return {
        "status": report.get("status"),
        "documents_count": report.get("documents_count"),
        "active_documents_count": report.get("active_documents_count"),
        "chunks_count": report.get("chunks_count"),
        "active_chunks_count": report.get("active_chunks_count"),
        "canonical_facts_count": report.get("canonical_facts_count"),
        "new_connections_count": len(report.get("comparison_opportunities") or []),
        "conflict_groups_count": report.get("conflict_groups_count"),
        "data_gaps_count": report.get("data_gaps_count"),
        "facts_without_evidence": report.get("facts_without_evidence"),
        "last_ingested_at": max((item.get("updated_at") or "" for item in report.get("documents") or []), default=None),
        "resources": report.get("resources"),
    }


@app.get("/knowledge/expansion-report")
async def knowledge_expansion_report(document_id: str | None = Query(default=None), active_only: bool = Query(default=True)):
    return _knowledge_report(active_only=active_only, document_id=document_id)


@app.post("/knowledge/rebuild")
async def knowledge_rebuild():
    rebuild = _rebuild_runtime_indexes_from_active_catalog()
    report = _knowledge_report(active_only=True)
    return {"status": "rebuilt", "reindexed": rebuild, "summary": await knowledge_summary(), "report": report}


@app.post("/knowledge/sync-neo4j")
async def knowledge_sync_neo4j():
    active_graph = _graph_db_for_repository()
    if _configured_kg_backend() == "neo4j" and active_graph is None:
        raise HTTPException(status_code=503, detail={**_kg_backend_diagnostics(active_graph), "error": "Neo4j is required by KG_BACKEND=neo4j"})
    sync = _sync_strict_graph_to_neo4j(active_graph)
    return {"status": "synced" if sync.get("status") == "synced" else "skipped", "strict_graph_projection": sync, "summary": await knowledge_summary()}


@app.get("/documents")
async def list_documents():
    return catalog.list_document_records()


@app.post("/documents/active")
async def set_documents_active_batch(request: DocumentsActiveBatchRequest):
    """Batch-update active flags without forcing a heavy Neo4j sync per checkbox."""

    if not request.changes:
        return {"status": "no_changes", "updated": 0, "missing": 0, "reindexed": _rebuild_runtime_indexes_from_active_catalog()}
    result = catalog.set_documents_active(request.changes)
    rebuild = _rebuild_runtime_indexes_from_active_catalog()
    active_graph = _graph_db_for_repository()
    if request.sync_neo4j:
        strict_graph_projection = _sync_strict_graph_to_neo4j(active_graph)
    else:
        strict_graph_projection = {"status": "skipped", "reason": "sync_neo4j_false"}
    return {
        "status": "updated",
        **result,
        "reindexed": rebuild,
        "strict_graph_projection": strict_graph_projection,
    }


@app.patch("/documents/{doc_id}/active")
async def set_document_active(doc_id: str, request: DocumentActiveRequest):
    """Enable/disable one document for retrieval, fallback graph and analytical answers."""

    if not catalog.set_document_active(doc_id, request.active):
        raise HTTPException(status_code=404, detail="Document not found")
    rebuild = _rebuild_runtime_indexes_from_active_catalog()
    active_graph = _graph_db_for_repository()
    strict_graph_projection = _sync_strict_graph_to_neo4j(active_graph)
    return {
        "doc_id": doc_id,
        "active": request.active,
        "reindexed": rebuild,
        "strict_graph_projection": strict_graph_projection,
    }


@app.delete("/documents/{doc_id}")
async def deactivate_document(doc_id: str):
    """Non-destructive delete: deactivate document so it no longer participates in answers."""

    if not catalog.set_document_active(doc_id, False):
        raise HTTPException(status_code=404, detail="Document not found")
    rebuild = _rebuild_runtime_indexes_from_active_catalog()
    active_graph = _graph_db_for_repository()
    strict_graph_projection = _sync_strict_graph_to_neo4j(active_graph)
    return {"doc_id": doc_id, "active": False, "deleted": False, "reindexed": rebuild, "strict_graph_projection": strict_graph_projection}


@app.post("/admin/reset-corpus")
async def reset_corpus(confirm: str = Query(default=""), clear_neo4j: bool = Query(default=True)):
    """Dangerous operation: clear local catalog/runtime and optionally Neo4j projection."""

    if confirm != "RESET_ACTIVE_CORPUS":
        raise HTTPException(status_code=400, detail="Pass confirm=RESET_ACTIVE_CORPUS to reset the active corpus")
    reset = _reset_runtime_corpus(clear_neo4j=clear_neo4j)
    return {"status": "reset", **reset}


@app.post("/graph/refresh")
async def refresh_graph_from_active_documents(sync_neo4j: bool = True):
    """Refresh in-memory retrieval/fallback state and optionally sync active catalog to Neo4j."""

    rebuild = _rebuild_runtime_indexes_from_active_catalog()
    active_graph = _graph_db_for_repository()
    if sync_neo4j:
        strict_graph_projection = _sync_strict_graph_to_neo4j(active_graph)
    else:
        strict_graph_projection = {"status": "skipped", "reason": "sync_neo4j_false"}
    return {
        "status": "refreshed",
        **rebuild,
        "catalog": catalog.counts(),
        "strict_graph_projection": strict_graph_projection,
        "kg": _kg_backend_diagnostics(active_graph),
    }


@app.post("/ingest/documents")
async def ingest_documents(files: List[UploadFile] = File(...), sync_graph: bool = True):
    """Accept multiple files, parse them, persist graph/catalog metadata and index chunks."""
    _validate_upload_batch(files)
    responses = []
    workspace_uid = "default-workspace"

    active_graph = _graph_db_for_repository()
    if _configured_kg_backend() == "neo4j" and active_graph is None:
        raise HTTPException(status_code=503, detail={**_kg_backend_diagnostics(active_graph), "error": "Neo4j is required by KG_BACKEND=neo4j"})

    if active_graph:
        active_graph.upsert_workspace(uid=workspace_uid, slug="default", name="Default Workspace")

    for file in files:
        safe_name = _validate_upload_name(file.filename or "uploaded.bin")
        content = await file.read()
        if not content:
            responses.append({"filename": safe_name, "status": "skipped_empty"})
            continue
        _validate_upload_size(content, safe_name)

        before_knowledge_report = _knowledge_report(active_only=True)
        content_hash = _sha256(content)
        doc_id = _stable_doc_id(content_hash)
        source_uid = _stable_source_id(content_hash)
        document_version = _document_version_for(safe_name, content_hash)
        ingested_at = datetime.now(timezone.utc).isoformat()
        tmp_path = Path(tempfile.gettempdir()) / f"{doc_id}_{safe_name}"
        tmp_path.write_bytes(content)

        try:
            parsed = parser_router.parse_document_intelligence(str(tmp_path), doc_id=doc_id, source_type="file")
        except Exception as exc:
            responses.append(
                {
                    "filename": safe_name,
                    "doc_id": doc_id,
                    "source_uid": source_uid,
                    "status": "parse_failed",
                    "parser": "unknown",
                    "chunks": 0,
                    "parser_error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    "parser_diagnostics": {"warnings": ["parser_failed"]},
                    "document_version": document_version,
                    "knowledge_expansion": {"status": "skipped", "reason": "parser_failed"},
                    "strict_graph_projection": {"status": "skipped", "reason": "parser_failed"},
                }
            )
            continue
        parsed_metadata = _parsed_metadata(parsed, source_name=safe_name)
        parser_name = parsed_metadata.get("parser", "unknown")
        document_status = _document_parse_status(parsed)
        source_metadata = parsed_metadata.get("source_metadata") or {}

        doc_meta = Document(
            doc_id=doc_id,
            workspace_uid=workspace_uid,
            title=safe_name,
            source_uid=source_uid,
            external_id=content_hash,
            parser=parser_name,
            language=parsed_metadata.get("language"),
            status=document_status,
            created_at=None,
            updated_at=ingested_at,
            version=document_version,
        )

        rich_chunks: List[Chunk] = []
        for idx, chunk in enumerate(parsed.chunks):
            chunk.doc_id = doc_id
            chunk.workspace_uid = workspace_uid
            chunk.ordinal = idx if chunk.ordinal is None else chunk.ordinal
            chunk.embedding_version = chunk.embedding_version or settings.embedding_model
            chunk.metadata.setdefault("parser", parser_name)
            chunk.metadata.setdefault("filename", safe_name)
            chunk.metadata.setdefault("source_name", safe_name)
            chunk.metadata.setdefault("source_title", safe_name)
            chunk.metadata.setdefault("source_type", "file")
            chunk.metadata.setdefault("source_url", None)
            chunk.metadata.setdefault("parser_name", parser_name)
            chunk.metadata.setdefault("parser_error", parsed_metadata.get("parser_error"))
            chunk.metadata.setdefault("source_metadata", source_metadata)
            chunk.metadata.setdefault("publication_year", source_metadata.get("publication_year"))
            chunk.metadata.setdefault("geographies", source_metadata.get("geographies") or [])
            chunk.metadata.setdefault("practice_scope", source_metadata.get("practice_scope"))
            chunk.metadata.setdefault("reliability_level", source_metadata.get("reliability_level"))
            chunk.metadata.setdefault("source_type_detected", source_metadata.get("source_type_detected"))
            rich_chunks.append(chunk)

        DOCUMENTS[doc_id] = doc_meta
        CHUNKS[doc_id] = rich_chunks
        catalog.upsert_document(
            doc_meta,
            metadata={
                "filename": safe_name,
                "source_type": "file",
                "content_hash": content_hash,
                "document_version": document_version,
                "ingested_at": ingested_at,
                "parse_status": document_status,
                "source_metadata": source_metadata,
                "parser_diagnostics": parsed.diagnostics,
                "document_intelligence": parsed_metadata["document_intelligence"],
            },
        )
        catalog.replace_chunks(doc_id, rich_chunks)
        knowledge_delta = KnowledgeExpansionEngine(catalog).build_delta_report(
            before_knowledge_report,
            document_id=doc_id,
            active_only=True,
        )

        if active_graph:
            active_graph.upsert_source(
                uid=source_uid,
                type_="file",
                uri=safe_name,
                checksum=content_hash,
                imported_at=None,
            )
            active_graph.link_workspace_source(workspace_uid=workspace_uid, source_uid=source_uid)
            active_graph.upsert_document(
                doc_id=doc_meta.doc_id,
                workspace_uid=doc_meta.workspace_uid,
                title=doc_meta.title,
                source_uid=doc_meta.source_uid,
                external_id=doc_meta.external_id,
                parser=doc_meta.parser,
                language=doc_meta.language,
                status=doc_meta.status,
                created_at=doc_meta.created_at,
                updated_at=doc_meta.updated_at,
                version=doc_meta.version,
            )
            active_graph.link_document_source(document_uid=doc_id, source_uid=source_uid)
            for chunk in rich_chunks:
                active_graph.upsert_chunk_node(
                    {
                        "uid": chunk.chunk_id,
                        "document_uid": doc_id,
                        "workspace_uid": workspace_uid,
                        "ordinal": chunk.ordinal,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "section_path": chunk.section_path,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "token_count": chunk.token_count,
                        "text_hash": chunk.text_hash,
                        "preview": chunk.preview or chunk.text[:200],
                        "embedding_version": chunk.embedding_version,
                        "updated_at": chunk.updated_at,
                    }
                )
                extraction = answer_extraction_pipeline.extract_from_chunk(chunk)
                for entity in extraction.entities:
                    uid = stable_entity_uid(workspace_uid, entity.entity_type, entity.canonical_name)
                    norm_name = normalise(entity.canonical_name)
                    active_graph.link_chunk_entity(
                        chunk_uid=chunk.chunk_id,
                        entity_uid=uid,
                        canonical_name=entity.canonical_name,
                        entity_type=entity.entity_type,
                        norm_name=norm_name,
                        confidence=0.75,
                        count=1,
                    )
            active_graph.link_chunk_sequence(doc_id)

        retrieval_engine.index_chunks(rich_chunks, replace_doc_id=doc_id)

        strict_graph_projection: Dict[str, Any] = {"status": "skipped", "reason": "neo4j_unavailable_or_fallback"}
        if active_graph and sync_graph:
            _project_document_to_graph(active_graph, doc_meta, rich_chunks, source_uri=safe_name)
            strict_graph_projection = _sync_strict_graph_to_neo4j(active_graph)
        elif active_graph:
            strict_graph_projection = {"status": "skipped", "reason": "sync_graph_false"}

        _enqueue_qdrant_projection(rich_chunks)

        responses.append(
            {
                "filename": safe_name,
                "doc_id": doc_id,
                "source_uid": source_uid,
                "status": doc_meta.status,
                "parse_status": document_status,
                "parser": parser_name,
                "chunks": len(rich_chunks),
                "parser_error": parsed_metadata.get("parser_error"),
                "parser_diagnostics": parsed.diagnostics,
                "source_metadata": source_metadata,
                "document_version": document_version,
                "knowledge_expansion": knowledge_delta,
                "strict_graph_projection": strict_graph_projection,
            }
        )

    return {"ingested": responses}


@app.post("/ingest/url")
async def ingest_url(url: str):
    """Fetch and ingest an online HTML resource without requiring external parsing services."""
    workspace_uid = "default-workspace"
    try:
        fetched = fetch_url_safely(
            url,
            allow_private=bool(getattr(settings, "ingest_url_allow_private", False)),
            max_bytes=int(getattr(settings, "ingest_url_max_bytes", 10_485_760)),
            timeout_seconds=int(getattr(settings, "ingest_url_timeout_seconds", 10)),
            request_get=requests.get,
        )
    except UnsafeUrlError as exc:
        raise HTTPException(status_code=400, detail=f"Unsafe URL: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"URL ingestion failed: {exc}") from exc

    url = fetched.url
    content_type = fetched.content_type
    content_type_lower = str(content_type or "").lower()
    if "text/html" not in content_type_lower and "application/xhtml+xml" not in content_type_lower:
        raise HTTPException(status_code=400, detail=f"URL content is not HTML: {content_type or 'unknown'}")

    content = fetched.content
    content_hash = _sha256(url.encode("utf-8") + content)
    doc_id = _stable_doc_id(content_hash)
    source_uid = _stable_source_id(content_hash)
    safe_name = Path(url.split("?")[0].rstrip("/")).name or "online_resource.html"
    if not safe_name.endswith((".html", ".htm")):
        safe_name = f"{safe_name}.html"
    before_knowledge_report = _knowledge_report(active_only=True)
    document_version = _document_version_for(safe_name, content_hash)
    ingested_at = datetime.now(timezone.utc).isoformat()
    tmp_path = Path(tempfile.gettempdir()) / f"{doc_id}_{safe_name}"
    tmp_path.write_bytes(content)

    parsed = parser_router.parse_document_intelligence(str(tmp_path), doc_id=doc_id, source_type="url", source_url=url)
    parsed_metadata = _parsed_metadata(parsed, source_url=url, source_name=safe_name)
    parser_name = parsed_metadata.get("parser", "html")
    title = parsed_metadata.get("title") or safe_name
    source_metadata = parsed_metadata.get("source_metadata") or {}
    doc_meta = Document(
        doc_id=doc_id,
        workspace_uid=workspace_uid,
        title=title,
        source_uid=source_uid,
        external_id=content_hash,
        parser=parser_name,
        language=parsed_metadata.get("language"),
        status="ingested" if parsed.chunks else "empty_or_parse_failed",
        created_at=None,
        updated_at=ingested_at,
        version=document_version,
    )

    rich_chunks: List[Chunk] = []
    for idx, chunk in enumerate(parsed.chunks):
        chunk.doc_id = doc_id
        chunk.workspace_uid = workspace_uid
        chunk.ordinal = idx if chunk.ordinal is None else chunk.ordinal
        chunk.embedding_version = chunk.embedding_version or settings.embedding_model
        chunk.metadata.setdefault("parser", parser_name)
        chunk.metadata.setdefault("parser_name", parser_name)
        chunk.metadata.setdefault("parser_error", parsed_metadata.get("parser_error"))
        chunk.metadata["filename"] = safe_name
        chunk.metadata["source_name"] = title
        chunk.metadata["source_title"] = title
        chunk.metadata["source_type"] = "url"
        chunk.metadata["source_url"] = url
        chunk.metadata.setdefault("source_metadata", source_metadata)
        chunk.metadata.setdefault("publication_year", source_metadata.get("publication_year"))
        chunk.metadata.setdefault("geographies", source_metadata.get("geographies") or [])
        chunk.metadata.setdefault("practice_scope", source_metadata.get("practice_scope"))
        chunk.metadata.setdefault("reliability_level", source_metadata.get("reliability_level"))
        chunk.metadata.setdefault("source_type_detected", source_metadata.get("source_type_detected"))
        rich_chunks.append(chunk)

    DOCUMENTS[doc_id] = doc_meta
    CHUNKS[doc_id] = rich_chunks
    catalog.upsert_document(
        doc_meta,
        metadata={
            "filename": safe_name,
            "source_name": title,
            "source_title": title,
            "source_url": url,
            "source_type": "url",
            "source_metadata": source_metadata,
            "content_hash": content_hash,
            "document_version": document_version,
            "ingested_at": ingested_at,
            "parser_diagnostics": parsed.diagnostics,
            "document_intelligence": parsed_metadata["document_intelligence"],
        },
    )
    catalog.replace_chunks(doc_id, rich_chunks)
    knowledge_delta = KnowledgeExpansionEngine(catalog).build_delta_report(
        before_knowledge_report,
        document_id=doc_id,
        active_only=True,
    )
    retrieval_engine.index_chunks(rich_chunks, replace_doc_id=doc_id)

    active_graph = _graph_db_for_repository()
    if _configured_kg_backend() == "neo4j" and active_graph is None:
        raise HTTPException(status_code=503, detail={**_kg_backend_diagnostics(active_graph), "error": "Neo4j is required by KG_BACKEND=neo4j"})
    strict_graph_projection: Dict[str, Any] = {"status": "skipped", "reason": "neo4j_unavailable_or_fallback"}
    if active_graph:
        _project_document_to_graph(active_graph, doc_meta, rich_chunks, source_uri=url)
        strict_graph_projection = _sync_strict_graph_to_neo4j(active_graph)

    _enqueue_qdrant_projection(rich_chunks)

    return {
        "ingested": {
            "url": url,
            "source_name": title,
            "doc_id": doc_id,
            "source_uid": source_uid,
            "status": doc_meta.status,
            "parser": parser_name,
            "chunks": len(rich_chunks),
            "parser_error": parsed_metadata.get("parser_error"),
            "parser_diagnostics": parsed.diagnostics,
            "source_metadata": source_metadata,
            "document_version": document_version,
            "knowledge_expansion": knowledge_delta,
            "strict_graph_projection": strict_graph_projection,
        }
    }


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    doc = _get_document_meta(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str):
    chunks = CHUNKS.get(doc_id) or catalog.list_chunks(doc_id, active_only=False)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found or has no chunks")
    return chunks


@app.post("/query")
async def query(question: str, top_k: int = 8):
    chunks = retrieval_engine.query(question, top_k=top_k)
    return [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "workspace_uid": c.workspace_uid,
            "page_range": f"{c.page_start}-{c.page_end}",
            "ordinal": c.ordinal,
            "section_path": c.section_path,
            "text": c.text[:700] + ("..." if len(c.text) > 700 else ""),
        }
        for c in chunks
    ]


@app.get("/debug/retrieval")
async def debug_retrieval(question: str, top_k: int = 10):
    from .retrieval.retrieval import expand_query

    search_query = expand_query(question)
    lex = retrieval_engine.lexical_retrieve(search_query, top_k=top_k)
    dense = retrieval_engine.dense_retrieve(search_query, top_k=top_k)
    return {
        "question": question,
        "expanded_query": search_query,
        "retrieval_stats": retrieval_engine.stats(),
        "lexical": [
            {
                "rank": i + 1,
                "score": score,
                "chunk_id": retrieval_engine.chunks[idx].chunk_id,
                "doc_id": retrieval_engine.chunks[idx].doc_id,
                "preview": retrieval_engine.chunks[idx].text[:300],
            }
            for i, (idx, score) in enumerate(lex)
            if 0 <= idx < len(retrieval_engine.chunks)
        ],
        "dense": [
            {"rank": i + 1, "score": score, "chunk_id": cid}
            for i, (cid, score) in enumerate(dense)
        ],
    }


@app.get("/sync/outbox/pending")
async def pending_outbox(limit: int = 100):
    return outbox.pending(limit=limit)


@app.post("/sync/outbox/process")
async def process_outbox(limit: int = 100):
    """Process pending projection events from the outbox."""
    pending = outbox.pending(limit=limit)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in retrieval_engine.chunks}
    processed = 0
    failed = 0
    errors = []

    for event in pending:
        event_id = event["event_id"]
        try:
            if event["aggregate_type"] == "Chunk" and event["op"] in {"upsert", "update"}:
                chunk = chunk_by_id.get(event["aggregate_uid"])
                if chunk is None:
                    raise RuntimeError(f"Chunk {event['aggregate_uid']} is not loaded in local index")
                ok = retrieval_engine.project_chunks_to_qdrant([chunk])
                if not ok:
                    raise RuntimeError("Qdrant projection unavailable")
            outbox.mark_processed(event_id)
            processed += 1
        except Exception as exc:
            outbox.mark_failed(event_id, str(exc))
            failed += 1
            errors.append({"event_id": event_id, "error": str(exc)})

    return {"processed": processed, "failed": failed, "errors": errors}


@app.post("/admin/rebuild-index")
async def rebuild_index():
    """Rebuild in-memory lexical retrieval from the durable SQLite catalog."""
    retrieval_engine.chunks.clear()
    for chunks in [catalog.list_chunks(doc.doc_id) for doc in catalog.list_documents()]:
        retrieval_engine.index_chunks(chunks)
    return {"status": "rebuilt", "retrieval": retrieval_engine.stats()}


@app.post("/admin/rebuild-graph")
async def rebuild_graph(limit: int = 50, offset: int = 0):
    """Re-project catalog documents, chunks, entities and relations to Neo4j.

    The endpoint is intentionally chunk-paginated. Neo4j projection can be slow
    on a laptop when every extracted relation is written one-by-one; pagination
    keeps the API responsive and makes demo setup observable.
    """
    active_graph = get_graph_db()
    if active_graph is None:
        return {"status": "skipped", "reason": "neo4j_unavailable", "projected": {"documents": 0, "chunks": 0, "entities": 0, "relations": 0}}

    totals = {"documents": 0, "chunks": 0, "entities": 0, "relations": 0}
    docs = catalog.list_documents()
    all_items: List[tuple[Document, Chunk]] = []
    for doc in docs:
        all_items.extend((doc, chunk) for chunk in catalog.list_chunks(doc.doc_id))

    safe_limit = max(1, min(int(limit or 50), 200))
    safe_offset = max(0, int(offset or 0))
    selected = all_items[safe_offset:safe_offset + safe_limit]
    chunks_by_doc: Dict[str, List[Chunk]] = {}
    doc_by_id: Dict[str, Document] = {}
    for doc, chunk in selected:
        doc_by_id[doc.doc_id] = doc
        chunks_by_doc.setdefault(doc.doc_id, []).append(chunk)

    for doc_id, chunks in chunks_by_doc.items():
        doc = doc_by_id[doc_id]
        projected = _project_document_to_graph(active_graph, doc, chunks, source_uri=doc.title)
        for key in totals:
            totals[key] += int(projected.get(key, 0))

    next_offset = safe_offset + len(selected)
    return {
        "status": "rebuilt_page",
        "offset": safe_offset,
        "limit": safe_limit,
        "next_offset": next_offset if next_offset < len(all_items) else None,
        "total_chunks": len(all_items),
        "projected": totals,
        "graph_stats": active_graph.stats(),
    }


@app.post("/admin/sync-strict-graph")
async def sync_strict_graph():
    """Materialize strict ontology facts from the SQLite catalog into Neo4j."""
    active_graph = _graph_db_for_repository()
    if active_graph is None:
        if _configured_kg_backend() == "neo4j":
            raise HTTPException(status_code=503, detail={**_kg_backend_diagnostics(active_graph), "error": "Neo4j is required by KG_BACKEND=neo4j"})
        return {"status": "skipped", **_kg_backend_diagnostics(active_graph)}
    result = _sync_strict_graph_to_neo4j(active_graph)
    return {**result, **_kg_backend_diagnostics(active_graph)}


@app.get("/graph/subgraph")
async def graph_subgraph(entity_ids: List[str] = Query(...), hops: int = 1):
    active_graph = _graph_db_for_repository()
    if active_graph is not None:
        return active_graph.fetch_subgraph(entity_ids, hops=hops)

    wanted = {normalise(entity_id) for entity_id in entity_ids}
    selected: List[tuple[Chunk, Any]] = []
    for chunk in retrieval_engine.chunks:
        extraction = answer_extraction_pipeline.extract_from_chunk(chunk)
        names = {normalise(entity.canonical_name) for entity in extraction.entities}
        ids = {
            _node_id(entity.entity_type, entity.canonical_name, chunk.workspace_uid or "default-workspace")
            for entity in extraction.entities
        }
        if not wanted or wanted.intersection(names) or wanted.intersection(ids):
            selected.append((chunk, extraction))
    return _build_local_subgraph(selected)


# --- Answer synthesis endpoint ---
def _intent(question: str) -> str:
    q = normalise(question)
    asks_inventory = (
        ("сплав" in q or "материал" in q)
        and any(term in q for term in ["какие", "что есть", "список", "перечень", "все", "имеются"])
    ) or any(term in q for term in ["что загружено", "что есть"])
    asks_activity = any(term in q for term in ["что делали", "что с каждым", "что с ними", "что с ним", "какие режим", "как обрабатывали", "какие свойства", "какие измерения"])
    has_technical_object = any(term in q for term in ["насос", "pump", "клапан", "valve", "npk-200", "dn50"])
    has_specific_lookup = any(
        term in q
        for term in [
            "параметр", "давлен", "температур", "материал", "корпус", "стандарт", "гост", "iso",
            "артикул", "деталь", "требован", "изображ", "схем", "монтаж", "оборудован", "лаборатор",
            "прочность", "твёрд", "тверд", "пластич", "корроз", "что делали", "режим",
        ]
    )
    if asks_inventory and asks_activity:
        return "material_activity_summary"
    if asks_activity and _query_terms(question).get("materials"):
        return "material_activity_summary"
    if any(term in q for term in ["изображ", "схем", "image", "монтаж"]):
        return "image_lookup"
    if any(term in q for term in ["оборудован", "установк", "печь", "прибор", "станок", "equipment"]):
        return "equipment_lookup"
    if any(term in q for term in ["лаборатор", "команд", "группа", "laboratory", "team"]):
        return "laboratory_lookup"
    if "герметич" in q:
        return "requirement_lookup"
    if any(term in q for term in ["противореч", "неоднород", "расход", "разные значения", "conflict", "different values"]):
        return "conflict_analysis"
    if any(term in q for term in ["пробел", "не хватает", "нет данных", "не измер", "не привед", "not reported", "gap"]):
        return "gap_analysis"
    if any(term in q for term in ["артикул", "article", "part number"]):
        return "part_article_lookup"
    if any(term in q for term in ["параметр", "давлен", "температур", "dn", "pn"]):
        return "parameter_lookup"
    if asks_inventory:
        return "material_inventory"
    if any(term in q for term in ["материал", "корпус"]):
        return "material_lookup"
    if any(term in q for term in ["стандарт", "гост", "iso", "astm", "en "]):
        return "standard_lookup"
    if any(term in q for term in ["требован", "requirement"]):
        return "requirement_lookup"
    if has_technical_object and not has_specific_lookup:
        return "object_overview"
    return "experiment_lookup"


def _focus_terms(question: str) -> List[str]:
    q = question or ""
    terms = []
    patterns = [
        r"\bDN\s*\d+\b",
        r"\bДу\s*\d+\b",
        r"\bNPK-\d+\b",
        r"\b7075-T6\b",
        r"\bВТ6\b",
        r"\bVT6\b",
        r"\b12[ХX]18[НH]10[ТT]\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, q, re.IGNORECASE):
            terms.append(match.group(0).replace(" ", ""))
    # Domain-constrained unknown material names: e.g. "сплав 911" should be
    # treated as a real requested constraint rather than ignored.  This avoids
    # returning unrelated demo facts when the user asks about a nonexistent
    # material.
    for match in re.finditer(
        r"\b(?:сплав(?:у|а|ом|е)?|сталь|стали|alloy|steel)\s+(?P<name>[A-Za-zА-Яа-я0-9\-]+)\b",
        q,
        re.IGNORECASE,
    ):
        terms.append(canonical_material_name(match.group("name")))
    for word in ["клапан", "насос", "valve", "pump"]:
        if word in q.lower():
            terms.append(word)
    return _unique([term for term in terms if term])


def _conflict_analysis_response(question: str, retrieval: Dict[str, Any], kg_diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """Build a deterministic conflict answer from canonical facts, not retrieval noise."""

    report = _knowledge_report(active_only=True)
    conflicts = [item for item in report.get("conflict_groups") or [] if isinstance(item, dict)]
    facts = _facts_for_conflicts(report.get("canonical_facts") or [], conflicts)
    sources = _sources_from_fact_rows(facts)
    if not conflicts:
        return {
            "answer": "В текущем canonical fact layer не найдено групп с расходящимися значениями для одного material + regime + property.",
            "status": "ok",
            "answer_mode": "graph_conflict_analysis",
            "analytical_intent": "conflict_analysis",
            "intent": "conflict_analysis",
            "constraints": {"raw_question": question, "materials": [], "regimes": [], "properties": []},
            "facts": [],
            "technical_objects": [],
            "parts": [],
            "parameters": [],
            "standards": [],
            "materials": [],
            "requirements": [],
            "equipment": [],
            "laboratories": [],
            "sources": [],
            "gaps": [],
            "data_gaps": [],
            "partial_matches": {},
            "decision_history": [],
            "subgraph": {"nodes": [], "edges": []},
            "retrieval": retrieval,
            "llm": llm_client.status(),
            "diagnostics": {**kg_diagnostics, "fact_conflicts": []},
        }
    return {
        "answer": _conflict_analysis_draft(conflicts),
        "status": "ok",
        "answer_mode": "graph_conflict_analysis",
        "analytical_intent": "conflict_analysis",
        "intent": "conflict_analysis",
        "constraints": {
            "raw_question": question,
            "materials": _unique(conflict.get("material") for conflict in conflicts),
            "regimes": _unique(conflict.get("regime") for conflict in conflicts),
            "properties": _unique(conflict.get("property") for conflict in conflicts),
        },
        "facts": facts,
        "technical_objects": [],
        "parts": [],
        "parameters": [],
        "standards": [],
        "materials": [{"name": item} for item in _unique(conflict.get("material") for conflict in conflicts)],
        "requirements": [],
        "equipment": [],
        "laboratories": [],
        "sources": sources,
        "evidence": sources,
        "gaps": [],
        "data_gaps": [],
        "partial_matches": {},
        "decision_history": [],
        "subgraph": _build_subgraph_from_facts(facts, sources, []),
        "graph_context": {
            "conflict_groups_count": len(conflicts),
            "canonical_facts_count": len(facts),
            "sources_count": len(sources),
        },
        "retrieval": {**retrieval, "analytical_intent": "conflict_analysis", "answer_mode": "graph_conflict_analysis"},
        "llm": llm_client.status(),
        "diagnostics": {**kg_diagnostics, "fact_conflicts": conflicts},
    }


def _facts_for_conflicts(rows: list[dict[str, Any]], conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for conflict in conflicts:
            if (
                normalise(str(row.get("material") or "")) == normalise(str(conflict.get("material") or ""))
                and normalise(str(row.get("regime") or "")) == normalise(str(conflict.get("regime") or ""))
                and normalise(str(row.get("property") or "")) == normalise(str(conflict.get("property") or ""))
            ):
                result.append(row)
                break
    return result


def _sources_from_fact_rows(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        for evidence in row.get("evidence") or []:
            if not isinstance(evidence, dict):
                continue
            key = (evidence.get("document_id") or evidence.get("doc_id"), evidence.get("chunk_id") or evidence.get("source_chunk_id"), evidence.get("quote"))
            if key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "source_name": evidence.get("source_name"),
                    "document_id": evidence.get("document_id") or evidence.get("doc_id"),
                    "chunk_id": evidence.get("chunk_id") or evidence.get("source_chunk_id"),
                    "page": evidence.get("page"),
                    "quote": evidence.get("quote"),
                    "score": evidence.get("confidence"),
                    "evidence_type": "graph_fact",
                }
            )
            if len(result) >= limit:
                return result
    return result


def _conflict_analysis_draft(conflicts: list[dict[str, Any]]) -> str:
    parts = []
    for conflict in conflicts[:6]:
        material = str(conflict.get("material") or "материал")
        regime = str(conflict.get("regime") or "режим")
        prop = str(conflict.get("property") or "свойство")
        values = []
        for item in conflict.get("values") or []:
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            unit = item.get("unit") or item.get("unit_normalized") or ""
            if value is not None:
                values.append(f"{float(value):g} {unit}".strip())
        values_text = ", ".join(_unique(values)[:8]) or "значения расходятся"
        parts.append(f"{material} / {regime} / {prop}: {values_text}")
    return "Найдены неоднородные значения для одинаковых связок material + regime + property: " + "; ".join(parts) + "."


def _material_property_inventory_response(
    question: str,
    constraints: Any,
    retrieval: Dict[str, Any],
    kg_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """Answer broad inventory questions from canonical facts instead of exact-match constraints."""

    requested_properties = [normalise(str(item)) for item in getattr(constraints, "properties", []) or []]
    requested_regimes = [normalise(str(item)) for item in getattr(constraints, "regimes", []) or []]
    rows = []
    for row in _knowledge_report(active_only=True).get("canonical_facts") or []:
        if not isinstance(row, dict):
            continue
        if requested_properties and normalise(str(row.get("property") or "")) not in requested_properties:
            continue
        if requested_regimes and not _inventory_regime_matches(row.get("regime"), requested_regimes):
            continue
        if not row.get("material"):
            continue
        rows.append(row)
    facts = rows[:20]
    sources = _sources_from_fact_rows(facts)
    material_names = _unique(row.get("material") for row in facts)
    regimes = _unique(row.get("regime") for row in facts)
    properties = _unique(row.get("property") for row in facts)
    if not facts:
        answer = "В canonical fact layer не найдено материалов с подтверждёнными данными по заданному свойству."
        status = "partial"
    else:
        answer = (
            "Найдены материалы с подтверждёнными данными по "
            f"{', '.join(properties) if properties else 'заданному свойству'}: "
            f"{', '.join(material_names)}."
        )
        status = "ok"
    return {
        "answer": answer,
        "status": status,
        "answer_mode": "material_inventory",
        "analytical_intent": "material_inventory",
        "intent": "material_inventory",
        "constraints": {
            "raw_question": question,
            "materials": material_names,
            "regimes": regimes,
            "properties": properties or list(getattr(constraints, "properties", []) or []),
        },
        "facts": facts,
        "technical_objects": [],
        "parts": [],
        "parameters": [],
        "standards": [],
        "materials": [{"name": item} for item in material_names],
        "requirements": [],
        "equipment": [],
        "laboratories": [],
        "sources": sources,
        "evidence": sources,
        "gaps": [],
        "data_gaps": [],
        "partial_matches": {},
        "decision_history": [],
        "subgraph": _build_subgraph_from_facts(facts, sources, []),
        "graph_context": {
            "canonical_facts_count": len(facts),
            "materials_count": len(material_names),
            "sources_count": len(sources),
        },
        "retrieval": {**retrieval, "analytical_intent": "material_inventory", "answer_mode": "material_inventory"},
        "llm": llm_client.status(),
        "diagnostics": kg_diagnostics,
    }


def _inventory_regime_matches(regime: Any, requested_regimes: list[str]) -> bool:
    value = normalise(str(regime or ""))
    if not requested_regimes:
        return True
    if "термообработка" in requested_regimes:
        return value in {"отжиг", "старение", "закалка", "термообработка"}
    return value in requested_regimes


def _question_constraints(question: str) -> Dict[str, List[str]]:
    """Extract explicit constraints from the user's question.

    Retrieval may be intentionally broad, but answer synthesis must stay
    grounded in chunks that match the material/object/process/property stated
    in the question. This prevents demo facts for ВТ6/12Х18Н10Т from being
    returned for a nonexistent query such as "сплав 911 при варении".
    """
    q = question or ""
    q_lower = normalise(q)
    base = _query_terms(q)
    constraints: Dict[str, List[str]] = {
        "materials": list(base.get("materials") or []),
        "processes": list(base.get("processes") or []),
        "properties": list(base.get("properties") or []),
        "objects": [],
    }

    for match in re.finditer(
        r"\b(?:сплав(?:у|а|ом|е)?|сталь|стали|материал|alloy|steel|material)\s+(?P<name>[A-Za-zА-Яа-я0-9\-]+)\b",
        q,
        re.IGNORECASE,
    ):
        constraints["materials"].append(canonical_material_name(match.group("name")))

    for pattern in [
        r"\b(?:клапан|valve)\s*(?:DN|Ду)?\s*\d*\b",
        r"\b(?:насос|pump)\s+[A-Za-zА-Яа-я0-9\-]+\b",
        r"\bNPK-\d+\b",
        r"\bDN\s*\d+\b",
        r"\bДу\s*\d+\b",
    ]:
        for match in re.finditer(pattern, q, re.IGNORECASE):
            value = re.sub(r"\s+", "", match.group(0).strip())
            if value:
                constraints["objects"].append(value)

    # Unknown processes in constructions like "при варении". Known processes
    # are canonicalised if their stem is present.
    for match in re.finditer(r"\bпри\s+(?P<name>[A-Za-zА-Яа-я0-9\-]+)", q, re.IGNORECASE):
        token = match.group("name")
        if token and token.lower() not in {"каких", "каком", "какой"}:
            canonical = None
            for term, value in PROCESS_TERMS.items():
                if term in token.lower():
                    canonical = value
                    break
            constraints["processes"].append(canonical or token)

    # Unknown properties in phrases like "эффект на прозрачность".
    for match in re.finditer(r"\b(?:эффект\s+на|влияни[ея]\s+на)\s+(?P<name>[A-Za-zА-Яа-я0-9\-]+)", q, re.IGNORECASE):
        token = match.group("name")
        if token:
            canonical = None
            for term, value in PROPERTY_TERMS.items():
                if term in token.lower():
                    canonical = value
                    break
            constraints["properties"].append(canonical or token)

    for term, value in PROPERTY_TERMS.items():
        if term in q_lower:
            constraints["properties"].append(value)
    for term, value in PROCESS_TERMS.items():
        if term in q_lower:
            constraints["processes"].append(value)

    return {key: _unique([str(v).strip() for v in values if str(v).strip()]) for key, values in constraints.items()}


def _constraint_text(chunk: Chunk, extraction: Any) -> str:
    entity_text = " ".join(entity.canonical_name for entity in extraction.entities)
    relation_text = " ".join(f"{rel.subject} {rel.predicate} {rel.object}" for rel in getattr(extraction, "relations", []))
    experiment_text_parts: List[str] = []
    for experiment in getattr(extraction, "experiments", []):
        experiment_text_parts.extend(entity.canonical_name for entity in experiment.materials)
        experiment_text_parts.extend(regime.canonical_name for regime in experiment.regimes)
        for measurement in experiment.measurements:
            experiment_text_parts.append(measurement.property_canonical)
            if measurement.value is not None:
                experiment_text_parts.append(f"{measurement.value:g} {measurement.unit or ''}")
            if measurement.effect and measurement.effect != "unknown":
                experiment_text_parts.append(measurement.effect)
    for gap in getattr(extraction, "data_gaps", []):
        experiment_text_parts.extend([gap.material or "", gap.regime or "", gap.property or "", gap.reason])
    return normalise(" ".join([chunk.text, entity_text, relation_text, " ".join(experiment_text_parts)])).replace("ё", "е")


def _constraint_value_matches(value: str, haystack: str) -> bool:
    needle = normalise(value).replace("ё", "е")
    haystack_norm = normalise(haystack).replace("ё", "е")
    compact_needle = re.sub(r"\s+", "", needle)
    compact_haystack = re.sub(r"\s+", "", haystack_norm)
    if not needle:
        return True
    if needle in haystack_norm or compact_needle in compact_haystack:
        return True
    aliases = {
        "вт6": ["vt6", "ti-6al-4v", "ti6al4v"],
        "vt6": ["вт6", "ti-6al-4v", "ti6al4v"],
        "ti-6al-4v": ["вт6", "vt6", "ti6al4v"],
        "12х18н10т": ["12x18h10t", "aisi321", "aisi 321"],
        "aisi 321": ["12х18н10т", "12x18h10t"],
        "7075": ["7075-t6"],
        "7075-t6": ["7075"],
        "отжиг": ["annealing", "anneal", "отожжен", "отожж"],
        "annealing": ["отжиг", "отожжен", "отожж"],
        "закалка": ["quenching", "quenched", "закален"],
        "quenching": ["закалка", "закален"],
        "старение": ["aging", "aged", "старен"],
        "aging": ["старение", "старен"],
        "прочность": ["strength", "mpa", "gpa"],
        "strength": ["прочность", "mpa", "gpa"],
        "твердость": ["твёрдость", "hardness", "hv", "hrc"],
        "hardness": ["твердость", "твёрдость", "hv", "hrc"],
        "коррозионная стойкость": ["corrosion", "корроз"],
        "corrosion resistance": ["корроз", "коррозионная стойкость"],
    }
    for alias in aliases.get(needle, []):
        if alias in haystack_norm or re.sub(r"\s+", "", alias) in compact_haystack:
            return True
    return False


def _matches_all_constraints(chunk: Chunk, extraction: Any, constraints: Dict[str, List[str]]) -> bool:
    haystack = _constraint_text(chunk, extraction)
    for key in ["materials", "processes", "properties", "objects"]:
        values = constraints.get(key) or []
        if values and not any(_constraint_value_matches(value, haystack) for value in values):
            return False
    return True


def _matches_any_primary_constraint(chunk: Chunk, extraction: Any, constraints: Dict[str, List[str]]) -> bool:
    haystack = _constraint_text(chunk, extraction)
    primary = (constraints.get("materials") or []) + (constraints.get("objects") or [])
    if not primary:
        primary = (constraints.get("processes") or []) + (constraints.get("properties") or [])
    return any(_constraint_value_matches(value, haystack) for value in primary)


def _select_answer_extractions(question: str, extractions: List[tuple[Chunk, Any]]) -> tuple[List[tuple[Chunk, Any]], Dict[str, Any]]:
    constraints = _question_constraints(question)
    has_constraints = any(constraints.values())
    if not has_constraints:
        return _focused_extractions(question, extractions), {"constraints": constraints, "match_level": "none"}

    strict = [(chunk, extraction) for chunk, extraction in extractions if _matches_all_constraints(chunk, extraction, constraints)]
    if strict:
        return strict, {"constraints": constraints, "match_level": "strict"}

    partial = [(chunk, extraction) for chunk, extraction in extractions if _matches_any_primary_constraint(chunk, extraction, constraints)]
    return partial, {"constraints": constraints, "match_level": "partial" if partial else "no_match"}


def _no_match_answer(question: str, match_info: Dict[str, Any]) -> str:
    constraints = match_info.get("constraints") or {}
    parts = []
    if constraints.get("materials"):
        parts.append("материал: " + ", ".join(constraints["materials"]))
    if constraints.get("objects"):
        parts.append("объект: " + ", ".join(constraints["objects"]))
    if constraints.get("processes"):
        parts.append("режим/процесс: " + ", ".join(constraints["processes"]))
    if constraints.get("properties"):
        parts.append("свойство: " + ", ".join(constraints["properties"]))
    detail = "; ".join(parts) if parts else question
    return (
        "Я не нашёл в загруженных документах данных, которые отвечают именно на этот вопрос. "
        f"Проверяемые условия: {detail}. "
        "В такой ситуации нельзя корректно подставлять сведения из других материалов, режимов или свойств: "
        "это выглядело бы как ответ, но фактически было бы неподтверждённой подменой. "
        "Загрузите документ, где явно упоминаются эти условия, либо уточните формулировку запроса."
    )


def _source_names_by_chunk(sources: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for source in sources:
        cid = source.get("chunk_id")
        if not cid:
            continue
        title = source.get("filename") or source.get("title") or source.get("doc_id") or cid
        section = source.get("section_path")
        row_id = source.get("row_id")
        parts = [str(title)]
        if section and section != "/":
            parts.append(f"раздел {section}")
        if row_id:
            parts.append(f"строка {row_id}")
        mapping[str(cid)] = ", ".join(parts)
    return mapping


def _format_source_list(chunk_ids: List[str], sources: List[Dict[str, Any]], limit: int = 3) -> str:
    names = _source_names_by_chunk(sources)
    values = _unique([names.get(str(cid)) for cid in chunk_ids if names.get(str(cid))])[:limit]
    return "; ".join(values)


def _human_property(name: str | None) -> str:
    value = normalise(str(name or "свойство"))
    mapping = {
        "tensile strength": "предел прочности",
        "yield strength": "предел текучести",
        "elongation": "относительное удлинение",
        "strength": "прочность",
        "прочность": "прочность",
        "hardness": "твёрдость",
        "твердость": "твёрдость",
        "corrosion resistance": "коррозионная стойкость",
        "коррозионная стойкость": "коррозионная стойкость",
        "ductility": "пластичность",
        "пластичность": "пластичность",
    }
    return mapping.get(value, str(name or "свойство").lower())


def _human_regime(name: str) -> str:
    result = str(name or "")
    replacements = {
        "Annealing": "отжиг",
        "Quenching": "закалка",
        "Aging": "старение",
    }
    for src, dst in replacements.items():
        result = re.sub(rf"\b{src}\b", dst, result, flags=re.IGNORECASE)
    return result


def _human_parameter(name: str) -> str:
    value = str(name or "")
    value = re.sub(r"^Parameter:\s*", "", value, flags=re.IGNORECASE)
    value = value.replace("P:", "давление:")
    value = value.replace("T:", "температура:")
    return value


def _dedupe_measurements(parts: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for part in parts:
        key = normalise(part)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)
    return deduped


def _dedupe_parameters(values: List[str]) -> List[str]:
    values = _dedupe_measurements(values)
    named_numbers = set()
    for value in values:
        if ":" in value:
            for number in re.findall(r"\d+(?:[\.,]\d+)?\s*(?:MPa|МПа|bar|бар|C|С)?", value, flags=re.IGNORECASE):
                named_numbers.add(normalise(number))
    result = []
    for value in values:
        if ":" not in value and normalise(value) in named_numbers:
            continue
        result.append(value)
    return result


def _compact_object_names(values: List[str]) -> List[str]:
    compact = _unique_norm([str(value).strip() for value in values if str(value).strip()])
    if any("npk-200" in normalise(value) for value in compact):
        compact = [value for value in compact if normalise(value) not in {"насос", "pump"}]
    if any("dn50" in normalise(value).replace(" ", "") for value in compact):
        compact = [value for value in compact if normalise(value) not in {"клапан", "valve"}]
    return compact


def _is_display_material(name: str) -> bool:
    value = normalise(name)
    blocked = {"t173", "t7351", "t73511", "t651", "t6", "vh330", "a240", "body", "seal", "imp"}
    if value in blocked:
        return False
    if any(term in value for term in ["mill-products", "gosts", "pipe", "wire", "rod", "bar", "strip", "foil", "sheet", "plate", "tube"]):
        return False
    if value.startswith("alloy 7075-t") and value != "alloy 7075-t6":
        return False
    if re.fullmatch(r"[a-zа-я]\d{3,4}", value) and not value.startswith("aisi"):
        return False
    allowed_markers = [
        "вт6", "vt6", "ti-6al-4v", "ti6al4v", "12х18н10т", "09г2с",
        "aisi 304", "aisi 321", "7075", "7075-t6", "сталь", "сплав",
    ]
    return any(marker in value for marker in allowed_markers)


def _canonical_display_material(name: str) -> str:
    value = str(name or "").strip()
    low = normalise(value)
    if low.startswith("al7075"):
        return "7075"
    replacements = {
        "сплав ti-6al-4v": "Ti-6Al-4V",
        "стали 12х18н10т": "12Х18Н10Т",
        "сталь 12х18н10т": "12Х18Н10Т",
        "алюминиевый сплав 7075": "7075-T6",
        "alloy 7075 bar": "7075",
        "alloy 7075 sheet": "7075",
        "alloy 7075-t6": "7075-T6",
    }
    return replacements.get(low, value)


def _focused_extractions(question: str, extractions: List[tuple[Chunk, Any]]) -> List[tuple[Chunk, Any]]:
    terms = _focus_terms(question)
    if not terms:
        return extractions
    selected = []
    for chunk, extraction in extractions:
        haystack = _constraint_text(chunk, extraction)
        if any(_constraint_value_matches(term, haystack) for term in terms):
            selected.append((chunk, extraction))
    return selected or extractions


def _dedupe_dicts(items: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = tuple(item.get(k) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _entity_items(extractions: List[tuple[Chunk, Any]], entity_type: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for chunk, extraction in extractions:
        for entity in extraction.entities:
            if entity.entity_type == entity_type:
                items.append(
                    {
                        "name": entity.canonical_name,
                        "type": entity.entity_type,
                        "source_chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "section_path": chunk.section_path,
                    }
                )
    return _dedupe_dicts(items, ["name", "source_chunk_id"])


def _relation_facts(extractions: List[tuple[Chunk, Any]]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for chunk, extraction in extractions:
        for fact in _accepted_relation_facts_from_bundle(chunk, extraction):
            facts.append(fact)
        for rel in getattr(extraction, "relations", []):
            facts.append(
                {
                    "subject": rel.subject,
                    "predicate": rel.predicate,
                    "object": rel.object,
                    "qualifiers": rel.qualifiers,
                    "confidence": rel.confidence,
                    "source_chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "section_path": chunk.section_path,
                    "fact_lifecycle_status": "legacy_relation",
                }
            )
    return _dedupe_dicts(facts, ["subject", "predicate", "object", "source_chunk_id"])


def _accepted_relation_facts_from_bundle(chunk: Chunk, extraction: Any) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    if not hasattr(extraction, "experiments") and not hasattr(extraction, "data_gaps"):
        return facts

    def evidence_chunk(default_chunk: Chunk, evidence_items: list[Any]) -> str:
        if evidence_items:
            source = getattr(evidence_items[0], "source", None)
            chunk_id = getattr(source, "chunk_id", None)
            if chunk_id:
                return str(chunk_id)
        return default_chunk.chunk_id

    def add(subject: str, predicate: str, obj: str, *, confidence: float, source_chunk_id: str, qualifiers: Dict[str, Any] | None = None, fact_type: str | None = None) -> None:
        if not subject or not obj:
            return
        facts.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "qualifiers": qualifiers or {},
                "confidence": confidence,
                "source_chunk_id": source_chunk_id,
                "doc_id": chunk.doc_id,
                "section_path": chunk.section_path,
                "fact_lifecycle_status": "accepted",
                "fact_type": fact_type,
            }
        )

    for experiment in getattr(extraction, "experiments", []):
        exp_label = experiment.experiment_id or f"experiment:{chunk.chunk_id}"
        exp_chunk_id = evidence_chunk(chunk, experiment.evidence)
        for material in experiment.materials:
            add(exp_label, "STUDIES", material.canonical_name, confidence=experiment.confidence, source_chunk_id=exp_chunk_id, fact_type="ExperimentResultFact")
        for regime in experiment.regimes:
            add(exp_label, "USES_REGIME", regime.canonical_name, confidence=regime.confidence, source_chunk_id=exp_chunk_id, fact_type="ExperimentResultFact")
        for equipment_item in getattr(experiment, "equipment", []):
            add(exp_label, "USES_EQUIPMENT", equipment_item.canonical_name, confidence=equipment_item.confidence, source_chunk_id=exp_chunk_id, fact_type="ExperimentResultFact")
        for laboratory in [*getattr(experiment, "laboratories", []), *getattr(experiment, "teams", [])]:
            add(exp_label, "PERFORMED_BY", laboratory.canonical_name, confidence=laboratory.confidence, source_chunk_id=exp_chunk_id, fact_type="ExpertiseFact")
        for measurement in experiment.measurements:
            measurement_chunk_id = evidence_chunk(chunk, measurement.evidence or experiment.evidence)
            value_text = _format_measurement_value(measurement)
            measurement_label = f"{measurement.property_canonical}: {value_text}" if value_text else measurement.property_canonical
            qualifiers = {
                "value": measurement.value,
                "unit": measurement.unit,
                "direction": measurement.effect if measurement.effect != "unknown" else None,
            }
            add(exp_label, "MEASURES", measurement_label, confidence=measurement.confidence, source_chunk_id=measurement_chunk_id, qualifiers=qualifiers, fact_type="ExperimentResultFact")
            add(measurement_label, "OF_PROPERTY", measurement.property_canonical, confidence=measurement.confidence, source_chunk_id=measurement_chunk_id, qualifiers=qualifiers, fact_type="ExperimentResultFact")
            if measurement.effect and measurement.effect != "unknown":
                add(measurement_label, "HAS_CHANGE", measurement.effect, confidence=measurement.confidence, source_chunk_id=measurement_chunk_id, qualifiers=qualifiers, fact_type="ExperimentResultFact")
            for material in experiment.materials:
                add(material.canonical_name, "HAS_MEASUREMENT", measurement_label, confidence=measurement.confidence, source_chunk_id=measurement_chunk_id, qualifiers=qualifiers, fact_type="ExperimentResultFact")
    return facts


def _format_measurement_value(measurement: Any) -> str:
    if getattr(measurement, "value", None) is None:
        return str(getattr(measurement, "effect", "") or "").strip()
    unit = getattr(measurement, "unit", None)
    return f"{float(measurement.value):g} {unit or ''}".strip()


def _clean_gap_text(text: str) -> str:
    cleaned = re.sub(r"^(data[_\s]+gap|gap|пробел)\s*[:\-]\s*", "", str(text or "").strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;|")
    return cleaned

def _gap_items(extractions: List[tuple[Chunk, Any]]) -> List[Dict[str, Any]]:
    gaps = []
    for chunk, extraction in extractions:
        for gap in getattr(extraction, "data_gaps", []):
            source_chunk_id = chunk.chunk_id
            if gap.evidence:
                source = getattr(gap.evidence[0], "source", None)
                source_chunk_id = getattr(source, "chunk_id", None) or source_chunk_id
            gaps.append(
                {
                    "gap": _clean_gap_text(gap.reason),
                    "missing_for": " · ".join(item for item in [gap.material, gap.regime, gap.property] if item) or None,
                    "source_chunk_id": source_chunk_id,
                    "doc_id": chunk.doc_id,
                    "fact_lifecycle_status": "accepted",
                }
            )
        for rel in getattr(extraction, "relations", []):
            if rel.predicate == "MISSING_FOR":
                gaps.append(
                    {
                        "gap": _clean_gap_text(rel.subject),
                        "missing_for": rel.object,
                        "source_chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "fact_lifecycle_status": "legacy_relation",
                    }
                )
        for entity in extraction.entities:
            if entity.entity_type == "DataGap":
                gaps.append(
                    {
                        "gap": _clean_gap_text(entity.canonical_name),
                        "missing_for": None,
                        "source_chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "fact_lifecycle_status": "legacy_relation",
                    }
                )
    return _dedupe_dicts(gaps, ["gap", "missing_for"])


def _experiment_summaries_from_facts(question: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse relation triples into experiment-level summaries."""
    terms = _query_terms(question)
    experiments: Dict[str, Dict[str, Any]] = {}
    prop_by_value: Dict[str, str] = {}
    change_by_value: Dict[str, str] = {}

    for fact in facts:
        pred = fact.get("predicate")
        subj = str(fact.get("subject") or "")
        obj = str(fact.get("object") or "")
        if pred == "OF_PROPERTY":
            prop_by_value[subj] = obj
        elif pred == "HAS_CHANGE":
            change_by_value[subj] = str((fact.get("qualifiers") or {}).get("direction") or obj)

    for fact in facts:
        pred = fact.get("predicate")
        subj = str(fact.get("subject") or "")
        obj = str(fact.get("object") or "")
        if pred not in {"STUDIES", "USES_REGIME", "MEASURES", "USES_EQUIPMENT", "PERFORMED_BY"}:
            continue
        exp = experiments.setdefault(
            subj,
            {"experiment": subj, "materials": [], "regimes": [], "measurements": [], "equipment": [], "laboratories": [], "source_chunk_ids": []},
        )
        if fact.get("source_chunk_id"):
            exp["source_chunk_ids"].append(fact.get("source_chunk_id"))
        if pred == "STUDIES":
            exp["materials"].append(obj)
        elif pred == "USES_REGIME":
            exp["regimes"].append(obj)
        elif pred == "USES_EQUIPMENT":
            exp["equipment"].append(obj)
        elif pred == "PERFORMED_BY":
            exp["laboratories"].append(obj)
        elif pred == "MEASURES":
            qualifiers = fact.get("qualifiers") or {}
            exp["measurements"].append(
                {
                    "value": obj,
                    "property": prop_by_value.get(obj),
                    "change": change_by_value.get(obj) or qualifiers.get("direction"),
                    "unit": qualifiers.get("unit"),
                    "raw_value": qualifiers.get("value"),
                }
            )

    summaries = []
    for exp in experiments.values():
        exp["materials"] = _unique(exp["materials"])
        exp["regimes"] = _unique(exp["regimes"])
        exp["equipment"] = _unique(exp["equipment"])
        exp["laboratories"] = _unique(exp["laboratories"])
        exp["source_chunk_ids"] = _unique(exp["source_chunk_ids"])

        probe = normalise(" ".join(
            exp["materials"]
            + exp["regimes"]
            + exp["equipment"]
            + exp["laboratories"]
            + [str(m.get("value")) + " " + str(m.get("property")) + " " + str(m.get("change")) for m in exp["measurements"]]
        ))
        keep = True
        if terms["materials"]:
            keep = keep and any(_constraint_value_matches(m, probe) for m in terms["materials"])
        if terms["processes"]:
            keep = keep and any(_constraint_value_matches(p, probe) for p in terms["processes"])
        if terms["properties"]:
            keep = keep and any(_constraint_value_matches(prop, probe) for prop in terms["properties"])
        if keep:
            summaries.append(exp)
    return summaries


def _fact_source_quote(source_chunk_id: str, sources: List[Dict[str, Any]]) -> str:
    for source in sources:
        if str(source.get("chunk_id") or "") == str(source_chunk_id):
            return str(source.get("quote") or "")
    return ""


def _fact_source_filename(source_chunk_id: str, sources: List[Dict[str, Any]]) -> str:
    for source in sources:
        if str(source.get("chunk_id") or "") == str(source_chunk_id):
            return normalise(str(source.get("filename") or source.get("title") or ""))
    return ""


def _source_is_reference(source_chunk_id: str, sources: List[Dict[str, Any]]) -> bool:
    filename = _fact_source_filename(source_chunk_id, sources)
    return any(marker in filename for marker in ["equipment", "laborator", "лаборатор"])


def _looks_like_pressure_not_strength(measurement: Dict[str, Any], sources: List[Dict[str, Any]]) -> bool:
    prop = normalise(str(measurement.get("property") or ""))
    if not _constraint_value_matches("прочность", prop):
        return False
    raw_value = str(measurement.get("raw_value") or "")
    quote = _fact_source_quote(str(measurement.get("source_chunk_id") or ""), sources).lower()
    if not quote:
        return False
    pressure_markers = ["pressure", "давление", "p=", "p =", "pn", "напор"]
    return raw_value and any(marker in quote for marker in pressure_markers)


def _clean_activity_label(value: str) -> str:
    text = str(value or "").strip(" .:;\"'«»")
    low = normalise(text)
    if not text:
        return ""
    blocked = {"laboratory", "oratory", "oratories", "lab", "team", "equipment"}
    if low in blocked:
        return ""
    if len(text) < 4:
        return ""
    if any(fragment in low for fragment in ["contact sales", "availability", "ility of"]):
        return ""
    if low == "сопротивления nabertherm":
        return "Печь сопротивления Nabertherm"
    return text


def _material_activity_group(name: str) -> tuple[str, str]:
    low = normalise(name)
    if low in {"вт6", "vt6", "ti-6al-4v", "ti6al4v"}:
        return "vt6_ti64", "ВТ6 / Ti-6Al-4V"
    if low in {"7075", "7075-t6"}:
        return "al7075", "7075 / 7075-T6"
    if low in {"12х18н10т", "12x18h10t", "aisi 321", "aisi321"}:
        return "steel_12x18n10t", "12Х18Н10Т / AISI 321"
    return low, name


def _activity_property_key(label: str) -> str:
    value = normalise(label)
    if value in {"прочность", "предел прочности", "tensile strength", "strength"}:
        return "прочность"
    if value in {"твердость", "твёрдость", "hardness"}:
        return "твёрдость"
    if value in {"пластичность", "относительное удлинение", "elongation", "ductility"}:
        return "пластичность"
    return value or "свойство"


def _activity_measurement_score(measurement: Dict[str, Any], sources: List[Dict[str, Any]]) -> int:
    score = 0
    if measurement.get("change"):
        score += 10
    filename = _fact_source_filename(str(measurement.get("source_chunk_id") or ""), sources)
    if any(marker in filename for marker in ["experiment", "synthetic", "article", "heat_treatment"]):
        score += 4
    if measurement.get("raw_value"):
        score += 1
    return score


def _activity_source_rank(source_id: str, sources: List[Dict[str, Any]]) -> tuple[int, str]:
    filename = _fact_source_filename(source_id, sources)
    if any(marker in filename for marker in ["synthetic", "experiment", "heat_treatment", "article"]):
        return (0, filename)
    if any(marker in filename for marker in ["equipment", "laborator", "readme_demo"]):
        return (2, filename)
    return (1, filename)


def _activity_regime_rank(regime: str) -> tuple[int, str]:
    value = str(regime or "")
    return (0 if "(" in value else 1, normalise(value))


def _compact_regimes(regimes: List[str]) -> List[str]:
    unique = sorted(_unique_norm([r for r in regimes if r]), key=_activity_regime_rank)
    detailed_bases = {
        normalise(regime.split("(", 1)[0].strip())
        for regime in unique
        if "(" in regime
    }
    compact = []
    for regime in unique:
        base = normalise(regime.split("(", 1)[0].strip())
        if "(" not in regime and base in detailed_bases:
            continue
        compact.append(regime)
    return compact


def _material_activity_answer(question: str, facts: List[Dict[str, Any]], sources: List[Dict[str, Any]], gaps: List[Dict[str, Any]]) -> str:
    prop_by_value: Dict[str, str] = {}
    change_by_value: Dict[str, str] = {}
    experiments: Dict[str, Dict[str, Any]] = {}
    materials: Dict[str, Dict[str, Any]] = {}

    def material_bucket(name: str) -> Dict[str, Any] | None:
        canonical = _canonical_display_material(name)
        if not _is_display_material(canonical):
            return None
        return materials.setdefault(
            canonical,
            {"name": canonical, "regimes": [], "measurements": [], "equipment": [], "labs": [], "gaps": [], "sources": []},
        )

    for fact in facts:
        pred = fact.get("predicate")
        subj = str(fact.get("subject") or "")
        obj = str(fact.get("object") or "")
        if pred == "OF_PROPERTY":
            prop_by_value[subj] = obj
        elif pred == "HAS_CHANGE":
            change_by_value[subj] = str((fact.get("qualifiers") or {}).get("direction") or obj)

    for fact in facts:
        pred = fact.get("predicate")
        subj = str(fact.get("subject") or "")
        obj = str(fact.get("object") or "")
        if pred not in {"STUDIES", "USES_REGIME", "MEASURES", "USES_EQUIPMENT", "PERFORMED_BY"}:
            continue
        exp = experiments.setdefault(subj, {"materials": [], "regimes": [], "measurements": [], "equipment": [], "labs": [], "sources": []})
        if fact.get("source_chunk_id"):
            exp["sources"].append(str(fact.get("source_chunk_id")))
        if pred == "STUDIES":
            exp["materials"].append(obj)
        elif pred == "USES_REGIME":
            exp["regimes"].append(_human_regime(obj))
        elif pred == "USES_EQUIPMENT":
            exp["equipment"].append(obj)
        elif pred == "PERFORMED_BY":
            exp["labs"].append(obj)
        elif pred == "MEASURES":
            qualifiers = fact.get("qualifiers") or {}
            exp["measurements"].append(
                {
                    "value": obj,
                    "property": prop_by_value.get(obj),
                    "change": change_by_value.get(obj) or qualifiers.get("direction"),
                    "unit": qualifiers.get("unit"),
                    "raw_value": qualifiers.get("value"),
                    "source_chunk_id": fact.get("source_chunk_id"),
                }
            )

    for exp in experiments.values():
        display_exp_materials = [
            _canonical_display_material(material_name)
            for material_name in _unique(exp["materials"])
            if _is_display_material(material_name)
        ]
        source_quotes = [_fact_source_quote(cid, sources).lower() for cid in _unique(exp["sources"])]
        source_filenames = [_fact_source_filename(cid, sources) for cid in _unique(exp["sources"])]
        reference_only = any(("equipment" in name or "laborator" in name or "лаборатор" in name) for name in source_filenames)
        gap_only = any(("data gap" in quote or "нет данных" in quote) for quote in source_quotes) and not any(
            marker in quote for quote in source_quotes for marker in ["experiment_id:", "experiment:", "process_regime:", "process:"]
        )
        safe_multi_material_context = len(set(display_exp_materials)) <= 1 or any("table columns:" in quote for quote in source_quotes)
        for material_name in _unique(exp["materials"]):
            bucket = material_bucket(material_name)
            if not bucket:
                continue
            bucket["sources"].extend(exp["sources"])
            if not safe_multi_material_context:
                continue
            bucket["equipment"].extend([item for item in exp["equipment"] if _clean_activity_label(item)])
            bucket["labs"].extend([item for item in exp["labs"] if _clean_activity_label(item)])
            if reference_only or gap_only:
                continue
            bucket["regimes"].extend(exp["regimes"])
            for measurement in exp["measurements"]:
                if _looks_like_pressure_not_strength(measurement, sources):
                    continue
                bucket["measurements"].append(measurement)

    for fact in facts:
        if fact.get("predicate") != "HAS_MEASUREMENT":
            continue
        bucket = material_bucket(str(fact.get("subject") or ""))
        if not bucket:
            continue
        qualifiers = fact.get("qualifiers") or {}
        measurement = {
            "value": fact.get("object"),
            "property": prop_by_value.get(str(fact.get("object") or "")),
            "change": change_by_value.get(str(fact.get("object") or "")) or qualifiers.get("direction"),
            "unit": qualifiers.get("unit"),
            "raw_value": qualifiers.get("value"),
            "source_chunk_id": fact.get("source_chunk_id"),
        }
        if _looks_like_pressure_not_strength(measurement, sources):
            continue
        bucket["measurements"].append(measurement)
        if fact.get("source_chunk_id"):
            bucket["sources"].append(str(fact.get("source_chunk_id")))

    for gap in gaps:
        target = str(gap.get("missing_for") or "")
        text = _clean_gap_text(str(gap.get("gap") or ""))
        for bucket in materials.values():
            if (target and _constraint_value_matches(bucket["name"], normalise(target))) or _constraint_value_matches(bucket["name"], normalise(text)):
                bucket["gaps"].append(text)
                if gap.get("source_chunk_id"):
                    bucket["sources"].append(str(gap.get("source_chunk_id")))

    grouped_materials: Dict[str, Dict[str, Any]] = {}
    for original_name, item in materials.items():
        group_key, label = _material_activity_group(original_name)
        grouped = grouped_materials.setdefault(
            group_key,
            {"name": label, "regimes": [], "measurements": [], "equipment": [], "labs": [], "gaps": [], "sources": []},
        )
        for field in ["regimes", "measurements", "equipment", "labs", "gaps", "sources"]:
            grouped[field].extend(item[field])
    materials = {item["name"]: item for item in grouped_materials.values()}

    preferred_order = ["ВТ6 / Ti-6Al-4V", "12Х18Н10Т / AISI 321", "09Г2С", "AISI 304", "7075 / 7075-T6"]
    ordered_names = [name for name in preferred_order if name in materials]
    ordered_names.extend(sorted(name for name in materials if name not in ordered_names))
    focus_materials = _question_constraints(question).get("materials") or []
    if focus_materials:
        ordered_names = [
            name
            for name in ordered_names
            if any(_constraint_value_matches(focus, name) or _constraint_value_matches(name, focus) for focus in focus_materials)
        ]

    if not ordered_names:
        focus_text = ", ".join(focus_materials)
        return f"По материалу {focus_text} не удалось выделить выполненные режимы, измерения или пробелы в найденных документах." if focus_text else "В найденных документах не удалось выделить материалы и действия с ними."

    sentences = []
    for name in ordered_names[:12]:
        item = materials[name]
        regimes = _compact_regimes(item["regimes"])[:4]
        equipment = _unique([_clean_activity_label(e) for e in item["equipment"] if _clean_activity_label(e)])[:3]
        labs = _unique([_clean_activity_label(lab_value) for lab_value in item["labs"] if _clean_activity_label(lab_value)])[:3]
        gaps_text = _unique([g for g in item["gaps"] if g])[:3]

        best_by_property: Dict[str, tuple[int, Dict[str, Any]]] = {}
        for measurement in item["measurements"]:
            prop = _human_property(str(measurement.get("property") or "свойство"))
            if prop == "свойство":
                continue
            prop_key = _activity_property_key(prop)
            score = _activity_measurement_score(measurement, sources)
            current = best_by_property.get(prop_key)
            if current is None or score > current[0]:
                best_by_property[prop_key] = (score, measurement)

        measurement_parts = []
        for prop_key, (_, measurement) in sorted(best_by_property.items(), key=lambda item: item[1][0], reverse=True):
            prop = _activity_property_key(_human_property(str(measurement.get("property") or "свойство")))
            value = str(measurement.get("value") or "")
            raw_prop = str(measurement.get("property") or "")
            value = re.sub(rf"^{re.escape(raw_prop)}\s*=\s*", "", value, flags=re.IGNORECASE)
            change = measurement.get("change")
            if change == "increase":
                measurement_parts.append(f"{prop} увеличилась до {value}")
            elif change == "decrease":
                measurement_parts.append(f"{prop} снизилась до {value}")
            elif change == "unchanged":
                measurement_parts.append(f"{prop} без существенных изменений ({value})")
            else:
                measurement_parts.append(f"{prop}: {value}")
            if len(measurement_parts) >= 4:
                break

        details = []
        if regimes:
            details.append("режимы: " + ", ".join(regimes))
        if measurement_parts:
            details.append("измерения: " + "; ".join(measurement_parts))
        if equipment:
            details.append("оборудование: " + ", ".join(equipment))
        if labs:
            details.append("лаборатории: " + ", ".join(labs))
        if gaps_text:
            details.append("пробелы: " + "; ".join(gaps_text))
        if not details:
            details.append("упоминается в документах, но действий или измерений в найденных фактах нет")

        source_ids = sorted(_unique(item["sources"]), key=lambda source_id: _activity_source_rank(str(source_id), sources))
        non_reference_sources = [source_id for source_id in source_ids if not _source_is_reference(source_id, sources)]
        source_text = _format_source_list(non_reference_sources or source_ids, sources, limit=2)
        sentence = f"{name}: " + "; ".join(details) + "."
        if source_text:
            sentence += f" Источники: {source_text}."
        sentences.append(sentence)

    return " ".join(sentences)


def _answer_text(
    question: str,
    intent: str,
    technical_objects: List[Dict[str, Any]],
    parts: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
    parameters: List[Dict[str, Any]],
    standards: List[Dict[str, Any]],
    materials: List[Dict[str, Any]],
    requirements: List[Dict[str, Any]],
    equipment: List[Dict[str, Any]],
    laboratories: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    gaps: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    sources: List[Dict[str, Any]] | None = None,
) -> str:
    sources = sources or []
    q = normalise(question)
    if "клапан" in q or "valve" in q:
        technical_objects = [item for item in technical_objects if any(term in normalise(str(item.get("name"))) for term in ["клапан", "valve", "dn50"])] or technical_objects
        if intent in {"parameter_lookup", "material_lookup"}:
            valve_materials = [item for item in materials if "12х18н10т" in normalise(str(item.get("name")))]
            materials = valve_materials or materials
    if "насос" in q or "pump" in q or "npk-200" in q:
        technical_objects = [item for item in technical_objects if any(term in normalise(str(item.get("name"))) for term in ["насос", "pump", "npk-200"])] or technical_objects
        pump_materials = [item for item in materials if any(term in normalise(str(item.get("name"))) for term in ["09г2с", "aisi 304", "aisi 321"])]
        materials = pump_materials or materials

    def names(items: List[Dict[str, Any]], key: str = "name") -> List[str]:
        return _unique([str(item.get(key)) for item in items if item.get(key)])

    object_names = _compact_object_names(names(technical_objects))
    prefix = f"Найдено по запросу: {', '.join(object_names[:3])}. " if object_names else ""

    if intent == "material_activity_summary":
        return _material_activity_answer(question, facts, sources, gaps)

    if intent == "object_overview":
        parameter_values = _dedupe_parameters([_human_parameter(value) for value in names(parameters)])
        material_values = names(materials)
        standard_values = names(standards)
        part_values = names(parts)
        article_values = names(articles)
        requirement_values = names(requirements)
        image_values = names(images)
        details = []
        if material_values:
            details.append("материалы: " + ", ".join(material_values[:6]))
        if parameter_values:
            details.append("параметры: " + ", ".join(parameter_values[:8]))
        if part_values or article_values:
            detail = []
            if part_values:
                detail.append("детали: " + ", ".join(part_values[:6]))
            if article_values:
                detail.append("артикулы: " + ", ".join(article_values[:8]))
            details.append("; ".join(detail))
        if standard_values:
            details.append("стандарты: " + ", ".join(standard_values[:6]))
        if requirement_values:
            details.append("требования: " + "; ".join(requirement_values[:4]))
        if image_values:
            details.append("изображения/схемы: " + "; ".join(image_values[:4]))
        source_ids = (
            [str(item.get("source_chunk_id")) for item in technical_objects + materials + parameters + parts + articles + standards + requirements + images]
            or [str(source.get("chunk_id")) for source in sources]
        )
        source_text = _format_source_list(source_ids, sources, limit=3)
        object_text = ", ".join(object_names[:4]) if object_names else "технический объект"
        if details:
            answer = f"По запросу найден объект: {object_text}. Краткая сводка: " + "; ".join(details) + "."
        else:
            answer = f"По запросу найден объект: {object_text}, но структурированные параметры, материалы или артикулы в найденных фрагментах не выделены."
        answer += " Уточните, что именно нужно: параметры, материалы, артикулы, требования, монтаж или связанные схемы."
        return answer + (f" Источники: {source_text}." if source_text else "")

    if intent == "parameter_lookup":
        values = _dedupe_parameters([_human_parameter(value) for value in names(parameters)])
        material_values = names(materials)
        standard_values = names(standards)
        if values:
            source_text = _format_source_list([str(item.get("source_chunk_id")) for item in parameters], sources)
            answer = prefix + "По найденным фрагментам указаны параметры: " + ", ".join(values[:12]) + (
                f". Материалы: {', '.join(material_values[:6])}." if material_values else ""
            ) + (f" Стандарты: {', '.join(standard_values[:6])}." if standard_values else "")
            return answer + (f" Источник: {source_text}." if source_text else "")
    if intent == "part_article_lookup":
        part_names = names(parts)
        article_values = names(articles)
        if part_names or article_values:
            source_text = _format_source_list([str(item.get("source_chunk_id")) for item in parts + articles], sources)
            answer = prefix + f"Найдены детали: {', '.join(part_names[:8])}. Артикулы: {', '.join(article_values[:12])}."
            return answer + (f" Источник: {source_text}." if source_text else "")
    if intent == "material_inventory":
        material_values = _unique([
            _canonical_display_material(value)
            for value in names(materials)
            if _is_display_material(value)
        ])
        if material_values:
            preferred_order = ["ВТ6", "Ti-6Al-4V", "12Х18Н10Т", "09Г2С", "AISI 304", "AISI 321", "7075-T6", "7075"]
            ordered = []
            for preferred in preferred_order:
                for value in material_values:
                    if normalise(preferred) == normalise(value) and value not in ordered:
                        ordered.append(value)
            ordered.extend([value for value in material_values if value not in ordered])
            source_text = _format_source_list([str(item.get("source_chunk_id")) for item in materials], sources)
            return "В найденных документах и таблицах упоминаются материалы и сплавы: " + ", ".join(ordered[:20]) + "." + (f" Источники: {source_text}." if source_text else "")
        return "В найденных фрагментах материалы или сплавы явно не выделены."
    if intent == "material_lookup":
        material_values = names(materials)
        if material_values:
            source_text = _format_source_list([str(item.get("source_chunk_id")) for item in materials], sources)
            return prefix + "В найденных фрагментах указаны материалы: " + ", ".join(material_values[:10]) + "." + (f" Источник: {source_text}." if source_text else "")
    if intent == "standard_lookup":
        standard_values = names(standards)
        if standard_values:
            source_text = _format_source_list([str(item.get("source_chunk_id")) for item in standards], sources)
            return "В документации упоминаются стандарты: " + ", ".join(standard_values[:12]) + "." + (f" Источник: {source_text}." if source_text else "")
    if intent == "requirement_lookup":
        req_values = names(requirements)
        if req_values:
            source_text = _format_source_list([str(item.get("source_chunk_id")) for item in requirements], sources)
            return prefix + "Найдены требования: " + "; ".join(req_values[:8]) + "." + (f" Источник: {source_text}." if source_text else "")
        if gaps:
            gap_texts = _unique([_clean_gap_text(str(gap.get("gap"))) for gap in gaps if gap.get("gap")])
            if gap_texts:
                return "Явного требования не найдено. Зафиксирован пробел: " + "; ".join(gap_texts[:5]) + "."
    if intent == "equipment_lookup":
        equipment_values = names(equipment)
        if equipment_values:
            source_text = _format_source_list([str(item.get("source_chunk_id")) for item in equipment], sources)
            return "Найдено оборудование/установки: " + "; ".join(equipment_values[:10]) + "." + (f" Источник: {source_text}." if source_text else "")
        source_text = _format_source_list([str(source.get("chunk_id")) for source in sources], sources)
        return "Я не нашёл явно выделенного оборудования в найденных фрагментах." + (f" Ближайшие источники: {source_text}." if source_text else "")
    if intent == "laboratory_lookup":
        lab_values = names(laboratories)
        if lab_values:
            source_text = _format_source_list([str(item.get("source_chunk_id")) for item in laboratories], sources)
            return "Найдены лаборатории/команды: " + "; ".join(lab_values[:10]) + "." + (f" Источник: {source_text}." if source_text else "")
        source_text = _format_source_list([str(source.get("chunk_id")) for source in sources], sources)
        return "Я не нашёл явно выделенных лабораторий или команд в найденных фрагментах." + (f" Ближайшие источники: {source_text}." if source_text else "")
    if intent == "image_lookup":
        image_values = names(images)
        if image_values:
            source_text = _format_source_list([str(item.get("source_chunk_id")) for item in images], sources)
            return "Найдены связанные изображения/схемы: " + "; ".join(image_values[:8]) + "." + (f" Источник: {source_text}." if source_text else "")
        source_text = _format_source_list([str(source.get("chunk_id")) for source in sources], sources)
        return "Я не нашёл явно извлечённых изображений или схем в найденных фрагментах." + (f" Ближайшие источники: {source_text}." if source_text else "")
    if intent == "gap_analysis":
        if gaps:
            gap_texts = _unique([_clean_gap_text(str(gap.get("gap"))) for gap in gaps if gap.get("gap")])
            return "Найдены пробелы в данных: " + "; ".join(gap_texts[:8]) + "."
    if gaps and any(term in q for term in ["корроз", "corrosion"]):
        gap_texts = _unique([_clean_gap_text(str(gap.get("gap"))) for gap in gaps if gap.get("gap")])
        if gap_texts:
            return "По коррозионной стойкости найдены пробелы в данных: " + "; ".join(gap_texts[:8]) + "."
    experiment_summaries = _experiment_summaries_from_facts(question, facts)
    if experiment_summaries:
        requested_terms = _query_terms(question)
        requested_props = [normalise(p) for p in requested_terms.get("properties", [])]
        sentences = []
        for exp in experiment_summaries[:3]:
            materials_text = ", ".join(exp["materials"][:4]) if exp["materials"] else "материал не указан"
            regimes_text = ", ".join(_human_regime(regime) for regime in exp["regimes"][:3]) if exp["regimes"] else "режим не указан"
            measurement_parts = []
            seen_properties = set()
            for measurement in exp["measurements"][:4]:
                prop = measurement.get("property") or "свойство"
                if requested_props and not any(_constraint_value_matches(rp, normalise(str(prop))) for rp in requested_props):
                    continue
                prop_key = normalise(str(prop))
                if prop_key in seen_properties:
                    continue
                seen_properties.add(prop_key)
                val = measurement.get("value")
                val_text = str(val or "")
                val_text = re.sub(rf"^{re.escape(str(prop))}\s*=\s*", "", val_text, flags=re.IGNORECASE)
                prop_text = _human_property(str(prop))
                change = measurement.get("change")
                if change == "increase":
                    change_text = "увеличилась" if normalise(prop_text).endswith("ость") else "увеличилось"
                elif change == "decrease":
                    change_text = "снизилась" if normalise(prop_text).endswith("ость") else "снизилось"
                elif change == "unchanged":
                    change_text = "существенно не изменилась" if normalise(prop_text).endswith("ость") else "существенно не изменилось"
                else:
                    change_text = None
                if change_text:
                    measurement_parts.append(f"{prop_text} {change_text}; зафиксированное значение — {val_text}")
                else:
                    measurement_parts.append(f"{prop_text}: {val_text}")
            measurement_parts = _dedupe_measurements(measurement_parts)
            measurements_text = "; ".join(measurement_parts) if measurement_parts else "численных измерений в найденном фрагменте нет"
            source_text = _format_source_list(exp.get("source_chunk_ids", []), sources)
            if exp["regimes"]:
                sentence = f"По {materials_text} найдено: режим обработки — {regimes_text}. Результат: {measurements_text}."
            else:
                sentence = f"По {materials_text} найдены свойства и измерения. Режим обработки в найденном фрагменте не указан. Результат: {measurements_text}."
            if source_text:
                sentence += f" Источник: {source_text}."
            sentences.append(sentence)
        return " ".join(sentences)
    if gaps:
        gap_texts = _unique([_clean_gap_text(str(gap.get("gap"))) for gap in gaps if gap.get("gap")])
        return "Точных фактов не найдено. Возможные пробелы: " + "; ".join(gap_texts[:5]) + "."
    return "По заданному вопросу структурированные факты не найдены. Проверьте, что demo_data загружена."


def _matches_requested(value: str, requested: List[str]) -> bool:
    if not requested:
        return True
    return any(_constraint_value_matches(req, normalise(str(value or ""))) for req in requested)


def _select_relevant_experiment_summaries(question: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only experiments that answer the explicit material/process/property request.

    This prevents the API from showing every entity located in the same retrieved
    documents. Retrieval may bring broad context, but the answer graph must be
    built only from experiment summaries that satisfy the user constraints.
    """
    constraints = _question_constraints(question)
    summaries = _experiment_summaries_from_facts(question, facts)
    if not summaries:
        return []

    selected: List[Dict[str, Any]] = []
    for exp in summaries:
        materials = exp.get("materials") or []
        regimes = exp.get("regimes") or []
        measurements = exp.get("measurements") or []
        properties = [m.get("property") for m in measurements if m.get("property")]
        values = [m.get("value") for m in measurements if m.get("value")]

        ok = True
        if constraints.get("materials"):
            ok = ok and any(_matches_requested(mat, constraints["materials"]) for mat in materials)
        if constraints.get("processes"):
            ok = ok and any(_matches_requested(regime, constraints["processes"]) for regime in regimes)
        if constraints.get("properties"):
            ok = ok and (
                any(_matches_requested(prop, constraints["properties"]) for prop in properties)
                or any(_matches_requested(val, constraints["properties"]) for val in values)
            )
        if ok:
            selected.append(exp)
    return selected


def _filter_facts_to_experiments(facts: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not summaries:
        return []
    allowed_names = set()
    allowed_chunk_ids = set()
    for exp in summaries:
        allowed_names.add(str(exp.get("experiment") or ""))
        allowed_names.update(str(v) for v in exp.get("materials", []) if v)
        allowed_names.update(str(v) for v in exp.get("regimes", []) if v)
        allowed_names.update(str(v) for v in exp.get("equipment", []) if v)
        allowed_names.update(str(v) for v in exp.get("laboratories", []) if v)
        allowed_chunk_ids.update(str(v) for v in exp.get("source_chunk_ids", []) if v)
        for measurement in exp.get("measurements", []) or []:
            allowed_names.add(str(measurement.get("value") or ""))
            allowed_names.add(str(measurement.get("property") or ""))
            allowed_names.add(str(measurement.get("change") or ""))
    allowed_names = {name for name in allowed_names if name}

    kept: List[Dict[str, Any]] = []
    for fact in facts:
        cid = str(fact.get("source_chunk_id") or "")
        subj = str(fact.get("subject") or "")
        obj = str(fact.get("object") or "")
        if allowed_chunk_ids and cid not in allowed_chunk_ids:
            continue
        if subj in allowed_names or obj in allowed_names:
            kept.append(fact)
    return _dedupe_dicts(kept, ["subject", "predicate", "object", "source_chunk_id"])


def _filter_items_by_facts(items: List[Dict[str, Any]], facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not facts:
        return []
    allowed_names = set()
    allowed_chunk_ids = set()
    for fact in facts:
        allowed_names.add(str(fact.get("subject") or ""))
        allowed_names.add(str(fact.get("object") or ""))
        if fact.get("source_chunk_id"):
            allowed_chunk_ids.add(str(fact.get("source_chunk_id")))
    filtered = []
    for item in items:
        name = str(item.get("name") or item.get("value") or "")
        cid = str(item.get("source_chunk_id") or "")
        if name in allowed_names or cid in allowed_chunk_ids:
            filtered.append(item)
    return _dedupe_dicts(filtered, ["name", "source_chunk_id"])


def _source_subset_for_facts(sources_by_chunk: Dict[str, Dict[str, Any]], facts: List[Dict[str, Any]], gaps: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    ids = {str(f.get("source_chunk_id")) for f in facts if f.get("source_chunk_id")}
    for gap in gaps or []:
        if gap.get("source_chunk_id"):
            ids.add(str(gap.get("source_chunk_id")))
    return [source for cid, source in sources_by_chunk.items() if cid in ids]


def _relation_endpoint_types(predicate: str) -> tuple[str, str]:
    hints = {
        "OBJECT_HAS_PARAMETER": ("TechnicalObject", "Parameter"),
        "OBJECT_HAS_PART": ("TechnicalObject", "Part"),
        "PART_HAS_ARTICLE_NUMBER": ("Part", "ArticleNumber"),
        "OBJECT_MADE_OF_MATERIAL": ("TechnicalObject", "Material"),
        "OBJECT_COMPLIES_WITH_STANDARD": ("TechnicalObject", "Standard"),
        "REQUIREMENT_APPLIES_TO_OBJECT": ("Requirement", "TechnicalObject"),
        "STUDIES": ("Experiment", "Material"),
        "USES_REGIME": ("Experiment", "ProcessRegime"),
        "USES_EQUIPMENT": ("Experiment", "Equipment"),
        "PERFORMED_BY": ("Experiment", "Laboratory"),
        "MEASURES": ("Experiment", "PropertyValue"),
        "OF_PROPERTY": ("PropertyValue", "Property"),
        "HAS_CHANGE": ("PropertyValue", "PropertyChange"),
        "HAS_MEASUREMENT": ("Material", "PropertyValue"),
        "MISSING_FOR": ("DataGap", "Entity"),
    }
    return hints.get(predicate, ("Entity", "Entity"))


def _build_subgraph_from_facts(facts: List[Dict[str, Any]], sources: List[Dict[str, Any]], gaps: List[Dict[str, Any]] | None = None) -> Dict[str, List[Dict[str, Any]]]:
    """Build a compact answer graph from final grounded facts only."""
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Dict[str, Any]] = {}

    def nid(type_: str, label: str) -> str:
        return f"{type_}:{hashlib.sha256(str(label).encode('utf-8')).hexdigest()[:18]}"

    def add_node(type_: str, label: str, **props: Any) -> str:
        node_id = nid(type_, label)
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "label": label, "type": type_, "properties": {}}
        nodes[node_id]["properties"].update({k: v for k, v in props.items() if v is not None})
        return node_id

    def add_edge(src: str, label: str, dst: str, **props: Any) -> None:
        edge_id = _edge_id(src, label, dst)
        if edge_id not in edges:
            edges[edge_id] = {"id": edge_id, "source": src, "target": dst, "label": label, "properties": {}}
        edges[edge_id]["properties"].update({k: v for k, v in props.items() if v is not None})

    source_nodes: Dict[str, str] = {}
    for source in sources:
        cid = str(source.get("chunk_id") or "")
        if not cid:
            continue
        title = source.get("filename") or source.get("title") or source.get("doc_id") or cid
        label = str(title)
        if source.get("row_id") is not None:
            label += f" / row {source.get('row_id')}"
        sid = add_node("SourceChunk", label, **source)
        source_nodes[cid] = sid

    for fact in facts:
        predicate = str(fact.get("predicate") or "RELATED_TO")
        subj = str(fact.get("subject") or "")
        obj = str(fact.get("object") or "")
        if not subj or not obj:
            continue
        stype, otype = _relation_endpoint_types(predicate)
        sid = add_node(stype, subj)
        oid = add_node(otype, obj)
        add_edge(sid, predicate, oid, confidence=fact.get("confidence"), qualifiers=fact.get("qualifiers"))
        cid = str(fact.get("source_chunk_id") or "")
        source_id = source_nodes.get(cid)
        if source_id:
            add_edge(sid, "FACT_SUPPORTED_BY_CHUNK", source_id)
            add_edge(oid, "FACT_SUPPORTED_BY_CHUNK", source_id)

    for gap in gaps or []:
        label = _clean_gap_text(str(gap.get("gap") or ""))
        if not label:
            continue
        gid = add_node("DataGap", label, missing_for=gap.get("missing_for"))
        cid = str(gap.get("source_chunk_id") or "")
        if cid in source_nodes:
            add_edge(gid, "FACT_SUPPORTED_BY_CHUNK", source_nodes[cid])

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _material_inventory_extractions(limit_chunks: int = 80) -> List[tuple[Chunk, Any]]:
    """Collect representative indexed chunks that introduce material names."""
    selected: List[tuple[Chunk, Any]] = []
    seen_materials = set()
    for chunk in retrieval_engine.chunks:
        extraction = _extract_answer_bundle(chunk, intent="material_inventory")
        display_materials = [
            _canonical_display_material(entity.canonical_name)
            for entity in extraction.entities
            if entity.entity_type == "Material" and _is_display_material(entity.canonical_name)
        ]
        new_materials = [name for name in display_materials if normalise(name) not in seen_materials]
        if not new_materials:
            continue
        for name in new_materials:
            seen_materials.add(normalise(name))
        selected.append((chunk, extraction))
        if len(selected) >= limit_chunks:
            break
    return selected


def _activity_signal_chunk(chunk: Chunk) -> bool:
    text = chunk.text or ""
    lower = text.lower()
    filename = normalise(str(chunk.metadata.get("filename") or ""))
    if "readme_demo_scenario" in filename:
        return False
    if chunk.metadata.get("chunk_kind") != "table_row" and len(re.findall(r"\bexperiment\s*:", lower)) > 1:
        return False
    if not any(_is_display_material(entity.canonical_name) for entity in _extract_answer_bundle(chunk, intent="material_activity_summary").entities if entity.entity_type == "Material"):
        return False
    if chunk.metadata.get("chunk_kind") == "table_row":
        return True
    if "table columns:" in lower and ("property" in lower or "process" in lower or "experiment" in lower or "material" in lower):
        return True
    if "tensile strength" in lower and "yield strength" in lower:
        return True
    if any(term in lower for term in ["experiment", "эксперимент", "опыт", "property:", "result:", "conclusion:", "вывод:"]):
        return True
    if any(term in lower for term in ["отжиг", "закал", "старен", "anneal", "quench", "aging", "aged"]):
        return True
    if any(term in lower for term in ["data gap", "нет данных", "не измер"]):
        return True
    return False


def _all_index_extractions(limit_chunks: int = 600) -> List[tuple[Chunk, Any]]:
    selected: List[tuple[Chunk, Any]] = []
    for chunk in retrieval_engine.chunks[:limit_chunks]:
        if not _activity_signal_chunk(chunk):
            continue
        selected.append((chunk, _extract_answer_bundle(chunk, intent="material_activity_summary")))
    return selected


def _should_use_analytical_engine(plan: AnalyticalQueryPlan) -> bool:
    if plan.intent == AnalyticalIntent.STRICT_MATERIAL_REGIME_PROPERTY:
        return False
    if plan.intent == AnalyticalIntent.UNKNOWN:
        return False
    if plan.intent == AnalyticalIntent.GENERAL_SEARCH and not (
        plan.constraints.materials
        or plan.constraints.regimes
        or plan.constraints.properties
        or plan.constraints.topic_tags
    ):
        return False
    return True


def _analytics_evidence_sources(evidence: list[Any], limit: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = set()
    for item in evidence[:limit]:
        key = (item.document_id, item.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "doc_id": item.document_id,
                "chunk_id": item.chunk_id,
                "title": item.source_name,
                "filename": item.source_name,
                "page_start": item.page,
                "page_end": item.page,
                "section_path": item.section_path,
                "quote": item.quote[:700] + ("..." if len(item.quote) > 700 else ""),
            }
        )
    return rows


def _analytics_entity_rows(context_entities: list[dict[str, Any]], entity_type: str) -> list[dict[str, Any]]:
    return [
        {"canonical_name": item.get("name"), "entity_type": entity_type}
        for item in context_entities
        if item.get("type") == entity_type
    ]


def _experiments_for_analytics(repository: Any, plan: AnalyticalQueryPlan) -> list[Any]:
    limit = max(50, int(getattr(settings, "analytics_max_facts", 30)) * 3)
    constraints = plan.constraints

    if plan.intent == AnalyticalIntent.DECISION_HISTORY and constraints.materials:
        return repository.find_experiments(material=constraints.materials[0], limit=limit)
    if plan.intent == AnalyticalIntent.GAP_ANALYSIS:
        return []

    if plan.intent == AnalyticalIntent.MATERIAL_COMPARISON and constraints.materials:
        return _dedupe_experiments(
            [
                exp
                for material in constraints.materials
                for exp in repository.find_experiments(
                    material=material,
                    property_name=constraints.properties[0] if constraints.properties else None,
                    limit=limit,
                )
            ]
        )

    if plan.intent == AnalyticalIntent.REGIME_COMPARISON and constraints.regimes:
        return _dedupe_experiments(
            [
                exp
                for regime in constraints.regimes
                for exp in repository.find_experiments(
                    regime=regime,
                    property_name=constraints.properties[0] if constraints.properties else None,
                    limit=limit,
                )
            ]
        )

    material = constraints.materials[0] if constraints.materials else None
    regime = constraints.regimes[0] if constraints.regimes else None
    property_name = constraints.properties[0] if constraints.properties else None
    if plan.intent == AnalyticalIntent.SIMILAR_EXPERIMENTS:
        return _score_similar_experiments(repository.find_experiments(limit=limit), plan)
    return repository.find_experiments(
        material=material,
        regime=regime,
        property_name=property_name,
        limit=limit,
    )


def _dedupe_experiments(experiments: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for exp in experiments:
        exp_id = getattr(exp, "experiment_id", None)
        if exp_id in seen:
            continue
        seen.add(exp_id)
        result.append(exp)
    return result


def _score_similar_experiments(experiments: list[Any], plan: AnalyticalQueryPlan) -> list[Any]:
    constraints = plan.constraints
    scored: list[tuple[float, Any]] = []
    for exp in experiments:
        score = 0.0
        if constraints.materials and any(item in exp.materials for item in constraints.materials):
            score += 0.35
        if constraints.regimes and any(item in exp.regimes for item in constraints.regimes):
            score += 0.25
        if constraints.properties and any(
            measurement.property_name in constraints.properties for measurement in exp.measurements
        ):
            score += 0.20
        if constraints.equipment and any(item in exp.equipment for item in constraints.equipment):
            score += 0.10
        if constraints.laboratories and any(item in exp.laboratories for item in constraints.laboratories):
            score += 0.05
        if score > 0:
            scored.append((score, exp))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [exp for _, exp in scored]


def _analytics_response(
    question: str,
    top_k: int,
    plan: AnalyticalQueryPlan,
    repository: Any,
    retrieval_meta: Dict[str, Any],
    kg_diagnostics: Dict[str, Any],
    answer_synthesis_mode: str | None = None,
) -> Dict[str, Any]:
    experiments = _experiments_for_analytics(repository, plan)
    gaps = repository.find_gaps(
        material=plan.constraints.materials[0] if plan.constraints.materials else None,
        regime=plan.constraints.regimes[0] if plan.constraints.regimes else None,
        property_name=plan.constraints.properties[0] if plan.constraints.properties else None,
    )
    history = []
    if plan.intent == AnalyticalIntent.DECISION_HISTORY and plan.constraints.materials:
        history = repository.get_decision_history(plan.constraints.materials[0])

    evidence_search = EvidenceSearch(retrieval_engine=retrieval_engine, catalog=catalog)
    evidence = evidence_search.search(question, plan.constraints, limit=max(top_k, 8))
    evidence = evidence_reranker.rerank(evidence, plan.constraints)
    context = graph_context_builder.from_experiments(
        plan=plan,
        experiments=experiments,
        gaps=gaps,
        evidence=evidence,
        decision_history=history,
    )

    if plan.intent == AnalyticalIntent.SIMILAR_EXPERIMENTS:
        _attach_similarity_scores(context, experiments, plan)

    if not context.sources and evidence:
        context.sources = _analytics_evidence_sources(evidence, limit=getattr(settings, "analytics_max_sources", 12))

    synthesis_mode = answer_synthesis_mode or getattr(settings, "answer_synthesis_mode", "template")
    draft_answer = AnswerSynthesizer(mode=synthesis_mode).synthesize(plan, context)
    llm_answer = None
    if synthesis_mode in {"hybrid", "llm"}:
        llm_answer = llm_client.synthesize_answer(
            question=question,
            intent=plan.intent.value,
            answer_draft=draft_answer,
            facts=context.facts,
            sources=context.sources,
            gaps=context.gaps,
        )
    answer = llm_answer or draft_answer
    analytics_diag = analytical_diagnostics(
        plan=plan,
        context=context,
        evidence_backend=evidence_search.last_backend,
        synthesis_mode=synthesis_mode,
    )
    status = "ok" if context.facts or context.gaps or context.decision_history else "partial"
    if not (context.facts or context.gaps or context.decision_history or context.evidence):
        status = "no_exact_match"

    retrieval = {
        **retrieval_meta,
        "analytical_intent": plan.intent.value,
        "answer_mode": plan.answer_mode,
        "evidence_backend": evidence_search.last_backend,
    }
    diagnostics = {**kg_diagnostics, **analytics_diag, "llm_answer_polished": bool(llm_answer)}
    return {
        "answer": answer,
        "status": status,
        "answer_mode": plan.answer_mode,
        "analytical_intent": plan.intent.value,
        "intent": plan.intent.value,
        "constraints": plan.constraints.model_dump(),
        "facts": context.facts,
        "experiments": [item.summary() for item in experiments[:10] if hasattr(item, "summary")],
        "technical_objects": [],
        "parts": [],
        "parameters": [],
        "standards": [],
        "materials": _analytics_entity_rows(context.entities, "Material"),
        "requirements": [],
        "equipment": _analytics_entity_rows(context.entities, "Equipment"),
        "laboratories": _analytics_entity_rows(context.entities, "Laboratory"),
        "sources": context.sources,
        "evidence": [item.model_dump() for item in evidence[: getattr(settings, "analytics_max_sources", 12)]],
        "gaps": context.gaps,
        "data_gaps": context.gaps,
        "partial_matches": context.partial_matches,
        "decision_history": context.decision_history,
        "subgraph": context.subgraph,
        "graph_context": context.stats(),
        "retrieval": retrieval,
        "llm": llm_client.status(),
        "diagnostics": diagnostics,
    }


def _attach_similarity_scores(context: Any, experiments: list[Any], plan: AnalyticalQueryPlan) -> None:
    scores: dict[str, float] = {}
    for exp in experiments:
        score = 0.0
        if plan.constraints.materials and any(item in exp.materials for item in plan.constraints.materials):
            score += 0.35
        if plan.constraints.regimes and any(item in exp.regimes for item in plan.constraints.regimes):
            score += 0.25
        if plan.constraints.properties and any(
            measurement.property_name in plan.constraints.properties for measurement in exp.measurements
        ):
            score += 0.20
        if exp.equipment:
            score += 0.10
        if exp.laboratories:
            score += 0.05
        scores[exp.experiment_id] = min(score, 1.0)
    for row in context.facts:
        exp_id = row.get("experiment_id")
        if exp_id in scores:
            row["similarity_score"] = scores[exp_id]


def _should_use_typed_fact_path(constraints: Any, ask_intent: str) -> bool:
    target_fact_types = list(getattr(constraints, "target_fact_types", []) or [])
    answer_mode = str(getattr(constraints, "answer_mode", "") or "")
    if not target_fact_types or not answer_mode:
        return False
    if getattr(constraints, "require_exact_match", False):
        return False
    if ask_intent in {
        "conflict_analysis",
        "gap_analysis",
        "material_inventory",
        "material_activity_summary",
        "object_overview",
        "parameter_lookup",
        "part_article_lookup",
        "standard_lookup",
        "requirement_lookup",
        "image_lookup",
    }:
        return False
    if answer_mode == "generic_typed_fact_summary":
        return False
    return answer_mode in {
        "technology_solution_search",
        "process_parameter_search",
        "experiment_catalog_search",
        "technology_comparison",
        "domestic_vs_foreign_practice",
        "expert_search",
        "knowledge_gap_search",
        "literature_review",
    }


def _decorate_ask_response(
    payload: Dict[str, Any],
    *,
    preset_id: RuntimePresetId | str | None,
    input_source: str,
    query_params_ignored: bool,
) -> Dict[str, Any]:
    """Attach runtime preset metadata without changing the answer contract."""
    preset = get_runtime_preset(preset_id)
    diagnostics = payload.get("diagnostics") or {}
    retrieval = payload.get("retrieval") or {}
    active_backend = retrieval.get("kg_backend_active") or diagnostics.get("kg_backend_active")
    neo4j_available = bool(retrieval.get("neo4j_available", diagnostics.get("neo4j_available", False)))
    runtime_diag = preset_diagnostics(
        preset,
        active_backend=active_backend,
        neo4j_available=neo4j_available,
        input_source=input_source,
        query_params_ignored=query_params_ignored,
    )
    payload["diagnostics"] = {**diagnostics, **runtime_diag}
    if isinstance(retrieval, dict):
        payload["retrieval"] = {
            **retrieval,
            "preset_id": preset.preset_id.value,
            "preset_title": preset.title,
            "effective_runtime_mode": runtime_diag["effective_runtime_mode"],
        }
    repairer = None
    if preset.answer_synthesis_mode in {"hybrid", "llm"} and not preset.strict_audit_mode:
        repairer = getattr(llm_client, "repair_grounded_answer", None)
    return enhance_answer_payload(payload, preset.preset_id, llm_repairer=repairer)


@app.post("/ask")
async def ask(
    request: AskRequest | None = Body(default=None),
    question: str | None = Query(default=None),
    top_k: int = Query(default=8, ge=1, le=50),
    preset_id: RuntimePresetId | None = Query(default=None),
):
    """Return a structured answer. Supports legacy query params and JSON body."""
    query_params_ignored = False
    if request is not None:
        query_params_ignored = bool(question or preset_id or top_k != 8)
        return await _ask_impl(
            question=request.question,
            top_k=request.top_k,
            preset_id=request.preset_id,
            input_source="json_body",
            query_params_ignored=query_params_ignored,
        )
    if not question:
        raise HTTPException(status_code=422, detail="question is required")
    return await _ask_impl(
        question=question,
        top_k=top_k,
        preset_id=preset_id,
        input_source="query_params",
        query_params_ignored=False,
    )


async def _ask_impl(
    question: str,
    top_k: int = 8,
    preset_id: RuntimePresetId | str | None = None,
    input_source: str = "internal",
    query_params_ignored: bool = False,
) -> Dict[str, Any]:
    """Return a structured technical-document answer with facts, sources, gaps and local graph."""
    preset = get_runtime_preset(preset_id)
    understanding = _question_understanding(question)
    planned_constraints = query_planner.parse(question)
    if understanding["needs_clarification"]:
        if _can_try_source_grounded_for_unclear(question, understanding):
            fallback = _source_grounded_fallback_from_query(
                question=question,
                top_k=top_k,
                planned_constraints=planned_constraints,
                kg_diagnostics={**_kg_backend_diagnostics(_graph_db_for_repository()), **_extraction_diagnostics()},
                ask_intent="source_grounded_answer",
                fallback_reason="retrieval_hits_for_unclear_domain_query",
            )
            if fallback is not None:
                return _decorate_ask_response(
                    fallback,
                    preset_id=preset.preset_id,
                    input_source=input_source,
                    query_params_ignored=query_params_ignored,
                )
        return _decorate_ask_response(
            _clarification_response(question, str(understanding.get("reason") or "unclear"), {"query_understanding": understanding}),
            preset_id=preset.preset_id,
            input_source=input_source,
            query_params_ignored=query_params_ignored,
        )

    repository_graph = _graph_db_for_repository(preset.kg_backend)
    try:
        graph_repository = GraphRepositoryFactory.create(
            catalog=catalog,
            extractor=extractor,
            graph_db=repository_graph,
            document_getter=_get_document_meta,
            configured_backend=preset.kg_backend,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={**_kg_backend_diagnostics(repository_graph, configured_backend=preset.kg_backend), **_extraction_diagnostics(), "error": str(exc)},
        ) from exc
    kg_diagnostics = {
        **_kg_backend_diagnostics(repository_graph, repository_backend_name(graph_repository), configured_backend=preset.kg_backend),
        **_extraction_diagnostics(),
    }
    graph_retriever = GraphRetriever(graph_repository)
    graph_retrieval_meta = {
        **retrieval_engine.stats(),
        "constraint_layer": "ontology_graph",
        "graph_repository": graph_repository.__class__.__name__,
        **kg_diagnostics,
        "query_constraints": planned_constraints.model_dump(),
    }
    analytics_plan = analytical_router.build_plan(question, planned_constraints)
    ask_intent = _intent(question)

    if planned_constraints.intent == QueryIntent.MATERIAL_REGIME_PROPERTY_EFFECT and planned_constraints.require_exact_match:
        graph_result = graph_retriever.material_regime_property(planned_constraints)
        if graph_result.exact:
            return _decorate_ask_response(answer_builder.exact_match_response(
                constraints=planned_constraints,
                facts=graph_result.exact,
                partial_matches=graph_result.partial_matches,
                gaps=graph_result.gaps,
                retrieval=graph_retrieval_meta,
                llm=llm_client.status(),
            ), preset_id=preset.preset_id, input_source=input_source, query_params_ignored=query_params_ignored)
        return _decorate_ask_response(answer_builder.no_match_response(
            constraints=planned_constraints,
            partial_matches=graph_result.partial_matches,
            gaps=graph_result.gaps,
            retrieval=graph_retrieval_meta,
            llm=llm_client.status(),
        ), preset_id=preset.preset_id, input_source=input_source, query_params_ignored=query_params_ignored)

    if ask_intent == "material_inventory" and planned_constraints.properties:
        return _decorate_ask_response(
            _material_property_inventory_response(question, planned_constraints, graph_retrieval_meta, kg_diagnostics),
            preset_id=preset.preset_id,
            input_source=input_source,
            query_params_ignored=query_params_ignored,
        )

    if _should_use_typed_fact_path(planned_constraints, ask_intent):
        typed_result = TypedFactRetriever(graph_repository, retrieval_engine=retrieval_engine).search(
            TypedFactQuery.from_constraints(question, planned_constraints),
            top_k=max(top_k, 12),
        )
        return _decorate_ask_response(
            build_typed_answer_payload(
                question=question,
                result=typed_result,
                retrieval=graph_retrieval_meta,
                kg_diagnostics=kg_diagnostics,
                llm=llm_client.status(),
            ),
            preset_id=preset.preset_id,
            input_source=input_source,
            query_params_ignored=query_params_ignored,
        )

    if _should_use_analytical_engine(analytics_plan):
        return _decorate_ask_response(_analytics_response(
            question=question,
            top_k=top_k,
            plan=analytics_plan,
            repository=graph_repository,
            retrieval_meta=graph_retrieval_meta,
            kg_diagnostics=kg_diagnostics,
            answer_synthesis_mode=preset.answer_synthesis_mode,
        ), preset_id=preset.preset_id, input_source=input_source, query_params_ignored=query_params_ignored)

    if planned_constraints.intent == QueryIntent.DECISION_HISTORY and planned_constraints.materials:
        history = graph_repository.get_decision_history(planned_constraints.materials[0])
        return _decorate_ask_response(answer_builder.decision_history_response(
            constraints=planned_constraints,
            history=history,
            retrieval=graph_retrieval_meta,
            llm=llm_client.status(),
        ), preset_id=preset.preset_id, input_source=input_source, query_params_ignored=query_params_ignored)

    if planned_constraints.intent == QueryIntent.GAP_ANALYSIS:
        gaps = graph_repository.find_gaps(
            material=planned_constraints.materials[0] if planned_constraints.materials else None,
            regime=planned_constraints.regimes[0] if planned_constraints.regimes else None,
            property_name=planned_constraints.properties[0] if planned_constraints.properties else None,
        )
        return _decorate_ask_response(answer_builder.gap_response(
            constraints=planned_constraints,
            gaps=gaps,
            retrieval=graph_retrieval_meta,
            llm=llm_client.status(),
        ), preset_id=preset.preset_id, input_source=input_source, query_params_ignored=query_params_ignored)

    if ask_intent == "conflict_analysis":
        return _decorate_ask_response(
            _conflict_analysis_response(question, graph_retrieval_meta, kg_diagnostics),
            preset_id=preset.preset_id,
            input_source=input_source,
            query_params_ignored=query_params_ignored,
        )
    rewrite = llm_client.rewrite_question_for_retrieval(question)
    retrieval_question = str((rewrite or {}).get("search_query") or question)
    extractions: List[tuple[Chunk, Any]] = []
    sources_by_chunk: Dict[str, Dict[str, Any]] = {}
    source_metadata_filter: Dict[str, Any] = {"applied": False, "reason": "not_used"}

    if ask_intent in {"material_inventory", "material_activity_summary"}:
        extractions = _material_inventory_extractions() if ask_intent == "material_inventory" else _all_index_extractions()
        sources_by_chunk = {chunk.chunk_id: _source_for_chunk(chunk) for chunk, _ in extractions}
    else:
        retrieval_top_k = max(top_k * 3, top_k) if (planned_constraints.geographies or planned_constraints.time_filters) else top_k
        candidate_chunks = retrieval_engine.query(retrieval_question, top_k=retrieval_top_k)
        candidate_chunks, source_metadata_filter = rerank_chunks_by_source_metadata(candidate_chunks, planned_constraints)
        candidate_chunks = candidate_chunks[:top_k]
        if not candidate_chunks:
            return _decorate_ask_response(_clarification_response(
                question,
                "no_retrieval_hits",
                {"query_understanding": understanding, "query_rewrite": rewrite},
            ), preset_id=preset.preset_id, input_source=input_source, query_params_ignored=query_params_ignored)
        for chunk in candidate_chunks:
            extraction = _extract_answer_bundle(chunk, intent=ask_intent)
            extractions.append((chunk, extraction))
            sources_by_chunk[chunk.chunk_id] = _source_for_chunk(chunk)

    if ask_intent in {"material_inventory", "material_activity_summary"}:
        answer_extractions = _focused_extractions(question, extractions) if ask_intent == "material_activity_summary" else extractions
        match_info = {"match_level": "inventory", "constraints": {}}
    else:
        answer_extractions, match_info = _select_answer_extractions(question, extractions)
    focused_chunk_ids = {chunk.chunk_id for chunk, _ in answer_extractions}
    sources_by_chunk = {cid: source for cid, source in sources_by_chunk.items() if cid in focused_chunk_ids} or sources_by_chunk

    # If explicit constraints were present but no chunk matched all of them,
    # return a grounded negative answer instead of stitching unrelated facts.
    # Partial evidence can be shown as sources only; it must not be converted
    # into positive facts for another material/process/property.
    if match_info.get("match_level") in {"partial", "no_match"} and any((match_info.get("constraints") or {}).values()):
        partial_sources = list(sources_by_chunk.values()) if match_info.get("match_level") == "partial" else []
        legacy_constraints = match_info.get("constraints") or {}
        negative_draft = _no_match_answer(question, match_info)
        gaps = [
            {
                "gap": negative_draft,
                "missing_for": legacy_constraints,
                "source_chunk_id": None,
                "doc_id": None,
            }
        ]
        llm_answer = None
        if preset.answer_synthesis_mode in {"hybrid", "llm"} and not preset.strict_audit_mode:
            llm_answer = llm_client.synthesize_answer(
                question=question,
                intent=ask_intent,
                answer_draft=negative_draft,
                facts=[],
                sources=partial_sources,
                gaps=gaps,
            )
        return _decorate_ask_response({
            "answer": llm_answer or negative_draft,
            "status": "no_exact_match",
            "answer_mode": "llm_grounded_negative" if llm_answer else "no_grounded_match",
            "intent": ask_intent,
            "constraints": planned_constraints.model_dump(),
            "facts": [],
            "technical_objects": [],
            "parts": [],
            "parameters": [],
            "standards": [],
            "materials": [],
            "requirements": [],
            "equipment": [],
            "laboratories": [],
            "sources": partial_sources,
            "gaps": gaps,
            "data_gaps": gaps,
            "partial_matches": {},
            "decision_history": [],
            "subgraph": _build_local_subgraph(answer_extractions) if partial_sources else {"nodes": [], "edges": []},
            "retrieval": {**retrieval_engine.stats(), **kg_diagnostics, "constraint_match": match_info, "query_rewrite": rewrite, "source_metadata_filter": source_metadata_filter},
            "llm": llm_client.status(),
            "diagnostics": kg_diagnostics,
        }, preset_id=preset.preset_id, input_source=input_source, query_params_ignored=query_params_ignored)

    facts = _relation_facts(answer_extractions)

    # For experiment-style questions, do not expose every entity from every
    # retrieved chunk. First collapse triples into experiments that match the
    # explicit material/process/property constraints, then keep only triples
    # connected to those experiments. This is the main guard against noisy,
    # demo-looking answers and unreadable graphs.
    relevant_experiments: List[Dict[str, Any]] = []
    if ask_intent == "experiment_lookup":
        relevant_experiments = _select_relevant_experiment_summaries(question, facts)
        if relevant_experiments:
            facts = _filter_facts_to_experiments(facts, relevant_experiments)
    accepted_facts = [fact for fact in facts if str(fact.get("fact_lifecycle_status") or "") == "accepted"]

    technical_objects = _entity_items(answer_extractions, "TechnicalObject")
    parts = _entity_items(answer_extractions, "Part")
    articles = _entity_items(answer_extractions, "ArticleNumber")
    parameters = _entity_items(answer_extractions, "Parameter")
    standards = _entity_items(answer_extractions, "Standard")
    materials = _entity_items(answer_extractions, "Material")
    requirements = _entity_items(answer_extractions, "Requirement")
    equipment = _entity_items(answer_extractions, "Equipment")
    laboratories = _entity_items(answer_extractions, "Laboratory")
    images = _entity_items(answer_extractions, "ImageArtifact")
    gaps = _gap_items(answer_extractions)

    if facts:
        technical_objects = _filter_items_by_facts(technical_objects, facts)
        parts = _filter_items_by_facts(parts, facts)
        articles = _filter_items_by_facts(articles, facts)
        parameters = _filter_items_by_facts(parameters, facts)
        standards = _filter_items_by_facts(standards, facts)
        if ask_intent != "material_inventory":
            materials = _filter_items_by_facts(materials, facts)
        requirements = _filter_items_by_facts(requirements, facts)
        equipment = _filter_items_by_facts(equipment, facts)
        laboratories = _filter_items_by_facts(laboratories, facts)
        images = _filter_items_by_facts(images, facts)

    if ask_intent == "gap_analysis" and not gaps:
        gaps = [{"gap": "В найденных фрагментах нет явного указания на отсутствие данных", "missing_for": question, "source_chunk_id": None, "doc_id": None}]

    sources = _source_subset_for_facts(sources_by_chunk, facts, gaps)
    if not sources and answer_extractions:
        # Keep at least one evidence source for exact constrained answers.
        sources = [_source_for_chunk(answer_extractions[0][0])]
    if not accepted_facts and sources and ask_intent == "experiment_lookup":
        return _decorate_ask_response(
            _source_grounded_payload(
                question=question,
                sources=sources,
                planned_constraints=planned_constraints,
                retrieval={
                    **retrieval_engine.stats(),
                    **kg_diagnostics,
                    "constraint_match": match_info,
                    "query_rewrite": rewrite,
                    "source_metadata_filter": source_metadata_filter,
                },
                kg_diagnostics=kg_diagnostics,
                llm=llm_client.status(),
                ask_intent=ask_intent,
                fallback_reason="retrieved_chunks_without_accepted_facts",
            ),
            preset_id=preset.preset_id,
            input_source=input_source,
            query_params_ignored=query_params_ignored,
        )

    rule_based_answer = _answer_text(
        question,
        ask_intent,
        technical_objects,
        parts,
        articles,
        parameters,
        standards,
        materials,
        requirements,
        equipment,
        laboratories,
        images,
        gaps,
        facts,
        sources,
    )
    # Recomputed after `sources` exists so rule-based output can cite actual files/rows.
    llm_answer = None
    if preset.answer_synthesis_mode in {"hybrid", "llm"} and not preset.strict_audit_mode:
        llm_answer = llm_client.synthesize_answer(
            question=question,
            intent=ask_intent,
            answer_draft=rule_based_answer,
            facts=facts,
            sources=sources,
            gaps=gaps,
        )
    answer = llm_answer or rule_based_answer

    # Use the compact answer graph built from the final facts, not from all
    # retrieved/extracted chunks. The full debug extraction graph is useful for
    # developers but too noisy for the main answer.
    subgraph = _build_subgraph_from_facts(facts, sources, gaps)

    return _decorate_ask_response({
        "answer": answer,
        "status": "ok",
        "answer_mode": "llm_grounded" if llm_answer else "rule_based",
        "intent": ask_intent,
        "constraints": planned_constraints.model_dump(),
        "facts": facts,
        "experiments": relevant_experiments,
        "technical_objects": technical_objects,
        "parts": parts + articles,
        "parameters": parameters,
        "standards": standards,
        "materials": materials,
        "requirements": requirements,
        "equipment": equipment,
        "laboratories": laboratories,
        "sources": sources,
        "gaps": gaps,
        "data_gaps": gaps,
        "partial_matches": {},
        "decision_history": [],
        "subgraph": subgraph,
        "retrieval": {**retrieval_engine.stats(), **kg_diagnostics, "constraint_match": match_info, "query_rewrite": rewrite, "source_metadata_filter": source_metadata_filter},
        "llm": llm_client.status(),
        "diagnostics": kg_diagnostics,
    }, preset_id=preset.preset_id, input_source=input_source, query_params_ignored=query_params_ignored)

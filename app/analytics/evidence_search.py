"""Supporting evidence search for analytical GraphRAG answers."""

from __future__ import annotations

import re
from typing import Any

from ..domain.query_constraints import QueryConstraints
from ..models.schemas import Chunk
from ..retrieval.retrieval import RetrievalEngine
from ..storage.catalog import SQLiteCatalog
from .query_models import EvidenceItem


TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


class EvidenceSearch:
    """Search document chunks as evidence without creating structured facts."""

    def __init__(
        self,
        retrieval_engine: RetrievalEngine | None = None,
        catalog: SQLiteCatalog | None = None,
    ) -> None:
        self.retrieval_engine = retrieval_engine
        self.catalog = catalog
        self.last_backend = "none"

    def search(
        self,
        query: str,
        constraints: QueryConstraints,
        limit: int = 10,
    ) -> list[EvidenceItem]:
        """Return evidence chunks relevant to the query and constraints."""
        if self.retrieval_engine is not None:
            chunks = self.retrieval_engine.query(query, top_k=max(limit, 1))
            if chunks:
                self.last_backend = _retrieval_backend_name(self.retrieval_engine)
                return [_chunk_to_evidence(chunk, score=1.0 / (idx + 1), backend=self.last_backend) for idx, chunk in enumerate(chunks)]

        if self.catalog is not None:
            chunks = self._sqlite_search(query, constraints, limit=limit)
            if chunks:
                self.last_backend = "sqlite_chunk_search"
                return [_chunk_to_evidence(chunk, score=1.0 / (idx + 1), backend=self.last_backend) for idx, chunk in enumerate(chunks)]

        self.last_backend = "none"
        return []

    def _sqlite_search(
        self,
        query: str,
        constraints: QueryConstraints,
        limit: int,
    ) -> list[Chunk]:
        if self.catalog is None:
            return []
        query_terms = set(_tokens(query))
        constraint_terms = set()
        for value in constraints.materials + constraints.regimes + constraints.properties:
            constraint_terms.update(_tokens(value))
        terms = query_terms | constraint_terms
        scored: list[tuple[float, Chunk]] = []
        for chunk in self.catalog.list_chunks():
            text_terms = set(_tokens(chunk.text))
            if not text_terms:
                continue
            overlap = len(terms & text_terms)
            if overlap <= 0:
                continue
            scored.append((float(overlap), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]


def _tokens(text: str) -> list[str]:
    return [item.lower().replace("ё", "е") for item in TOKEN_RE.findall(text or "")]


def _retrieval_backend_name(engine: RetrievalEngine) -> str:
    stats: dict[str, Any] = engine.stats()
    if stats.get("qdrant_ready"):
        return "hybrid_qdrant"
    if stats.get("local_embeddings_ready"):
        return "hybrid_local_embeddings"
    return str(stats.get("effective_retrieval_mode") or stats.get("retrieval_mode") or "bm25")


def _chunk_to_evidence(chunk: Chunk, score: float, backend: str) -> EvidenceItem:
    metadata = chunk.metadata or {}
    return EvidenceItem(
        source_name=str(metadata.get("source_name") or metadata.get("filename") or chunk.doc_id),
        document_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        page=chunk.page_start,
        section_path=chunk.section_path,
        quote=(chunk.text or "").strip(),
        score=score,
        retrieval_backend=backend,
        metadata={
            "chunk_kind": metadata.get("chunk_kind"),
            "table_id": metadata.get("table_id"),
            "row_id": metadata.get("row_id"),
            "source_metadata": metadata.get("source_metadata") or {},
            "publication_year": metadata.get("publication_year"),
            "geographies": metadata.get("geographies") or [],
            "practice_scope": metadata.get("practice_scope"),
            "reliability_level": metadata.get("reliability_level"),
        },
    )

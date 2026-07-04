"""
Hybrid retrieval implementation.

The preferred production path is BM25S + dense embeddings in Qdrant +
RRF fusion. The module also contains a dependency-free in-memory
fallback. This is intentional: the hackathon demo must start even when
Qdrant, sentence-transformers or bm25s have not been installed yet.
"""

from __future__ import annotations

import math
import re
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..config import settings
from ..models.schemas import Chunk

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore


TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


QUERY_EXPANSION_TERMS = {
    "dn50": ["DN50", "DN 50", "Ду 50", "условный проход 50", "номинальный диаметр 50"],
    "dn": ["DN", "Ду", "условный проход", "номинальный диаметр"],
    "pn": ["PN", "номинальное давление", "pressure rating", "давление"],
    "клапан": ["клапан", "valve", "арматура"],
    "valve": ["valve", "клапан", "арматура"],
    "насос": ["насос", "pump", "агрегат"],
    "pump": ["pump", "насос", "агрегат"],
    "артикул": ["артикул", "part number", "article", "обозначение", "код"],
    "стандарт": ["стандарт", "ГОСТ", "ISO", "ASTM", "EN", "standard"],
    "материал": ["материал", "material", "сталь", "сплав", "корпус"],
    "прочность": ["прочность", "strength", "ultimate strength", "yield strength", "MPa"],
    "прочк": ["прочность", "прочке", "strength", "ultimate strength", "yield strength", "MPa"],
    "твердость": ["твёрдость", "твердость", "hardness", "HV", "HRC"],
    "корроз": ["коррозионная стойкость", "corrosion resistance", "corrosion", "нет данных"],
    "вт6": ["ВТ6", "VT6", "Ti-6Al-4V"],
    "vt6": ["VT6", "ВТ6", "Ti-6Al-4V"],
    "12х18н10т": ["12Х18Н10Т", "12X18H10T", "AISI 321"],
    "7075": ["7075", "7075-T6", "aluminum alloy 7075"],
}


def expand_query(query: str) -> str:
    """Add bilingual/technical synonyms to improve BM25 and dense retrieval.

    This is deliberately lightweight and deterministic; it makes the demo
    robust even when embeddings are disabled or a local model is unavailable.
    """
    if not getattr(settings, "retrieval_query_expansion", True):
        return query
    q_norm = (query or "").lower().replace("ё", "е")
    additions: List[str] = []
    for trigger, terms in QUERY_EXPANSION_TERMS.items():
        if trigger in q_norm:
            additions.extend(terms)
    # Normalize common compact forms.
    if re.search(r"\bду\s*50\b", q_norm):
        additions.extend(["DN50", "DN 50"])
    if re.search(r"\bdn\s*50\b", q_norm):
        additions.extend(["Ду 50", "DN50"])
    deduped = list(dict.fromkeys(term for term in additions if term and term.lower() not in q_norm))
    return (query + " " + " ".join(deduped)).strip() if deduped else query


def tokenize(text: str) -> List[str]:
    """Simple multilingual tokeniser suitable for BM25 fallback."""
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class SimpleBM25:
    """Small BM25 implementation used when external BM25S is absent."""

    def __init__(self, documents: Iterable[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs_tokens = [tokenize(doc) for doc in documents]
        self.doc_lens = [len(toks) for toks in self.docs_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.docs_tokens]
        df: Dict[str, int] = defaultdict(int)
        for tokens in self.docs_tokens:
            for token in set(tokens):
                df[token] += 1
        self.df = dict(df)
        self.n_docs = len(self.docs_tokens)

    def get_scores(self, query: str) -> List[float]:
        if not self.docs_tokens:
            return []
        query_terms = tokenize(query)
        scores: List[float] = []
        for tf, dl in zip(self.term_freqs, self.doc_lens):
            score = 0.0
            for term in query_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                score += idf * f * (self.k1 + 1) / denom
            scores.append(score)
        return scores


class PersistentVectorCache:
    """Small SQLite cache for local chunk embeddings.

    The cache is intentionally local and optional.  It lives under DATA_DIR and
    is never a source of truth: if it is unavailable, dense retrieval simply
    rebuilds missing vectors or degrades to BM25.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load_vectors(
        self,
        chunks: list[Chunk],
        *,
        model_id: str,
        dimensions: int,
    ) -> tuple[dict[str, list[float]], dict[str, int]]:
        stats = {"vectors_total": len(chunks), "vectors_cached": 0, "vectors_missing": 0, "vectors_stale": 0}
        if not chunks:
            return {}, stats
        try:
            self._ensure_schema()
            vectors: dict[str, list[float]] = {}
            with sqlite3.connect(self.path) as conn:
                for chunk in chunks:
                    text_hash = _chunk_text_hash(chunk)
                    key = _vector_cache_key(text_hash, model_id, dimensions)
                    row = conn.execute(
                        "SELECT vector_json FROM vectors WHERE cache_key = ?",
                        (key,),
                    ).fetchone()
                    if row:
                        try:
                            vector = json.loads(str(row[0]))
                            if isinstance(vector, list) and len(vector) == dimensions:
                                vectors[chunk.chunk_id] = [float(item) for item in vector]
                                stats["vectors_cached"] += 1
                                continue
                        except Exception:
                            pass
                    stale = conn.execute(
                        """
                        SELECT 1 FROM vectors
                        WHERE text_hash = ? AND (model_id != ? OR dimensions != ?)
                        LIMIT 1
                        """,
                        (text_hash, model_id, dimensions),
                    ).fetchone()
                    if stale:
                        stats["vectors_stale"] += 1
                    stats["vectors_missing"] += 1
            return vectors, stats
        except Exception:
            stats["vectors_missing"] = len(chunks)
            return {}, stats

    def store_vectors(
        self,
        rows: list[tuple[Chunk, list[float]]],
        *,
        model_id: str,
        dimensions: int,
    ) -> None:
        if not rows:
            return
        self._ensure_schema()
        payload = []
        for chunk, vector in rows:
            text_hash = _chunk_text_hash(chunk)
            payload.append(
                (
                    _vector_cache_key(text_hash, model_id, dimensions),
                    text_hash,
                    model_id,
                    dimensions,
                    json.dumps([float(item) for item in vector], separators=(",", ":")),
                )
            )
        with sqlite3.connect(self.path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO vectors(cache_key, text_hash, model_id, dimensions, vector_json, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                payload,
            )

    def inspect_vectors(self, chunks: list[Chunk], *, model_id: str) -> dict[str, int]:
        stats = {"vectors_total": len(chunks), "vectors_cached": 0, "vectors_missing": 0, "vectors_stale": 0}
        if not chunks:
            return stats
        try:
            self._ensure_schema()
            with sqlite3.connect(self.path) as conn:
                for chunk in chunks:
                    text_hash = _chunk_text_hash(chunk)
                    row = conn.execute(
                        "SELECT 1 FROM vectors WHERE text_hash = ? AND model_id = ? LIMIT 1",
                        (text_hash, model_id),
                    ).fetchone()
                    if row:
                        stats["vectors_cached"] += 1
                        continue
                    stale = conn.execute(
                        "SELECT 1 FROM vectors WHERE text_hash = ? LIMIT 1",
                        (text_hash,),
                    ).fetchone()
                    if stale:
                        stats["vectors_stale"] += 1
                    stats["vectors_missing"] += 1
            return stats
        except Exception:
            stats["vectors_missing"] = len(chunks)
            return stats

    def _ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    cache_key TEXT PRIMARY KEY,
                    text_hash TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_text_hash ON vectors(text_hash)")


class RetrievalEngine:
    """Hybrid retrieval over a corpus of document chunks."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._bm25 = SimpleBM25([])
        self._embedding_model = None
        self._qdrant = None
        self._qdrant_ready = False
        self._local_embeddings: Dict[str, List[float]] = {}
        self._local_embeddings_ready = False
        self._local_embedding_index_status = "not_started"
        self._last_embedding_error = ""
        self._last_qdrant_error = ""
        self._last_dense_candidates = 0
        self._last_lexical_candidates = 0
        self._last_fused_candidates = 0
        self._last_returned_chunks = 0
        self._last_fused_scores: Dict[str, float] = {}
        self._last_selected_debug: list[dict[str, Any]] = []
        self._vector_cache = PersistentVectorCache(Path(settings.data_dir) / "cache" / "embedding_vectors.sqlite3")
        self._vector_cache_stats: dict[str, int] = {
            "vectors_total": 0,
            "vectors_cached": 0,
            "vectors_missing": 0,
            "vectors_stale": 0,
        }

    @property
    def chunks(self) -> List[Chunk]:
        return self._chunks

    def _ensure_embeddings(self) -> bool:
        """Load embedding model if available. Return False on missing deps."""
        if SentenceTransformer is None:
            self._last_embedding_error = "dependency missing: sentence-transformers is not installed"
            return False
        if self._embedding_model is None:
            target = _embedding_model_load_target()
            model_path = _embedding_model_path()
            if model_path and not model_path.exists() and _is_real_sentence_transformer_class(SentenceTransformer):
                self._last_embedding_error = (
                    f"model path missing: {model_path}. "
                    "Mount local model to /models/... or allow build-time download."
                )
                return False
            try:
                kwargs: dict[str, Any] = {}
                local_only = _embedding_local_files_only()
                if local_only:
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    kwargs["local_files_only"] = True
                self._embedding_model = SentenceTransformer(target, **kwargs)
            except TypeError:
                try:
                    self._embedding_model = SentenceTransformer(target)
                except Exception as exc:
                    self._embedding_model = None
                    self._last_embedding_error = _embedding_load_error(target, exc)
                    return False
            except Exception as exc:
                self._embedding_model = None
                self._last_embedding_error = _embedding_load_error(target, exc)
                return False
        self._last_embedding_error = ""
        return True

    def _ensure_qdrant(self) -> bool:
        """Connect to Qdrant if available and reachable.

        The connection check is intentionally performed before loading
        an embedding model. Loading BGE-M3 can be expensive and may fail
        on machines without internet/cache; if Qdrant itself is not
        reachable there is no reason to load the model.
        """
        if self._qdrant_ready:
            return True
        if QdrantClient is None or qmodels is None:
            return False
        try:
            self._qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=3.0)
            collections = self._qdrant.get_collections().collections
        except Exception:
            self._qdrant = None
            self._qdrant_ready = False
            self._last_qdrant_error = "qdrant_unreachable"
            return False

        if not self._ensure_embeddings():
            return False

        try:
            dim = _embedding_dimension_from_model(self._embedding_model)
            names = {c.name for c in collections}
            if settings.qdrant_collection not in names:
                self._qdrant.create_collection(
                    collection_name=settings.qdrant_collection,
                    vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
                )
            self._qdrant_ready = True
        except Exception:
            self._qdrant_ready = False
            self._last_qdrant_error = "qdrant_collection_setup_failed"
        return self._qdrant_ready

    def _ensure_payload_indexes(self) -> None:
        """Create Qdrant payload indexes used by filters.

        This is best-effort because older qdrant-client/server versions
        differ slightly in enum names and APIs. Retrieval still works if
        payload indexes cannot be created.
        """
        if not self._qdrant or qmodels is None:
            return
        fields = ["workspace_uid", "doc_id", "section_path", "embedding_version"]
        for field in fields:
            try:
                self._qdrant.create_payload_index(
                    collection_name=settings.qdrant_collection,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                continue

    @staticmethod
    def _to_vector_list(vector) -> List[float]:
        """Convert numpy/torch/list vector to a plain Python list."""
        if hasattr(vector, "tolist"):
            return [float(x) for x in vector.tolist()]
        return [float(x) for x in vector]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def build_local_embedding_index(
        self,
        chunks: List[Chunk],
        *,
        batch_size: int = 32,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Build/cache local dense vectors for active chunks.

        The persistent cache key is text_hash + model identifier + dimensions.
        Dense retrieval remains optional: failures are reported as diagnostics
        and callers can continue with BM25.
        """
        if not getattr(settings, "enable_local_embeddings", False):
            self._local_embedding_index_status = "disabled"
            self._last_embedding_error = "disabled by config"
            return {
                "ok": False,
                "vectors_total": len(chunks),
                "vectors_created": 0,
                "vectors_cached": 0,
                "vectors_missing": len(chunks),
                "vectors_stale": 0,
                "model_id": _embedding_model_identifier(),
                "model_path": str(_embedding_model_path() or ""),
                "dimensions": None,
                "error": self._last_embedding_error,
            }
        if not chunks:
            self._local_embedding_index_status = "empty_corpus"
            self._vector_cache_stats = {"vectors_total": 0, "vectors_cached": 0, "vectors_missing": 0, "vectors_stale": 0}
            return {
                "ok": False,
                "vectors_total": 0,
                "vectors_created": 0,
                "vectors_cached": 0,
                "vectors_missing": 0,
                "vectors_stale": 0,
                "model_id": _embedding_model_identifier(),
                "model_path": str(_embedding_model_path() or ""),
                "dimensions": None,
                "error": "empty corpus",
            }
        if not self._ensure_embeddings():
            self._local_embedding_index_status = "failed"
            self._vector_cache_stats = {
                "vectors_total": len(chunks),
                "vectors_cached": len(self._local_embeddings),
                "vectors_missing": max(0, len(chunks) - len(self._local_embeddings)),
                "vectors_stale": 0,
            }
            return {
                "ok": False,
                **self._vector_cache_stats,
                "vectors_created": 0,
                "model_id": _embedding_model_identifier(),
                "model_path": str(_embedding_model_path() or ""),
                "dimensions": None,
                "error": self._last_embedding_error,
            }
        try:
            self._local_embedding_index_status = "building"
            dimensions = _embedding_dimension_from_model(self._embedding_model)
            model_id = _embedding_model_identifier()
            cached, cache_stats = self._vector_cache.load_vectors(chunks, model_id=model_id, dimensions=dimensions)
            self._local_embeddings.update(cached)
            missing_chunks = [chunk for chunk in chunks if chunk.chunk_id not in self._local_embeddings]
            created = 0
            if missing_chunks:
                total = len(chunks)
                done = len(chunks) - len(missing_chunks)
                for start in range(0, len(missing_chunks), max(1, batch_size)):
                    batch = missing_chunks[start : start + max(1, batch_size)]
                    texts = [chunk.text for chunk in batch]
                    vectors = self._embedding_model.encode(
                        texts,
                        batch_size=max(1, batch_size),
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                    if len(vectors) == 0:
                        self._local_embedding_index_status = "failed"
                        self._last_embedding_error = "indexing failed: embedding model returned no vectors"
                        self._vector_cache_stats = {**cache_stats, "vectors_missing": len(missing_chunks) - created}
                        return {
                            "ok": False,
                            **self._vector_cache_stats,
                            "vectors_created": created,
                            "model_id": model_id,
                            "model_path": str(_embedding_model_path() or ""),
                            "dimensions": dimensions,
                            "error": self._last_embedding_error,
                        }
                    new_rows: list[tuple[Chunk, list[float]]] = []
                    for chunk, vector in zip(batch, vectors):
                        vector_list = self._to_vector_list(vector)
                        self._local_embeddings[chunk.chunk_id] = vector_list
                        new_rows.append((chunk, vector_list))
                    self._vector_cache.store_vectors(new_rows, model_id=model_id, dimensions=dimensions)
                    created += len(new_rows)
                    done += len(new_rows)
                    if progress_callback:
                        progress_callback(
                            {
                                "done": done,
                                "total": total,
                                "vectors_created": created,
                                "vectors_cached": cache_stats.get("vectors_cached", 0),
                            }
                        )
            missing_after = sum(1 for chunk in chunks if chunk.chunk_id not in self._local_embeddings)
            self._vector_cache_stats = {
                "vectors_total": len(chunks),
                "vectors_cached": cache_stats.get("vectors_cached", 0),
                "vectors_missing": missing_after,
                "vectors_stale": cache_stats.get("vectors_stale", 0),
            }
            self._local_embeddings_ready = bool(self._local_embeddings)
            self._local_embedding_index_status = "ready" if self._local_embeddings_ready else "failed"
            return {
                "ok": self._local_embeddings_ready,
                **self._vector_cache_stats,
                "vectors_created": created,
                "model_id": model_id,
                "model_path": str(_embedding_model_path() or ""),
                "dimensions": dimensions,
                "error": "" if self._local_embeddings_ready else "no vectors built",
            }
        except Exception as exc:
            self._local_embeddings_ready = False
            self._local_embedding_index_status = "failed"
            self._last_embedding_error = f"indexing failed: {type(exc).__name__}"
            self._vector_cache_stats = {
                "vectors_total": len(chunks),
                "vectors_cached": len(self._local_embeddings),
                "vectors_missing": max(0, len(chunks) - len(self._local_embeddings)),
                "vectors_stale": 0,
            }
            return {
                "ok": False,
                **self._vector_cache_stats,
                "vectors_created": 0,
                "model_id": _embedding_model_identifier(),
                "model_path": str(_embedding_model_path() or ""),
                "dimensions": self._embedding_dimension(),
                "error": self._last_embedding_error,
            }

    def project_chunks_to_local_embeddings(self, chunks: List[Chunk]) -> bool:
        """Build an in-memory dense index when Qdrant is unavailable.

        This gives a realistic embeddings mode for hackathon demos on a single
        machine. It is optional and best-effort: if sentence-transformers is not
        installed or the model is not cached, retrieval silently falls back to
        BM25.
        """
        result = self.build_local_embedding_index(chunks)
        return bool(result.get("ok"))

    def local_dense_retrieve(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        if not getattr(settings, "enable_local_embeddings", False):
            self._last_dense_candidates = 0
            self._last_embedding_error = "disabled by config"
            return []
        if self._chunks and len(self._local_embeddings) < len({chunk.chunk_id for chunk in self._chunks}):
            self.project_chunks_to_local_embeddings(self._chunks)
        if not self._local_embeddings or not self._ensure_embeddings():
            self._last_dense_candidates = 0
            return []
        try:
            q_vector = self._embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            q_list = self._to_vector_list(q_vector)
            scored = [(chunk_id, self._cosine(q_list, vector)) for chunk_id, vector in self._local_embeddings.items()]
            scored.sort(key=lambda item: item[1], reverse=True)
            result = scored[:top_k]
            self._last_dense_candidates = len(result)
            return result
        except Exception as exc:
            self._last_dense_candidates = 0
            self._last_embedding_error = f"dense query failed: {type(exc).__name__}"
            return []

    def project_chunks_to_qdrant(self, chunks: List[Chunk]) -> bool:
        """Project chunks to Qdrant without changing the local BM25 index.

        This is used by the outbox processor. It keeps the canonical
        write-path separate from the retrieval projection and avoids
        duplicating chunks in the in-memory lexical index.
        """
        if not chunks or not self._ensure_qdrant():
            return False
        try:
            texts = [chunk.text for chunk in chunks]
            embeddings = self._embedding_model.encode(texts, batch_size=32, convert_to_numpy=True)
            points = []
            for chunk, vector in zip(chunks, embeddings):
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "workspace_uid": chunk.workspace_uid,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_path": chunk.section_path,
                    "ordinal": chunk.ordinal,
                    "text_hash": chunk.text_hash,
                    "embedding_version": chunk.embedding_version,
                    "text": chunk.text[:2000],
                }
                points.append(qmodels.PointStruct(id=chunk.chunk_id, vector=vector.tolist(), payload=payload))
            self._ensure_payload_indexes()
            self._qdrant.upsert(collection_name=settings.qdrant_collection, points=points)
            return True
        except Exception:
            self._qdrant_ready = False
            self._last_qdrant_error = "qdrant_projection_failed"
            return False

    def index_chunks(self, chunks: List[Chunk], replace_doc_id: str | None = None) -> None:
        """Index chunks for lexical retrieval.

        If `replace_doc_id` is supplied, old chunks of the same document
        are removed first. This makes re-ingestion idempotent and avoids
        duplicated lexical hits during demos.
        """
        if not chunks:
            return
        if replace_doc_id:
            removed_ids = {chunk.chunk_id for chunk in self._chunks if chunk.doc_id == replace_doc_id}
            self._chunks = [chunk for chunk in self._chunks if chunk.doc_id != replace_doc_id]
            for chunk_id in removed_ids:
                self._local_embeddings.pop(chunk_id, None)
            self._local_embeddings_ready = bool(self._local_embeddings)
        self._chunks.extend(chunks)
        self._bm25 = SimpleBM25([chunk.text for chunk in self._chunks])
        if getattr(settings, "enable_local_embeddings", False):
            inspected = self._vector_cache.inspect_vectors(self._chunks, model_id=_embedding_model_identifier())
            self._vector_cache_stats = inspected
        else:
            self._vector_cache_stats["vectors_total"] = len(self._chunks)
            self._vector_cache_stats["vectors_cached"] = len(self._local_embeddings)
            self._vector_cache_stats["vectors_missing"] = max(0, len(self._chunks) - len(self._local_embeddings))

        if settings.direct_qdrant_projection:
            # Best-effort direct projection for demo mode. The canonical
            # production path is still the outbox processor.
            self.project_chunks_to_qdrant(chunks)
        if getattr(settings, "eager_local_embeddings", False):
            # Keep this opt-in. Building sentence-transformers vectors during
            # API import can block Docker startup while a model is downloaded.
            self.project_chunks_to_local_embeddings(chunks)

    def lexical_retrieve(self, query: str, top_k: int = 30) -> List[Tuple[int, float]]:
        scores = self._bm25.get_scores(query)
        idx_scores = [(idx, score) for idx, score in enumerate(scores) if score > 0]
        idx_scores.sort(key=lambda x: x[1], reverse=True)
        result = idx_scores[:top_k]
        self._last_lexical_candidates = len(result)
        return result

    def dense_retrieve(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        if getattr(settings, "enable_local_embeddings", False):
            return self.local_dense_retrieve(query, top_k=top_k)
        self._last_dense_candidates = 0
        self._last_embedding_error = "disabled by config"
        if not getattr(settings, "direct_qdrant_projection", False):
            return []
        if not self._ensure_qdrant():
            return self.local_dense_retrieve(query, top_k=top_k)
        try:
            q_emb = self._embedding_model.encode([query], convert_to_numpy=True)[0]
            # qdrant-client changed the high-level API over time. Support
            # both old `.search` and new `.query_points` interfaces.
            if hasattr(self._qdrant, "search"):
                result = self._qdrant.search(
                    collection_name=settings.qdrant_collection,
                    query_vector=q_emb.tolist(),
                    limit=top_k,
                )
                hits = result
            else:
                result = self._qdrant.query_points(
                    collection_name=settings.qdrant_collection,
                    query=q_emb.tolist(),
                    limit=top_k,
                )
                hits = getattr(result, "points", result)
            result = [(hit.payload["chunk_id"], float(hit.score)) for hit in hits if hit.payload]
            self._last_dense_candidates = len(result)
            return result
        except Exception as exc:
            self._last_dense_candidates = 0
            self._last_embedding_error = f"dense query failed: {type(exc).__name__}"
            return []

    def stats(self) -> Dict[str, object]:
        mode = (settings.retrieval_mode or "bm25").lower()
        dependency_available = SentenceTransformer is not None
        local_enabled = bool(getattr(settings, "enable_local_embeddings", False))
        cached_vectors_ready = bool(
            local_enabled
            and self._chunks
            and int(self._vector_cache_stats.get("vectors_cached", 0)) >= len({chunk.chunk_id for chunk in self._chunks})
        )
        model_path = _embedding_model_path()
        local_dense_capable = bool(
            local_enabled
            and dependency_available
            and (model_path is None or model_path.exists())
            and not self._last_embedding_error
        )
        dense_enabled = bool(self._qdrant_ready or self._local_embeddings_ready or cached_vectors_ready or local_dense_capable)
        degraded_reason = self._hybrid_degraded_reason(mode, dependency_available, local_enabled, dense_enabled)
        effective_mode = "hybrid_degraded_to_bm25" if mode == "hybrid" and degraded_reason else mode
        return {
            "chunks": len(self._chunks),
            "retrieval_mode": mode,
            "effective_retrieval_mode": effective_mode,
            "bm25_ready": bool(self._chunks),
            "qdrant_ready": self._qdrant_ready,
            "qdrant_last_error": self._last_qdrant_error,
            "embedding_dependency_available": dependency_available,
            "embedding_model_loaded": self._embedding_model is not None,
            "embedding_model": settings.embedding_model,
            "embedding_model_path": str(_embedding_model_path() or ""),
            "embedding_model_path_exists": bool(_embedding_model_path() and _embedding_model_path().exists()),
            "embedding_offline_mode": _embedding_offline_mode(),
            "embedding_dimension": self._embedding_dimension(),
            "embedding_last_error": self._last_embedding_error,
            "direct_qdrant_projection": settings.direct_qdrant_projection,
            "local_embeddings_enabled": local_enabled,
            "eager_local_embeddings": getattr(settings, "eager_local_embeddings", False),
            "local_embeddings_ready": self._local_embeddings_ready,
            "local_embedding_index_status": self._local_embedding_index_status,
            "local_embedding_vectors": len(self._local_embeddings),
            "embedding_status": {
                "embedding_model_id": settings.embedding_model,
                "embedding_model_path": str(_embedding_model_path() or ""),
                "offline_mode": _embedding_offline_mode(),
                "dimensions": self._embedding_dimension(),
                "vectors_total": int(self._vector_cache_stats.get("vectors_total", len(self._chunks))),
                "vectors_cached": int(self._vector_cache_stats.get("vectors_cached", 0)),
                "vectors_missing": int(
                    self._vector_cache_stats.get("vectors_missing", max(0, len(self._chunks) - len(self._local_embeddings)))
                ),
                "vectors_stale": int(self._vector_cache_stats.get("vectors_stale", 0)),
                "cache_path": str(self._vector_cache.path),
            },
            "vectors_total": int(self._vector_cache_stats.get("vectors_total", len(self._chunks))),
            "vectors_cached": int(self._vector_cache_stats.get("vectors_cached", 0)),
            "vectors_missing": int(self._vector_cache_stats.get("vectors_missing", max(0, len(self._chunks) - len(self._local_embeddings)))),
            "vectors_stale": int(self._vector_cache_stats.get("vectors_stale", 0)),
            "hybrid_dense_enabled": dense_enabled,
            "hybrid_degraded_reason": degraded_reason,
            "chunks_found_bm25": self._last_lexical_candidates,
            "chunks_found_dense": self._last_dense_candidates,
            "chunks_after_fusion": self._last_fused_candidates,
            "chunks_returned": self._last_returned_chunks,
            "last_dense_candidates": self._last_dense_candidates,
            "top_fused_chunks": self._last_selected_debug,
            "query_expansion": getattr(settings, "retrieval_query_expansion", True),
        }

    def _hybrid_degraded_reason(
        self,
        mode: str,
        dependency_available: bool,
        local_enabled: bool,
        dense_enabled: bool,
    ) -> str:
        if mode != "hybrid" or dense_enabled:
            return ""
        if not local_enabled and not getattr(settings, "direct_qdrant_projection", False):
            return "disabled by config"
        if local_enabled and not dependency_available:
            return "dependency missing"
        if self._local_embedding_index_status == "building":
            return "indexing in progress"
        if self._last_embedding_error:
            return self._last_embedding_error
        if self._last_qdrant_error:
            return self._last_qdrant_error
        return "dense retrieval not ready"

    def _embedding_dimension(self) -> int | None:
        if self._local_embeddings:
            first = next(iter(self._local_embeddings.values()), None)
            if isinstance(first, list):
                return len(first)
        if self._embedding_model is not None:
            try:
                return _embedding_dimension_from_model(self._embedding_model)
            except Exception:
                return None
        return None

    def _rrf_fusion(
        self,
        lexical: List[Tuple[int, float]],
        dense: List[Tuple[str, float]],
        k: int = 60,
        lambda_weight: float = 60.0,
    ) -> List[str]:
        scores: Dict[str, float] = {}
        for rank, (idx, _) in enumerate(lexical, start=1):
            if 0 <= idx < len(self._chunks):
                cid = self._chunks[idx].chunk_id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (lambda_weight + rank)
        for rank, (cid, _) in enumerate(dense, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (lambda_weight + rank)
        self._last_fused_scores = dict(scores)
        return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:k]

    def query(self, question: str, top_k: int = 10) -> List[Chunk]:
        """Hybrid retrieval with document collapse.

        If dense/Qdrant is unavailable, this still returns lexical BM25
        results. At most one chunk per document is returned to increase
        diversity.
        """
        if not self._chunks:
            return []
        candidate_k = max(50, top_k * 5)
        search_query = expand_query(question)
        mode = (settings.retrieval_mode or "hybrid").lower()
        lex = [] if mode == "embedding" else self.lexical_retrieve(search_query, top_k=candidate_k)
        dens = [] if mode == "bm25" else self.dense_retrieve(search_query, top_k=candidate_k)
        if mode in {"embedding", "hybrid"} and not dens and not lex:
            lex = self.lexical_retrieve(search_query, top_k=candidate_k)
        if mode == "embedding" and not dens:
            lex = self.lexical_retrieve(search_query, top_k=candidate_k)
        fused_ids = self._rrf_fusion(lex, dens, k=candidate_k)
        self._last_fused_candidates = len(fused_ids)
        id_to_chunk = {chunk.chunk_id: chunk for chunk in self._chunks}
        selected: List[Chunk] = []
        # Keep diversity, but do not collapse a table document to a single row.
        # In technical docs one CSV/XLSX may contain several independent facts;
        # strict one-chunk-per-doc collapse can select the wrong row and hide the
        # exact answer.  Two or three chunks per document is a safer compromise.
        per_doc_counts: Dict[str, int] = defaultdict(int)
        max_chunks_per_doc = 3
        for cid in fused_ids:
            chunk = id_to_chunk.get(cid)
            if chunk is None or per_doc_counts[chunk.doc_id] >= max_chunks_per_doc:
                continue
            selected.append(chunk)
            per_doc_counts[chunk.doc_id] += 1
            if len(selected) >= top_k:
                break

        # If the corpus currently contains only one or two documents, strict
        # collapse would return too few chunks. Fill remaining slots with the
        # best still-unselected chunks while preserving the fused order.
        selected_ids = {chunk.chunk_id for chunk in selected}
        for cid in fused_ids:
            if len(selected) >= top_k:
                break
            chunk = id_to_chunk.get(cid)
            if chunk is None or chunk.chunk_id in selected_ids:
                continue
            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
        self._last_returned_chunks = len(selected)
        self._last_selected_debug = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_name": (chunk.metadata or {}).get("source_name") or (chunk.metadata or {}).get("filename") or chunk.doc_id,
                "page": chunk.page_start,
                "section_path": chunk.section_path,
                "score": round(float(self._last_fused_scores.get(chunk.chunk_id, 0.0)), 6),
            }
            for chunk in selected[:10]
        ]
        return selected


def _chunk_text_hash(chunk: Chunk) -> str:
    value = str(chunk.text_hash or "").strip()
    if value:
        return value
    return hashlib.sha256((chunk.text or "").encode("utf-8")).hexdigest()


def _embedding_model_path() -> Path | None:
    value = str(getattr(settings, "embedding_model_path", "") or os.getenv("EMBEDDING_MODEL_PATH", "") or "").strip()
    return Path(value) if value else None


def _embedding_model_load_target() -> str:
    path = _embedding_model_path()
    return str(path) if path else str(settings.embedding_model)


def _embedding_model_identifier() -> str:
    path = _embedding_model_path()
    return str(path) if path else str(settings.embedding_model)


def _embedding_offline_mode() -> bool:
    return _bool_env("TRANSFORMERS_OFFLINE", False) or _bool_env("HF_HUB_OFFLINE", False)


def _embedding_local_files_only() -> bool:
    return bool(_embedding_model_path()) or _embedding_offline_mode() or not _bool_env("EMBEDDINGS_ALLOW_DOWNLOAD", False)


def _is_real_sentence_transformer_class(cls: Any) -> bool:
    module = str(getattr(cls, "__module__", ""))
    return module.startswith("sentence_transformers")


def _embedding_dimension_from_model(model: Any) -> int:
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def _embedding_load_error(target: str, exc: BaseException) -> str:
    detail = str(exc).splitlines()[0][:220] if str(exc) else type(exc).__name__
    action = " Mount local model to /models/... or allow build-time download."
    return f"model load failed: {target}: {type(exc).__name__}: {detail}.{action}"


def _vector_cache_key(text_hash: str, model_id: str, dimensions: int) -> str:
    return hashlib.sha256(f"{text_hash}|{model_id}|{dimensions}".encode("utf-8")).hexdigest()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

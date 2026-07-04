from __future__ import annotations

from app.models.schemas import Chunk
from app.retrieval.retrieval import RetrievalEngine


def _chunk(chunk_id: str, text: str, doc_id: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id or f"doc_{chunk_id}",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="test",
        embedding_version="test-model",
    )


def _retrieval_env(monkeypatch, *, mode: str = "bm25", local_embeddings: bool = False) -> None:
    import app.retrieval.retrieval as retrieval

    monkeypatch.setattr(retrieval.settings, "retrieval_mode", mode, raising=False)
    monkeypatch.setattr(retrieval.settings, "enable_local_embeddings", local_embeddings, raising=False)
    monkeypatch.setattr(retrieval.settings, "eager_local_embeddings", False, raising=False)
    monkeypatch.setattr(retrieval.settings, "direct_qdrant_projection", False, raising=False)
    monkeypatch.setattr(retrieval.settings, "retrieval_query_expansion", False, raising=False)
    monkeypatch.setattr(
        retrieval.settings,
        "embedding_model",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        raising=False,
    )


def test_bm25_mode_does_not_require_sentence_transformers(monkeypatch) -> None:
    import app.retrieval.retrieval as retrieval

    _retrieval_env(monkeypatch, mode="bm25", local_embeddings=False)
    monkeypatch.setattr(retrieval, "SentenceTransformer", None)
    engine = RetrievalEngine()
    engine.index_chunks([_chunk("c1", "прочность после старения материала")])

    result = engine.query("прочность", top_k=3)
    stats = engine.stats()

    assert [item.chunk_id for item in result] == ["c1"]
    assert stats["embedding_dependency_available"] is False
    assert stats["local_embeddings_ready"] is False
    assert stats["hybrid_degraded_reason"] == ""


def test_hybrid_missing_sentence_transformers_degrades_to_bm25(monkeypatch) -> None:
    import app.retrieval.retrieval as retrieval

    _retrieval_env(monkeypatch, mode="hybrid", local_embeddings=True)
    monkeypatch.setattr(retrieval, "SentenceTransformer", None)
    engine = RetrievalEngine()
    engine.index_chunks([_chunk("c1", "прочность после старения материала")])

    result = engine.query("прочность", top_k=3)
    stats = engine.stats()

    assert [item.chunk_id for item in result] == ["c1"]
    assert stats["effective_retrieval_mode"] == "hybrid_degraded_to_bm25"
    assert stats["hybrid_dense_enabled"] is False
    assert stats["hybrid_degraded_reason"] == "dependency missing"


def test_hybrid_merge_keeps_bm25_and_adds_dense_candidates(monkeypatch) -> None:
    _retrieval_env(monkeypatch, mode="hybrid", local_embeddings=False)
    engine = RetrievalEngine()
    engine.index_chunks(
        [
            _chunk("bm25", "прочность после старения материала"),
            _chunk("dense", "tensile strength after aging treatment"),
        ]
    )
    monkeypatch.setattr(engine, "dense_retrieve", lambda query, top_k=20: [("dense", 0.95)])

    result = engine.query("прочность", top_k=5)
    ids = [item.chunk_id for item in result]

    assert "bm25" in ids
    assert "dense" in ids


def test_hybrid_deduplicates_chunk_ids(monkeypatch) -> None:
    _retrieval_env(monkeypatch, mode="hybrid", local_embeddings=False)
    engine = RetrievalEngine()
    engine.index_chunks(
        [
            _chunk("shared", "прочность после старения материала"),
            _chunk("dense", "tensile strength after aging treatment"),
        ]
    )
    monkeypatch.setattr(engine, "dense_retrieve", lambda query, top_k=20: [("shared", 0.99), ("dense", 0.95)])

    result = engine.query("прочность", top_k=5)
    ids = [item.chunk_id for item in result]

    assert ids.count("shared") == 1
    assert ids.count("dense") == 1


class _FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            if "corrosion" in lowered or "корроз" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors


def test_mocked_local_embeddings_build_lazy_index(monkeypatch) -> None:
    import app.retrieval.retrieval as retrieval

    _retrieval_env(monkeypatch, mode="hybrid", local_embeddings=True)
    monkeypatch.setattr(retrieval, "SentenceTransformer", _FakeSentenceTransformer)
    engine = RetrievalEngine()
    engine.index_chunks(
        [
            _chunk("strength", "tensile strength after aging treatment"),
            _chunk("corrosion", "corrosion resistance after heat treatment"),
        ]
    )

    result = engine.query("устойчивость к коррозии после обработки", top_k=2)
    stats = engine.stats()

    assert result[0].chunk_id == "corrosion"
    assert stats["embedding_dependency_available"] is True
    assert stats["embedding_model_loaded"] is True
    assert stats["local_embeddings_ready"] is True
    assert stats["local_embedding_vectors"] == 2
    assert stats["hybrid_dense_enabled"] is True
    assert stats["hybrid_degraded_reason"] == ""


def test_hybrid_mode_does_not_enable_qdrant_projection_by_default(monkeypatch) -> None:
    import app.api as api

    monkeypatch.setattr(api.settings, "retrieval_mode", "hybrid", raising=False)
    monkeypatch.setattr(api.settings, "direct_qdrant_projection", False, raising=False)

    assert api._qdrant_outbox_enabled() is False

    monkeypatch.setattr(api.settings, "direct_qdrant_projection", True, raising=False)
    assert api._qdrant_outbox_enabled() is True

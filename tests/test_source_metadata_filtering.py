from __future__ import annotations

from app.models.schemas import Chunk
from app.retrieval.metadata_filters import rerank_chunks_by_source_metadata
from app.retrieval.query_planner import QueryPlanner


def _chunk(chunk_id: str, text: str, metadata: dict) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="/",
        metadata=metadata,
    )


def test_rerank_chunks_prioritizes_domestic_and_foreign_source_metadata() -> None:
    constraints = QueryPlanner().parse(
        "Какие способы закачки шахтных вод применялись в России и за рубежом?"
    )
    chunks = [
        _chunk("plain", "закачка шахтных вод", {"source_name": "unknown.txt"}),
        _chunk(
            "domestic",
            "российская практика закачки шахтных вод",
            {"source_metadata": {"geographies": ["Россия"], "practice_scope": "domestic"}},
        ),
        _chunk(
            "foreign",
            "foreign mine water injection practice",
            {"source_metadata": {"geographies": ["зарубежная практика"], "practice_scope": "foreign_or_global"}},
        ),
    ]

    ranked, diagnostics = rerank_chunks_by_source_metadata(chunks, constraints, now_year=2026)

    assert diagnostics["applied"] is True
    assert diagnostics["matched_chunks"] == 2
    assert {ranked[0].chunk_id, ranked[1].chunk_id} == {"domestic", "foreign"}
    assert ranked[-1].chunk_id == "plain"


def test_rerank_chunks_prioritizes_recent_publication_year() -> None:
    constraints = QueryPlanner().parse(
        "Покажите публикации по распределению Au и Ag между штейном и шлаком за последние 5 лет"
    )
    chunks = [
        _chunk("old", "распределение Au Ag", {"source_metadata": {"publication_year": 2017}}),
        _chunk("recent", "распределение Au Ag", {"source_metadata": {"publication_year": 2024}}),
    ]

    ranked, diagnostics = rerank_chunks_by_source_metadata(chunks, constraints, now_year=2026)

    assert diagnostics["matched_chunks"] == 1
    assert ranked[0].chunk_id == "recent"


def test_rerank_chunks_preserves_order_when_metadata_missing() -> None:
    constraints = QueryPlanner().parse("Какая мировая практика по циркуляции католита?")
    chunks = [_chunk("a", "католит", {}), _chunk("b", "циркуляция", {})]

    ranked, diagnostics = rerank_chunks_by_source_metadata(chunks, constraints, now_year=2026)

    assert diagnostics["reason"] == "no_chunks_matched_source_metadata"
    assert [chunk.chunk_id for chunk in ranked] == ["a", "b"]

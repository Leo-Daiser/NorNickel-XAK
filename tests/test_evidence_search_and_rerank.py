from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.analytics.evidence_reranker import EvidenceReranker
from app.analytics.evidence_search import EvidenceSearch
from app.analytics.query_models import EvidenceItem
from app.models.schemas import Chunk, Document
from app.retrieval.query_planner import QueryPlanner
from app.storage.catalog import SQLiteCatalog


def test_evidence_item_requires_non_empty_quote() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(quote="", score=0.1, retrieval_backend="test")


def test_reranker_boosts_matching_constraints_and_penalizes_conflict() -> None:
    constraints = QueryPlanner().parse("Что делали по ВТ6 при отжиге и какой эффект на прочность?")
    items = [
        EvidenceItem(
            source_name="a",
            quote="ВТ6 отжиг прочность 1120 MPa",
            score=0.1,
            retrieval_backend="test",
        ),
        EvidenceItem(
            source_name="b",
            quote="7075-T6 старение твёрдость 180 HV",
            score=0.2,
            retrieval_backend="test",
        ),
    ]
    reranked = EvidenceReranker().rerank(items, constraints)
    assert reranked[0].source_name == "a"
    assert reranked[0].score > items[0].score
    assert reranked[1].score <= items[1].score


def test_reranker_boosts_matching_source_metadata() -> None:
    constraints = QueryPlanner().parse("Какая мировая практика по циркуляции католита?")
    items = [
        EvidenceItem(
            source_name="local",
            quote="Циркуляция католита описана без географии.",
            score=0.3,
            retrieval_backend="test",
            metadata={"source_metadata": {"practice_scope": "domestic"}},
        ),
        EvidenceItem(
            source_name="global",
            quote="World practice describes catholyte circulation.",
            score=0.2,
            retrieval_backend="test",
            metadata={"source_metadata": {"practice_scope": "foreign_or_global", "geographies": ["мировая практика"]}},
        ),
    ]

    reranked = EvidenceReranker().rerank(items, constraints)

    assert reranked[0].source_name == "global"


def test_evidence_search_fallback_returns_stable_structure(tmp_path: Path) -> None:
    catalog = SQLiteCatalog(tmp_path / "catalog.sqlite3")
    doc = Document(doc_id="doc1", workspace_uid="ws", title="demo.txt", parser="txt")
    chunk = Chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        text="ВТ6 отжиг прочность 1120 MPa",
        page_start=1,
        page_end=1,
        section_path="/",
        metadata={"filename": "demo.txt"},
    )
    catalog.upsert_document(doc)
    catalog.replace_chunks("doc1", [chunk])
    constraints = QueryPlanner().parse("Что известно по ВТ6?")
    result = EvidenceSearch(catalog=catalog).search("ВТ6 прочность", constraints, limit=3)
    assert result
    assert result[0].quote
    assert result[0].retrieval_backend == "sqlite_chunk_search"


def test_evidence_does_not_create_facts() -> None:
    item = EvidenceItem(quote="ВТ6 отжиг прочность 1120 MPa", score=1.0, retrieval_backend="test")
    assert not hasattr(item, "measurements")
    assert not hasattr(item, "materials")

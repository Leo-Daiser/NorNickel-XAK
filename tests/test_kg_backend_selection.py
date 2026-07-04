from __future__ import annotations

import pytest

from app.config import settings
from app.graph.graph_repository import CatalogGraphRepository, GraphRepositoryFactory
from app.graph.neo4j_repository import Neo4jGraphRepository
from app.extraction.extraction import EntityRelationExtractor
from app.storage.catalog import SQLiteCatalog


class DummyGraphDB:
    pass


def test_backend_fallback_mode_selects_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "kg_backend", "fallback")
    repo = GraphRepositoryFactory.create(
        catalog=SQLiteCatalog(tmp_path / "catalog.sqlite3"),
        extractor=EntityRelationExtractor(),
        graph_db=DummyGraphDB(),
    )
    assert isinstance(repo, CatalogGraphRepository)


def test_backend_neo4j_unavailable_is_explicit_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "kg_backend", "neo4j")
    with pytest.raises(RuntimeError):
        GraphRepositoryFactory.create(
            catalog=SQLiteCatalog(tmp_path / "catalog.sqlite3"),
            extractor=EntityRelationExtractor(),
            graph_db=None,
        )


def test_backend_auto_unavailable_falls_back(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "kg_backend", "auto")
    repo = GraphRepositoryFactory.create(
        catalog=SQLiteCatalog(tmp_path / "catalog.sqlite3"),
        extractor=EntityRelationExtractor(),
        graph_db=None,
    )
    assert isinstance(repo, CatalogGraphRepository)


def test_backend_auto_available_uses_neo4j(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "kg_backend", "auto")
    repo = GraphRepositoryFactory.create(
        catalog=SQLiteCatalog(tmp_path / "catalog.sqlite3"),
        extractor=EntityRelationExtractor(),
        graph_db=DummyGraphDB(),
    )
    assert isinstance(repo, Neo4jGraphRepository)


def test_ask_diagnostics_contains_active_backend(tmp_path) -> None:
    from tests.strict_qa_helpers import seeded_client

    client = seeded_client(tmp_path)
    response = client.post(
        "/ask",
        params={"question": "Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?", "top_k": 5},
    )
    assert response.status_code == 200
    retrieval = response.json()["retrieval"]
    assert retrieval["kg_backend_active"] == "fallback"
    assert retrieval["kg_backend_configured"] in {"auto", "fallback"}


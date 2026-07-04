from __future__ import annotations

import time

import app.api as api


class DummyGraphDB:
    pass


def test_auto_backend_selects_neo4j_after_retry_success(monkeypatch) -> None:
    dummy = DummyGraphDB()
    monkeypatch.setattr(api.settings, "kg_backend", "auto")
    monkeypatch.setattr(api, "init_graph_db", lambda: dummy)
    api.graph_db = None
    api.graph_db_error = "old connection refused"
    api.graph_db_last_failure_at = 0.0

    active_graph = api._graph_db_for_repository(force_retry=True)
    diagnostics = api._kg_backend_diagnostics(active_graph, configured_backend="auto")

    assert active_graph is dummy
    assert diagnostics["kg_backend_active"] == "neo4j"
    assert diagnostics["neo4j_available"] is True
    assert diagnostics["neo4j_error"] == ""
    assert diagnostics["kg_backend_decision"]["selected"] == "neo4j"


def test_auto_backend_falls_back_during_short_failure_ttl(monkeypatch) -> None:
    monkeypatch.setattr(api.settings, "kg_backend", "auto")
    monkeypatch.setattr(api, "init_graph_db", lambda: None)
    api.graph_db = None
    api.graph_db_error = "connection refused"
    api.graph_db_last_failure_at = time.monotonic()

    active_graph = api._graph_db_for_repository()
    diagnostics = api._kg_backend_diagnostics(active_graph, configured_backend="auto")

    assert active_graph is None
    assert diagnostics["kg_backend_active"] == "fallback"
    assert diagnostics["neo4j_available"] is False
    assert "connection refused" in diagnostics["kg_backend_decision"]["reason"]


def test_health_diagnostics_do_not_expose_password(monkeypatch) -> None:
    monkeypatch.setattr(api.settings, "neo4j_password", "password")
    diagnostics = api._kg_backend_diagnostics(DummyGraphDB(), configured_backend="auto")
    assert diagnostics["neo4j_password_configured"] is True
    assert "neo4j_password" not in diagnostics
    assert "'password'" not in str(diagnostics)

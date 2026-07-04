from __future__ import annotations

from app.graph.neo4j_connection import check_neo4j_connection, create_neo4j_driver


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def run(self, query: str):
        assert query == "RETURN 1 AS ok"
        return [{"ok": 1}]


class FakeDriver:
    def __init__(self) -> None:
        self.closed = False

    def verify_connectivity(self) -> None:
        return None

    def session(self, **kwargs):
        assert kwargs.get("database") == "neo4j"
        return FakeSession()

    def close(self) -> None:
        self.closed = True


def test_check_neo4j_connection_success_with_mock_driver() -> None:
    calls: dict[str, object] = {}

    def factory(uri: str, auth):
        calls["uri"] = uri
        calls["auth"] = auth
        return FakeDriver()

    status = check_neo4j_connection("bolt://neo4j:7687", "neo4j", "password", "neo4j", driver_factory=factory)
    assert status.available is True
    assert status.error == ""
    assert status.reason == "Neo4j connection check succeeded with RETURN 1"
    assert status.password_configured is True
    assert calls == {"uri": "bolt://neo4j:7687", "auth": ("neo4j", "password")}


def test_empty_neo4j_password_has_clear_error() -> None:
    status = check_neo4j_connection("bolt://neo4j:7687", "neo4j", "", "neo4j", driver_factory=lambda *args, **kwargs: FakeDriver())
    assert status.available is False
    assert "Neo4j password is missing" in status.error


def test_create_driver_rejects_missing_password() -> None:
    try:
        create_neo4j_driver("bolt://neo4j:7687", "neo4j", "", driver_factory=lambda *args, **kwargs: FakeDriver())
    except ValueError as exc:
        assert "NEO4J_PASSWORD" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("missing password should fail")

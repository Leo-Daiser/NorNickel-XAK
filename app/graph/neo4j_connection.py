"""Canonical Neo4j connection helpers used by API, repositories and scripts."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

try:
    from neo4j import GraphDatabase, basic_auth
except ImportError:  # pragma: no cover - exercised only without dependency
    GraphDatabase = None  # type: ignore[assignment]
    basic_auth = None  # type: ignore[assignment]


class Neo4jConnectionStatus(BaseModel):
    """Result of a direct Neo4j connectivity check."""

    available: bool
    uri: str
    user: str
    database: str = "neo4j"
    password_configured: bool
    error: str = ""
    reason: str = ""


DriverFactory = Callable[..., Any]


def create_neo4j_driver(uri: str, user: str, password: str, driver_factory: DriverFactory | None = None) -> Any:
    """Create a Neo4j driver with the official tuple/basic_auth flow."""

    if not password:
        raise ValueError("Neo4j password is missing. Set NEO4J_PASSWORD.")
    factory = driver_factory
    if factory is None:
        if GraphDatabase is None:
            raise ImportError("neo4j driver not installed; install via requirements.txt")
        factory = GraphDatabase.driver
    auth = basic_auth(user, password) if basic_auth is not None and driver_factory is None else (user, password)
    return factory(uri, auth=auth)


def check_neo4j_connection(
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
    driver_factory: DriverFactory | None = None,
) -> Neo4jConnectionStatus:
    """Run a direct RETURN 1 check and return structured diagnostics."""

    status_kwargs = {
        "uri": uri,
        "user": user,
        "database": database or "neo4j",
        "password_configured": bool(password),
    }
    driver = None
    try:
        driver = create_neo4j_driver(uri, user, password, driver_factory=driver_factory)
        if hasattr(driver, "verify_connectivity"):
            driver.verify_connectivity()
        session_kwargs = {"database": database} if database else {}
        with driver.session(**session_kwargs) as session:
            list(session.run("RETURN 1 AS ok"))
        return Neo4jConnectionStatus(
            available=True,
            reason="Neo4j connection check succeeded with RETURN 1",
            **status_kwargs,
        )
    except Exception as exc:
        return Neo4jConnectionStatus(
            available=False,
            error=str(exc),
            reason=f"Neo4j connection check failed: {exc}",
            **status_kwargs,
        )
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass

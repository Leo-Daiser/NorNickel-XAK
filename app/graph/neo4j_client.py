"""Neo4j utility functions used by strict graph backend and scripts."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from .graph_db import GraphDB


def split_cypher_statements(text: str) -> list[str]:
    """Split a simple schema cypher file into executable statements."""
    statements: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(current).strip().rstrip(";").strip()
            if statement:
                statements.append(statement)
            current = []
    tail = "\n".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def apply_schema(graph_db: GraphDB, schema_path: str | Path | None = None) -> dict[str, Any]:
    """Apply the strict ontology schema to Neo4j."""
    path = Path(schema_path) if schema_path else Path(__file__).with_name("schema.cypher")
    statements = split_cypher_statements(path.read_text(encoding="utf-8"))
    applied = 0
    errors: list[str] = []
    with graph_db.session() as session:
        for stale_index in _stale_document_chunk_text_indexes(session):
            try:
                session.run(f"DROP INDEX {stale_index} IF EXISTS")
                applied += 1
            except Exception as exc:  # pragma: no cover - depends on Neo4j version
                errors.append(f"DROP INDEX {stale_index} IF EXISTS: {exc}")
        for statement in statements:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    session.run(statement)
                    applied += 1
                    last_error = None
                    break
                except Exception as exc:  # pragma: no cover - depends on Neo4j version
                    last_error = exc
                    if "DeadlockDetected" not in str(exc) or attempt == 2:
                        break
                    time.sleep(0.25 * (attempt + 1))
            if last_error is not None:
                errors.append(f"{statement.splitlines()[0]}: {last_error}")
        try:
            session.run("CALL db.awaitIndexes()")
        except Exception as exc:  # pragma: no cover - depends on Neo4j edition/version
            errors.append(f"CALL db.awaitIndexes(): {exc}")
    return {"schema_path": str(path), "statements": len(statements), "applied": applied, "errors": errors}


def _stale_document_chunk_text_indexes(session) -> list[str]:
    """Return RANGE indexes on DocumentChunk.text that break long chunk projection."""
    rows = session.run(
        """
        SHOW INDEXES
        YIELD name, type, labelsOrTypes, properties
        RETURN name, type, labelsOrTypes, properties
        """
    )
    result: list[str] = []
    for row in rows:
        try:
            labels = list(row["labelsOrTypes"] or [])
            properties = list(row["properties"] or [])
            index_type = str(row["type"] or "").upper()
            name = str(row["name"] or "")
        except Exception:
            continue
        if index_type == "RANGE" and labels == ["DocumentChunk"] and properties == ["text"] and name:
            result.append(name)
    return result


def ping(graph_db: GraphDB) -> tuple[bool, str | None]:
    """Check that Neo4j accepts a trivial query."""
    try:
        graph_db.run("RETURN 1 AS ok")
        return True, None
    except Exception as exc:
        return False, str(exc)

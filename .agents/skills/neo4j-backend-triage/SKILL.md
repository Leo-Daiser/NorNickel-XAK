---
name: neo4j-backend-triage
description: Use when Neo4j, kg_backend_active, fallback graph, connection refused, missing credentials, graph schema sync, or Cypher diagnostics are involved.
---

# Neo4j Backend Triage

## Purpose

Stabilize the project's Neo4j integration without breaking fallback mode.

## Read first

- `app/graph/neo4j_connection.py`
- `app/graph/graph_db.py`
- `app/graph/graph_repository.py`
- `app/api.py`
- `scripts/check_neo4j_connection.py`
- `scripts/init_neo4j_schema.py`
- `scripts/sync_graph_to_neo4j.py`
- `scripts/smoke_neo4j_graph.py`
- `tests/test_neo4j_connection_helper.py`
- `tests/test_neo4j_backend_activation.py`

## Invariants

- `GraphDatabase.driver` must use `auth=(user, password)` or `basic_auth(user, password)`.
- Do not use custom auth dictionaries.
- Do not expose Neo4j passwords in health, logs, reports, or tests.
- `KG_BACKEND=auto` must choose Neo4j if `RETURN 1` succeeds.
- Fallback graph must remain functional if Neo4j is unavailable.
- Health must not cache a permanent Neo4j failure.

## Required checks

Run:

```bash
python -m pytest -q tests/test_neo4j_connection_helper.py tests/test_neo4j_backend_activation.py
python scripts/check_neo4j_connection.py
python scripts/init_neo4j_schema.py
python scripts/sync_graph_to_neo4j.py
python scripts/smoke_neo4j_graph.py
```

If Docker is required:

```bash
docker compose exec api python scripts/check_neo4j_connection.py
docker compose exec api python scripts/init_neo4j_schema.py
docker compose exec api python scripts/sync_graph_to_neo4j.py
docker compose exec api python scripts/smoke_neo4j_graph.py
```

## Failure conditions

The task is not complete if:

- `/health` reports `neo4j_available=false` while direct `RETURN 1` works from the API container.
- `kg_backend_active=fallback` when `KG_BACKEND=auto` and Neo4j is available.
- Neo4j password appears in output.
- Existing fallback tests fail.

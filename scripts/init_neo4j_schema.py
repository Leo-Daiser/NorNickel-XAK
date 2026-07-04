from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.graph_db import GraphDB  # noqa: E402
from app.graph.neo4j_client import apply_schema  # noqa: E402


def main() -> int:
    try:
        graph_db = GraphDB()
    except Exception as exc:
        print(f"Neo4j unavailable: {exc}")
        return 2
    try:
        result = apply_schema(graph_db)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not result.get("errors") else 1
    finally:
        graph_db.close()


if __name__ == "__main__":
    raise SystemExit(main())


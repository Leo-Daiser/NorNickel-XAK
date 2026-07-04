from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.extraction.extraction import EntityRelationExtractor  # noqa: E402
from app.extraction.pipeline import ExtractionPipeline  # noqa: E402
from app.graph.graph_db import GraphDB  # noqa: E402
from app.graph.graph_writer import sync_catalog_to_neo4j  # noqa: E402
from app.graph.neo4j_client import apply_schema  # noqa: E402
from app.storage.catalog import SQLiteCatalog  # noqa: E402


def main() -> int:
    try:
        graph_db = GraphDB()
    except Exception as exc:
        print(f"Neo4j unavailable: {exc}")
        return 2
    try:
        schema = apply_schema(graph_db)
        catalog = SQLiteCatalog(settings.catalog_db_path)
        extractor = EntityRelationExtractor()
        pipeline = ExtractionPipeline(deterministic_extractor=None)
        stats = sync_catalog_to_neo4j(
            graph_db=graph_db,
            catalog=catalog,
            extractor=extractor,
            document_getter=catalog.get_document,
            pipeline=pipeline,
        )
        print(json.dumps({"schema": schema, "sync": stats}, ensure_ascii=False, indent=2))
        return 0 if not schema.get("errors") else 1
    finally:
        graph_db.close()


if __name__ == "__main__":
    raise SystemExit(main())

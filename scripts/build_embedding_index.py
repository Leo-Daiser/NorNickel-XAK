from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.retrieval.retrieval import RetrievalEngine  # noqa: E402
from app.storage.catalog import SQLiteCatalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build persistent local embedding vectors for active chunks.")
    parser.add_argument("--catalog", default=str(settings.catalog_db_path), help="SQLite catalog path.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive document chunks.")
    parser.add_argument("--json", action="store_true", help="Print compact JSON only.")
    args = parser.parse_args()

    started = time.monotonic()
    catalog = SQLiteCatalog(args.catalog)
    chunks = catalog.list_chunks(active_only=not args.include_inactive)
    engine = RetrievalEngine()

    def progress(row: dict[str, Any]) -> None:
        if args.json:
            return
        done = int(row.get("done", 0))
        total = int(row.get("total", 0))
        created = int(row.get("vectors_created", 0))
        cached = int(row.get("vectors_cached", 0))
        print(f"progress: {done}/{total} chunks, created={created}, cached={cached}", flush=True)

    result = engine.build_local_embedding_index(
        chunks,
        batch_size=max(1, args.batch_size),
        progress_callback=progress,
    )
    elapsed = round(time.monotonic() - started, 3)
    output = {
        "status": "PASS" if result.get("ok") and int(result.get("vectors_missing", 0)) == 0 else "WARN",
        "vectors_total": int(result.get("vectors_total", len(chunks))),
        "vectors_created": int(result.get("vectors_created", 0)),
        "vectors_cached": int(result.get("vectors_cached", 0)),
        "vectors_missing": int(result.get("vectors_missing", len(chunks))),
        "vectors_stale": int(result.get("vectors_stale", 0)),
        "model_id": result.get("model_id") or getattr(settings, "embedding_model", ""),
        "model_path": result.get("model_path") or getattr(settings, "embedding_model_path", ""),
        "dimensions": result.get("dimensions"),
        "cache_path": str(Path(settings.data_dir) / "cache" / "embedding_vectors.sqlite3"),
        "elapsed_sec": elapsed,
        "error": result.get("error", ""),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

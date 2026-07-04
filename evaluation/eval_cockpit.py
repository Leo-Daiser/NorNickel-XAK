from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["DIRECT_QDRANT_PROJECTION"] = "false"
os.environ["ENABLE_LLM"] = "false"
os.environ["ENABLE_LOCAL_EMBEDDINGS"] = "false"
os.environ["RETRIEVAL_MODE"] = "bm25"
os.environ["ANSWER_SYNTHESIS_MODE"] = "template"

from fastapi.testclient import TestClient  # noqa: E402


def _reset_api(tmp: str):
    import app.api as api
    from app.retrieval.retrieval import RetrievalEngine
    from app.storage.catalog import SQLiteCatalog
    from app.storage.outbox import SQLiteOutbox

    api.graph_db = None
    api.catalog = SQLiteCatalog(Path(tmp) / "catalog.sqlite3")
    api.outbox = SQLiteOutbox(Path(tmp) / "outbox.sqlite3")
    api.retrieval_engine = RetrievalEngine()
    api.retrieval_engine.dense_retrieve = lambda question, top_k=20: []
    api.DOCUMENTS.clear()
    api.CHUNKS.clear()
    return api


def _load_demo(client: TestClient) -> None:
    allowed = {".csv", ".xlsx", ".txt", ".html", ".htm", ".docx", ".md"}
    files = [
        ("files", (path.name, path.read_bytes(), "application/octet-stream"))
        for path in sorted((ROOT / "demo_data").iterdir())
        if path.suffix.lower() in allowed
    ]
    response = client.post("/ingest/documents", files=files)
    if response.status_code != 200:
        raise RuntimeError(response.text)


def _has_diagnostics(payload: dict[str, Any]) -> bool:
    if payload.get("diagnostics"):
        return True
    if payload.get("retrieval", {}).get("kg_backend_active"):
        return True
    return False


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["METADATA_DB_PATH"] = str(Path(tmp) / "outbox.sqlite3")
        os.environ["CATALOG_DB_PATH"] = str(Path(tmp) / "catalog.sqlite3")
        api = _reset_api(tmp)
        client = TestClient(api.app)
        _load_demo(client)

        capabilities = client.get("/system/capabilities")
        graph_stats = client.get("/graph/stats")
        entities = client.get("/graph/entities", params={"entity_type": "Material", "limit": 20})
        scenarios = client.get("/demo/scenarios")

        strict_positive = client.post("/demo/run-scenario", params={"scenario_id": "strict_positive_vt6_annealing_strength"})
        strict_negative = client.post("/demo/run-scenario", params={"scenario_id": "strict_negative_vt6_cryo_toughness"})
        overview = client.post("/ask", params={"question": "Что уже делали по ВТ6?", "top_k": 12})
        neighborhood = client.post("/ask", params={"question": "Покажи связанные сущности по ВТ6.", "top_k": 12})

        payloads = {
            "capabilities_available": int(capabilities.status_code == 200 and bool(capabilities.json().get("analytics"))),
            "graph_stats_available": int(graph_stats.status_code == 200 and graph_stats.json().get("experiments", 0) > 0),
            "entities_available": int(entities.status_code == 200 and bool(entities.json().get("items"))),
            "demo_scenarios_available": int(scenarios.status_code == 200 and bool(scenarios.json().get("items"))),
            "strict_positive_pass": int(
                strict_positive.status_code == 200
                and strict_positive.json().get("status") == "ok"
                and bool(strict_positive.json().get("facts"))
            ),
            "strict_negative_pass": int(
                strict_negative.status_code == 200
                and strict_negative.json().get("status") == "no_exact_match"
                and strict_negative.json().get("facts") == []
            ),
            "analytics_pass": int(
                overview.status_code == 200
                and overview.json().get("analytical_intent") == "material_overview"
                and neighborhood.status_code == 200
                and bool(neighborhood.json().get("subgraph", {}).get("nodes"))
            ),
            "diagnostics_presence": int(
                all(
                    _has_diagnostics(response.json())
                    for response in [strict_positive, strict_negative, overview, neighborhood]
                    if response.status_code == 200
                )
            ),
        }

        print("Cockpit evaluation:")
        for key, value in payloads.items():
            print(f"{key}: {value}")
        passed = all(value == 1 for value in payloads.values())
        print("PASS" if passed else "FAIL")
        print("SUMMARY", json.dumps(payloads, ensure_ascii=False))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

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


def _selected_text(payload: dict[str, Any]) -> str:
    slim_facts = [
        {key: value for key, value in row.items() if key not in {"evidence"}}
        for row in payload.get("facts", [])
    ]
    subgraph = payload.get("subgraph") or {}
    semantic_nodes = [
        node
        for node in subgraph.get("nodes", [])
        if node.get("type") not in {"SourceChunk", "Document"}
    ]
    selected = {
        "answer": payload.get("answer"),
        "facts": slim_facts,
        "gaps": payload.get("gaps"),
        "subgraph": {"nodes": semantic_nodes, "edges": subgraph.get("edges", [])},
    }
    return json.dumps(selected, ensure_ascii=False, sort_keys=True).lower()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["METADATA_DB_PATH"] = str(Path(tmp) / "outbox.sqlite3")
        os.environ["CATALOG_DB_PATH"] = str(Path(tmp) / "catalog.sqlite3")
        api = _reset_api(tmp)
        client = TestClient(api.app)
        _load_demo(client)

        questions = json.loads((ROOT / "evaluation" / "analytics_questions.json").read_text(encoding="utf-8"))
        intent_hits = 0
        non_empty = 0
        evidence_hits = 0
        graph_context_hits = 0
        no_unrelated_hits = 0
        no_unrelated_total = 0

        for item in questions:
            response = client.post("/ask", params={"question": item["question"], "top_k": 12})
            payload = response.json() if response.status_code == 200 else {}
            analytical_intent = payload.get("analytical_intent")
            intent_ok = analytical_intent == item["expected_intent"]
            intent_hits += int(intent_ok)
            non_empty += int(bool(payload.get("answer")))
            evidence_hits += int(bool(payload.get("evidence") or payload.get("sources")))
            graph_context = payload.get("graph_context") or {}
            graph_context_hits += int({"facts_count", "sources_count", "subgraph_nodes"} <= set(graph_context))

            focus_material = item.get("focus_material")
            if focus_material:
                no_unrelated_total += 1
                text = _selected_text(payload)
                forbidden = ["7075-t6", "12х18н10т", "вт6"]
                forbidden = [value for value in forbidden if value != focus_material.lower()]
                no_unrelated_hits += int(not any(value in text for value in forbidden))

            print(
                ("PASS" if intent_ok else "FAIL"),
                item["id"],
                "intent=",
                analytical_intent,
                "status=",
                payload.get("status"),
            )
            print("  answer=", str(payload.get("answer", ""))[:260])

        strict = client.post(
            "/ask",
            params={
                "question": "Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?",
                "top_k": 12,
            },
        )
        strict_payload = strict.json() if strict.status_code == 200 else {}
        strict_preserved = int(
            strict_payload.get("status") == "no_exact_match"
            and strict_payload.get("facts") == []
            and "точных данных не найдено" in str(strict_payload.get("answer", "")).lower()
        )

        total = len(questions)
        summary = {
            "intent_accuracy": round(intent_hits / total, 3),
            "non_empty_answer_rate": round(non_empty / total, 3),
            "evidence_presence_rate": round(evidence_hits / total, 3),
            "graph_context_presence_rate": round(graph_context_hits / total, 3),
            "no_unrelated_material_rate": round(no_unrelated_hits / no_unrelated_total, 3)
            if no_unrelated_total
            else 1.0,
            "strict_behavior_preserved": float(strict_preserved),
        }
        print("Analytics evaluation:")
        for key, value in summary.items():
            print(f"{key}: {value:.3f}")
        passed = (
            summary["intent_accuracy"] >= 0.95
            and summary["non_empty_answer_rate"] == 1.0
            and summary["graph_context_presence_rate"] == 1.0
            and summary["no_unrelated_material_rate"] == 1.0
            and summary["strict_behavior_preserved"] == 1.0
        )
        print("PASS" if passed else "FAIL")
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

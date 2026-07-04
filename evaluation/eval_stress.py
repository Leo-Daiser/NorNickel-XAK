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

from fastapi.testclient import TestClient  # noqa: E402


def _payload_text(payload: dict[str, Any]) -> str:
    selected = {
        "answer": payload.get("answer"),
        "status": payload.get("status"),
        "constraints": payload.get("constraints"),
        "facts": payload.get("facts"),
        "gaps": payload.get("gaps"),
        "data_gaps": payload.get("data_gaps"),
        "decision_history": payload.get("decision_history"),
        "subgraph": payload.get("subgraph"),
    }
    return json.dumps(selected, ensure_ascii=False, sort_keys=True).lower()


def _answer_and_facts_text(payload: dict[str, Any]) -> str:
    selected = {"answer": payload.get("answer"), "facts": payload.get("facts")}
    return json.dumps(selected, ensure_ascii=False, sort_keys=True).lower()


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


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["METADATA_DB_PATH"] = str(Path(tmp) / "outbox.sqlite3")
        os.environ["CATALOG_DB_PATH"] = str(Path(tmp) / "catalog.sqlite3")
        api = _reset_api(tmp)
        client = TestClient(api.app)
        _load_demo(client)

        questions = json.loads((ROOT / "evaluation" / "stress_questions.json").read_text(encoding="utf-8"))
        passed = 0
        no_exact_total = 0
        no_exact_passed = 0

        for item in questions:
            response = client.post("/ask", params={"question": item["question"], "top_k": 12})
            payload = response.json() if response.status_code == 200 else {}
            text = _payload_text(payload)
            answer_facts_text = _answer_and_facts_text(payload)
            contract_ok = all(
                key in payload
                for key in ["answer", "status", "constraints", "facts", "sources", "gaps", "subgraph", "retrieval"]
            )
            retrieval = payload.get("retrieval") or {}
            contract_ok = contract_ok and bool(retrieval.get("kg_backend_active"))
            expected_status = item.get("expected_status")
            status_ok = not expected_status or payload.get("status") == expected_status
            missing = [term for term in item.get("expected_terms", []) if term.lower() not in text]
            forbidden = [term for term in item.get("forbidden_answer_terms", []) if term.lower() in answer_facts_text]
            no_exact_facts_ok = True
            if item.get("expect_no_exact_facts"):
                no_exact_total += 1
                no_exact_facts_ok = payload.get("facts") == []
                if status_ok and no_exact_facts_ok:
                    no_exact_passed += 1
            ok = response.status_code == 200 and contract_ok and status_ok and not missing and not forbidden and no_exact_facts_ok
            passed += int(ok)
            print(("PASS" if ok else "FAIL"), item["id"], "status=", payload.get("status"), "missing=", missing, "forbidden=", forbidden)
            print("  answer=", str(payload.get("answer", ""))[:260])

        summary = {
            "total": len(questions),
            "passed": passed,
            "failed": len(questions) - passed,
            "no_exact_pass_rate": round(no_exact_passed / no_exact_total, 3) if no_exact_total else 1.0,
        }
        print("SUMMARY", json.dumps(summary, ensure_ascii=False))
        return 0 if passed == len(questions) else 1


if __name__ == "__main__":
    raise SystemExit(main())

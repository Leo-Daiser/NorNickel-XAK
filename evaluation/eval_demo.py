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
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()


def _load_demo(client: TestClient) -> None:
    demo_dir = ROOT / "demo_data"
    allowed = {".csv", ".xlsx", ".txt", ".html", ".htm", ".docx", ".md"}
    if demo_dir.exists():
        files = [
            ("files", (path.name, path.read_bytes(), "application/octet-stream"))
            for path in sorted(demo_dir.iterdir())
            if path.suffix.lower() in allowed
        ]
    else:
        files = _fallback_demo_files()
    response = client.post("/ingest/documents", files=files)
    if response.status_code != 200:
        raise RuntimeError(f"demo ingestion failed: {response.status_code} {response.text}")


def _fallback_demo_files() -> list[tuple[str, tuple[str, bytes, str]]]:
    """Small in-memory fixture used when the old demo_data folder is absent."""

    docs = {
        "valve_and_standards.txt": (
            "Клапан DN50 имеет PN16, материал корпуса 12Х18Н10Т. "
            "Документация ссылается на ГОСТ 33259 и ISO 9001. "
            "В разделе монтажа есть image: схема монтажа клапана."
        ),
        "pump_articles.csv": (
            "item,article,material\n"
            "насос NPK-200,ART-NPK-200-BODY,12Х18Н10Т\n"
            "насос NPK-200,ART-NPK-200-IMP,12Х18Н10Т\n"
        ),
        "vt6_strength.txt": (
            "После отжига сплава ВТ6 при 900 °C предел прочности составил 1120 MPa; "
            "эффект: increase. The 7075-T6 aluminum alloy showed tensile strength of 77 ksi after aging treatment."
        ),
        "al7075_corrosion_gap.txt": (
            "For 7075-T6, corrosion resistance after heat treatment was discussed, "
            "but no numerical corrosion data were reported."
        ),
    }
    return [
        ("files", (name, text.encode("utf-8"), "text/plain" if name.endswith(".txt") else "text/csv"))
        for name, text in docs.items()
    ]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["METADATA_DB_PATH"] = str(Path(tmp) / "outbox.sqlite3")
        os.environ["CATALOG_DB_PATH"] = str(Path(tmp) / "catalog.sqlite3")

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

        client = TestClient(api.app)
        _load_demo(client)

        gold = json.loads((ROOT / "evaluation" / "gold_questions.json").read_text(encoding="utf-8"))
        results = []
        source_hits = 0
        gap_total = 0
        gap_passed = 0

        for item in gold:
            response = client.post("/ask", params={"question": item["question"], "top_k": 12})
            ok_contract = response.status_code == 200
            payload = response.json() if ok_contract else {}
            for key in ["answer", "facts", "sources", "gaps", "subgraph"]:
                ok_contract = ok_contract and key in payload
            text = _payload_text(payload)
            missing_terms = [term for term in item["expected_terms"] if term.lower() not in text]
            passed = ok_contract and not missing_terms and bool(payload.get("subgraph", {}).get("nodes")) and bool(payload.get("subgraph", {}).get("edges"))
            if payload.get("sources"):
                source_hits += 1
            if "gap" in [term.lower() for term in item["expected_terms"]]:
                gap_total += 1
                if "gap" in text or "нет данных" in text or "не хватает" in text:
                    gap_passed += 1
            results.append({"id": item["id"], "passed": passed, "missing_terms": missing_terms})
            status = "PASS" if passed else "FAIL"
            print(f"{status} {item['id']} missing={missing_terms}")

        passed_count = sum(1 for result in results if result["passed"])
        total = len(results)
        summary = {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "source_coverage": round(source_hits / total, 3) if total else 0.0,
            "gap_detection_pass_rate": round(gap_passed / gap_total, 3) if gap_total else 1.0,
        }
        print("SUMMARY", json.dumps(summary, ensure_ascii=False))
        return 0 if passed_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

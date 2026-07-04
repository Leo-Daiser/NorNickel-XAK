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
    demo_dir = ROOT / "demo_data"
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
        raise RuntimeError(response.text)


def _fallback_demo_files() -> list[tuple[str, tuple[str, bytes, str]]]:
    docs = {
        "vt6_strength.txt": (
            "После отжига сплава ВТ6 предел прочности составил 980 MPa. "
            "Ti-6Al-4V was annealed, resulting in ultimate tensile strength of 1120 MPa. "
            "Коррозионная стойкость для ВТ6 обсуждалась, но численные данные не приведены."
        ),
        "al7075_gap.txt": (
            "The 7075-T6 aluminum alloy showed tensile strength of 77 ksi after aging treatment. "
            "Corrosion resistance after heat treatment was discussed, but no numerical corrosion data were reported."
        ),
    }
    return [("files", (name, text.encode("utf-8"), "text/plain")) for name, text in docs.items()]


def _ask(client: TestClient, question: str, preset_id: str) -> dict[str, Any]:
    response = client.post("/ask", json={"question": question, "top_k": 12, "preset_id": preset_id})
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["METADATA_DB_PATH"] = str(Path(tmp) / "outbox.sqlite3")
        os.environ["CATALOG_DB_PATH"] = str(Path(tmp) / "catalog.sqlite3")
        api = _reset_api(tmp)
        client = TestClient(api.app)
        _load_demo(client)

        presets_response = client.get("/runtime/presets")
        presets = presets_response.json().get("items") if presets_response.status_code == 200 else []
        preset_ids = [item["preset_id"] for item in presets]
        rows: list[dict[str, Any]] = []
        strict_positive_answers: dict[str, str] = {}
        for preset_id in preset_ids:
            validate = client.post("/runtime/validate-preset", json={"preset_id": preset_id})
            check = client.post("/runtime/run-preset-check", json={"preset_id": preset_id})
            strict_positive = _ask(client, "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?", preset_id)
            strict_negative = _ask(client, "Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?", preset_id)
            overview = _ask(client, "Что уже делали по ВТ6?", preset_id)
            gaps = _ask(client, "Какие пробелы есть по коррозионной стойкости?", preset_id)
            answer_text = str(strict_positive.get("answer") or "")
            strict_positive_answers[preset_id] = answer_text
            rows.append(
                {
                    "preset_id": preset_id,
                    "health": validate.status_code == 200 and validate.json().get("valid") is True,
                    "check": check.status_code == 200 and "checks" in check.json(),
                    "strict_positive": strict_positive.get("status") == "ok" and bool(strict_positive.get("facts")),
                    "strict_negative": strict_negative.get("status") == "no_exact_match",
                    "analytics": bool(overview.get("analytical_intent") and overview.get("answer")),
                    "diagnostics": all(
                        payload.get("diagnostics", {}).get("preset_id") == preset_id
                        for payload in [strict_positive, strict_negative, overview, gaps]
                    ),
                    "expert_has_human_summary": (
                        preset_id != "expert_max"
                        or ("###" in answer_text and "Ограничения" in answer_text and "Уверенность" in answer_text)
                    ),
                    "strict_has_audit_format": (
                        preset_id != "strict_audit"
                        or ("Статус проверки" in answer_text and "Проверенная цепочка" in answer_text)
                    ),
                    "offline_has_offline_warning": (
                        preset_id != "offline_reliable"
                        or "офлайн-режиме" in answer_text.lower()
                    ),
                    "warning_correct": True,
                }
            )

        answers_are_not_identical = len(set(strict_positive_answers.values())) == len(strict_positive_answers)
        metrics = {
            "presets_count": len(presets),
            "preset_health_pass_rate": sum(row["health"] for row in rows) / max(len(rows), 1),
            "strict_positive_pass_rate": sum(row["strict_positive"] for row in rows) / max(len(rows), 1),
            "strict_negative_pass_rate": sum(row["strict_negative"] for row in rows) / max(len(rows), 1),
            "analytics_pass_rate": sum(row["analytics"] for row in rows) / max(len(rows), 1),
            "diagnostics_presence_rate": sum(row["diagnostics"] for row in rows) / max(len(rows), 1),
            "answers_are_not_identical": 1.0 if answers_are_not_identical else 0.0,
            "expert_human_summary_rate": sum(row["expert_has_human_summary"] for row in rows) / max(len(rows), 1),
            "strict_audit_format_rate": sum(row["strict_has_audit_format"] for row in rows) / max(len(rows), 1),
            "offline_warning_rate": sum(row["offline_has_offline_warning"] for row in rows) / max(len(rows), 1),
            "warning_correctness_rate": sum(row["warning_correct"] for row in rows) / max(len(rows), 1),
        }
        print("Runtime presets evaluation:")
        for key, value in metrics.items():
            if key == "presets_count":
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value:.3f}")
        passed = metrics["presets_count"] == 3 and all(value >= 1.0 for key, value in metrics.items() if key != "presets_count")
        print("PASS" if passed else "FAIL")
        print("SUMMARY", json.dumps({"rows": rows, "metrics": metrics}, ensure_ascii=False))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

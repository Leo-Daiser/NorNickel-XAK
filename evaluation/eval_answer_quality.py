from __future__ import annotations

import json
import os
import re
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


QUESTIONS = [
    "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?",
    "Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?",
    "Что уже делали по ВТ6?",
    "Сравни ВТ6 и 7075-T6 по прочности.",
    "Какие пробелы есть по коррозионной стойкости?",
    "Найди похожие эксперименты на ВТ6 при отжиге.",
    "Какая лаборатория занималась 12Х18Н10Т?",
    "Что есть по теме титановых сплавов?",
]

INTERNAL_RE = re.compile(r"\b(?:doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|SCI-[A-Za-z0-9_-]+|EXP-[A-Za-z0-9_-]+|Experiment\s+doc_)\b")


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
            "Лаборатория ЛМ-12 исследовала тему титановых сплавов. "
            "После отжига сплава ВТ6 предел прочности составил 980 MPa. "
            "Ti-6Al-4V was annealed, resulting in ultimate tensile strength of 1120 MPa. "
            "Нужны дополнительные данные по вязкости ВТ6 после криообработки."
        ),
        "al7075_strength_gap.txt": (
            "The 7075-T6 aluminum alloy showed tensile strength of 77 ksi after aging treatment. "
            "For 7075-T6, corrosion resistance after heat treatment was discussed, but no numerical corrosion data were reported."
        ),
        "steel_lab.txt": (
            "Лаборатория коррозионных испытаний занималась 12Х18Н10Т после закалки; "
            "твердость составила 240 HV."
        ),
    }
    return [("files", (name, text.encode("utf-8"), "text/plain")) for name, text in docs.items()]


def _ask(client: TestClient, question: str, preset_id: str = "expert_max") -> dict[str, Any]:
    response = client.post("/ask", json={"question": question, "top_k": 12, "preset_id": preset_id})
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def _readable(answer: str) -> bool:
    lowered = answer.lower()
    return bool(answer.strip()) and "###" in answer and "что найдено" in lowered and "уверенность" in lowered


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["METADATA_DB_PATH"] = str(Path(tmp) / "outbox.sqlite3")
        os.environ["CATALOG_DB_PATH"] = str(Path(tmp) / "catalog.sqlite3")
        api = _reset_api(tmp)
        client = TestClient(api.app)
        _load_demo(client)

        rows = []
        for question in QUESTIONS:
            payload = _ask(client, question)
            answer = str(payload.get("answer") or "")
            rows.append(
                {
                    "question": question,
                    "readable": _readable(answer),
                    "no_internal_ids": not INTERNAL_RE.search(answer)
                    and "effect:" not in answer
                    and "unknown" not in answer
                    and "increase" not in answer
                    and "decrease" not in answer
                    and "technical_answer" not in answer,
                    "evidence_populated": bool(payload.get("evidence")) or not payload.get("sources"),
                    "negative_clear": payload.get("status") != "no_exact_match"
                    or ("точных данных" in answer.lower() and "нельзя считать ответом" in answer.lower()),
                    "comparison_quality": not question.startswith("Сравни")
                    or (
                        "Сравнение ограничено" in answer
                        and "MPa" in answer
                        and "прямое экспериментальное сравнение" in answer
                        and "technical_answer" not in answer
                        and "unknown" not in answer
                        and "increase" not in answer
                        and "decrease" not in answer
                    ),
                    "graph_present": bool((payload.get("subgraph") or {}).get("nodes")),
                    "details_present": all(key in payload for key in ["facts", "sources", "data_gaps", "partial_matches", "diagnostics"]),
                }
            )
            print(
                "PASS" if all(value for key, value in rows[-1].items() if key != "question") else "FAIL",
                question,
            )

        preset_answers = {
            preset: _ask(client, QUESTIONS[0], preset).get("answer")
            for preset in ["expert_max", "strict_audit", "offline_reliable"]
        }
        preset_difference = len(set(preset_answers.values())) == 3
        metrics = {
            "human_readability_pass_rate": sum(row["readable"] for row in rows) / len(rows),
            "no_internal_ids_in_main_answer_rate": sum(row["no_internal_ids"] for row in rows) / len(rows),
            "evidence_population_rate": sum(row["evidence_populated"] for row in rows) / len(rows),
            "preset_difference_rate": 1.0 if preset_difference else 0.0,
            "negative_answer_clarity_rate": sum(row["negative_clear"] for row in rows) / len(rows),
            "comparison_answer_quality_rate": sum(row["comparison_quality"] for row in rows) / len(rows),
            "graph_presence_rate": sum(row["graph_present"] for row in rows) / len(rows),
            "details_presence_rate": sum(row["details_present"] for row in rows) / len(rows),
        }
        print("\nAnswer quality evaluation:")
        for key, value in metrics.items():
            print(f"{key}: {value:.3f}")
        passed = all(value >= 0.95 for value in metrics.values())
        print("PASS" if passed else "FAIL")
        print("SUMMARY", json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end dirty demo corpus evaluation through API ingestion and /ask."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("KG_BACKEND", "fallback")
os.environ.setdefault("RUNTIME_PROFILE", "economy_core")
os.environ.setdefault("ENABLE_LLM", "false")
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("ENABLE_LOCAL_EMBEDDINGS", "false")
os.environ.setdefault("RETRIEVAL_MODE", "bm25")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.graph.answer_graph import build_answer_graph  # noqa: E402
from app.parsing.file_profile import profile_corpus  # noqa: E402
from app.parsing.text_quality import SUPPORTED_FILE_EXTENSIONS  # noqa: E402
from evaluation.eval_corpus_truthfulness import raw_leaks, unit_number_claims  # noqa: E402
from tests.strict_qa_helpers import reset_api  # noqa: E402


def _check(rows: list[dict[str, Any]], condition: bool, name: str, reason: str, *, warn: bool = False) -> None:
    rows.append({"check": name, "status": "PASS" if condition else "WARN" if warn else "FAIL", "reason": reason})


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and not path.name.startswith("."))


def _mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".csv":
        return "text/csv"
    if ext in {".html", ".htm"}:
        return "text/html"
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext == ".pptx":
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if ext == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "text/plain"


def _ask(client: TestClient, question: str) -> dict[str, Any]:
    response = client.post("/ask", json={"question": question, "preset_id": "offline_reliable", "top_k": 12})
    if response.status_code != 200:
        return {"status": "request_failed", "answer": response.text, "http_status": response.status_code}
    return response.json()


def _raw_leaks_in_answer_and_graph(payload: dict[str, Any]) -> list[str]:
    answer = str(payload.get("answer") or "")
    graph = build_answer_graph(payload)
    labels = "\n".join(str(node.label) for node in graph.nodes)
    return raw_leaks(answer) + raw_leaks(labels)


def run_eval(input_dir: str | Path = ROOT / "demo_data") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    corpus_dir = Path(input_dir)
    if not corpus_dir.exists():
        corpus_dir = ROOT / "evaluation" / "test_corpus"
    readiness = profile_corpus(corpus_dir)
    readiness_summary = readiness.get("summary") or {}
    with tempfile.TemporaryDirectory() as tmp:
        api = reset_api(Path(tmp))
        api.settings.kg_backend = "fallback"
        api.settings.runtime_profile = "economy_core"
        api.settings.enable_llm = False
        api.settings.llm_provider = "offline"
        api.settings.enable_local_embeddings = False
        api.settings.retrieval_mode = "bm25"
        client = TestClient(api.app)

        ingest_rows = []
        unsupported = []
        for path in _files(corpus_dir):
            if path.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
                unsupported.append(str(path.relative_to(ROOT)))
                continue
            response = client.post(
                "/ingest/documents",
                files=[("files", (path.name, path.read_bytes(), _mime(path)))],
            )
            payload = response.json() if response.status_code == 200 else {"error": response.text}
            item = (payload.get("ingested") or [{}])[0] if isinstance(payload, dict) else {}
            ingest_rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "http_status": response.status_code,
                    "status": item.get("status"),
                    "parse_status": item.get("parse_status") or item.get("status"),
                    "chunks": item.get("chunks", 0),
                    "parser_error": item.get("parser_error"),
                }
            )

        _check(rows, bool(ingest_rows), "supported_files_ingested", f"supported_files={len(ingest_rows)}")
        _check(rows, all(row["http_status"] == 200 for row in ingest_rows), "no_http_ingest_crash", "All supported files returned HTTP 200.")
        parse_failed = [row for row in ingest_rows if row.get("status") == "parse_failed"]
        _check(rows, not parse_failed, "no_parser_crashes", f"parse_failed={len(parse_failed)}")
        ocr_needed = [row for row in ingest_rows if row.get("status") == "ocr_required"]
        _check(rows, bool(ocr_needed) or int(readiness_summary.get("ocr_required_count") or 0) >= 1, "ocr_required_controlled", f"ocr_required={len(ocr_needed)}", warn=True)
        if unsupported:
            _check(rows, False, "unsupported_formats_controlled", f"unsupported_files={len(unsupported)}", warn=True)

        report_response = client.get("/knowledge/expansion-report")
        report = report_response.json() if report_response.status_code == 200 else {}
        _check(rows, report_response.status_code == 200, "knowledge_report_available", f"status={report_response.status_code}")
        _check(rows, int(report.get("facts_without_evidence") or 0) == 0, "facts_have_evidence", f"facts_without_evidence={report.get('facts_without_evidence')}")
        _check(rows, int(report.get("canonical_facts_count") or 0) > 0, "canonical_facts_present", f"canonical_facts={report.get('canonical_facts_count')}")
        _check(rows, int(report.get("conflict_groups_count") or 0) > 0, "conflicts_found", f"conflicts={report.get('conflict_groups_count')}")
        _check(rows, int(report.get("data_gaps_count") or 0) > 0, "data_gaps_found", f"data_gaps={report.get('data_gaps_count')}")

        queries = {
            "comparison": "Сравни ВТ6 и 7075-T6 по прочности.",
            "gaps": "Какие пробелы в данных обнаружены?",
            "negative": "Что известно о сплаве X999 при лазерной обработке?",
        }
        answers: dict[str, Any] = {name: _ask(client, question) for name, question in queries.items()}
        for name, payload in answers.items():
            leaks = _raw_leaks_in_answer_and_graph(payload)
            _check(rows, payload.get("status") not in {"request_failed"}, f"ask_{name}_status", f"status={payload.get('status')}")
            _check(rows, not leaks, f"ask_{name}_no_raw_leaks", f"raw_leaks={len(leaks)}")
        negative_answer = str((answers.get("negative") or {}).get("answer") or "")
        unsupported_numbers = unit_number_claims(negative_answer)
        _check(rows, not unsupported_numbers, "negative_no_numeric_hallucination", f"unit_numbers={unsupported_numbers}")
        _check(
            rows,
            re.search(r"(нет данных|не найден|отсутств|не удалось)", negative_answer, re.IGNORECASE) is not None,
            "negative_controlled_no_data",
            "Negative answer is explicit no-data/partial-match response.",
        )
        _check(rows, not api.settings.enable_llm and not api.settings.enable_local_embeddings, "economy_core_no_llm_embeddings", "LLM and embeddings disabled.")

    failed = any(row["status"] == "FAIL" for row in rows)
    warned = any(row["status"] == "WARN" for row in rows)
    return {
        "summary": "FAIL" if failed else "WARN" if warned else "PASS",
        "checks": rows,
        "ingestion": ingest_rows,
        "unsupported_files": unsupported,
        "readiness_summary": readiness_summary,
    }


def main() -> int:
    result = run_eval()
    path = ROOT / "artifacts" / "eval_dirty_demo_corpus.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY: {result['summary']}")
    for row in result["checks"]:
        print(f"[{row['status']}] {row['check']}: {row['reason']}")
    print(f"JSON report: {path}")
    return 1 if result["summary"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.answering.grounding_guard import validate_text_against_payload  # noqa: E402


API_BASE = os.getenv("API_BASE", "http://localhost:8000")
PRESET_ID = os.getenv("REAL_CORPUS_EVAL_PRESET", "strict_audit")
QUESTIONS_PATH = ROOT / "artifacts" / "real_corpus_ui_questions.md"
ARTIFACT_PATH = ROOT / "artifacts" / "eval_real_corpus_ui_questions.json"
RAW_LEAK_RE = re.compile(
    r"\b(?:doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|EXP-[A-Za-z0-9_-]+|SCI-[A-Za-z0-9_-]+|"
    r"PropertyValue|SourceChunk|MEASURES|OF_PROPERTY|STUDIES|technical_answer)\b",
    re.IGNORECASE,
)
INSUFFICIENT_EVIDENCE_RE = re.compile(
    r"(недостаточно\s+структур|структурированные\s+факты\s+не\s+найдены|"
    r"структурированн\w*\s+(?:exact-)?факт\w*\s+недостаточно|"
    r"точных\s+данных\s+не\s+найдено|нет\s+подтвержд|не\s+наш[её]л|insufficient\s+structured)",
    re.IGNORECASE,
)


def main() -> int:
    result, code = run_eval()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY: {result['summary']}")
    if result.get("error"):
        print(result["error"])
    for row in result.get("rows", []):
        status = "PASS" if row["passed"] else "FAIL"
        reasons = "; ".join(row["reasons"][:3])
        print(
            f"[{status}] {row['case_id']}: {reasons} | "
            f"status={row.get('response_status')} facts={row.get('facts_count')} "
            f"sources={row.get('sources_count')} latency_ms={row.get('latency_ms')} raw_leaks={row.get('raw_leaks_count')}"
        )
    print(f"JSON report: {ARTIFACT_PATH}")
    return code


def run_eval(api_base: str = API_BASE) -> tuple[dict[str, Any], int]:
    questions = load_questions(QUESTIONS_PATH)
    if not questions:
        return {"summary": "FAIL", "error": f"no questions found in {QUESTIONS_PATH}", "rows": []}, 1
    try:
        health = request_json("GET", f"{api_base.rstrip('/')}/health", timeout=10)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return {
            "summary": "FAIL",
            "error": f"API target is unavailable; run docker compose up first. Details: {type(exc).__name__}",
            "rows": [],
        }, 1

    rows = [run_case(api_base, idx, question) for idx, question in enumerate(questions, start=1)]
    failed = [row for row in rows if not row["passed"]]
    warned = [row for row in rows if row.get("warnings")]
    summary = "FAIL" if failed else ("WARN" if warned else "PASS")
    return {
        "summary": summary,
        "preset_id": PRESET_ID,
        "api_base": api_base,
        "health": {
            "kg_backend_active": health.get("kg_backend_active"),
            "effective_retrieval_mode": (health.get("retrieval") or {}).get("effective_retrieval_mode"),
            "local_embedding_vectors": (health.get("retrieval") or {}).get("local_embedding_vectors"),
        },
        "rows": rows,
        "failures_count": len(failed),
        "warnings_count": sum(len(row.get("warnings") or []) for row in rows),
    }, 1 if failed else 0


def run_case(api_base: str, idx: int, question: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = request_json(
            "POST",
            f"{api_base.rstrip('/')}/ask",
            payload={"question": question, "top_k": 12, "preset_id": PRESET_ID},
            timeout=180,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        errors, warnings = validate_payload(payload)
        return {
            "case_id": f"real_ui_{idx:02d}",
            "question": question,
            "passed": not errors,
            "reasons": errors or ["ok"],
            "warnings": warnings,
            "response_status": payload.get("status"),
            "answer_mode": payload.get("answer_mode"),
            "facts_count": len(payload.get("facts") or []),
            "sources_count": len(payload.get("sources") or payload.get("evidence") or []),
            "gaps_count": len(payload.get("data_gaps") or payload.get("gaps") or []),
            "raw_leaks_count": len(RAW_LEAK_RE.findall(str(payload.get("answer") or ""))),
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        return {
            "case_id": f"real_ui_{idx:02d}",
            "question": question,
            "passed": False,
            "reasons": [f"request failed: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "response_status": "request_failed",
            "answer_mode": "",
            "facts_count": 0,
            "sources_count": 0,
            "gaps_count": 0,
            "raw_leaks_count": 0,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


def validate_payload(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    answer = str(payload.get("answer") or "")
    facts = payload.get("facts") or []
    sources = payload.get("sources") or payload.get("evidence") or []

    if payload.get("status") not in {"ok", "partial", "no_exact_match"}:
        errors.append(f"unexpected response status: {payload.get('status')}")
    leaks = RAW_LEAK_RE.findall(answer)
    if leaks:
        errors.append(f"raw leak in main answer: {len(leaks)}")
    unsupported = validate_text_against_payload(answer, payload)
    if unsupported.violations:
        kinds = ", ".join(sorted({item["kind"] for item in unsupported.violations}))
        errors.append(f"unsupported grounded claims in main answer: {kinds}")
    for fact in facts:
        status = str(fact.get("fact_lifecycle_status") or "accepted")
        if status in {"rejected", "quarantine", "quarantined", "legacy_relation"}:
            errors.append(f"non-accepted fact exposed in answer payload: {status}")
            break
    for gap in payload.get("data_gaps") or payload.get("gaps") or []:
        status = str(gap.get("fact_lifecycle_status") or "accepted")
        if status in {"rejected", "quarantine", "quarantined", "legacy_relation"}:
            errors.append(f"non-accepted gap exposed in answer payload: {status}")
            break
    if facts and not sources:
        errors.append("factual answer has facts but no evidence/source payload")
    if not facts and sources and payload.get("status") == "ok" and not INSUFFICIENT_EVIDENCE_RE.search(answer):
        warnings.append("sources found without accepted facts; answer should emphasize insufficient structured evidence")
    if "Офлайн-режим" in answer:
        errors.append("answer contains offline mode banner")
    return errors, warnings


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_questions(path: Path) -> list[str]:
    if not path.exists():
        return []
    questions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*\d+\.\s+(.+?)\s*$", line)
        if match:
            questions.append(match.group(1))
    return questions


if __name__ == "__main__":
    raise SystemExit(main())

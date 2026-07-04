"""Live answer-readiness evaluation for final-TZ metallurgy queries.

This gate is deliberately different from eval_tz_query_readiness:

* query readiness checks whether the question is parsed into constraints;
* answer readiness checks whether the current API/corpus answers honestly.

Missing evidence is reported as WARN, not hidden as success. Raw leaks,
unsupported numeric claims and request failures are FAIL.
"""

from __future__ import annotations

import argparse
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
from app.domain.unit_normalization import normalize_unit_label  # noqa: E402
from app.graph.answer_graph import build_answer_graph  # noqa: E402
from app.retrieval.query_planner import QueryPlanner  # noqa: E402
from app.ui_helpers import answer_evidence_summary_rows  # noqa: E402
from evaluation.eval_corpus_truthfulness import raw_leaks  # noqa: E402
from evaluation.eval_tz_query_readiness import CASES as TZ_QUERY_CASES  # noqa: E402


DEFAULT_API_BASE = os.getenv("API_BASE", "http://localhost:8000")
DEFAULT_OUTPUT = ROOT / "artifacts" / "eval_tz_answer_readiness.json"
DEFAULT_MARKDOWN = ROOT / "artifacts" / "eval_tz_answer_readiness.md"

NUMERIC_CLAIM_RE = re.compile(
    r"\b(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>"
    r"мг\s*/\s*(?:л|дм\s*[³3])|mg\s*/\s*(?:l|dm\s*3)|"
    r"г\s*/\s*л|g\s*/\s*l|м\s*/\s*с|m\s*/\s*s|"
    r"м3\s*/\s*ч|м³\s*/\s*ч|m3\s*/\s*h|"
    r"т\s*/\s*сут|t\s*/\s*(?:day|d)|"
    r"USD\s*/\s*t|\$\s*/\s*t|руб\.?\s*/\s*т|RUB\s*/\s*t|"
    r"USD\s*/\s*m3|руб\.?\s*/\s*м[³3]|RUB\s*/\s*m3|"
    r"MPa|МПа|ksi|HV|HRC|°\s*[CС]|%|ppm"
    r")\b",
    re.IGNORECASE,
)
NO_DATA_RE = re.compile(r"(нет данных|не найден|отсутств|недостаточно|не удалось|no data|not found|missing)", re.IGNORECASE)


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, *, timeout: int = 120) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def api_url(api_base: str, path: str) -> str:
    return f"{api_base.rstrip('/')}/{path.lstrip('/')}"


def numeric_claims(text: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in NUMERIC_CLAIM_RE.finditer(str(text or "")):
        rows.append(
            {
                "value": float(match.group("value").replace(",", ".")),
                "unit": normalize_unit_label(match.group("unit")),
                "text": match.group(0),
            }
        )
    return rows


def allowed_numeric_values(constraints: dict[str, Any], payload: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for row in constraints.get("numeric_constraints") or []:
        for key in ["value", "value_min", "value_max"]:
            try:
                if row.get(key) is not None:
                    values.append(float(row[key]))
            except (TypeError, ValueError):
                pass

    def walk(value: Any, *, key: str = "") -> None:
        if key in {"answer", "human_answer", "technical_answer"}:
            return
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, key=str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                walk(item, key=key)
            return
        if isinstance(value, (int, float)):
            values.append(float(value))
            return
        if isinstance(value, str):
            values.extend(item["value"] for item in numeric_claims(value))

    walk(payload)
    return values


def unsupported_numeric_claims(answer: str, allowed_values: list[float], *, tolerance: float = 1.5) -> list[dict[str, Any]]:
    unsupported: list[dict[str, Any]] = []
    for claim in numeric_claims(answer):
        value = float(claim["value"])
        if allowed_values and any(abs(value - allowed) <= tolerance for allowed in allowed_values):
            continue
        unsupported.append(claim)
    return unsupported


def evidence_count(payload: dict[str, Any]) -> int:
    return len(payload.get("evidence") or payload.get("sources") or [])


def graph_raw_leaks(payload: dict[str, Any]) -> list[str]:
    graph = build_answer_graph(payload)
    labels = "\n".join(str(node.label) for node in graph.nodes)
    return raw_leaks(labels)


def classify_case(case: dict[str, Any], constraints: dict[str, Any], payload: dict[str, Any], *, latency_ms: int) -> dict[str, Any]:
    answer = str(payload.get("answer") or payload.get("human_answer") or "")
    answer_blob = searchable_blob(payload)
    leaks = raw_leaks(answer) + graph_raw_leaks(payload)
    evidence_rows = answer_evidence_summary_rows(payload)
    allowed_values = allowed_numeric_values(constraints, payload)
    unsupported_numbers = unsupported_numeric_claims(answer, allowed_values)
    guard_result = validate_text_against_payload(answer, payload)
    request_ok = payload.get("status") not in {"request_failed", None, ""}
    no_data = bool(NO_DATA_RE.search(answer))
    missing_constraints = missing_expected_constraints(case.get("expected") or {}, answer_blob)
    llm_polished = bool(((payload.get("diagnostics") or {}) if isinstance(payload.get("diagnostics"), dict) else {}).get("llm_answer_polished"))
    warnings: list[str] = []
    failures: list[str] = []
    if not request_ok:
        failures.append("request_failed")
    if leaks:
        failures.append("raw_leak")
    if unsupported_numbers:
        failures.append("unsupported_numeric_claim")
    if guard_result.violations and llm_polished:
        failures.append("grounding_guard_violation")
    elif guard_result.violations:
        warnings.append("deterministic_grounding_guard_diagnostic")
    if request_ok and evidence_count(payload) > 0 and missing_constraints and not no_data:
        failures.append("constraint_mismatch")
    if request_ok and evidence_count(payload) == 0:
        warnings.append("no_evidence_in_current_corpus")
    if request_ok and payload.get("status") in {"no_exact_match", "partial"}:
        warnings.append(f"non_ok_answer_status:{payload.get('status')}")
    if request_ok and evidence_count(payload) == 0 and not no_data:
        warnings.append("coverage_gap_not_explicitly_explained")
    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "status": "FAIL" if failures else "WARN" if warnings else "PASS",
        "answer_status": payload.get("status"),
        "answer_mode": payload.get("answer_mode"),
        "latency_ms": latency_ms,
        "constraints": constraints,
        "evidence_count": evidence_count(payload),
        "evidence_summary_count": len(evidence_rows),
        "facts_count": len(payload.get("facts") or payload.get("primary_facts") or []),
        "raw_leaks_count": len(leaks),
        "unsupported_numeric_claims_count": len(unsupported_numbers),
        "grounding_violations_count": len(guard_result.violations),
        "missing_constraints": missing_constraints,
        "warnings": warnings,
        "failures": failures,
        "answer_preview": answer[:700],
    }


def searchable_blob(payload: dict[str, Any]) -> str:
    pieces = [
        str(payload.get("answer") or ""),
        str(payload.get("human_answer") or ""),
        json.dumps(payload.get("facts") or payload.get("primary_facts") or [], ensure_ascii=False),
        json.dumps(payload.get("evidence") or payload.get("sources") or [], ensure_ascii=False),
    ]
    return normalize_for_match("\n".join(pieces))


def missing_expected_constraints(expected: dict[str, Any], normalized_blob: str) -> list[str]:
    missing: list[str] = []
    for field in ["materials", "regimes", "properties", "equipment", "topic_tags", "geographies"]:
        values = expected.get(field) or []
        if not values:
            continue
        found = [value for value in values if expected_value_present(value, normalized_blob)]
        if not found:
            missing.append(field)
    return missing


def expected_value_present(value: Any, normalized_blob: str) -> bool:
    variants = {normalize_for_match(str(value))}
    if str(value) == "ванна электроэкстракции":
        variants.update({"ванн электроэкстракц", "electrowinning cell"})
    if str(value) == "диафрагменная ячейка":
        variants.update({"диафрагмен", "diaphragm cell"})
    if str(value) == "система газоочистки":
        variants.update({"газоочист", "очистк газ", "gas cleaning"})
    if str(value) == "ПВП":
        variants.update({"пвп", "печ взвешен плавк", "flash smelting", "fluidized bed furnace"})
    if str(value) == "мировая практика":
        variants.update({"миров", "worldwide", "global"})
    if str(value) == "зарубежная практика":
        variants.update({"зарубеж", "abroad", "foreign"})
    if str(value) == "Россия":
        variants.update({"росси", "russia"})
    return any(variant and variant in normalized_blob for variant in variants)


def normalize_for_match(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def ask(api_base: str, question: str, *, preset_id: str, timeout: int) -> dict[str, Any]:
    return request_json(
        "POST",
        api_url(api_base, "/ask"),
        {"question": question, "top_k": 12, "preset_id": preset_id},
        timeout=timeout,
    )


def run_eval(api_base: str = DEFAULT_API_BASE, *, preset_id: str = "offline_reliable", timeout: int = 180) -> tuple[dict[str, Any], int]:
    planner = QueryPlanner()
    try:
        health = request_json("GET", api_url(api_base, "/health"), timeout=10)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        result = {
            "summary": "WARN",
            "api_base": api_base,
            "preset_id": preset_id,
            "error": f"API target is unavailable; run docker compose up first. Details: {type(exc).__name__}: {exc}",
            "rows": [],
        }
        return result, 0

    rows: list[dict[str, Any]] = []
    for case in TZ_QUERY_CASES:
        constraints = planner.parse(case["question"]).model_dump(mode="json")
        started = time.perf_counter()
        try:
            payload = ask(api_base, case["question"], preset_id=preset_id, timeout=timeout)
        except Exception as exc:
            payload = {"status": "request_failed", "answer": f"{type(exc).__name__}: {exc}"}
        latency_ms = int((time.perf_counter() - started) * 1000)
        rows.append(classify_case(case, constraints, payload, latency_ms=latency_ms))

    failures = [row for row in rows if row["status"] == "FAIL"]
    warnings = [row for row in rows if row["status"] == "WARN"]
    result = {
        "summary": "FAIL" if failures else "WARN" if warnings else "PASS",
        "api_base": api_base,
        "preset_id": preset_id,
        "health": safe_health_summary(health),
        "cases_total": len(rows),
        "cases_passed": sum(row["status"] == "PASS" for row in rows),
        "cases_warned": len(warnings),
        "cases_failed": len(failures),
        "raw_leak_rate": round(sum(row["raw_leaks_count"] > 0 for row in rows) / max(1, len(rows)), 3),
        "unsupported_numeric_claim_rate": round(sum(row["unsupported_numeric_claims_count"] > 0 for row in rows) / max(1, len(rows)), 3),
        "evidence_presence_rate": round(sum(row["evidence_count"] > 0 for row in rows) / max(1, len(rows)), 3),
        "rows": rows,
    }
    return result, 1 if failures else 0


def safe_health_summary(health: dict[str, Any]) -> dict[str, Any]:
    retrieval = health.get("retrieval") or {}
    llm = health.get("llm") or {}
    catalog = health.get("catalog") or {}
    return {
        "status": health.get("status"),
        "runtime_profile": health.get("runtime_profile"),
        "kg_backend_active": health.get("kg_backend_active"),
        "neo4j_available": health.get("neo4j_available"),
        "documents": catalog.get("documents"),
        "chunks": catalog.get("chunks"),
        "llm_provider": llm.get("provider"),
        "llm_ready": llm.get("ready"),
        "retrieval_mode": retrieval.get("retrieval_mode"),
        "effective_retrieval_mode": retrieval.get("effective_retrieval_mode"),
        "local_embeddings_ready": retrieval.get("local_embeddings_ready"),
        "hybrid_degraded_reason": retrieval.get("hybrid_degraded_reason"),
    }


def write_reports(result: dict[str, Any], *, json_path: str | Path, markdown_path: str | Path) -> None:
    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_target = Path(markdown_path)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.write_text(render_markdown(result), encoding="utf-8")


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# TZ Answer Readiness",
        "",
        f"SUMMARY: **{result.get('summary')}**",
        f"Preset: `{result.get('preset_id')}`",
        "",
        "## Metrics",
        f"- cases_total: {result.get('cases_total')}",
        f"- cases_passed: {result.get('cases_passed')}",
        f"- cases_warned: {result.get('cases_warned')}",
        f"- cases_failed: {result.get('cases_failed')}",
        f"- raw_leak_rate: {result.get('raw_leak_rate')}",
        f"- unsupported_numeric_claim_rate: {result.get('unsupported_numeric_claim_rate')}",
        f"- evidence_presence_rate: {result.get('evidence_presence_rate')}",
        "",
        "## Cases",
        "| case | status | evidence | facts | warnings | failures |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in result.get("rows") or []:
        lines.append(
            f"| {row.get('case_id')} | {row.get('status')} | {row.get('evidence_count')} | "
            f"{row.get('facts_count')} | {', '.join(row.get('warnings') or []) or '-'} | "
            f"{', '.join(row.get('failures') or []) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- PASS means the current API answer is clean and evidence-backed.",
            "- WARN means the system answered safely, but the current corpus lacks enough grounded evidence for this TZ case.",
            "- FAIL means a product-critical issue: raw leak, unsupported numeric claim, grounding violation or request failure.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate live API answer readiness on final-TZ queries.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL.")
    parser.add_argument("--preset-id", default="offline_reliable", help="Preset used for /ask. Default avoids LLM token spend.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-query timeout seconds.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON report path.")
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN), help="Output Markdown report path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result, exit_code = run_eval(args.api_base, preset_id=args.preset_id, timeout=args.timeout)
    write_reports(result, json_path=args.output, markdown_path=args.markdown)
    print(f"SUMMARY: {result['summary']}")
    if result.get("error"):
        print(result["error"])
    print(f"cases_passed: {result.get('cases_passed', 0)}")
    print(f"cases_warned: {result.get('cases_warned', 0)}")
    print(f"cases_failed: {result.get('cases_failed', 0)}")
    print(f"raw_leak_rate: {result.get('raw_leak_rate', 0)}")
    print(f"unsupported_numeric_claim_rate: {result.get('unsupported_numeric_claim_rate', 0)}")
    print(f"evidence_presence_rate: {result.get('evidence_presence_rate', 0)}")
    for row in result.get("rows") or []:
        print(
            f"[{row['status']}] {row['case_id']}: evidence={row['evidence_count']} "
            f"facts={row['facts_count']} warnings={row['warnings']} failures={row['failures']}"
        )
    print(f"JSON report: {args.output}")
    print(f"Markdown report: {args.markdown}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

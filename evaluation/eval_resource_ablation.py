from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
import argparse
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARTIFACT_PATH = ROOT / "artifacts" / "eval_resource_ablation.json"
DEFAULT_API_BASE = os.getenv("API_BASE", "http://localhost:8000")

PROFILES = ["economy_core", "economy_guarded_llm", "balanced_hybrid", "quality_full"]


PROFILE_ENV: dict[str, dict[str, str]] = {
    "economy_core": {
        "RETRIEVAL_MODE": "bm25",
        "ENABLE_LOCAL_EMBEDDINGS": "false",
        "EAGER_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "false",
        "LLM_PROVIDER": "offline",
        "ANSWER_SYNTHESIS_MODE": "template",
    },
    "economy_guarded_llm": {
        "RETRIEVAL_MODE": "bm25",
        "ENABLE_LOCAL_EMBEDDINGS": "false",
        "EAGER_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "true",
        "LLM_PROVIDER": "auto",
        "LLM_TIMEOUT_SECONDS": "12",
        "MISTRAL_TIMEOUT_SECONDS": "12",
        "ANSWER_SYNTHESIS_MODE": "hybrid",
    },
    "balanced_hybrid": {
        "RETRIEVAL_MODE": "hybrid",
        "ENABLE_LOCAL_EMBEDDINGS": "true",
        "EAGER_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "false",
        "LLM_PROVIDER": "auto",
        "ANSWER_SYNTHESIS_MODE": "hybrid",
        "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
    "quality_full": {
        "RETRIEVAL_MODE": "hybrid",
        "ENABLE_LOCAL_EMBEDDINGS": "true",
        "EAGER_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "true",
        "LLM_PROVIDER": "auto",
        "LLM_TIMEOUT_SECONDS": "12",
        "MISTRAL_TIMEOUT_SECONDS": "12",
        "ANSWER_SYNTHESIS_MODE": "hybrid",
        "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
}

COMMON_ENV = {
    "KG_BACKEND": "fallback",
    "EXTRACTION_MODE": "deterministic",
    "EXTRACTION_ENABLE_LLM": "false",
    "RETRIEVAL_QUERY_EXPANSION": "true",
}

_RUNNER_CODE = r"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(os.environ["RESOURCE_ABLATION_ROOT"])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from evaluation.eval_demo_regression import DEMO_CASES, PRESET_ID, validate_case

with tempfile.TemporaryDirectory() as tmp:
    os.environ["METADATA_DB_PATH"] = str(Path(tmp) / "outbox.sqlite3")
    os.environ["CATALOG_DB_PATH"] = str(Path(tmp) / "catalog.sqlite3")

    import app.api as api
    from app.retrieval.retrieval import RetrievalEngine
    from app.storage.catalog import SQLiteCatalog
    from app.storage.outbox import SQLiteOutbox

    api.graph_db = None
    api.graph_db_error = None
    api.catalog = SQLiteCatalog(Path(tmp) / "catalog.sqlite3")
    api.outbox = SQLiteOutbox(Path(tmp) / "outbox.sqlite3")
    api.retrieval_engine = RetrievalEngine()
    api.DOCUMENTS.clear()
    api.CHUNKS.clear()

    client = TestClient(api.app)
    demo_dir = ROOT / "demo_data"
    if demo_dir.exists():
        files = [
            ("files", (path.name, path.read_bytes(), "application/octet-stream"))
            for path in sorted(demo_dir.iterdir())
            if path.suffix.lower() in {".csv", ".xlsx", ".txt", ".html", ".htm", ".docx", ".md"}
        ]
    else:
        fallback_docs = {
            "vt6_strength.txt": (
                "Лаборатория ЛМ-12 исследовала тему титановых сплавов. "
                "После отжига сплава ВТ6 предел прочности составил 980 MPa. "
                "Ti-6Al-4V was annealed, resulting in ultimate tensile strength of 1120 MPa."
            ),
            "al7075_strength_gap.txt": (
                "The 7075-T6 aluminum alloy showed tensile strength of 77 ksi after aging treatment. "
                "For 7075-T6, corrosion resistance after heat treatment was discussed, but no numerical corrosion data were reported."
            ),
            "conflicts_and_gaps.txt": (
                "Для ВТ6 после отжига также указана прочность 1120 MPa. "
                "Какие есть противоречия или неоднородные данные по прочности? "
                "Нужны дополнительные данные по вязкости ВТ6 после криообработки."
            ),
        }
        files = [("files", (name, text.encode("utf-8"), "text/plain")) for name, text in fallback_docs.items()]
    ingest = client.post("/ingest/documents", files=files)
    if ingest.status_code != 200:
        raise RuntimeError(ingest.text)

    health = client.get("/health").json()
    rows = []
    for case in DEMO_CASES:
        started = time.perf_counter()
        response = client.post("/ask", json={"question": case.question, "top_k": 12, "preset_id": PRESET_ID})
        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code != 200:
            rows.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "passed": False,
                    "reasons": [f"HTTP {response.status_code}: {response.text[:160]}"],
                    "raw_leaks_count": 0,
                    "graph_nodes": 0,
                    "graph_edges": 0,
                    "evidence_count": 0,
                    "latency_ms": latency_ms,
                    "llm_grounding_guard_status": "skipped",
                    "guard_repair_attempted": False,
                    "guard_fallback_used": False,
                    "guard_violations_count": 0,
                    "llm_polished": False,
                    "warnings": [],
                }
            )
            continue
        payload = response.json()
        row = validate_case(case, payload)
        diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
        row["latency_ms"] = latency_ms
        row["llm_polished"] = bool(diagnostics.get("llm_answer_polished"))
        rows.append(row)

    print(json.dumps({"health": health, "rows": rows}, ensure_ascii=False))
"""


def profile_environment(profile: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env.update(COMMON_ENV)
    env.update(PROFILE_ENV[profile])
    env["RUNTIME_PROFILE"] = profile
    env["RESOURCE_ABLATION_ROOT"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_profile(profile: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", _RUNNER_CODE],
            cwd=ROOT,
            env=profile_environment(profile),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=360,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _skipped_profile(profile, "Profile run timed out; external LLM/model loading may be too slow in host mode.")
    if result.returncode != 0:
        return {
            "profile": profile,
            "status": "FAIL",
            "error": (result.stderr or result.stdout).strip()[-1200:],
            "rows": [],
            "warnings": [],
        }
    payload = _parse_last_json_line(result.stdout)
    rows = payload.get("rows") or []
    health = payload.get("health") or {}
    summary = summarize_profile(profile, rows, health)
    return {**summary, "rows": rows, "health": _health_digest(health)}


def _skipped_profile(profile: str, reason: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "status": "WARN",
        "skipped": True,
        "error": reason,
        "queries_passed": 0,
        "queries_failed": 0,
        "raw_leaks_count": 0,
        "unsupported_numeric_claims_count": 0,
        "average_latency_ms": None,
        "llm_calls_count": 0,
        "guard_fallback_count": 0,
        "guard_repaired_count": 0,
        "evidence_count": 0,
        "graph_contract_pass": None,
        "effective_retrieval_mode": None,
        "resource_notes": [],
        "warnings": [reason],
        "failed_cases": [],
        "rows": [],
    }


def _parse_last_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("profile runner did not emit JSON")


def summarize_profile(profile: str, rows: list[dict[str, Any]], health: dict[str, Any]) -> dict[str, Any]:
    failed = [row for row in rows if not row.get("passed")]
    warnings = _profile_warnings(profile, health)
    status = "FAIL" if failed else ("WARN" if warnings else "PASS")
    latencies = [int(row.get("latency_ms") or 0) for row in rows if row.get("latency_ms") is not None]
    guard_statuses = [str(row.get("llm_grounding_guard_status") or "skipped") for row in rows]
    unsupported = sum(
        int(row.get("guard_violations_count") or 0)
        + sum("unsupported" in str(reason).lower() or "hallucinated" in str(reason).lower() for reason in row.get("reasons") or [])
        for row in rows
    )
    llm_calls = sum(1 for row in rows if row.get("llm_polished")) + sum(1 for row in rows if row.get("guard_repair_attempted"))
    return {
        "profile": profile,
        "status": status,
        "queries_passed": len(rows) - len(failed),
        "queries_failed": len(failed),
        "raw_leaks_count": sum(int(row.get("raw_leaks_count") or 0) for row in rows),
        "unsupported_numeric_claims_count": unsupported,
        "average_latency_ms": int(mean(latencies)) if latencies else None,
        "llm_calls_count": llm_calls,
        "guard_fallback_count": guard_statuses.count("fallback"),
        "guard_repaired_count": guard_statuses.count("repaired"),
        "evidence_count": sum(int(row.get("evidence_count") or 0) for row in rows),
        "graph_contract_pass": all(int(row.get("graph_nodes") or 0) <= 10 and int(row.get("graph_edges") or 0) <= 12 for row in rows),
        "effective_retrieval_mode": ((health.get("retrieval") or {}).get("effective_retrieval_mode")),
        "resource_notes": _resource_notes(health),
        "warnings": warnings,
        "failed_cases": [row.get("case_id") for row in failed],
    }


def _profile_warnings(profile: str, health: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    retrieval = health.get("retrieval") or {}
    llm = health.get("llm") or {}
    if profile in {"economy_guarded_llm", "quality_full"} and not llm.get("ready"):
        warnings.append(f"LLM mode ran without ready LLM provider: {llm.get('last_error') or 'not ready'}")
    if profile in {"balanced_hybrid", "quality_full"} and retrieval.get("effective_retrieval_mode") != "hybrid":
        warnings.append(f"Hybrid retrieval degraded: {retrieval.get('hybrid_degraded_reason') or 'unknown reason'}")
    if profile == "economy_core" and (retrieval.get("local_embeddings_enabled") or llm.get("enabled")):
        warnings.append("economy_core has embeddings or LLM enabled")
    return warnings


def _resource_notes(health: dict[str, Any]) -> list[str]:
    retrieval = health.get("retrieval") or {}
    llm = health.get("llm") or {}
    return [
        f"retrieval={retrieval.get('effective_retrieval_mode') or retrieval.get('retrieval_mode')}",
        f"vectors={retrieval.get('local_embedding_vectors', 0)}",
        f"llm_provider={llm.get('provider')}",
        f"llm_ready={llm.get('ready')}",
    ]


def _health_digest(health: dict[str, Any]) -> dict[str, Any]:
    retrieval = health.get("retrieval") or {}
    llm = health.get("llm") or {}
    return {
        "runtime_profile": health.get("runtime_profile"),
        "retrieval": {
            key: retrieval.get(key)
            for key in [
                "retrieval_mode",
                "effective_retrieval_mode",
                "local_embeddings_enabled",
                "local_embeddings_ready",
                "local_embedding_vectors",
                "hybrid_dense_enabled",
                "hybrid_degraded_reason",
            ]
        },
        "llm": {key: llm.get(key) for key in ["enabled", "provider", "ready", "model", "last_error"]},
    }


def run_eval() -> tuple[dict[str, Any], int]:
    profiles = [run_profile(profile) for profile in PROFILES]
    failed = [row for row in profiles if row.get("status") == "FAIL"]
    result = {
        "summary": "FAIL" if failed else ("WARN" if any(row.get("status") == "WARN" for row in profiles) else "PASS"),
        "profiles": profiles,
    }
    return result, 1 if failed else 0


def _api_url(api_base: str, path: str) -> str:
    return f"{api_base.rstrip('/')}/{path.lstrip('/')}"


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 180) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_docker_target(api_base: str = DEFAULT_API_BASE) -> tuple[dict[str, Any], int]:
    try:
        health = _request_json("GET", _api_url(api_base, "/health"), timeout=10)
    except Exception as exc:
        message = f"API target is unavailable; run docker compose up first. Details: {type(exc).__name__}"
        return {"summary": "FAIL", "target": "docker", "api_base": api_base, "error": message, "profiles": []}, 1

    from evaluation.eval_demo_regression import DEMO_CASES, PRESET_ID, validate_case

    rows: list[dict[str, Any]] = []
    for case in DEMO_CASES:
        started = time.perf_counter()
        try:
            payload = _request_json(
                "POST",
                _api_url(api_base, "/ask"),
                {"question": case.question, "top_k": 12, "preset_id": PRESET_ID},
                timeout=240,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            row = validate_case(case, payload)
            row["latency_ms"] = latency_ms
            diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
            row["llm_polished"] = bool(diagnostics.get("llm_answer_polished"))
        except Exception as exc:
            row = {
                "case_id": case.case_id,
                "question": case.question,
                "passed": False,
                "reasons": [f"request failed: {type(exc).__name__}: {exc}"],
                "raw_leaks_count": 0,
                "graph_nodes": 0,
                "graph_edges": 0,
                "evidence_count": 0,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "llm_grounding_guard_status": "skipped",
                "guard_repair_attempted": False,
                "guard_fallback_used": False,
                "guard_violations_count": 0,
                "llm_polished": False,
                "warnings": [],
            }
        rows.append(row)
    profile_name = str(health.get("runtime_profile") or (health.get("runtime_profile_summary") or {}).get("runtime_profile") or "docker_api")
    profile_summary = summarize_profile(profile_name, rows, health)
    profile_summary["profile"] = f"docker:{profile_name}"
    result = {
        "summary": "FAIL" if profile_summary["queries_failed"] else ("WARN" if profile_summary["warnings"] else "PASS"),
        "target": "docker",
        "api_base": api_base,
        "profiles": [{**profile_summary, "rows": rows, "health": _health_digest(health)}],
    }
    return result, 1 if profile_summary["queries_failed"] else 0


def _print_table(result: dict[str, Any]) -> None:
    headers = ["profile", "status", "passed", "failed", "latency_ms", "retrieval", "llm_calls", "guard_fb", "warnings"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in result.get("profiles", []):
        print(
            "| "
            + " | ".join(
                [
                    str(row.get("profile")),
                    str(row.get("status")),
                    str(row.get("queries_passed")),
                    str(row.get("queries_failed")),
                    str(row.get("average_latency_ms")),
                    str(row.get("effective_retrieval_mode")),
                    str(row.get("llm_calls_count")),
                    str(row.get("guard_fallback_count")),
                    str(len(row.get("warnings") or [])),
                ]
            )
            + " |"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare demo quality/resource behavior across runtime profiles.")
    parser.add_argument("--target", choices=["host", "docker"], default="host", help="host uses isolated local subprocess profiles; docker uses running API at --api-base.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL for --target docker.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.target == "docker":
        result, exit_code = run_docker_target(args.api_base)
    else:
        result, exit_code = run_eval()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY: {result['summary']}")
    if result.get("error"):
        print(result["error"])
    _print_table(result)
    for profile in result.get("profiles", []):
        for warning in profile.get("warnings") or []:
            print(f"[WARN] {profile['profile']}: {warning}")
        if profile.get("error"):
            print(f"[FAIL] {profile['profile']}: {profile['error']}")
    print(f"JSON report: {ARTIFACT_PATH}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

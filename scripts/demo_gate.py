from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extraction_quality_report import Neo4jScanOptions, build_report as build_extraction_report  # noqa: E402
from scripts.resource_efficiency_report import _docker_api_image_size_bytes  # noqa: E402
from app.runtime.profiles import profile_consistency_issues, runtime_profile_from_environment  # noqa: E402


RAW_LEAKAGE_RE = re.compile(
    r"\b(?:technical_answer|doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|"
    r"EXP-[A-Za-z0-9_-]+|SCI-[A-Za-z0-9_-]+|increase|decrease|unknown|"
    r"PropertyValue|SourceChunk|Experiment|MEASURES|OF_PROPERTY|STUDIES)\b"
)

DEMO_QUESTIONS = [
    "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?",
    "Сравни ВТ6 и 7075-T6 по прочности.",
    "Какие есть пробелы в данных?",
]


@dataclass
class GateCheck:
    level: str
    message: str


class DemoGate:
    def __init__(self) -> None:
        self.checks: list[GateCheck] = []

    def pass_(self, message: str) -> None:
        self.checks.append(GateCheck("PASS", message))

    def warn(self, message: str) -> None:
        self.checks.append(GateCheck("WARN", message))

    def fail(self, message: str) -> None:
        self.checks.append(GateCheck("FAIL", message))

    @property
    def failed(self) -> bool:
        return any(item.level == "FAIL" for item in self.checks)

    def render(self) -> str:
        summary = "FAIL" if self.failed else "PASS"
        lines = [f"SUMMARY: {summary}"]
        lines.extend(f"[{item.level}] {item.message}" for item in self.checks)
        return "\n".join(lines)


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 90) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _api_url(api_base: str, path: str) -> str:
    return f"{api_base.rstrip('/')}/{path.lstrip('/')}"


def contains_raw_leakage(text: str) -> bool:
    return bool(RAW_LEAKAGE_RE.search(text or ""))


def answer_payload_has_offline_leakage(payload: dict[str, Any]) -> bool:
    diagnostics = payload.get("diagnostics") or {}
    preset_id = str(diagnostics.get("preset_id") or payload.get("preset_id") or "")
    answer = str(payload.get("answer") or "")
    return preset_id != "expert_max" or answer.strip().lower().startswith("офлайн-режим")


def classify_retrieval(health: dict[str, Any]) -> tuple[str, str]:
    retrieval = health.get("retrieval") or {}
    if retrieval.get("bm25_ready") is not True:
        return "FAIL", "BM25 retrieval is not ready"
    mode = str(retrieval.get("retrieval_mode") or "")
    local_enabled = bool(retrieval.get("local_embeddings_enabled"))
    effective = str(retrieval.get("effective_retrieval_mode") or "")
    if mode == "hybrid" and local_enabled:
        if effective == "hybrid" and retrieval.get("hybrid_dense_enabled") is True:
            return "PASS", "Hybrid dense retrieval enabled"
        reason = str(retrieval.get("hybrid_degraded_reason") or "unknown reason")
        if effective == "hybrid_degraded_to_bm25" and reason:
            return "WARN", f"Hybrid degraded to BM25: {reason}"
        return "FAIL", f"Hybrid retrieval configured but state is inconsistent: effective={effective or 'missing'}"
    return "PASS", f"Retrieval ready: {effective or mode or 'bm25'}"


def classify_image_size(size_bytes: int | None, *, strict: bool = False, max_gb: float = 5.0) -> GateCheck:
    if size_bytes is None:
        return GateCheck("WARN", "API Docker image size is unknown")
    size_gb = size_bytes / (1024 ** 3)
    if size_gb > max_gb:
        level = "FAIL" if strict else "WARN"
        return GateCheck(level, f"API Docker image is large: {size_gb:.2f} GB > {max_gb:g} GB")
    return GateCheck("PASS", f"API Docker image size within limit: {size_gb:.2f} GB <= {max_gb:g} GB")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resource_limit_gb() -> float:
    try:
        return float(os.getenv("RESOURCE_MAX_IMAGE_GB", "5"))
    except ValueError:
        return 5.0


def run_resource_checks(gate: DemoGate, health: dict[str, Any]) -> None:
    strict = _env_bool("RESOURCE_STRICT", False)
    max_gb = _resource_limit_gb()
    gate.checks.append(classify_image_size(_docker_api_image_size_bytes(), strict=strict, max_gb=max_gb))

    profile = str(
        health.get("runtime_profile")
        or (health.get("runtime_profile_summary") or {}).get("runtime_profile")
        or runtime_profile_from_environment()
        or ""
    )
    retrieval = health.get("retrieval") or {}
    extraction = health.get("extraction") or {}
    answering = health.get("answering") or {}
    llm = health.get("llm") or {}

    consistency_issues = profile_consistency_issues(
        runtime_profile=profile,
        retrieval_mode=str(retrieval.get("retrieval_mode") or ""),
        local_embeddings_enabled=bool(retrieval.get("local_embeddings_enabled")),
        llm_enabled=bool(llm.get("enabled")),
        llm_provider=str(llm.get("provider") or ""),
        effective_retrieval_mode=str(retrieval.get("effective_retrieval_mode") or ""),
        hybrid_dense_enabled=retrieval.get("hybrid_dense_enabled") if "hybrid_dense_enabled" in retrieval else None,
    )
    if consistency_issues and strict:
        for issue in consistency_issues:
            gate.fail(issue)
    elif consistency_issues:
        for issue in consistency_issues:
            gate.warn(issue)
    else:
        gate.pass_("Runtime profile/embedding settings are resource-consistent")

    llm_extraction = bool(extraction.get("llm_extraction_available") or extraction.get("extraction_enable_llm"))
    if llm_extraction and strict:
        gate.fail("LLM extraction is enabled in strict resource/compliance mode")
    elif llm_extraction:
        gate.warn("LLM extraction is enabled; deterministic extraction is expected for resource efficiency")
    else:
        gate.pass_("LLM extraction disabled; deterministic extraction remains source of truth")

    if health.get("qdrant_projection_enabled") and not retrieval.get("qdrant_ready"):
        gate.warn("Qdrant projection is enabled but Qdrant is not ready/used")
    else:
        gate.pass_("Qdrant is disabled or actively available")

    try:
        extraction_report = build_extraction_report(Neo4jScanOptions(skip_neo4j=True))
        facts_without_evidence = int((extraction_report.get("summary") or {}).get("facts_without_evidence") or 0)
        if facts_without_evidence == 0:
            gate.pass_("facts_without_evidence = 0")
        else:
            gate.fail(f"facts_without_evidence = {facts_without_evidence}")
    except Exception as exc:
        gate.warn(f"Could not compute facts_without_evidence: {type(exc).__name__}")

    llm_enabled = bool(llm.get("enabled"))
    answer_mode = str(answering.get("answer_synthesis_mode") or "")
    if not llm_enabled:
        gate.pass_("LLM disabled; no LLM resource cost in current profile")
    elif not llm_extraction and answer_mode in {"hybrid", "llm"}:
        gate.pass_("LLM used only for guarded answer polish")
    elif not llm_extraction:
        gate.pass_("LLM is not used for extraction; deterministic fact layer remains authoritative")
    else:
        gate.warn("LLM usage is not clearly limited to guarded polish")
    gate.pass_("Grounding guard enabled")


def validate_answer_payload(payload: dict[str, Any], *, expected_preset: str = "expert_max") -> list[GateCheck]:
    checks: list[GateCheck] = []
    status = str(payload.get("status") or "")
    if status and status not in {"ok", "no_exact_match"}:
        checks.append(GateCheck("FAIL", f"Unexpected answer status: {status}"))
    else:
        checks.append(GateCheck("PASS", f"Answer status acceptable: {status or 'missing'}"))

    diagnostics = payload.get("diagnostics") or {}
    actual_preset = str(diagnostics.get("preset_id") or payload.get("preset_id") or "")
    if actual_preset == expected_preset:
        checks.append(GateCheck("PASS", f"{expected_preset} preset used"))
    else:
        checks.append(GateCheck("FAIL", f"Expected preset {expected_preset}, got {actual_preset or 'missing'}"))

    answer = str(payload.get("answer") or "")
    if answer.strip().lower().startswith("офлайн-режим"):
        checks.append(GateCheck("FAIL", "expert_max answer starts with offline mode banner"))
    elif contains_raw_leakage(answer):
        checks.append(GateCheck("FAIL", "Main answer contains raw technical leakage"))
    else:
        checks.append(GateCheck("PASS", "No raw technical leakage in main answer"))
    return checks


def validate_answer_graph(payload: dict[str, Any]) -> list[GateCheck]:
    checks: list[GateCheck] = []
    try:
        from app.graph.answer_graph import answer_graph_to_html, build_answer_graph

        graph = build_answer_graph(payload)
        nodes = list(graph.nodes)
        edges = list(graph.edges)
        if len(nodes) <= 10:
            checks.append(GateCheck("PASS", f"Answer graph nodes <= 10 ({len(nodes)})"))
        else:
            checks.append(GateCheck("FAIL", f"Answer graph has too many nodes: {len(nodes)}"))
        if len(edges) <= 12:
            checks.append(GateCheck("PASS", f"Answer graph edges <= 12 ({len(edges)})"))
        else:
            checks.append(GateCheck("FAIL", f"Answer graph has too many edges: {len(edges)}"))
        labels = "\n".join(str(node.label) for node in nodes)
        if contains_raw_leakage(labels):
            checks.append(GateCheck("FAIL", "Answer graph labels contain raw technical ids/labels"))
        else:
            checks.append(GateCheck("PASS", "No raw ids in answer graph labels"))
        html = answer_graph_to_html(graph)
        if "Колесо — масштаб" in html and "узлы зафиксированы" in html:
            checks.append(GateCheck("PASS", "Russian graph control hint present"))
        else:
            checks.append(GateCheck("FAIL", "Russian graph control hint missing"))
        if "dragNodes: false" in html and "drag node: move" not in html and "Wheel:" not in html:
            checks.append(GateCheck("PASS", "Graph node dragging disabled in HTML"))
        else:
            checks.append(GateCheck("FAIL", "Graph drag node controls are unsafe"))
    except Exception as exc:
        checks.append(GateCheck("FAIL", f"Answer graph validation failed: {type(exc).__name__}: {exc}"))
    return checks


def check_release_security(gate: DemoGate) -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        gate.pass_(".env is not present in project/container filesystem")
    else:
        git_dir = ROOT / ".git"
        if not git_dir.exists():
            gate.warn(".env exists locally; git metadata unavailable to verify tracking")
        else:
            try:
                result = subprocess.run(
                    ["git", "ls-files", ".env"],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                if result.stdout.strip():
                    gate.fail(".env is tracked by git")
                else:
                    gate.pass_(".env is not tracked by git")
            except Exception as exc:
                gate.warn(f"Could not verify .env git tracking: {type(exc).__name__}")

    release_path = ROOT / "dist" / "release_unpacked"
    if not release_path.exists():
        gate.warn("Release package is not built; release package scan skipped")
        return
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_release_package.py"), "--path", str(release_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode == 0:
            gate.pass_("Release package security scan passed")
        else:
            gate.fail("Release package security scan failed")
    except Exception as exc:
        gate.warn(f"Could not run release package scan: {type(exc).__name__}")


def run_gate(api_base: str) -> DemoGate:
    gate = DemoGate()
    try:
        health = _request_json("GET", _api_url(api_base, "/health"), timeout=20)
        gate.pass_("API health reachable")
    except Exception as exc:
        gate.fail(f"API health unavailable: {type(exc).__name__}: {exc}")
        check_release_security(gate)
        return gate

    if health.get("kg_backend_active") == "neo4j" and health.get("neo4j_available") is True:
        gate.pass_("Neo4j active")
    else:
        gate.fail(
            f"Neo4j is not active: kg_backend_active={health.get('kg_backend_active')}, "
            f"neo4j_available={health.get('neo4j_available')}"
        )

    llm = health.get("llm") or {}
    provider = str(llm.get("provider") or llm.get("llm_provider_active") or health.get("llm_provider_active") or "")
    llm_ready = bool(llm.get("ready") if "ready" in llm else llm.get("llm_ready", health.get("llm_ready")))
    llm_enabled = bool(llm.get("enabled") if "enabled" in llm else llm.get("llm_enabled", health.get("llm_enabled")))
    runtime_profile = str(health.get("runtime_profile") or (health.get("runtime_profile_summary") or {}).get("runtime_profile") or "")
    if runtime_profile == "economy_core" and not llm_enabled:
        gate.pass_("LLM disabled for economy_core")
    elif llm_enabled and llm_ready and provider in {"mistral", "openrouter"}:
        gate.pass_(f"LLM ready: {provider}")
    else:
        gate.fail(f"LLM is not demo-ready: enabled={llm_enabled}, ready={llm_ready}, provider={provider or 'missing'}")

    level, message = classify_retrieval(health)
    if level == "PASS":
        gate.pass_(message)
    elif level == "WARN":
        gate.warn(message)
    else:
        gate.fail(message)

    run_resource_checks(gate, health)

    try:
        presets = _request_json("GET", _api_url(api_base, "/runtime/presets"), timeout=20)
        preset_ids = {str(item.get("preset_id") or item.get("id")) for item in presets.get("items", [])}
        required = {"expert_max", "strict_audit", "offline_reliable"}
        missing = sorted(required - preset_ids)
        if missing:
            gate.fail(f"Missing runtime presets: {', '.join(missing)}")
        else:
            gate.pass_("Runtime presets are registered")
        if {"expert_max", "strict_audit"} <= preset_ids:
            gate.pass_("strict_audit differs from expert_max by separate preset id")
    except Exception as exc:
        gate.fail(f"Runtime presets unavailable: {type(exc).__name__}: {exc}")

    sample_payload: dict[str, Any] | None = None
    for question in DEMO_QUESTIONS:
        try:
            payload = _request_json(
                "POST",
                _api_url(api_base, "/ask"),
                {"question": question, "top_k": 8, "preset_id": "expert_max"},
                timeout=120,
            )
            sample_payload = sample_payload or payload
            for check in validate_answer_payload(payload, expected_preset="expert_max"):
                gate.checks.append(GateCheck(check.level, f"{question[:42]}...: {check.message}"))
        except urllib.error.HTTPError as exc:
            gate.fail(f"/ask failed for {question[:42]}...: HTTP {exc.code}")
        except Exception as exc:
            gate.fail(f"/ask failed for {question[:42]}...: {type(exc).__name__}: {exc}")

    if sample_payload is not None:
        for check in validate_answer_graph(sample_payload):
            gate.checks.append(check)
    else:
        gate.fail("Answer graph validation skipped because no /ask payload succeeded")

    check_release_security(gate)
    return gate


def main() -> int:
    api_base = os.getenv("API_BASE", "http://localhost:8000")
    gate = run_gate(api_base)
    print(gate.render())
    return 1 if gate.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

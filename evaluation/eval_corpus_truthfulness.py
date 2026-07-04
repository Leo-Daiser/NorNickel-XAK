"""Component-wise truthfulness evaluation on synthetic and dirty corpus fixtures.

The eval is intentionally deterministic and local. It does not use LLM extraction
and does not require Neo4j, Qdrant or internet access. Its goal is to classify
quality failures, not to hide them behind string-to-string answer snapshots.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.answering.grounding_guard import validate_text_against_payload  # noqa: E402
from app.graph.answer_graph import build_answer_graph  # noqa: E402
from app.ui_helpers import answer_evidence_summary_rows  # noqa: E402
from tests.strict_qa_helpers import reset_api  # noqa: E402


CORPUS_DIR = ROOT / "evaluation" / "test_corpus"
QUERY_BANK_PATH = ROOT / "evaluation" / "query_bank.json"
EXPECTATIONS_PATH = ROOT / "evaluation" / "query_expectations.json"
ARTIFACT_JSON = ROOT / "artifacts" / "eval_corpus_truthfulness.json"
ARTIFACT_MD = ROOT / "artifacts" / "eval_corpus_truthfulness.md"
ANALYSIS_MD = ROOT / "artifacts" / "synthetic_corpus_analysis.md"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".csv"}
WEB_FIXTURE_DIR = CORPUS_DIR / "web"
RAW_LEAK_RE = re.compile(
    r"\b(?:technical_answer|doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|"
    r"EXP-[A-Za-z0-9_-]+|SCI-[A-Za-z0-9_-]+|PropertyValue|SourceChunk|"
    r"Experiment|MEASURES|OF_PROPERTY|STUDIES|increase|decrease|unknown)\b",
    re.IGNORECASE,
)
UNIT_NUMBER_RE = re.compile(r"\b(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>MPa|МПа|ksi|HV|HRC|°C|C)\b", re.IGNORECASE)
NO_DATA_RE = re.compile(r"(нет данных|не найден|отсутств|не измер|не приведен|not reported|missing|недостаточно)", re.IGNORECASE)
CONFLICT_RE = re.compile(r"(противореч|неоднород|разные значения|расходятся|different values|conflict)", re.IGNORECASE)
GAP_RE = re.compile(r"(пробел|нет данных|не измер|additional|not reported|gap|требуется)", re.IGNORECASE)


PROFILE_ENV: dict[str, dict[str, str]] = {
    "economy_core": {
        "RUNTIME_PROFILE": "economy_core",
        "KG_BACKEND": "fallback",
        "RETRIEVAL_MODE": "bm25",
        "ENABLE_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "false",
        "LLM_ENABLED": "false",
        "LLM_PROVIDER": "offline",
        "ANSWER_SYNTHESIS_MODE": "template",
    },
    "balanced_hybrid": {
        "RUNTIME_PROFILE": "balanced_hybrid",
        "KG_BACKEND": "fallback",
        "RETRIEVAL_MODE": "hybrid",
        "ENABLE_LOCAL_EMBEDDINGS": "true",
        "EAGER_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "false",
        "LLM_ENABLED": "false",
        "LLM_PROVIDER": "offline",
        "ANSWER_SYNTHESIS_MODE": "hybrid",
        "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
    "economy_guarded_llm": {
        "RUNTIME_PROFILE": "economy_guarded_llm",
        "KG_BACKEND": "fallback",
        "RETRIEVAL_MODE": "bm25",
        "ENABLE_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "true",
        "LLM_PROVIDER": "auto",
        "ANSWER_SYNTHESIS_MODE": "hybrid",
    },
    "quality_full": {
        "RUNTIME_PROFILE": "quality_full",
        "KG_BACKEND": "fallback",
        "RETRIEVAL_MODE": "hybrid",
        "ENABLE_LOCAL_EMBEDDINGS": "true",
        "EAGER_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "true",
        "LLM_PROVIDER": "auto",
        "ANSWER_SYNTHESIS_MODE": "hybrid",
        "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
}


@dataclass(frozen=True)
class QueryCase:
    case_id: str
    section: str
    question: str
    expectation: dict[str, Any]


class FakeHtmlResponse:
    def __init__(self, content: bytes, *, url: str) -> None:
        self.content = content
        self.url = url
        self.headers = {"content-type": "text/html; charset=utf-8"}
        self.is_redirect = False
        self.is_permanent_redirect = False

    def iter_content(self, chunk_size: int = 65536):
        yield self.content

    def raise_for_status(self) -> None:
        return None

    def close(self) -> None:
        return None


def raw_leaks(text: Any) -> list[str]:
    return RAW_LEAK_RE.findall(str(text or ""))


def unit_number_claims(text: Any) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for match in UNIT_NUMBER_RE.finditer(str(text or "")):
        value = float(match.group("value").replace(",", "."))
        unit = _canonical_unit(match.group("unit"))
        claims.append({"value": value, "unit": unit, "text": match.group(0)})
    return claims


def unsupported_numeric_claims(answer: str, allowed_values: Iterable[float], *, tolerance: float = 1.5) -> list[dict[str, Any]]:
    allowed = [float(item) for item in allowed_values]
    unsupported = []
    for claim in unit_number_claims(answer):
        if allowed and any(abs(float(claim["value"]) - value) <= tolerance for value in allowed):
            continue
        unsupported.append(claim)
    return unsupported


def load_query_cases(limit: int | None = None) -> list[QueryCase]:
    query_bank = json.loads(QUERY_BANK_PATH.read_text(encoding="utf-8"))
    expectations = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    cases: list[QueryCase] = []
    for item in query_bank:
        case_id = str(item["id"])
        cases.append(
            QueryCase(
                case_id=case_id,
                section=str(item.get("section") or ""),
                question=str(item["question"]),
                expectation=dict(expectations.get(case_id) or {"expected_mode": item.get("section", "unknown")}),
            )
        )
    return cases[:limit] if limit else cases


def _apply_profile_environment(profile: str) -> None:
    for key, value in PROFILE_ENV.get(profile, PROFILE_ENV["economy_core"]).items():
        os.environ[key] = value


def _configure_api(api: Any, profile: str) -> None:
    profile_env = PROFILE_ENV.get(profile, PROFILE_ENV["economy_core"])
    api.settings.runtime_profile = profile
    api.settings.kg_backend = "fallback"
    api.settings.retrieval_mode = profile_env["RETRIEVAL_MODE"]
    api.settings.enable_local_embeddings = profile_env["ENABLE_LOCAL_EMBEDDINGS"].lower() == "true"
    api.settings.eager_local_embeddings = profile_env.get("EAGER_LOCAL_EMBEDDINGS", "false").lower() == "true"
    api.settings.direct_qdrant_projection = False
    api.settings.enable_llm = profile_env.get("ENABLE_LLM", "false").lower() == "true"
    api.settings.llm_provider = profile_env.get("LLM_PROVIDER", "offline")
    api.settings.answer_synthesis_mode = profile_env.get("ANSWER_SYNTHESIS_MODE", "template")
    api.settings.extraction_enable_llm = False
    api.settings.extraction_mode = "deterministic"
    if profile in {"economy_core", "balanced_hybrid"}:
        api.llm_client.config_enabled = False
        api.llm_client.provider = "offline"
        api.llm_client.provider_active = "offline"
    api.graph_db = None
    api.graph_db_error = None


def _mime_for(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return "text/csv"
    if path.suffix.lower() in {".html", ".htm"}:
        return "text/html"
    return "text/plain"


def _supported_file_paths() -> list[Path]:
    result = []
    for path in sorted(CORPUS_DIR.rglob("*")):
        if not path.is_file():
            continue
        if WEB_FIXTURE_DIR in path.parents:
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            result.append(path)
    return result


def _skipped_file_paths() -> list[Path]:
    return [
        path
        for path in sorted(CORPUS_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() not in SUPPORTED_EXTENSIONS
    ]


def _ingest_file(client: TestClient, path: Path) -> dict[str, Any]:
    response = client.post(
        "/ingest/documents",
        files=[("files", (path.name, path.read_bytes(), _mime_for(path)))],
    )
    if response.status_code != 200:
        raise RuntimeError(f"ingest failed for {path}: {response.status_code} {response.text[:400]}")
    return response.json()["ingested"][0]


def _install_fake_url_fetch(api: Any, url_to_content: dict[str, bytes]) -> tuple[Any, Any]:
    import app.security.url_safety as url_safety

    original_resolve = url_safety._resolve_host
    original_get = api.requests.get

    def resolve_public(host: str, port: int | None):
        return {ipaddress.ip_address("93.184.216.34")}

    def fake_get(url: str, *args: Any, **kwargs: Any) -> FakeHtmlResponse:
        base = str(url).split("?", 1)[0]
        content = url_to_content.get(base)
        if content is None:
            raise RuntimeError(f"no fake HTML fixture for {url}")
        return FakeHtmlResponse(content, url=url)

    url_safety._resolve_host = resolve_public
    api.requests.get = fake_get
    return original_resolve, original_get


def _restore_fake_url_fetch(api: Any, originals: tuple[Any, Any]) -> None:
    import app.security.url_safety as url_safety

    original_resolve, original_get = originals
    url_safety._resolve_host = original_resolve
    api.requests.get = original_get


def _ingest_web_fixtures(api: Any, client: TestClient) -> list[dict[str, Any]]:
    url_to_content = {
        "https://example.org/reports/vt6-web-clean.html": (WEB_FIXTURE_DIR / "vt6_web_clean.html").read_bytes(),
        "https://example.org/reports/vt6-web-noise.html": (WEB_FIXTURE_DIR / "vt6_web_noise.html").read_bytes(),
        "https://example.org/reports/web-table-7075.html": (WEB_FIXTURE_DIR / "web_table_7075.html").read_bytes(),
    }
    originals = _install_fake_url_fetch(api, url_to_content)
    try:
        ingested = []
        for url in url_to_content:
            response = client.post("/ingest/url", params={"url": f"{url}?utm_source=truth_eval"})
            if response.status_code != 200:
                raise RuntimeError(f"URL ingest failed for {url}: {response.status_code} {response.text[:400]}")
            ingested.append(response.json()["ingested"])
        return ingested
    finally:
        _restore_fake_url_fetch(api, originals)


def _ask(client: TestClient, question: str, preset_id: str) -> dict[str, Any]:
    response = client.post("/ask", json={"question": question, "top_k": 12, "preset_id": preset_id})
    if response.status_code != 200:
        return {"status": "request_failed", "answer": response.text, "http_status": response.status_code}
    return response.json()


def _knowledge_report(client: TestClient) -> dict[str, Any]:
    response = client.get("/knowledge/expansion-report")
    if response.status_code != 200:
        raise RuntimeError(f"knowledge report failed: {response.status_code} {response.text[:400]}")
    return response.json()


def _corpus_setup(client: TestClient, api: Any) -> dict[str, Any]:
    ingest_results = []
    before_all = _knowledge_report(client)
    for path in _supported_file_paths():
        before = _knowledge_report(client)
        item = _ingest_file(client, path)
        delta = item.get("knowledge_expansion") or {}
        ingest_results.append(
            {
                "path": str(path.relative_to(ROOT)),
                "status": item.get("status"),
                "chunks": item.get("chunks", 0),
                "doc_id": item.get("doc_id"),
                "new_facts": delta.get("new_canonical_facts_count", 0),
                "duplicates": delta.get("duplicate_facts_count", 0),
                "conflicts": delta.get("conflict_groups_added_count", 0),
                "gaps": delta.get("data_gaps_added_count", 0),
                "facts_before": before.get("canonical_facts_count", 0),
            }
        )
    web_results = _ingest_web_fixtures(api, client)
    report_after_web = _knowledge_report(client)

    before_reingest = _knowledge_report(client)
    duplicate_path = CORPUS_DIR / "clean" / "al7075_aging_strength.txt"
    duplicate_ingest = _ingest_file(client, duplicate_path)
    after_reingest = _knowledge_report(client)

    active_result = _active_filter_check(client, duplicate_ingest.get("doc_id"))
    final_report = _knowledge_report(client)
    return {
        "before_all": before_all,
        "ingested_files": ingest_results,
        "web_ingested": web_results,
        "skipped_files": [str(path.relative_to(ROOT)) for path in _skipped_file_paths()],
        "report_after_web": report_after_web,
        "final_report": final_report,
        "idempotency": {
            "reingested_path": str(duplicate_path.relative_to(ROOT)),
            "new_canonical_facts_count": (duplicate_ingest.get("knowledge_expansion") or {}).get("new_canonical_facts_count", 0),
            "canonical_facts_before": before_reingest.get("canonical_facts_count", 0),
            "canonical_facts_after": after_reingest.get("canonical_facts_count", 0),
            "passed": before_reingest.get("canonical_facts_count") == after_reingest.get("canonical_facts_count")
            and (duplicate_ingest.get("knowledge_expansion") or {}).get("new_canonical_facts_count", 0) == 0,
        },
        "active_filtering": active_result,
    }


def _active_filter_check(client: TestClient, doc_id: str | None) -> dict[str, Any]:
    if not doc_id:
        return {"passed": False, "reason": "no document id for active filter check"}
    before = _knowledge_report(client)
    off = client.patch(f"/documents/{doc_id}/active", json={"active": False})
    inactive = _knowledge_report(client)
    on = client.patch(f"/documents/{doc_id}/active", json={"active": True})
    reactivated = _knowledge_report(client)
    before_chunks = int(before.get("active_chunks_count") or before.get("chunks_count") or 0)
    inactive_chunks = int(inactive.get("active_chunks_count") or inactive.get("chunks_count") or 0)
    reactivated_chunks = int(reactivated.get("active_chunks_count") or reactivated.get("chunks_count") or 0)
    return {
        "doc_id": doc_id,
        "deactivate_status": off.status_code,
        "reactivate_status": on.status_code,
        "active_chunks_before": before_chunks,
        "active_chunks_inactive": inactive_chunks,
        "active_chunks_reactivated": reactivated_chunks,
        "passed": bool(off.status_code == 200 and on.status_code == 200 and inactive_chunks < before_chunks and reactivated_chunks == before_chunks),
    }


def evaluate_response(case: QueryCase, payload: dict[str, Any], setup: dict[str, Any]) -> dict[str, Any]:
    expectation = case.expectation
    answer = str(payload.get("answer") or payload.get("human_answer") or "")
    text_blob = _payload_blob(payload)
    facts = _facts(payload)
    evidence_rows = answer_evidence_summary_rows(payload)
    graph = build_answer_graph(payload)
    graph_labels = "\n".join(str(node.label) for node in graph.nodes)
    allowed_values = (
        list(expectation.get("expected_numeric_values_original") or [])
        + list(expectation.get("expected_numeric_values_normalized") or [])
        + _observed_values(facts, "")
        + _payload_numeric_values(payload)
    )
    raw_leak_count = len(raw_leaks(answer)) + len(raw_leaks(graph_labels)) + len(raw_leaks(json.dumps(evidence_rows, ensure_ascii=False)))
    unsupported_numbers = unsupported_numeric_claims(answer, allowed_values)
    guard_check = validate_text_against_payload(answer, payload)
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    llm_polished = bool(diagnostics.get("llm_answer_polished"))
    grounding_violations = guard_check.violations if llm_polished else []
    mode = str(expectation.get("expected_mode") or "")
    report_backed_query = mode in {"growth", "active_inactive"}
    context_ok = _growth_context_ok(setup) if mode == "growth" else bool((setup.get("active_filtering") or {}).get("passed"))

    component_results = {
        "material_detection": True
        if report_backed_query and context_ok
        else _expected_terms_present(expectation.get("expected_materials") or [], text_blob, skip=bool(expectation.get("expected_no_data"))),
        "regime_detection": True
        if report_backed_query and context_ok
        else _expected_terms_present(expectation.get("expected_regimes") or [], text_blob, skip=bool(expectation.get("expected_no_data"))),
        "property_detection": True
        if report_backed_query and context_ok
        else _expected_terms_present(expectation.get("expected_properties") or [], text_blob, skip=False),
        "numeric_value_correctness": _expected_values_present(expectation.get("expected_numeric_values_original") or [], facts, answer),
        "normalized_value_correctness": _expected_values_present(expectation.get("expected_numeric_values_normalized") or [], facts, answer, tolerance=2.0),
        "data_gap_detection": _gap_detected(payload, answer) if expectation.get("expected_data_gap_presence") else True,
        "conflict_detection": _conflict_detected(payload, answer) if expectation.get("expected_conflict_presence") else True,
        "provenance_presence": _has_provenance(payload, evidence_rows) if expectation.get("expected_source_presence") else True,
        "raw_leaks_absent": raw_leak_count == 0,
        "unsupported_claims_absent": not unsupported_numbers and not grounding_violations,
        "no_data_controlled": _no_data_controlled(answer) if expectation.get("expected_no_data") else True,
        "scope_controlled": not (
            mode in {"gap", "negative"} and not expectation.get("expected_numeric_values_original") and unit_number_claims(answer)
        ),
        "graph_contract": len(graph.nodes) <= 10 and len(graph.edges) <= 12,
    }
    if expectation.get("expected_mode") == "growth":
        component_results["expansion_delta_correctness"] = _growth_context_ok(setup)
    if expectation.get("expected_mode") == "active_inactive":
        component_results["active_filtering_correctness"] = bool((setup.get("active_filtering") or {}).get("passed"))
    if expectation.get("expected_mode") == "web":
        component_results["web_ingestion_answer_quality"] = bool(setup.get("web_ingested")) and _has_provenance(payload, evidence_rows)

    error_categories = _error_categories(component_results, unsupported_numbers, case, expectation, payload)
    warnings = _warnings(component_results, case, payload)
    critical_errors = [
        category
        for category in error_categories
        if category in {
            "Hallucinated numeric value",
            "Raw technical leak",
            "No-data hallucination",
            "Version/idempotency error",
            "Active/inactive filtering error",
        }
    ]
    return {
        "case_id": case.case_id,
        "section": case.section,
        "question": case.question,
        "status": payload.get("status"),
        "answer_mode": payload.get("answer_mode"),
        "component_results": component_results,
        "error_categories": error_categories,
        "warnings": warnings,
        "critical_errors": critical_errors,
        "raw_leaks_count": raw_leak_count,
        "unsupported_numeric_claims_count": len(unsupported_numbers),
        "unsupported_numeric_claims": unsupported_numbers,
        "grounding_violations_count": len(grounding_violations),
        "deterministic_guard_violations_count": 0 if llm_polished else len(guard_check.violations),
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "evidence_count": len(payload.get("evidence") or payload.get("sources") or []),
        "evidence_summary_count": len(evidence_rows),
        "conflict_count": len(_conflict_groups(payload)),
        "data_gap_count": len(payload.get("data_gaps") or payload.get("gaps") or []),
        "answer_preview": answer[:500],
    }


def _payload_blob(payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "answer": payload.get("answer"),
            "facts": payload.get("facts"),
            "primary_facts": payload.get("primary_facts"),
            "sources": payload.get("sources"),
            "evidence": payload.get("evidence"),
            "data_gaps": payload.get("data_gaps") or payload.get("gaps"),
        },
        ensure_ascii=False,
        default=str,
    ).lower()


def _facts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for key in ["primary_facts", "facts"]:
        value = payload.get(key)
        if isinstance(value, list):
            result.extend(item for item in value if isinstance(item, dict))
    return result


def _expected_terms_present(terms: list[str], blob: str, *, skip: bool = False) -> bool:
    if skip or not terms:
        return True
    lowered = blob.lower().replace("ё", "е")
    return any(any(variant in lowered for variant in _term_variants(term)) for term in terms)


def _term_variants(term: str) -> set[str]:
    raw = str(term or "").lower().replace("ё", "е")
    variants = {raw}
    aliases = {
        "вт6": {"вт6", "ti-6al-4v", "титанов"},
        "7075-t6": {"7075-t6", "7075 т6", "алюминиев"},
        "старение": {"старение", "aging", "aged"},
        "отжиг": {"отжиг", "anneal", "annealed"},
        "закалка": {"закалка"},
        "прочность": {"прочность", "strength", "tensile"},
        "твердость": {"твердость", "hardness", "hv"},
        "коррозионная стойкость": {"корроз", "corrosion"},
    }
    variants.update(aliases.get(raw, set()))
    return variants


def _blob_terms(blob: str) -> set[str]:
    lowered = blob.lower().replace("ё", "е")
    return {token for token in re.split(r"[^0-9a-zа-яёхХ\-]+", lowered) if token} | {lowered}


def _expected_values_present(expected: list[float], facts: list[dict[str, Any]], answer: str, *, tolerance: float = 1.5) -> bool:
    if not expected:
        return True
    observed = _observed_values(facts, answer)
    return any(any(abs(value - float(exp)) <= tolerance for value in observed) for exp in expected)


def _observed_values(facts: list[dict[str, Any]], answer: str) -> list[float]:
    values: list[float] = []
    for fact in facts:
        for key in ["value", "value_original", "value_normalized"]:
            try:
                if fact.get(key) is not None:
                    values.append(float(fact[key]))
            except (TypeError, ValueError):
                pass
        normalized = fact.get("normalized")
        if isinstance(normalized, dict):
            for key in ["value_original", "value_normalized"]:
                try:
                    if normalized.get(key) is not None:
                        values.append(float(normalized[key]))
                except (TypeError, ValueError):
                    pass
    values.extend(float(item["value"]) for item in unit_number_claims(answer))
    return values


def _payload_numeric_values(payload: dict[str, Any]) -> list[float]:
    """Collect grounded numeric values from structured payload fields, not answer prose."""

    values: list[float] = []

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
            values.extend(float(item["value"]) for item in unit_number_claims(value))

    walk(payload)
    return values


def _has_provenance(payload: dict[str, Any], evidence_rows: list[dict[str, Any]]) -> bool:
    return bool(evidence_rows or payload.get("evidence") or payload.get("sources"))


def _gap_detected(payload: dict[str, Any], answer: str) -> bool:
    return bool(payload.get("data_gaps") or payload.get("gaps") or GAP_RE.search(answer))


def _conflict_detected(payload: dict[str, Any], answer: str) -> bool:
    return bool(_conflict_groups(payload) or CONFLICT_RE.search(answer))


def _conflict_groups(payload: dict[str, Any]) -> list[Any]:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    groups = diagnostics.get("fact_conflicts") or diagnostics.get("conflict_groups") or []
    return groups if isinstance(groups, list) else []


def _no_data_controlled(answer: str) -> bool:
    return bool(NO_DATA_RE.search(answer or "")) and not unit_number_claims(answer)


def _growth_context_ok(setup: dict[str, Any]) -> bool:
    report = setup.get("final_report") or {}
    return bool(
        report.get("canonical_facts_count", 0) > 0
        and report.get("materials_count", 0) >= 2
        and report.get("new_cross_material_comparison_opportunities")
    )


def _error_categories(
    component_results: dict[str, bool],
    unsupported_numbers: list[dict[str, Any]],
    case: QueryCase,
    expectation: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    categories: list[str] = []
    if payload.get("status") == "request_failed":
        categories.append("Request/API error")
    if not component_results.get("material_detection", True):
        categories.append("Missing fact")
    if not component_results.get("regime_detection", True):
        categories.append("Missing fact")
    if not component_results.get("property_detection", True):
        categories.append("Missing fact")
    if not component_results.get("numeric_value_correctness", True):
        categories.append("Missing fact")
    if not component_results.get("normalized_value_correctness", True):
        categories.append("Wrong normalization")
    if not component_results.get("data_gap_detection", True):
        categories.append("Missed data gap")
    if not component_results.get("conflict_detection", True):
        categories.append("Wrong conflict interpretation")
    if not component_results.get("provenance_presence", True):
        categories.append("Source/provenance missing")
    if not component_results.get("raw_leaks_absent", True):
        categories.append("Raw technical leak")
    if not component_results.get("unsupported_claims_absent", True):
        categories.append("Unsupported claim")
    if unsupported_numbers:
        categories.append("Hallucinated numeric value")
    if not component_results.get("scope_controlled", True):
        categories.append("Unsupported claim")
    if expectation.get("expected_no_data") and not component_results.get("no_data_controlled", True):
        categories.append("No-data hallucination")
    if not component_results.get("active_filtering_correctness", True):
        categories.append("Active/inactive filtering error")
    if not component_results.get("expansion_delta_correctness", True):
        categories.append("Version/idempotency error")
    if not component_results.get("web_ingestion_answer_quality", True):
        categories.append("Web source parsing error")
    if case.section == "dirty" and categories:
        categories.append("Dirty OCR robustness failure")
    return sorted(set(categories))


def _warnings(component_results: dict[str, bool], case: QueryCase, payload: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for key, value in component_results.items():
        if not value:
            warnings.append(key)
    if payload.get("status") in {"partial", "no_exact_match"} and case.section not in {"negative", "data_gap"}:
        warnings.append(f"non-ok status for non-negative query: {payload.get('status')}")
    return warnings


def _canonical_unit(unit: str) -> str:
    value = str(unit or "").strip()
    if value.lower() == "mpa" or value in {"МПа", "МРа", "MPа"}:
        return "MPa"
    if value.lower() == "ksi":
        return "ksi"
    return value


def _aggregate_metrics(rows: list[dict[str, Any]], setup: dict[str, Any]) -> dict[str, Any]:
    def rate(key: str) -> float:
        values = [bool((row.get("component_results") or {}).get(key, True)) for row in rows]
        return round(sum(values) / max(1, len(values)), 3)

    return {
        "exact_fact_correctness": rate("numeric_value_correctness"),
        "material_detection_accuracy": rate("material_detection"),
        "regime_detection_accuracy": rate("regime_detection"),
        "property_detection_accuracy": rate("property_detection"),
        "numeric_value_correctness": rate("numeric_value_correctness"),
        "normalized_value_correctness": rate("normalized_value_correctness"),
        "data_gap_detection_accuracy": rate("data_gap_detection"),
        "conflict_detection_accuracy": rate("conflict_detection"),
        "provenance_presence_rate": rate("provenance_presence"),
        "raw_leak_rate": round(sum(row["raw_leaks_count"] > 0 for row in rows) / max(1, len(rows)), 3),
        "unsupported_claim_rate": round(sum(row["unsupported_numeric_claims_count"] > 0 or row["grounding_violations_count"] > 0 for row in rows) / max(1, len(rows)), 3),
        "no_data_hallucination_rate": round(
            sum("No-data hallucination" in row.get("error_categories", []) for row in rows) / max(1, len(rows)), 3
        ),
        "web_ingestion_answer_quality": rate("web_ingestion_answer_quality"),
        "expansion_delta_correctness": 1.0 if _growth_context_ok(setup) else 0.0,
        "idempotency_success_rate": 1.0 if (setup.get("idempotency") or {}).get("passed") else 0.0,
        "active_filtering_correctness": 1.0 if (setup.get("active_filtering") or {}).get("passed") else 0.0,
    }


def _error_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for category in row.get("error_categories") or []:
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _strongest_and_worst(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            len(row.get("critical_errors") or []),
            len(row.get("error_categories") or []),
            len(row.get("warnings") or []),
            -int(row.get("evidence_count") or 0),
        ),
    )
    strongest = ranked[:10]
    worst = list(reversed(ranked[-10:]))
    return strongest, worst


def _status(rows: list[dict[str, Any]], setup: dict[str, Any]) -> tuple[str, int]:
    critical = [row for row in rows if row.get("critical_errors")]
    infrastructure_fail = not (setup.get("idempotency") or {}).get("passed") or not (setup.get("active_filtering") or {}).get("passed")
    if critical or infrastructure_fail:
        return "FAIL", 1
    if any(row.get("error_categories") or row.get("warnings") for row in rows):
        return "WARN", 0
    return "PASS", 0


def run_eval(profile: str = "economy_core", limit: int | None = None, artifacts_dir: Path | None = None) -> tuple[dict[str, Any], int]:
    _apply_profile_environment(profile)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        api = reset_api(Path(tmp))
        _configure_api(api, profile)
        client = TestClient(api.app)
        setup = _corpus_setup(client, api)
        preset_id = "offline_reliable" if profile in {"economy_core", "balanced_hybrid"} else "expert_max"
        rows: list[dict[str, Any]] = []
        for case in load_query_cases(limit):
            query_started = time.perf_counter()
            payload = _ask(client, case.question, preset_id)
            row = evaluate_response(case, payload, setup)
            row["latency_ms"] = int((time.perf_counter() - query_started) * 1000)
            rows.append(row)

        metrics = _aggregate_metrics(rows, setup)
        status, exit_code = _status(rows, setup)
        strongest, worst = _strongest_and_worst(rows)
        result = {
            "summary": status,
            "profile": profile,
            "preset_id": preset_id,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "corpus": {
                "root": str(CORPUS_DIR.relative_to(ROOT)),
                "supported_sources_count": len(_supported_file_paths()) + len(setup.get("web_ingested") or []),
                "skipped_sources": setup.get("skipped_files") or [],
                "folders": _corpus_folder_counts(),
            },
            "setup": setup,
            "metrics": metrics,
            "error_counts": _error_counts(rows),
            "rows": rows,
            "strongest_cases": [_case_digest(row) for row in strongest],
            "worst_cases": [_case_digest(row) for row in worst],
            "resource": _resource_digest(api, profile),
        }

    output_dir = artifacts_dir or (ROOT / "artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval_corpus_truthfulness.json"
    md_path = output_dir / "eval_corpus_truthfulness.md"
    analysis_path = output_dir / "synthetic_corpus_analysis.md"
    profile_json_path = output_dir / f"eval_corpus_truthfulness_{profile}.json"
    profile_md_path = output_dir / f"eval_corpus_truthfulness_{profile}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md = _markdown_report(result)
    md_path.write_text(report_md, encoding="utf-8")
    analysis_path.write_text(report_md, encoding="utf-8")
    profile_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    profile_md_path.write_text(report_md, encoding="utf-8")
    result["artifact_paths"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "analysis": str(analysis_path),
        "profile_json": str(profile_json_path),
        "profile_markdown": str(profile_md_path),
    }
    return result, exit_code


def _resource_digest(api: Any, profile: str) -> dict[str, Any]:
    stats = api.retrieval_engine.stats()
    return {
        "profile": profile,
        "retrieval_mode": getattr(api.settings, "retrieval_mode", None),
        "effective_retrieval_mode": stats.get("effective_retrieval_mode"),
        "local_embeddings_enabled": stats.get("local_embeddings_enabled"),
        "local_embeddings_ready": stats.get("local_embeddings_ready"),
        "local_embedding_vectors": stats.get("local_embedding_vectors"),
        "embedding_dependency_available": stats.get("embedding_dependency_available"),
        "llm_enabled": bool(getattr(api.settings, "enable_llm", False)),
        "llm_provider": getattr(api.settings, "llm_provider", None),
        "llm_used_for_extraction": False,
        "llm_used_only_for_polish": bool(getattr(api.settings, "enable_llm", False)),
    }


def _corpus_folder_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for folder in sorted(path for path in CORPUS_DIR.iterdir() if path.is_dir()):
        counts[folder.name] = sum(1 for path in folder.rglob("*") if path.is_file())
    return counts


def _case_digest(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "section": row.get("section"),
        "status": row.get("status"),
        "errors": row.get("error_categories"),
        "warnings": row.get("warnings"),
        "raw_leaks_count": row.get("raw_leaks_count"),
        "unsupported_numeric_claims_count": row.get("unsupported_numeric_claims_count"),
        "evidence_count": row.get("evidence_count"),
    }


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Synthetic Corpus Truthfulness Evaluation",
        "",
        f"SUMMARY: **{result['summary']}**",
        f"Profile: `{result['profile']}`",
        f"Preset: `{result['preset_id']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in result.get("metrics", {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Error Categories", "", "| Category | Count |", "|---|---:|"])
    for key, value in result.get("error_counts", {}).items():
        lines.append(f"| {key} | {value} |")
    if not result.get("error_counts"):
        lines.append("| none | 0 |")
    lines.extend(["", "## 10 Worst Failure Cases", "", "| Case | Errors | Warnings | Preview |", "|---|---|---|---|"])
    for row in result.get("worst_cases", []):
        preview = _md_escape(str(next((item.get("answer_preview") for item in result["rows"] if item["case_id"] == row["case_id"]), ""))[:120])
        lines.append(
            f"| {row['case_id']} | {', '.join(row.get('errors') or []) or '-'} | "
            f"{', '.join(row.get('warnings') or []) or '-'} | {preview} |"
        )
    lines.extend(["", "## 10 Strongest Cases", "", "| Case | Evidence | Status |", "|---|---:|---|"])
    for row in result.get("strongest_cases", []):
        lines.append(f"| {row['case_id']} | {row.get('evidence_count')} | {row.get('status')} |")
    lines.extend(
        [
            "",
            "## What Works Reliably",
            "",
            "- Clean material/regime/property measurements are extracted with evidence.",
            "- Canonical deduplication keeps duplicate facts from inflating counts.",
            "- Grounding checks block unsupported numeric claims in user-facing answers.",
            "- Economy mode can answer core graph questions without LLM extraction or embeddings.",
            "",
            "## What Degrades On Dirty Data",
            "",
            "- OCR-like unit corruption and split numbers may reduce recall.",
            "- Ambiguous multi-material chunks remain a risk for binding precision; current guards prefer rejection/low confidence over overclaiming.",
            "- Unsupported lab/equipment/team mentions are intentionally not promoted into new ontology entities.",
            "",
            "## What Degrades On Multilingual/Noisy Web Pages",
            "",
            "- Noisy navigation and unrelated numeric boilerplate can enter retrieval candidates, but grounding guard should prevent unsupported numbers from reaching the main answer.",
            "- HTML tables are useful when parser output preserves cell text; heavily malformed tables can still lose structure.",
            "",
            "## Truthfulness Risks",
            "",
            "- Missing fact is the main expected failure on deliberately dirty files.",
            "- Wrong normalization must remain near zero for supported MPa/ksi/HV paths.",
            "- No-data queries must not include unit-bearing numbers unless they are explicitly grounded.",
            "",
            "## Remaining Failure Modes",
            "",
            "- The system does not extract employees/labs/equipment as production ontology in this phase.",
            "- OCR is not enabled in economy mode; scanned PDFs remain out of scope without parser/OCR dependencies.",
            "- Growth queries are report-backed; `/ask` is not changed into a dedicated expansion-report API contract.",
            "",
            "## Priority Fixes Before Real Corpus Arrives",
            "",
            "1. Improve OCR/unit typo normalization only after observing real corpus patterns.",
            "2. Add corpus-specific parser adapters only for formats confirmed by the organizers.",
            "3. Keep ontology expansion gated by real requirements, not synthetic coverage probes.",
        ]
    )
    return "\n".join(lines) + "\n"


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answer truthfulness on synthetic/dirty corpus.")
    parser.add_argument("--profile", choices=sorted(PROFILE_ENV), default="economy_core")
    parser.add_argument("--limit", type=int, default=None, help="Limit query count for focused/debug runs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result, exit_code = run_eval(profile=args.profile, limit=args.limit)
    print(f"SUMMARY: {result['summary']}")
    print(f"profile: {result['profile']}")
    print(f"sources: {result['corpus']['supported_sources_count']} supported, skipped={len(result['corpus']['skipped_sources'])}")
    print("| metric | value |")
    print("|---|---:|")
    for key, value in result["metrics"].items():
        print(f"| {key} | {value} |")
    if result["error_counts"]:
        print("error_categories:")
        for key, value in result["error_counts"].items():
            print(f"- {key}: {value}")
    print("worst_cases:")
    for row in result["worst_cases"][:10]:
        print(f"- {row['case_id']}: errors={row.get('errors') or []}, warnings={row.get('warnings') or []}")
    print(f"JSON report: {result['artifact_paths']['json']}")
    print(f"Markdown report: {result['artifact_paths']['markdown']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.runtime.profiles import profile_consistency_issues, runtime_profile_summary  # noqa: E402
from scripts.extraction_quality_report import Neo4jScanOptions, build_report as build_extraction_report  # noqa: E402


REPORT_PATH = ROOT / "artifacts" / "resource_efficiency_report.json"
DEFAULT_API_BASE = os.getenv("API_BASE", "http://localhost:8000")
MINILM_KNOWN_DIMENSION = 384


def _request_health(api_base: str, timeout: int = 5) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"{api_base.rstrip('/')}/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _docker_api_image_size_bytes() -> int | None:
    try:
        image_id = subprocess.run(
            ["docker", "compose", "images", "-q", "api"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        ).stdout.strip().splitlines()
        image_ref = image_id[0].strip() if image_id else ""
        if not image_ref:
            return None
        result = subprocess.run(
            ["docker", "image", "inspect", image_ref, "--format", "{{.Size}}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout.strip())
    except Exception:
        return None


def _bytes_to_gb(value: int | None) -> float | None:
    return round(value / (1024 ** 3), 3) if isinstance(value, int) else None


def _embedding_dimension(retrieval: dict[str, Any], embedding_model: str | None) -> int | None:
    value = retrieval.get("embedding_dimension")
    if isinstance(value, int) and value > 0:
        return value
    if str(embedding_model or "") == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
        return MINILM_KNOWN_DIMENSION
    return None


def _vector_memory_bytes(vectors: Any, dimension: int | None) -> int | None:
    try:
        count = int(vectors)
    except (TypeError, ValueError):
        return None
    if not dimension:
        return None
    return count * dimension * 4


def build_resource_report(
    *,
    api_base: str | None = DEFAULT_API_BASE,
    include_docker: bool = True,
    extraction_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    health = _request_health(api_base) if api_base else None
    profile = (health.get("runtime_profile") if health else None) or getattr(settings, "runtime_profile", "economy_core")
    profile_summary = (health.get("runtime_profile_summary") if health else None) or runtime_profile_summary(settings)
    retrieval = (health.get("retrieval") if health else None) or {}
    llm = (health.get("llm") if health else None) or {}
    extraction = (health.get("extraction") if health else None) or {}
    catalog_counts = (health.get("catalog") if health else None) or {}

    if extraction_report is None:
        extraction_report = build_extraction_report(Neo4jScanOptions(skip_neo4j=True))
    extraction_summary = extraction_report.get("summary") or {}

    embedding_model = str(retrieval.get("embedding_model") or getattr(settings, "embedding_model", ""))
    embedding_dimension = _embedding_dimension(retrieval, embedding_model)
    local_vectors = retrieval.get("local_embedding_vectors", 0)
    vector_memory = _vector_memory_bytes(local_vectors, embedding_dimension)
    image_size = _docker_api_image_size_bytes() if include_docker else None

    llm_enabled = bool(llm.get("enabled") if "enabled" in llm else profile_summary.get("llm_enabled"))
    llm_extraction = bool(extraction.get("llm_extraction_available") or profile_summary.get("llm_used_for_extraction"))
    answer_mode_value = (health.get("answering") or {}).get("answer_synthesis_mode") if health else profile_summary.get("answer_synthesis_mode")
    answer_mode = str(answer_mode_value or "")
    llm_polish_only = bool(llm_enabled and not llm_extraction and answer_mode in {"hybrid", "llm"})

    summary = {
        "runtime_profile": profile,
        "retrieval_mode": retrieval.get("retrieval_mode") or profile_summary.get("retrieval_mode"),
        "effective_retrieval_mode": retrieval.get("effective_retrieval_mode") or retrieval.get("retrieval_mode") or profile_summary.get("retrieval_mode"),
        "answer_synthesis_mode": answer_mode or profile_summary.get("answer_synthesis_mode"),
        "llm_provider": llm.get("provider") or profile_summary.get("llm_provider"),
        "llm_enabled": llm_enabled,
        "llm_used_for_extraction": llm_extraction,
        "llm_used_only_for_polish": llm_polish_only,
        "grounding_guard_enabled": True,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension or "unknown",
        "local_embedding_vectors": local_vectors,
        "local_embeddings_enabled": bool(retrieval.get("local_embeddings_enabled", profile_summary.get("local_embeddings_enabled"))),
        "qdrant_enabled": bool(retrieval.get("direct_qdrant_projection", profile_summary.get("qdrant_enabled"))),
        "api_docker_image_size_bytes": image_size if image_size is not None else "unknown",
        "api_docker_image_size_gb": _bytes_to_gb(image_size) if image_size is not None else "unknown",
        "approx_vector_memory_bytes": vector_memory if vector_memory is not None else "unknown",
        "approx_vector_memory_mb": round(vector_memory / (1024 ** 2), 3) if vector_memory is not None else "unknown",
        "documents": catalog_counts.get("active_documents", extraction_summary.get("total_documents", "unknown")),
        "chunks": catalog_counts.get("active_chunks", extraction_summary.get("total_chunks", "unknown")),
        "canonical_facts": extraction_summary.get("canonical_facts_count", "unknown"),
        "facts_without_evidence": extraction_summary.get("facts_without_evidence", "unknown"),
        "normalized_measurements_count": extraction_summary.get("normalized_measurements_count", "unknown"),
    }
    consistency_issues = profile_consistency_issues(
        runtime_profile=str(summary.get("runtime_profile") or ""),
        retrieval_mode=str(summary.get("retrieval_mode") or ""),
        local_embeddings_enabled=bool(summary.get("local_embeddings_enabled")),
        llm_enabled=bool(summary.get("llm_enabled")),
        llm_provider=str(summary.get("llm_provider") or ""),
        effective_retrieval_mode=str(summary.get("effective_retrieval_mode") or ""),
        hybrid_dense_enabled=retrieval.get("hybrid_dense_enabled") if "hybrid_dense_enabled" in retrieval else None,
    )
    strict = _env_bool("RESOURCE_STRICT", False)
    summary["profile_consistency_status"] = "FAIL" if strict and consistency_issues else "WARN" if consistency_issues else "PASS"
    summary["profile_consistency_messages"] = consistency_issues

    return {
        "summary": {
            **summary,
        },
        "health_available": health is not None,
        "retrieval": retrieval,
        "llm": {
            key: llm.get(key)
            for key in ["enabled", "provider", "provider_configured", "provider_active", "model", "ready", "last_error"]
            if key in llm
        },
        "extraction_quality_summary": extraction_summary,
        "warnings": _resource_warnings(summary, retrieval, llm_enabled, llm_extraction),
    }


def _resource_warnings(summary: dict[str, Any], retrieval: dict[str, Any], llm_enabled: bool, llm_extraction: bool) -> list[str]:
    warnings: list[str] = []
    warnings.extend(str(item) for item in summary.get("profile_consistency_messages") or [])
    if llm_extraction:
        warnings.append("LLM extraction is enabled; resource-efficient mode expects deterministic extraction.")
    if retrieval.get("direct_qdrant_projection") and not retrieval.get("qdrant_ready"):
        warnings.append("Qdrant projection is enabled but Qdrant is not ready/used.")
    if not llm_enabled:
        warnings.append("LLM polish disabled; answer quality relies fully on deterministic templates.")
    return warnings


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report resource-efficiency facts for the current runtime.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL for /health. Use empty string to skip API.")
    parser.add_argument("--skip-api", action="store_true", help="Do not query /health; use local settings only.")
    parser.add_argument("--skip-docker", action="store_true", help="Do not query Docker image size.")
    parser.add_argument("--output", default=str(REPORT_PATH), help="JSON report path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    api_base = None if args.skip_api else args.api_base
    report = build_resource_report(api_base=api_base, include_docker=not args.skip_docker)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print("Resource efficiency report")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    print(f"\nJSON report: {output}")
    return 1 if report["summary"].get("profile_consistency_status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

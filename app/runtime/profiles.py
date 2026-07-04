"""Resource-oriented runtime profile defaults.

The selected RUNTIME_PROFILE is enforced for runtime-heavy knobs after
Settings/BaseSettings reads explicit environment variables.  Conflicting env
values are kept as warnings, while secrets and connection settings are never
touched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


MINILM_MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RUNTIME_PROFILE_IDS = {
    "economy_core",
    "economy_guarded_llm",
    "balanced_hybrid",
    "quality_full",
}

DEFAULT_RUNTIME_PROFILE = "economy_core"

PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "economy_core": {
        "RETRIEVAL_MODE": "bm25",
        "ENABLE_LOCAL_EMBEDDINGS": False,
        "EAGER_LOCAL_EMBEDDINGS": False,
        "DIRECT_QDRANT_PROJECTION": False,
        "EMBEDDING_MODEL": MINILM_MULTILINGUAL_MODEL,
        "ANSWER_SYNTHESIS_MODE": "template",
        "ENABLE_LLM": False,
        "LLM_PROVIDER": "offline",
        "EXTRACTION_MODE": "deterministic",
        "EXTRACTION_ENABLE_LLM": False,
        "RETRIEVAL_QUERY_EXPANSION": True,
    },
    "economy_guarded_llm": {
        "RETRIEVAL_MODE": "bm25",
        "ENABLE_LOCAL_EMBEDDINGS": False,
        "EAGER_LOCAL_EMBEDDINGS": False,
        "DIRECT_QDRANT_PROJECTION": False,
        "EMBEDDING_MODEL": MINILM_MULTILINGUAL_MODEL,
        "ANSWER_SYNTHESIS_MODE": "hybrid",
        "ENABLE_LLM": True,
        "LLM_PROVIDER": "auto",
        "EXTRACTION_MODE": "deterministic",
        "EXTRACTION_ENABLE_LLM": False,
        "RETRIEVAL_QUERY_EXPANSION": True,
    },
    "balanced_hybrid": {
        "RETRIEVAL_MODE": "hybrid",
        "ENABLE_LOCAL_EMBEDDINGS": True,
        "EAGER_LOCAL_EMBEDDINGS": False,
        "DIRECT_QDRANT_PROJECTION": False,
        "EMBEDDING_MODEL": MINILM_MULTILINGUAL_MODEL,
        "ANSWER_SYNTHESIS_MODE": "template",
        "ENABLE_LLM": False,
        "LLM_PROVIDER": "offline",
        "EXTRACTION_MODE": "deterministic",
        "EXTRACTION_ENABLE_LLM": False,
        "RETRIEVAL_QUERY_EXPANSION": True,
    },
    "quality_full": {
        "RETRIEVAL_MODE": "hybrid",
        "ENABLE_LOCAL_EMBEDDINGS": True,
        "EAGER_LOCAL_EMBEDDINGS": False,
        "DIRECT_QDRANT_PROJECTION": False,
        "EMBEDDING_MODEL": MINILM_MULTILINGUAL_MODEL,
        "ANSWER_SYNTHESIS_MODE": "hybrid",
        "ENABLE_LLM": True,
        "LLM_PROVIDER": "auto",
        "EXTRACTION_MODE": "deterministic",
        "EXTRACTION_ENABLE_LLM": False,
        "RETRIEVAL_QUERY_EXPANSION": True,
    },
}

PROFILE_RUNTIME_FIELDS = {
    "RETRIEVAL_MODE": "retrieval_mode",
    "ENABLE_LOCAL_EMBEDDINGS": "enable_local_embeddings",
    "EAGER_LOCAL_EMBEDDINGS": "eager_local_embeddings",
    "DIRECT_QDRANT_PROJECTION": "direct_qdrant_projection",
    "EMBEDDING_MODEL": "embedding_model",
    "ANSWER_SYNTHESIS_MODE": "answer_synthesis_mode",
    "ENABLE_LLM": "enable_llm",
    "LLM_PROVIDER": "llm_provider",
    "EXTRACTION_MODE": "extraction_mode",
    "EXTRACTION_ENABLE_LLM": "extraction_enable_llm",
    "RETRIEVAL_QUERY_EXPANSION": "retrieval_query_expansion",
}


def normalize_runtime_profile(value: str | None) -> str:
    """Return a supported profile id, falling back to the resource-light profile."""

    profile = str(value or "").strip().lower()
    return profile if profile in RUNTIME_PROFILE_IDS else DEFAULT_RUNTIME_PROFILE


def runtime_profile_from_environment(env_file: str | Path = ".env") -> str:
    """Resolve RUNTIME_PROFILE from environment or local .env for Settings defaults."""

    explicit = os.getenv("RUNTIME_PROFILE")
    if explicit:
        return normalize_runtime_profile(explicit)
    dotenv_value = _read_dotenv_key(Path(env_file), "RUNTIME_PROFILE")
    return normalize_runtime_profile(dotenv_value)


def profile_defaults(profile: str | None = None) -> dict[str, Any]:
    resolved = normalize_runtime_profile(profile or runtime_profile_from_environment())
    return dict(PROFILE_DEFAULTS[resolved])


def profile_default(name: str, fallback: Any = None, profile: str | None = None) -> Any:
    return profile_defaults(profile).get(name, fallback)


def apply_runtime_profile_policy(settings: Any) -> list[str]:
    """Make runtime profiles deterministic after environment loading.

    Pydantic BaseSettings correctly reads explicit env values, but for demo and
    reproducible evals the selected RUNTIME_PROFILE is the contract.  We keep a
    warning list for observability and then apply the profile values to runtime
    knobs only. Secrets and connection settings are never touched.
    """

    profile = normalize_runtime_profile(getattr(settings, "runtime_profile", None))
    defaults = profile_defaults(profile)
    warnings: list[str] = []
    for env_key, attr in PROFILE_RUNTIME_FIELDS.items():
        desired = defaults.get(env_key)
        current = getattr(settings, attr, None)
        if _normalize_compare(current) != _normalize_compare(desired):
            warnings.append(f"{env_key}={current!r} overridden by RUNTIME_PROFILE={profile}")
            setattr(settings, attr, desired)
    setattr(settings, "runtime_profile", profile)
    object.__setattr__(settings, "profile_policy_warnings", warnings)
    return warnings


def effective_profile_value(name: str, fallback: Any = None, profile: str | None = None) -> Any:
    """Return explicit env value when set, otherwise the selected profile default."""

    if name in os.environ and str(os.environ.get(name) or "").strip() != "":
        return os.environ[name]
    return profile_default(name, fallback, profile=profile)


def bool_from_env_or_profile(name: str, fallback: bool = False, profile: str | None = None) -> bool:
    return _to_bool(effective_profile_value(name, fallback, profile=profile))


def str_from_env_or_profile(name: str, fallback: str = "", profile: str | None = None) -> str:
    value = effective_profile_value(name, fallback, profile=profile)
    return str(value if value is not None else fallback)


def runtime_profile_summary(settings: Any) -> dict[str, Any]:
    """Return profile/resource facts safe for health and reports."""

    llm_enabled = bool(getattr(settings, "enable_llm", False))
    extraction_llm = bool(getattr(settings, "extraction_enable_llm", False))
    answer_mode = str(getattr(settings, "answer_synthesis_mode", "template") or "template").lower()
    return {
        "runtime_profile": normalize_runtime_profile(getattr(settings, "runtime_profile", None)),
        "effective_runtime_mode": "profile_enforced",
        "profile_policy_warnings": list(getattr(settings, "profile_policy_warnings", []) or []),
        "retrieval_mode": getattr(settings, "retrieval_mode", None),
        "local_embeddings_enabled": bool(getattr(settings, "enable_local_embeddings", False)),
        "eager_local_embeddings": bool(getattr(settings, "eager_local_embeddings", False)),
        "embedding_model": getattr(settings, "embedding_model", None),
        "qdrant_enabled": bool(getattr(settings, "direct_qdrant_projection", False)),
        "llm_enabled": llm_enabled,
        "llm_provider": getattr(settings, "llm_provider", None),
        "llm_used_for_extraction": extraction_llm,
        "llm_used_only_for_polish": bool(llm_enabled and not extraction_llm and answer_mode in {"hybrid", "llm"}),
        "answer_synthesis_mode": answer_mode,
        "grounding_guard_enabled": True,
    }


def profile_consistency_issues(
    *,
    runtime_profile: str | None,
    retrieval_mode: str | None,
    local_embeddings_enabled: bool,
    llm_enabled: bool,
    llm_provider: str | None,
    effective_retrieval_mode: str | None = None,
    hybrid_dense_enabled: bool | None = None,
) -> list[str]:
    """Return user-facing profile/runtime mismatch warnings."""

    profile = normalize_runtime_profile(runtime_profile)
    mode = str(retrieval_mode or "").strip().lower()
    provider = str(llm_provider or "").strip().lower()
    effective = str(effective_retrieval_mode or mode).strip().lower()
    issues: list[str] = []
    if profile == "economy_core" and (
        local_embeddings_enabled
        or mode == "hybrid"
        or llm_enabled
        or (provider and provider != "offline")
    ):
        issues.append("Profile economy_core is overridden by explicit env settings.")
    if profile == "balanced_hybrid" and (
        not local_embeddings_enabled
        or mode != "hybrid"
        or effective != "hybrid"
        or hybrid_dense_enabled is False
    ):
        issues.append("balanced_hybrid requested but dense retrieval is disabled/degraded.")
    return issues


def _read_dotenv_key(path: Path, key: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    prefix = f"{key}="
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        return value.strip("'\"")
    return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_compare(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value if value is not None else "").strip().lower()

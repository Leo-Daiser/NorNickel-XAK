from __future__ import annotations

from app.runtime.profiles import (
    MINILM_MULTILINGUAL_MODEL,
    apply_runtime_profile_policy,
    normalize_runtime_profile,
    profile_consistency_issues,
    profile_defaults,
)


def test_economy_core_disables_embeddings_and_llm_polish() -> None:
    defaults = profile_defaults("economy_core")

    assert defaults["RETRIEVAL_MODE"] == "bm25"
    assert defaults["ENABLE_LOCAL_EMBEDDINGS"] is False
    assert defaults["ENABLE_LLM"] is False
    assert defaults["LLM_PROVIDER"] == "offline"
    assert defaults["ANSWER_SYNTHESIS_MODE"] == "template"
    assert defaults["EXTRACTION_MODE"] == "deterministic"
    assert defaults["EXTRACTION_ENABLE_LLM"] is False


def test_balanced_hybrid_enables_lazy_minilm_embeddings_without_qdrant() -> None:
    defaults = profile_defaults("balanced_hybrid")

    assert defaults["RETRIEVAL_MODE"] == "hybrid"
    assert defaults["ENABLE_LOCAL_EMBEDDINGS"] is True
    assert defaults["EAGER_LOCAL_EMBEDDINGS"] is False
    assert defaults["DIRECT_QDRANT_PROJECTION"] is False
    assert defaults["EMBEDDING_MODEL"] == MINILM_MULTILINGUAL_MODEL
    assert defaults["ENABLE_LLM"] is False
    assert defaults["LLM_PROVIDER"] == "offline"
    assert defaults["ANSWER_SYNTHESIS_MODE"] == "template"


def test_unknown_runtime_profile_falls_back_to_economy_core() -> None:
    assert normalize_runtime_profile("unknown-heavy-profile") == "economy_core"


def test_profile_consistency_detects_economy_overrides() -> None:
    issues = profile_consistency_issues(
        runtime_profile="economy_core",
        retrieval_mode="hybrid",
        local_embeddings_enabled=True,
        llm_enabled=True,
        llm_provider="mistral",
        effective_retrieval_mode="hybrid",
        hybrid_dense_enabled=True,
    )

    assert issues == ["Profile economy_core is overridden by explicit env settings."]


def test_profile_consistency_detects_balanced_degradation() -> None:
    issues = profile_consistency_issues(
        runtime_profile="balanced_hybrid",
        retrieval_mode="hybrid",
        local_embeddings_enabled=True,
        llm_enabled=False,
        llm_provider="offline",
        effective_retrieval_mode="hybrid_degraded_to_bm25",
        hybrid_dense_enabled=False,
    )

    assert issues == ["balanced_hybrid requested but dense retrieval is disabled/degraded."]


def test_runtime_profile_policy_overrides_conflicting_runtime_env_values() -> None:
    class DummySettings:
        runtime_profile = "economy_core"
        retrieval_mode = "hybrid"
        enable_local_embeddings = True
        eager_local_embeddings = False
        direct_qdrant_projection = False
        embedding_model = "custom"
        answer_synthesis_mode = "hybrid"
        enable_llm = True
        llm_provider = "mistral"
        extraction_mode = "deterministic"
        extraction_enable_llm = False
        retrieval_query_expansion = True

    settings = DummySettings()
    warnings = apply_runtime_profile_policy(settings)

    assert warnings
    assert settings.retrieval_mode == "bm25"
    assert settings.enable_local_embeddings is False
    assert settings.enable_llm is False
    assert settings.llm_provider == "offline"
    assert settings.answer_synthesis_mode == "template"

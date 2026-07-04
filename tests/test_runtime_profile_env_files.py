from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_env_economy_example_has_no_llm_or_embeddings() -> None:
    text = _read(".env.economy.example")

    assert "RUNTIME_PROFILE=economy_core" in text
    assert "RETRIEVAL_MODE=bm25" in text
    assert "ENABLE_LOCAL_EMBEDDINGS=false" in text
    assert "ENABLE_LLM=false" in text
    assert "LLM_PROVIDER=offline" in text
    assert "MISTRAL_API_KEY=" not in text
    assert "OPENROUTER_API_KEY=" not in text


def test_env_balanced_example_enables_lazy_hybrid_embeddings() -> None:
    text = _read(".env.balanced.example")

    assert "RUNTIME_PROFILE=balanced_hybrid" in text
    assert "RETRIEVAL_MODE=hybrid" in text
    assert "ENABLE_LOCAL_EMBEDDINGS=true" in text
    assert "EAGER_LOCAL_EMBEDDINGS=false" in text
    assert "DIRECT_QDRANT_PROJECTION=false" in text
    assert "paraphrase-multilingual-MiniLM-L12-v2" in text


def test_env_quality_example_requires_grounding_guard_in_comment() -> None:
    text = _read(".env.quality.example")

    assert "RUNTIME_PROFILE=quality_full" in text
    assert "LLM grounding guard is required" in text
    assert "EXTRACTION_ENABLE_LLM=false" in text

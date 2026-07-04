from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_api_build_accepts_optional_embeddings_requirements() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "EXTRA_REQUIREMENTS: ${EXTRA_REQUIREMENTS:-requirements.txt}" in compose
    assert "ARG EXTRA_REQUIREMENTS=requirements.txt" in dockerfile
    assert "pip install -r /code/$EXTRA_REQUIREMENTS" in dockerfile
    assert "COPY requirements-embeddings.txt" in dockerfile


def test_requirements_embeddings_keeps_sentence_transformers_optional() -> None:
    base_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    embeddings_requirements = (ROOT / "requirements-embeddings.txt").read_text(encoding="utf-8")

    assert "sentence-transformers" not in base_requirements
    assert "-r requirements.txt" in embeddings_requirements
    assert "sentence-transformers" in embeddings_requirements

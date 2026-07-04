from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.llm.structured_llm import StructuredLLM
from app.runtime.presets import get_runtime_preset
from tests.strict_qa_helpers import reset_api
from tests.test_llm_provider_selection import _clear_llm_env


def test_health_masks_mistral_and_openrouter_secrets(tmp_path, monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/test-model")
    api = reset_api(tmp_path)
    monkeypatch.setattr(api, "_graph_db_for_repository", lambda force_retry=False: None)
    monkeypatch.setattr(api, "llm_client", StructuredLLM())

    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    payload = response.json()
    dumped = json.dumps(payload, ensure_ascii=False)

    assert payload["llm_provider_configured"] == "auto"
    assert payload["llm_provider_active"] == "mistral"
    assert payload["mistral_api_key_configured"] is True
    assert "mistral-secret" not in dumped
    assert "sk-or-secret" not in dumped
    assert "Authorization" not in dumped


def test_expert_max_still_uses_hybrid_answer_synthesis() -> None:
    preset = get_runtime_preset("expert_max")

    assert preset.answer_synthesis_mode == "hybrid"


def test_offline_reliable_still_uses_template_without_keys(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    client = StructuredLLM()
    preset = get_runtime_preset("offline_reliable")

    assert preset.answer_synthesis_mode == "template"
    assert client.status()["ready"] is False


def test_docker_compose_passes_mistral_env_to_api_service() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "working_dir: /code/hackathon_project" in compose
    assert "env_file:" in compose
    assert "- .env" in compose
    assert "RUNTIME_PROFILE: ${RUNTIME_PROFILE:-economy_core}" in compose
    assert "NEO4J_URI: ${NEO4J_DOCKER_URI:-bolt://neo4j:7687}" in compose
    assert "LLM_PROVIDER: none" not in compose
    for name in [
        "MISTRAL_API_KEY:",
        "OPENROUTER_API_KEY:",
        "LLM_API_KEY:",
    ]:
        assert name not in compose

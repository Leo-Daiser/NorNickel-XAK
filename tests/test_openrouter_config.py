from __future__ import annotations

import pytest

from app.llm.structured_llm import StructuredLLM


@pytest.fixture(autouse=True)
def _clear_llm_enabled_flags(monkeypatch):
    monkeypatch.delenv("LLM_ENABLED", raising=False)
    monkeypatch.delenv("ENABLE_LLM", raising=False)


def test_openrouter_aliases_configure_provider(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/test-model")
    client = StructuredLLM()
    status = client.status()
    assert status["provider"] == "openrouter"
    assert status["base_url"] == "https://openrouter.ai/api/v1"
    assert status["model"] == "openrouter/test-model"
    assert status["api_key_configured"] is True
    assert status["ready"] is True


def test_openrouter_missing_model_reports_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client = StructuredLLM()
    status = client.status()
    assert status["provider"] == "openrouter"
    assert status["ready"] is False
    assert "LLM_MODEL/OPENROUTER_MODEL is missing" in status["last_error"]


def test_llm_enabled_alias_is_supported(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_MODEL", "openrouter/test-model")
    client = StructuredLLM()
    assert client.status()["ready"] is True


def test_openrouter_detected_from_llm_api_key_and_model(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_MODEL", "openrouter/free")
    monkeypatch.setenv("LLM_BASE_URL", "http://host.docker.internal:11434")
    client = StructuredLLM()
    status = client.status()
    assert status["provider"] == "openrouter"
    assert status["base_url"] == "https://openrouter.ai/api/v1"
    assert status["model"] == "openrouter/free"
    assert status["ready"] is True


def test_openrouter_placeholder_model_is_not_ready(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "replace-with-openrouter-model-slug")
    client = StructuredLLM()
    status = client.status()
    assert status["provider"] == "openrouter"
    assert status["ready"] is False
    assert "placeholder" in status["last_error"]

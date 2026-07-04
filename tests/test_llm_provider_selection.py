from __future__ import annotations

from app.llm.structured_llm import StructuredLLM


_LLM_ENV_KEYS = [
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LLM_ENABLED",
    "ENABLE_LLM",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_BASE_URL",
    "MISTRAL_API_KEY",
    "MISTRAL_BASE_URL",
    "MISTRAL_MODEL",
    "MISTRAL_TIMEOUT_SECONDS",
    "MISTRAL_MAX_TOKENS",
    "MISTRAL_TEMPERATURE",
]


def _clear_llm_env(monkeypatch) -> None:
    for key in _LLM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    import app.llm.structured_llm as structured_llm

    monkeypatch.setattr(structured_llm.settings, "llm_provider", "none", raising=False)
    monkeypatch.setattr(structured_llm.settings, "llm_api_key", "", raising=False)
    monkeypatch.setattr(structured_llm.settings, "llm_model", "", raising=False)
    monkeypatch.setattr(structured_llm.settings, "llm_base_url", "", raising=False)
    monkeypatch.setattr(structured_llm.settings, "enable_llm", False, raising=False)
    monkeypatch.setattr(structured_llm.settings, "mistral_api_key", "", raising=False)
    monkeypatch.setattr(structured_llm.settings, "mistral_model", "mistral-small-latest", raising=False)
    monkeypatch.setattr(structured_llm.settings, "mistral_base_url", "https://api.mistral.ai/v1", raising=False)
    monkeypatch.setattr(structured_llm.settings, "openrouter_api_key", "", raising=False)
    monkeypatch.setattr(structured_llm.settings, "openrouter_model", "", raising=False)
    monkeypatch.setattr(structured_llm.settings, "openrouter_base_url", "https://openrouter.ai/api/v1", raising=False)


def test_llm_provider_mistral_selects_mistral(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")

    client = StructuredLLM()
    status = client.status()

    assert status["provider"] == "mistral"
    assert status["llm_provider_configured"] == "mistral"
    assert status["llm_provider_active"] == "mistral"
    assert status["mistral_model"] == "mistral-small-latest"
    assert status["mistral_api_key_configured"] is True
    assert status["ready"] is True


def test_llm_provider_openrouter_keeps_openrouter(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/test-model")

    client = StructuredLLM()
    status = client.status()

    assert status["provider"] == "openrouter"
    assert status["base_url"] == "https://openrouter.ai/api/v1"
    assert status["model"] == "openrouter/test-model"
    assert status["ready"] is True


def test_llm_provider_auto_prefers_mistral_when_key_exists(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/test-model")

    client = StructuredLLM()
    status = client.status()

    assert status["provider"] == "mistral"
    assert status["provider_configured"] == "auto"
    assert status["ready"] is True


def test_llm_provider_auto_uses_openrouter_without_mistral_key(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/test-model")

    client = StructuredLLM()
    status = client.status()

    assert status["provider"] == "openrouter"
    assert status["provider_configured"] == "auto"
    assert status["ready"] is True


def test_llm_provider_auto_without_keys_is_offline_not_ready(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "auto")

    client = StructuredLLM()
    status = client.status()

    assert status["provider"] == "offline"
    assert status["provider_configured"] == "auto"
    assert status["ready"] is False
    assert "offline" in status["last_error"]


def test_llm_provider_offline_disables_llm(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "offline")

    client = StructuredLLM()
    status = client.status()

    assert status["provider"] == "offline"
    assert status["ready"] is False
    assert "offline" in status["last_error"]


def test_llm_enabled_false_disables_provider_even_with_mistral_key(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")

    client = StructuredLLM()
    status = client.status()

    assert status["provider"] == "mistral"
    assert status["ready"] is False
    assert "LLM is disabled" in status["last_error"]


def test_enable_llm_false_alias_disables_provider_even_with_mistral_key(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("ENABLE_LLM", "false")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")

    client = StructuredLLM()
    status = client.status()

    assert status["provider"] == "mistral"
    assert status["ready"] is False
    assert "LLM is disabled" in status["last_error"]

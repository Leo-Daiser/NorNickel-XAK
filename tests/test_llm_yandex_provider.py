from __future__ import annotations

import json

from app.llm.structured_llm import StructuredLLM


def _clear_conflicting_llm_env(monkeypatch) -> None:
    for name in [
        "LLM_ENABLED",
        "ENABLE_LLM",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "MISTRAL_API_KEY",
        "MISTRAL_MODEL",
        "YANDEX_API_KEY",
        "YC_API_KEY",
        "YANDEX_FOLDER_ID",
        "YANDEX_MODEL_URI",
        "YANDEX_BASE_URL",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_yandex_provider_status_is_ready_without_secret_leak(monkeypatch) -> None:
    _clear_conflicting_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "yandex")
    monkeypatch.setenv("ENABLE_LLM", "true")
    monkeypatch.setenv("YANDEX_API_KEY", "secret-yandex-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder-id")
    monkeypatch.setenv("YANDEX_MODEL_URI", "gpt://folder-id/yandexgpt-5.1")
    monkeypatch.setenv("YANDEX_BASE_URL", "https://ai.api.cloud.yandex.net/v1")

    client = StructuredLLM()
    status = client.status()
    rendered = json.dumps(status, ensure_ascii=False)

    assert status["provider"] == "yandex"
    assert status["ready"] is True
    assert status["yandex_api_key_configured"] is True
    assert status["yandex_model_uri_configured"] is True
    assert status["yandex_base_url"] == "https://ai.api.cloud.yandex.net/v1"
    assert "secret-yandex-key" not in rendered


def test_yandex_provider_accepts_yc_api_key_alias(monkeypatch) -> None:
    _clear_conflicting_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "yandex")
    monkeypatch.setenv("ENABLE_LLM", "true")
    monkeypatch.setenv("YC_API_KEY", "secret-yc-key")
    monkeypatch.setenv("YANDEX_MODEL_URI", "gpt://folder-id/yandexgpt-5.1")

    client = StructuredLLM()
    status = client.status()

    assert status["provider"] == "yandex"
    assert status["ready"] is True
    assert status["yandex_api_key_configured"] is True

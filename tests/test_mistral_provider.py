from __future__ import annotations

import requests

from app.llm.structured_llm import StructuredLLM
from tests.test_llm_provider_selection import _clear_llm_env


class _OKResponse:
    status_code = 200
    text = '{"choices":[{"message":{"content":"{\\"answer\\":\\"OK\\"}"}}]}'

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"choices": [{"message": {"content": '{"answer":"OK"}'}}]}


class _ErrorResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        raise requests.HTTPError(response=self)

    def json(self):
        return {}


def test_mistral_provider_builds_chat_completion_request(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret")
    monkeypatch.setenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    monkeypatch.setenv("MISTRAL_MODEL", "mistral-small-latest")
    monkeypatch.setenv("MISTRAL_MAX_TOKENS", "1200")
    monkeypatch.setenv("MISTRAL_TEMPERATURE", "0.2")

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _OKResponse()

    monkeypatch.setattr("app.llm.structured_llm.requests.post", fake_post)

    client = StructuredLLM()
    content = client._chat("system", "user")

    assert content == '{"answer":"OK"}'
    assert captured["url"] == "https://api.mistral.ai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer mistral-secret"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["json"]["model"] == "mistral-small-latest"
    assert captured["json"]["temperature"] == 0.2
    assert captured["json"]["max_tokens"] == 1200
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert "response_format" not in captured["json"]


def test_mistral_missing_api_key_reports_not_ready(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mistral")

    client = StructuredLLM()
    status = client.status()

    assert status["provider"] == "mistral"
    assert status["ready"] is False
    assert status["mistral_api_key_configured"] is False
    assert "MISTRAL_API_KEY is missing" in status["last_error"]


def test_mistral_http_errors_are_safe_and_do_not_expose_key(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret")

    def fake_post(*args, **kwargs):
        return _ErrorResponse(401, "invalid token mistral-secret")

    monkeypatch.setattr("app.llm.structured_llm.requests.post", fake_post)

    client = StructuredLLM()
    assert client._chat("system", "user") is None
    status = client.status()

    assert "http_401" in status["last_error"]
    assert "mistral-secret" not in status["last_error"]
    assert "[redacted]" in status["last_error"]


def test_mistral_rate_limit_error_is_controlled(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret")

    def fake_post(*args, **kwargs):
        return _ErrorResponse(429, "rate limit")

    monkeypatch.setattr("app.llm.structured_llm.requests.post", fake_post)

    client = StructuredLLM()
    assert client._chat("system", "user") is None
    assert "http_429" in client.status()["last_error"]


def test_mistral_timeout_is_controlled(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret")

    def fake_post(*args, **kwargs):
        raise requests.Timeout()

    monkeypatch.setattr("app.llm.structured_llm.requests.post", fake_post)

    client = StructuredLLM()
    assert client._chat("system", "user") is None
    assert client.status()["last_error"] == "timeout"


def test_mistral_falls_back_to_openrouter_when_configured(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "openrouter/test-model")
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(url)
        if "mistral.ai" in url:
            return _ErrorResponse(429, "rate limit")
        return _OKResponse()

    monkeypatch.setattr("app.llm.structured_llm.requests.post", fake_post)

    client = StructuredLLM()
    assert client._chat("system", "user") == '{"answer":"OK"}'
    status = client.status()

    assert calls == [
        "https://api.mistral.ai/v1/chat/completions",
        "https://openrouter.ai/api/v1/chat/completions",
    ]
    assert status["llm_provider_configured"] == "mistral"
    assert status["llm_provider_active"] == "openrouter"
    assert "mistral_failed:http_429" in status["fallback_reason"]

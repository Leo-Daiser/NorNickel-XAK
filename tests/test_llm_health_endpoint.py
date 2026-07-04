from __future__ import annotations

from fastapi.testclient import TestClient

from tests.strict_qa_helpers import reset_api


class FakeLLM:
    def status(self):
        return {
            "enabled": True,
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openrouter/test-model",
            "api_key_configured": True,
            "ready": True,
            "last_error": "",
        }

    def test_connection(self):
        return {**self.status(), "success": True, "latency_ms": 1, "short_response": "OK", "response_preview": "OK", "error": ""}


class FakeNotReadyLLM:
    def status(self):
        return {
            "enabled": True,
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": None,
            "api_key_configured": True,
            "ready": False,
            "last_error": "OpenRouter API key is configured, but LLM_MODEL/OPENROUTER_MODEL is missing.",
        }

    def test_connection(self):
        return {**self.status(), "success": False, "latency_ms": None, "response_preview": None, "error": self.status()["last_error"]}


def test_system_capabilities_include_llm_status(tmp_path, monkeypatch) -> None:
    api = reset_api(tmp_path)
    monkeypatch.setattr(api, "llm_client", FakeLLM())
    client = TestClient(api.app)
    response = client.get("/system/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["llm"]["provider"] == "openrouter"
    assert payload["optional_features"]["llm_available"] is True


def test_system_test_llm_uses_configured_client(tmp_path, monkeypatch) -> None:
    api = reset_api(tmp_path)
    monkeypatch.setattr(api, "llm_client", FakeLLM())
    client = TestClient(api.app)
    response = client.post("/system/test-llm")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["provider"] == "openrouter"
    assert payload["short_response"] == "OK"
    assert payload["response_preview"] == "OK"


def test_system_test_llm_returns_not_ready_without_network(tmp_path, monkeypatch) -> None:
    api = reset_api(tmp_path)
    monkeypatch.setattr(api, "llm_client", FakeNotReadyLLM())
    client = TestClient(api.app)
    response = client.post("/system/test-llm")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["provider"] == "openrouter"
    assert "LLM_MODEL/OPENROUTER_MODEL is missing" in payload["error"]

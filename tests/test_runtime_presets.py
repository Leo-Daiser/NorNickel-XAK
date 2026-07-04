from __future__ import annotations

from fastapi.testclient import TestClient


def test_runtime_presets_contract() -> None:
    import app.api as api

    client = TestClient(api.app)
    response = client.get("/runtime/presets")
    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["preset_id"] for item in items] == ["expert_max", "strict_audit", "offline_reliable"]
    assert all(item["title"] for item in items)


def test_runtime_validate_preset() -> None:
    import app.api as api

    client = TestClient(api.app)
    response = client.post("/runtime/validate-preset", json={"preset_id": "offline_reliable"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["valid"] is True
    assert payload["diagnostics"]["preset_id"] == "offline_reliable"
    assert payload["diagnostics"]["effective_runtime_mode"]["kg_backend"] == "fallback"

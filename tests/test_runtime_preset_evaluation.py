from __future__ import annotations

from fastapi.testclient import TestClient


def test_runtime_preset_check_endpoint_shape() -> None:
    import app.api as api

    client = TestClient(api.app)
    response = client.post("/runtime/run-preset-check", json={"preset_id": "offline_reliable"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["preset"]["preset_id"] == "offline_reliable"
    assert "checks" in payload
    assert payload["diagnostics"]["preset_id"] == "offline_reliable"


def test_runtime_presets_eval_module_imports() -> None:
    import evaluation.eval_runtime_presets as eval_runtime_presets

    assert eval_runtime_presets.ROOT.exists()

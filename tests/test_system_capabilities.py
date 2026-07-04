from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_system_capabilities_contains_major_sections(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.get("/system/capabilities")
    assert response.status_code == 200, response.text
    payload = response.json()
    for key in ["kg_backend", "parser", "extraction", "analytics", "optional_features"]:
        assert key in payload
    assert payload["analytics"]["supported_intents"]


def test_optional_features_are_booleans(tmp_path) -> None:
    client = seeded_client(tmp_path)
    payload = client.get("/system/capabilities").json()
    optional = payload["optional_features"]
    assert optional
    assert all(isinstance(value, bool) for value in optional.values())

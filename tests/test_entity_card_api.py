from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_material_entity_card_contains_related_experiments_regimes_properties(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.get("/graph/entity/Material/ВТ6")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["entity"]["type"] == "Material"
    assert payload["related"]["experiments"]
    assert payload["related"]["regimes"]
    assert payload["related"]["properties"]
    assert payload["subgraph"]["nodes"]
    assert payload["subgraph"]["edges"]
    assert payload["diagnostics"]["kg_backend_active"]


def test_unknown_entity_returns_empty_card_gracefully(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.get("/graph/entity/Material/UNKNOWN-XYZ")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["entity"]["canonical_name"] == "UNKNOWN-XYZ"
    assert payload["related"]["experiments"] == []
    assert payload["subgraph"]["nodes"]


def test_entity_type_whitelist_is_enforced(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.get("/graph/entity/BadLabel/ВТ6")
    assert response.status_code == 400

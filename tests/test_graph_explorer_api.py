from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_graph_stats_returns_counts_and_backend(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.get("/graph/stats")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["experiments"] >= 2
    assert payload["materials"] >= 1
    assert payload["kg_backend_active"]


def test_graph_entities_returns_items_and_filters_by_type(tmp_path) -> None:
    client = seeded_client(tmp_path)
    all_response = client.get("/graph/entities", params={"limit": 20})
    assert all_response.status_code == 200, all_response.text
    assert all_response.json()["items"]

    material_response = client.get("/graph/entities", params={"entity_type": "Material", "limit": 20})
    assert material_response.status_code == 200, material_response.text
    items = material_response.json()["items"]
    assert items
    assert all(item["type"] == "Material" for item in items)
    assert any(item["canonical_name"] == "ВТ6" for item in items)


def test_graph_neighborhood_returns_nodes_and_edges(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.get("/graph/neighborhood", params={"entity_type": "Material", "entity_id": "ВТ6"})
    assert response.status_code == 200, response.text
    subgraph = response.json()["subgraph"]
    assert subgraph["nodes"]
    assert subgraph["edges"]


def test_invalid_entity_type_returns_400(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.get("/graph/entities", params={"entity_type": "BadLabel"})
    assert response.status_code == 400

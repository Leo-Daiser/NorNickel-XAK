from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_ask_material_overview_returns_analytical_fields(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post("/ask", params={"question": "Что уже делали по ВТ6?", "top_k": 6})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["analytical_intent"] == "material_overview"
    assert payload["answer_mode"] == "overview"
    assert payload["graph_context"]["facts_count"] > 0
    assert payload["sources"] or payload["evidence"]
    assert payload["diagnostics"]["answer_synthesis_mode"]


def test_expert_max_analytics_path_uses_llm_polish_when_ready(tmp_path, monkeypatch) -> None:
    client = seeded_client(tmp_path)
    import app.api as api

    class FakeLLM:
        def status(self):
            return {"ready": True, "provider": "mistral", "last_error": ""}

        def synthesize_answer(self, **kwargs):
            assert kwargs["facts"]
            return "LLM polished comparison answer"

    monkeypatch.setattr(api, "llm_client", FakeLLM())

    response = client.post(
        "/ask",
        json={"question": "Сравни ВТ6 и 7075-T6 по прочности.", "top_k": 12, "preset_id": "expert_max"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["diagnostics"]["preset_id"] == "expert_max"
    assert payload["diagnostics"]["effective_runtime_mode"]["answer_synthesis_mode"] == "hybrid"
    assert payload["diagnostics"]["llm_answer_polished"] is True
    assert "LLM polished comparison answer" in payload["answer"]


def test_ask_graph_neighborhood_returns_subgraph_stats(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post("/ask", params={"question": "Покажи связанные сущности по ВТ6", "top_k": 6})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["analytical_intent"] == "graph_neighborhood"
    assert payload["graph_context"]["subgraph_nodes"] > 0
    assert payload["subgraph"]["nodes"]


def test_ask_strict_no_exact_match_still_has_no_facts(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post(
        "/ask",
        params={"question": "Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?", "top_k": 6},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "no_exact_match"
    assert payload["facts"] == []
    assert payload["analytical_intent"] == "strict_material_regime_property"
    assert payload["retrieval"]["kg_backend_active"]
    assert "точных данных не найдено" in payload["answer"].lower()


def test_dense_candidates_do_not_bypass_exact_constraints(tmp_path, monkeypatch) -> None:
    client = seeded_client(tmp_path)
    import app.api as api

    monkeypatch.setattr(api.retrieval_engine, "dense_retrieve", lambda question, top_k=20: [(chunk.chunk_id, 1.0) for chunk in api.retrieval_engine.chunks])

    response = client.post(
        "/ask",
        json={
            "question": "Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?",
            "top_k": 12,
            "preset_id": "expert_max",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "no_exact_match"
    assert payload["facts"] == []
    assert payload["diagnostics"]["preset_id"] == "expert_max"

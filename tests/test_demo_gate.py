from __future__ import annotations

from scripts.demo_gate import (
    DemoGate,
    classify_image_size,
    classify_retrieval,
    contains_raw_leakage,
    run_resource_checks,
    validate_answer_graph,
    validate_answer_payload,
)


def test_demo_gate_classifies_hybrid_degradation_as_warning() -> None:
    level, message = classify_retrieval(
        {
            "retrieval": {
                "retrieval_mode": "hybrid",
                "effective_retrieval_mode": "hybrid_degraded_to_bm25",
                "bm25_ready": True,
                "local_embeddings_enabled": True,
                "hybrid_dense_enabled": False,
                "hybrid_degraded_reason": "dependency missing",
            }
        }
    )

    assert level == "WARN"
    assert "dependency missing" in message


def test_demo_gate_fails_when_expert_max_falls_to_offline_preset() -> None:
    checks = validate_answer_payload(
        {
            "status": "ok",
            "answer": "Офлайн-режим: шаблонный ответ.",
            "diagnostics": {"preset_id": "offline_reliable"},
        },
        expected_preset="expert_max",
    )

    assert any(check.level == "FAIL" and "Expected preset expert_max" in check.message for check in checks)
    assert any(check.level == "FAIL" and "offline mode" in check.message for check in checks)


def test_demo_gate_catches_raw_technical_leakage() -> None:
    assert contains_raw_leakage("Found chunk_abc and PropertyValue via MEASURES") is True

    checks = validate_answer_payload(
        {
            "status": "ok",
            "answer": "Main answer leaked chunk_123.",
            "diagnostics": {"preset_id": "expert_max"},
        },
        expected_preset="expert_max",
    )

    assert any(check.level == "FAIL" and "raw technical leakage" in check.message for check in checks)


def test_demo_gate_summary_fails_only_on_failures() -> None:
    gate = DemoGate()
    gate.pass_("Neo4j active")
    gate.warn("Hybrid degraded to BM25: dependency missing")

    assert gate.failed is False
    assert "SUMMARY: PASS" in gate.render()

    gate.fail("LLM is not ready")
    assert gate.failed is True
    assert "SUMMARY: FAIL" in gate.render()


def test_demo_gate_large_image_warns_unless_resource_strict() -> None:
    six_gb = 6 * 1024 ** 3

    relaxed = classify_image_size(six_gb, strict=False, max_gb=5)
    strict = classify_image_size(six_gb, strict=True, max_gb=5)

    assert relaxed.level == "WARN"
    assert strict.level == "FAIL"


def test_demo_gate_warns_on_profile_override_by_default(monkeypatch) -> None:
    import scripts.demo_gate as demo_gate

    monkeypatch.delenv("RESOURCE_STRICT", raising=False)
    monkeypatch.setattr(demo_gate, "_docker_api_image_size_bytes", lambda: None)
    gate = DemoGate()

    run_resource_checks(
        gate,
        {
            "runtime_profile": "economy_core",
            "retrieval": {
                "retrieval_mode": "hybrid",
                "effective_retrieval_mode": "hybrid",
                "local_embeddings_enabled": True,
                "hybrid_dense_enabled": True,
            },
            "llm": {"enabled": True, "provider": "mistral"},
            "extraction": {"llm_extraction_available": False},
            "answering": {"answer_synthesis_mode": "hybrid"},
            "qdrant_projection_enabled": False,
        },
    )

    assert any(check.level == "WARN" and "Profile economy_core is overridden" in check.message for check in gate.checks)


def test_demo_gate_resource_strict_fails_on_profile_override(monkeypatch) -> None:
    import scripts.demo_gate as demo_gate

    monkeypatch.setenv("RESOURCE_STRICT", "true")
    monkeypatch.setattr(demo_gate, "_docker_api_image_size_bytes", lambda: None)
    gate = DemoGate()

    run_resource_checks(
        gate,
        {
            "runtime_profile": "economy_core",
            "retrieval": {
                "retrieval_mode": "hybrid",
                "effective_retrieval_mode": "hybrid",
                "local_embeddings_enabled": True,
                "hybrid_dense_enabled": True,
            },
            "llm": {"enabled": True, "provider": "mistral"},
            "extraction": {"llm_extraction_available": False},
            "answering": {"answer_synthesis_mode": "hybrid"},
            "qdrant_projection_enabled": False,
        },
    )

    assert any(check.level == "FAIL" and "Profile economy_core is overridden" in check.message for check in gate.checks)


def test_demo_gate_validates_answer_graph_contract() -> None:
    checks = validate_answer_graph(
        {
            "status": "ok",
            "answer_mode": "comparison",
            "constraints": {"materials": ["ВТ6", "7075-T6"], "properties": ["прочность"]},
            "facts": [
                {"material": "ВТ6", "property": "прочность", "value": 950, "unit": "MPa", "effect": "increase"},
                {"material": "7075-T6", "property": "прочность", "value": 83, "unit": "ksi", "effect": "unknown"},
            ],
            "sources": [{"id": "doc_1"}, {"id": "doc_2"}],
        }
    )

    assert not [check for check in checks if check.level == "FAIL"]
    assert any("Russian graph control hint" in check.message for check in checks)
    assert any("Graph node dragging disabled" in check.message for check in checks)

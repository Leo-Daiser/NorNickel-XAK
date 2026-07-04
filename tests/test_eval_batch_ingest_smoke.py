from __future__ import annotations

from pathlib import Path

from evaluation.eval_batch_ingest_smoke import health_profile_warnings, select_ready_sample
from scripts.batch_ingest_corpus import build_ingest_plan


def test_select_ready_sample_balances_source_groups(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    group_a = root / "Статьи"
    group_b = root / "Обзоры"
    group_a.mkdir(parents=True)
    group_b.mkdir(parents=True)
    (group_a / "a1.txt").write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")
    (group_a / "a2.txt").write_text("После отжига сплава ВТ6 предел прочности составил 1120 MPa.", encoding="utf-8")
    (group_b / "b1.txt").write_text("The 7075-T6 aluminum alloy showed tensile strength of 77 ksi.", encoding="utf-8")
    (group_b / "skip.rar").write_bytes(b"rar")

    plan, _ = build_ingest_plan(root, max_file_mb=25.0)
    sample = select_ready_sample(plan, limit=3)

    assert len(sample) == 3
    assert [item.source_group for item in sample[:2]] == ["Обзоры", "Статьи"]
    assert all(item.planned_status == "ready" for item in sample)
    assert all(item.extension == ".txt" for item in sample)


def test_health_profile_warnings_detect_economy_overrides() -> None:
    health = {
        "runtime_profile": "economy_core",
        "retrieval": {
            "retrieval_mode": "hybrid",
            "effective_retrieval_mode": "hybrid_degraded_to_bm25",
            "local_embeddings_enabled": True,
        },
        "llm": {"enabled": True, "provider": "mistral"},
    }

    warnings = health_profile_warnings(health)

    assert "runtime_profile_economy_core_overridden_by_env" in warnings
    assert "hybrid_degraded_to_bm25" in warnings


def test_health_profile_warnings_pass_clean_economy() -> None:
    health = {
        "runtime_profile": "economy_core",
        "retrieval": {"retrieval_mode": "bm25", "effective_retrieval_mode": "bm25", "local_embeddings_enabled": False},
        "llm": {"enabled": False, "provider": "offline"},
    }

    assert health_profile_warnings(health) == []

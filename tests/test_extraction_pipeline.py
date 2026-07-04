from __future__ import annotations

import pytest

from app.extraction.pipeline import ExtractionPipeline
from app.models.schemas import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="chunk",
        doc_id="doc",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="test",
        metadata={"filename": "test.txt"},
    )


def test_deterministic_mode_works_without_llm() -> None:
    bundle = ExtractionPipeline(mode="deterministic", audit_enabled=False).extract_from_chunk(
        _chunk("Эксперимент EXP-001: ВТ6 после отжига показал прочность 1120 MPa.")
    )

    assert bundle.experiments
    assert bundle.diagnostics["extractor_version"] == "pipeline_v1"


def test_hybrid_mode_does_not_fail_when_llm_unavailable() -> None:
    bundle = ExtractionPipeline(mode="hybrid", enable_llm=False, audit_enabled=False).extract_from_chunk(
        _chunk("Эксперимент EXP-001: ВТ6 после отжига показал прочность 1120 MPa.")
    )

    assert bundle.experiments
    assert "llm_extractor_unavailable" in bundle.diagnostics["warnings"]


def test_llm_mode_fails_clearly_when_llm_unavailable() -> None:
    pipeline = ExtractionPipeline(mode="llm", enable_llm=False, audit_enabled=False)

    with pytest.raises(RuntimeError, match="LLM extractor is unavailable"):
        pipeline.extract_from_chunk(_chunk("ВТ6 отжиг прочность 1120 MPa"))


def test_accepted_rejected_split_works() -> None:
    bundle = ExtractionPipeline(mode="deterministic", audit_enabled=False).extract_from_chunk(
        _chunk("Сплав ВТ6 относится к титановым материалам.")
    )

    assert bundle.experiments == []
    assert bundle.quarantined_items
    assert not bundle.accepted_facts
    assert bundle.diagnostics["accepted_experiments"] == 0


def test_diagnostics_include_extractor_version() -> None:
    bundle = ExtractionPipeline(mode="deterministic", audit_enabled=False).extract_from_chunk(
        _chunk("Эксперимент EXP-001: ВТ6 после отжига показал прочность 1120 MPa.")
    )

    assert bundle.extractor_version == "pipeline_v1"
    assert bundle.diagnostics["min_confidence"] == 0.55

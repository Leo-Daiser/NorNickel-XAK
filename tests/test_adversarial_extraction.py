from __future__ import annotations

import json
from pathlib import Path

from app.extraction.pipeline import ExtractionPipeline
from app.models.schemas import Chunk


def _chunk(case: dict) -> Chunk:
    return Chunk(
        doc_id=f"doc_{case['id']}",
        chunk_id=f"chunk_{case['id']}",
        workspace_uid="test",
        text=case["text"],
        page_start=1,
        page_end=1,
        section_path="/",
        metadata={"filename": f"{case['id']}.txt"},
    )


def test_adversarial_extraction_gold_cases() -> None:
    cases = json.loads((Path(__file__).resolve().parents[1] / "evaluation" / "adversarial_extraction_gold.json").read_text(encoding="utf-8"))
    pipeline = ExtractionPipeline(mode="deterministic", audit_enabled=False)
    for case in cases:
        bundle = pipeline.extract_from_chunk(_chunk(case))
        materials = {item.canonical_name for exp in bundle.experiments for item in exp.materials}
        assert case["expected_material"] in materials, case["id"]
        assert not (materials & set(case.get("forbidden_materials") or [])), case["id"]
        measurements = [
            (measurement.property_canonical, measurement.value, measurement.unit, measurement.effect)
            for exp in bundle.experiments
            for measurement in exp.measurements
        ]
        for expected in case.get("expected_measurements") or []:
            assert any(
                prop == expected["property"]
                and value == expected["value"]
                and unit == expected["unit"]
                and (not expected.get("effect") or effect == expected["effect"])
                for prop, value, unit, effect in measurements
            ), case["id"]
        if case.get("expected_gap_property"):
            assert any(gap.property == case["expected_gap_property"] for gap in bundle.data_gaps), case["id"]

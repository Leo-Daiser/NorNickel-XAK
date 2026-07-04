from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.extraction.pipeline import ExtractionPipeline  # noqa: E402
from app.models.schemas import Chunk  # noqa: E402


GOLD_PATH = Path(__file__).with_name("extraction_gold.json")


def _chunk(case: dict[str, Any]) -> Chunk:
    metadata = {"filename": f"{case['id']}.txt"}
    if case.get("kind") == "table_row":
        metadata.update({"filename": f"{case['id']}.csv", "chunk_kind": "table_row", "row_id": 1})
    return Chunk(
        doc_id=f"doc_{case['id']}",
        chunk_id=f"chunk_{case['id']}",
        workspace_uid="eval",
        text=case["text"],
        page_start=1,
        page_end=1,
        section_path="evaluation",
        metadata=metadata,
    )


def _has_expected_experiment(bundle, case: dict[str, Any]) -> bool:
    material = case.get("expected_material")
    regime = case.get("expected_regime")
    prop = case.get("expected_property")
    if not any([material, regime, prop]):
        return bool(bundle.experiments) == bool(case.get("expected_experiments"))
    for experiment in bundle.experiments:
        materials = {item.canonical_name for item in experiment.materials}
        regimes = {item.canonical_name for item in experiment.regimes}
        properties = {item.property_canonical for item in experiment.measurements}
        if material and material not in materials:
            continue
        if regime and regime not in regimes:
            continue
        if prop and prop not in properties:
            continue
        return True
    return False


def _property_accuracy(bundle, case: dict[str, Any]) -> float:
    expected = case.get("expected_property")
    if not expected:
        return 1.0
    properties = {
        measurement.property_canonical
        for experiment in bundle.experiments
        for measurement in experiment.measurements
    }
    gap_properties = {gap.property for gap in bundle.data_gaps if gap.property}
    return 1.0 if expected in properties or expected in gap_properties else 0.0


def _unit_accuracy(bundle, case: dict[str, Any]) -> float:
    expected = case.get("expected_unit")
    if not expected:
        return 1.0
    units = {
        measurement.unit
        for experiment in bundle.experiments
        for measurement in experiment.measurements
        if measurement.unit
    }
    return 1.0 if expected in units else 0.0


def _evidence_presence(bundle) -> float:
    checked = 0
    ok = 0
    for experiment in bundle.experiments:
        checked += 1
        ok += int(bool(experiment.evidence and all(item.quote for item in experiment.evidence)))
        for measurement in experiment.measurements:
            checked += 1
            ok += int(bool(measurement.evidence and all(item.quote for item in measurement.evidence)))
    for gap in bundle.data_gaps:
        checked += 1
        ok += int(bool(gap.evidence and all(item.quote for item in gap.evidence)))
    return 1.0 if checked == 0 else ok / checked


def _rejection_ok(bundle, case: dict[str, Any]) -> bool:
    if case.get("expect_rejection"):
        return bool(bundle.rejected_items)
    if case.get("expected_no_merged_materials"):
        forbidden = set(case["expected_no_merged_materials"])
        return all(not forbidden.issubset({item.canonical_name for item in exp.materials}) for exp in bundle.experiments)
    return True


def main() -> int:
    cases = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    pipeline = ExtractionPipeline(mode="deterministic", audit_enabled=False)
    positive_cases = 0
    positive_hits = 0
    negative_cases = 0
    false_experiments = 0
    property_scores: list[float] = []
    unit_scores: list[float] = []
    evidence_scores: list[float] = []
    rejection_hits = 0

    for case in cases:
        bundle = pipeline.extract_from_chunk(_chunk(case))
        expected_experiments = int(case.get("expected_experiments", 0))
        if expected_experiments > 0:
            positive_cases += 1
            positive_hits += int(len(bundle.experiments) >= expected_experiments and _has_expected_experiment(bundle, case))
        elif not case.get("expected_gaps"):
            negative_cases += 1
            false_experiments += int(bool(bundle.experiments))

        if case.get("expected_gaps"):
            positive_cases += 1
            positive_hits += int(bool(bundle.data_gaps))

        property_scores.append(_property_accuracy(bundle, case))
        unit_scores.append(_unit_accuracy(bundle, case))
        evidence_scores.append(_evidence_presence(bundle))
        rejection_hits += int(_rejection_ok(bundle, case))

        status = "PASS" if _rejection_ok(bundle, case) else "FAIL"
        print(
            f"{status} {case['id']}: experiments={len(bundle.experiments)} "
            f"gaps={len(bundle.data_gaps)} rejected={len(bundle.rejected_items)}"
        )

    metrics = {
        "accepted_experiment_recall": positive_hits / max(positive_cases, 1),
        "false_experiment_rate": false_experiments / max(negative_cases, 1),
        "measurement_property_accuracy": mean(property_scores),
        "unit_normalization_accuracy": mean(unit_scores),
        "evidence_presence_rate": mean(evidence_scores),
        "rejection_accuracy": rejection_hits / max(len(cases), 1),
    }
    print("\nExtraction evaluation:")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
    passed = (
        metrics["accepted_experiment_recall"] >= 0.875
        and metrics["false_experiment_rate"] == 0.0
        and metrics["measurement_property_accuracy"] >= 0.875
        and metrics["unit_normalization_accuracy"] >= 1.0
        and metrics["evidence_presence_rate"] >= 1.0
        and metrics["rejection_accuracy"] >= 0.875
    )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

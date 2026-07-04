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


GOLD_PATH = Path(__file__).with_name("adversarial_extraction_gold.json")


def _chunk(case: dict[str, Any]) -> Chunk:
    return Chunk(
        doc_id=f"doc_{case['id']}",
        chunk_id=f"chunk_{case['id']}",
        workspace_uid="adversarial",
        text=case["text"],
        page_start=1,
        page_end=1,
        section_path="adversarial",
        metadata={"filename": f"{case['id']}.txt"},
    )


def _materials(bundle) -> set[str]:
    return {item.canonical_name for exp in bundle.experiments for item in exp.materials}


def _measurements(bundle) -> list[dict[str, Any]]:
    return [
        {
            "property": measurement.property_canonical,
            "value": measurement.value,
            "unit": measurement.unit,
            "effect": measurement.effect,
        }
        for exp in bundle.experiments
        for measurement in exp.measurements
    ]


def _has_measurement(actual: list[dict[str, Any]], expected: dict[str, Any]) -> bool:
    for row in actual:
        if row["property"] != expected["property"]:
            continue
        if expected.get("unit") and row.get("unit") != expected["unit"]:
            continue
        if expected.get("effect") and row.get("effect") != expected["effect"]:
            continue
        if expected.get("value") is not None and row.get("value") is not None:
            if abs(float(row["value"]) - float(expected["value"])) > 1e-6:
                continue
        return True
    return False


def main() -> int:
    cases = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    pipeline = ExtractionPipeline(mode="deterministic", audit_enabled=False)
    material_scores: list[float] = []
    not_material_scores: list[float] = []
    measurement_scores: list[float] = []
    employee_lab_scores: list[float] = []
    gap_scores: list[float] = []
    evidence_scores: list[float] = []

    for case in cases:
        bundle = pipeline.extract_from_chunk(_chunk(case))
        materials = _materials(bundle)
        actual_measurements = _measurements(bundle)
        material_scores.append(1.0 if case.get("expected_material") in materials else 0.0)
        forbidden = set(case.get("forbidden_materials") or [])
        not_material_scores.append(1.0 if not (materials & forbidden) else 0.0)
        expected_measurements = case.get("expected_measurements") or []
        measurement_scores.append(
            1.0
            if all(_has_measurement(actual_measurements, expected) for expected in expected_measurements)
            else 0.0
        )
        expected_labs = set(case.get("expected_laboratories") or [])
        expected_employees = set(case.get("expected_employees") or [])
        labs = {item.canonical_name for exp in bundle.experiments for item in exp.laboratories}
        employees = {item.canonical_name for exp in bundle.experiments for item in exp.employees}
        if expected_labs or expected_employees:
            employee_lab_scores.append(1.0 if expected_labs.issubset(labs) and expected_employees.issubset(employees) else 0.0)
        else:
            employee_lab_scores.append(1.0)
        if case.get("expected_gap_property"):
            gap_scores.append(1.0 if any(gap.property == case["expected_gap_property"] for gap in bundle.data_gaps) else 0.0)
        else:
            gap_scores.append(1.0)
        evidence_scores.append(
            1.0
            if all(exp.evidence and all(item.quote for item in exp.evidence) for exp in bundle.experiments)
            else 0.0
        )
        status = "PASS" if min(material_scores[-1], not_material_scores[-1], measurement_scores[-1], employee_lab_scores[-1], gap_scores[-1], evidence_scores[-1]) == 1.0 else "FAIL"
        print(
            f"{status} {case['id']}: materials={sorted(materials)} "
            f"measurements={actual_measurements} gaps={len(bundle.data_gaps)}"
        )

    metrics = {
        "material_accuracy": mean(material_scores),
        "experiment_id_not_material_rate": mean(not_material_scores),
        "measurement_property_value_accuracy": mean(measurement_scores),
        "employee_lab_extraction_rate": mean(employee_lab_scores),
        "gap_detection_accuracy": mean(gap_scores),
        "evidence_presence_rate": mean(evidence_scores),
    }
    print("\nAdversarial extraction evaluation:")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
    passed = all(value >= 1.0 for value in metrics.values())
    print("PASS" if passed else "FAIL")
    print("SUMMARY", json.dumps(metrics, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

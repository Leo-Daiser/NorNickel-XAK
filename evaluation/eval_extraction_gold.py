from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.unit_normalization import normalize_strength_to_mpa  # noqa: E402
from app.extraction.pipeline import ExtractionPipeline  # noqa: E402
from app.models.schemas import Chunk  # noqa: E402


ARTIFACT_PATH = ROOT / "artifacts" / "eval_extraction_gold.json"
CONFIDENCE_THRESHOLD = 0.70

GOLD_CASES: list[dict[str, Any]] = [
    {
        "id": "vt6_annealing_strength_ru",
        "text": "После отжига сплава ВТ6 предел прочности составил 980 MPa.",
        "expected": {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 980.0, "unit": "MPa"},
        "expect_numeric": True,
    },
    {
        "id": "al7075_aging_strength_ksi",
        "text": "The 7075-T6 aluminum alloy showed tensile strength of 77 ksi after aging treatment.",
        "expected": {
            "material": "7075-T6",
            "regime": "старение",
            "property": "прочность",
            "value": 77.0,
            "unit": "ksi",
            "normalized_value_mpa": 531.0,
        },
        "expect_numeric": True,
    },
    {
        "id": "ti64_annealed_uts",
        "text": "Ti-6Al-4V was annealed, resulting in ultimate tensile strength of 1120 MPa.",
        "expected": {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 1120.0, "unit": "MPa"},
        "expect_numeric": True,
    },
    {
        "id": "corrosion_qualitative_no_number",
        "text": "Коррозионная стойкость после обработки повысилась, но численные значения не приведены.",
        "expected": {"property": "коррозионная стойкость", "effect": "increase"},
        "expect_numeric": False,
    },
]


def _chunk(case: dict[str, Any]) -> Chunk:
    return Chunk(
        doc_id=f"doc_{case['id']}",
        chunk_id=f"chunk_{case['id']}",
        workspace_uid="eval",
        text=case["text"],
        page_start=1,
        page_end=1,
        section_path="gold",
        metadata={"filename": f"{case['id']}.txt"},
    )


def _measurements(bundle) -> list[Any]:
    return [measurement for experiment in bundle.experiments for measurement in experiment.measurements]


def _materials(bundle) -> set[str]:
    return {material.canonical_name for experiment in bundle.experiments for material in experiment.materials} | {
        entity.canonical_name for entity in bundle.entities if entity.entity_type == "Material"
    }


def _regimes(bundle) -> set[str]:
    return {regime.canonical_name for experiment in bundle.experiments for regime in experiment.regimes} | {
        entity.canonical_name for entity in bundle.entities if entity.entity_type == "ProcessRegime"
    }


def _properties(bundle) -> set[str]:
    return {measurement.property_canonical for measurement in _measurements(bundle)} | {
        entity.canonical_name for entity in bundle.entities if entity.entity_type == "Property"
    }


def _has_evidence(bundle) -> bool:
    spans = []
    for experiment in bundle.experiments:
        spans.extend(experiment.evidence)
        for measurement in experiment.measurements:
            spans.extend(measurement.evidence)
    for gap in bundle.data_gaps:
        spans.extend(gap.evidence)
    spans.extend(span for entity in bundle.entities for span in entity.evidence)
    return bool(spans) and all(span.source.chunk_id and span.quote for span in spans)


def _evaluate_case(case: dict[str, Any], pipeline: ExtractionPipeline) -> dict[str, Any]:
    bundle = pipeline.extract_from_chunk(_chunk(case))
    expected = case["expected"]
    checks: dict[str, bool] = {}
    missed: list[str] = []

    if expected.get("material"):
        checks["material"] = expected["material"] in _materials(bundle)
    if expected.get("regime"):
        checks["regime"] = expected["regime"] in _regimes(bundle)
    if expected.get("property"):
        checks["property"] = expected["property"] in _properties(bundle)

    measurements = _measurements(bundle)
    if case.get("expect_numeric"):
        expected_value = float(expected["value"])
        expected_unit = expected["unit"]
        matching = [
            item for item in measurements
            if item.property_canonical == expected["property"]
            and item.value is not None
            and abs(float(item.value) - expected_value) < 0.001
            and item.unit == expected_unit
        ]
        checks["measurement"] = bool(matching)
        checks["confidence"] = bool(matching and max(item.confidence for item in matching) >= CONFIDENCE_THRESHOLD)
        if "normalized_value_mpa" in expected:
            converted = [normalize_strength_to_mpa(item.value, item.unit)[0] for item in matching]
            checks["unit_normalization"] = any(value is not None and abs(value - float(expected["normalized_value_mpa"])) <= 1.0 for value in converted)
    else:
        numeric_measurements = [item for item in measurements if item.value is not None]
        qualitative = [item for item in measurements if item.property_canonical == expected.get("property") and item.effect == expected.get("effect")]
        checks["no_hallucinated_numeric"] = not numeric_measurements
        checks["qualitative_effect"] = bool(qualitative) or expected.get("property") in _properties(bundle)

    checks["evidence_bound"] = _has_evidence(bundle)
    for name, ok in checks.items():
        if not ok:
            missed.append(name)

    return {
        "id": case["id"],
        "passed": not missed,
        "missed": missed,
        "experiments": len(bundle.experiments),
        "measurements": [
            {
                "property": item.property_canonical,
                "value": item.value,
                "unit": item.unit,
                "effect": item.effect,
                "confidence": item.confidence,
            }
            for item in measurements
        ],
        "materials": sorted(_materials(bundle)),
        "regimes": sorted(_regimes(bundle)),
        "properties": sorted(_properties(bundle)),
        "rejected": [{"item_type": item.item_type, "reason": item.reason} for item in bundle.rejected_items],
        "checks": checks,
    }


def main() -> int:
    pipeline = ExtractionPipeline(mode="deterministic", audit_enabled=False)
    rows = [_evaluate_case(case, pipeline) for case in GOLD_CASES]
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Golden extraction evaluation:")
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{status} {row['id']} missed={row['missed']}")
    passed = all(row["passed"] for row in rows)
    print("PASS" if passed else "FAIL")
    print(f"JSON report: {ARTIFACT_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

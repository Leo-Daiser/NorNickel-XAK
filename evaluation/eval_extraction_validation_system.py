from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.extraction.models import EvidenceSpan, ExtractedEntity, ExtractedExperiment, ExtractedMeasurement, ExtractionSource  # noqa: E402
from app.extraction.pipeline import ExtractionPipeline  # noqa: E402
from app.extraction.validators import validate_entity, validate_experiment  # noqa: E402
from app.models.schemas import Chunk  # noqa: E402


ARTIFACT = ROOT / "artifacts" / "eval_extraction_validation_system.json"


def main() -> int:
    checks = [
        _check_pdf_font_code_rejected(),
        _check_chemical_formula_preserved_but_not_mechanical_subject(),
        _check_bare_percent_no_plasticity(),
        _check_capacity_reference_no_mechanical_mrp(),
        _check_explicit_elongation_accepted(),
        _check_long_chunk_schema_safe(),
        _check_accepted_facts_have_evidence(),
    ]
    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps({"summary": status, "checks": checks}, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['case_id']}: {item['reason']}")
    print(f"SUMMARY: {status}")
    print(f"JSON report: {ARTIFACT}")
    return 0 if status == "PASS" else 1


def _check_pdf_font_code_rejected() -> dict[str, Any]:
    evidence = _evidence("/MT255 12 0 R /FontDescriptor /CIDFontType2 /Encoding Identity-H")
    accepted, rejected = validate_entity(
        ExtractedEntity(entity_type="Material", raw_name="MT255", canonical_name="MT255", confidence=0.8, evidence=[evidence])
    )
    passed = accepted is None and rejected is not None and rejected.reason == "pdf_font_code_without_domain_context"
    return _row("pdf_font_code_rejected", passed, "" if passed else f"accepted={accepted!r} rejected={rejected!r}")


def _check_chemical_formula_preserved_but_not_mechanical_subject() -> dict[str, Any]:
    evidence = _evidence("SO2 concentration is valid chemistry, but ductility is not reported for a material sample.")
    material = ExtractedEntity(entity_type="Material", raw_name="SO2", canonical_name="SO2", confidence=0.8, evidence=[evidence])
    accepted_entity, entity_rejection = validate_entity(material)
    experiment = ExtractedExperiment(
        experiment_id="exp-so2",
        materials=[material],
        regimes=[],
        measurements=[
            ExtractedMeasurement(
                property_raw="ductility",
                property_canonical="пластичность",
                value=12.0,
                unit="%",
                confidence=0.8,
                evidence=[evidence],
            )
        ],
        evidence=[evidence],
        confidence=0.8,
    )
    accepted_exp, rejections = validate_experiment(experiment, min_confidence=0.55)
    passed = accepted_entity is not None and entity_rejection is None and accepted_exp is None and any(
        item.reason == "chemical_substance_incompatible_with_mechanical_property" for item in rejections
    )
    return _row("chemical_formula_subject_compatibility", passed, "" if passed else [item.reason for item in rejections])


def _check_bare_percent_no_plasticity() -> dict[str, Any]:
    bundle = _bundle("Directory of Copper Mines and Plants. Production capacity data: copper 35% summary country basis.")
    measurements = [item for experiment in bundle.experiments for item in experiment.measurements]
    passed = all(item.property_canonical != "пластичность" for item in measurements)
    return _row("bare_percent_no_plasticity", passed, "" if passed else [item.model_dump() for item in measurements])


def _check_capacity_reference_no_mechanical_mrp() -> dict[str, Any]:
    bundle = _bundle(
        "Directory of Copper Mines and Plants. Facility-by-facility production capacity data. "
        "Copper capacity 35% by summary country basis; aging table number 12 is a reference code."
    )
    measurements = [item for experiment in bundle.experiments for item in experiment.measurements]
    passed = all(item.property_canonical not in {"пластичность", "прочность", "твёрдость"} for item in measurements)
    return _row(
        "capacity_reference_no_mechanical_mrp",
        passed,
        "" if passed else [item.model_dump() for item in measurements],
    )


def _check_explicit_elongation_accepted() -> dict[str, Any]:
    bundle = _bundle("Эксперимент: сплав ВТ6 после отжига; относительное удлинение составило 14%.")
    passed = any(
        item.property_canonical == "пластичность" and item.value == 14.0 and item.unit == "%"
        for experiment in bundle.experiments
        for item in experiment.measurements
    )
    return _row("explicit_elongation_accepted", passed, "" if passed else bundle.model_dump())


def _check_long_chunk_schema_safe() -> dict[str, Any]:
    schema = (ROOT / "app" / "graph" / "schema.cypher").read_text(encoding="utf-8")
    passed = (
        "DROP INDEX chunk_text_index IF EXISTS" in schema
        and "FOR (n:DocumentChunk) ON (n.text);" not in schema
        and "CREATE FULLTEXT INDEX chunk_fulltext" in schema
    )
    return _row("long_chunk_schema_safe", passed, "" if passed else "DocumentChunk.text RANGE index still present")


def _check_accepted_facts_have_evidence() -> dict[str, Any]:
    bundle = _bundle("The 7075-T6 aluminum alloy showed tensile strength of 77 ksi after aging treatment.")
    accepted_measurements = [item for experiment in bundle.experiments for item in experiment.measurements]
    passed = bool(accepted_measurements) and all(item.evidence and item.evidence[0].quote for item in accepted_measurements)
    return _row("accepted_facts_have_evidence", passed, "" if passed else bundle.model_dump())


def _bundle(text: str):
    chunk = Chunk(
        chunk_id="chunk",
        doc_id="doc",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="/",
        metadata={"filename": "eval-fixture.txt", "source_name": "eval-fixture.txt"},
    )
    return ExtractionPipeline(mode="deterministic", audit_enabled=False).extract_from_chunk(chunk)


def _evidence(text: str) -> EvidenceSpan:
    return EvidenceSpan(source=ExtractionSource(document_id="doc", chunk_id="chunk", source_name="eval"), quote=text, confidence=0.9)


def _row(case_id: str, passed: bool, reason: Any) -> dict[str, Any]:
    return {"case_id": case_id, "passed": bool(passed), "reason": reason or "ok"}


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.extraction.models import (
    EvidenceSpan,
    ExtractionBundle,
    ExtractionSource,
    ExtractedEntity,
    RejectedExtraction,
)


def test_evidence_span_requires_quote() -> None:
    source = ExtractionSource(document_id="doc", chunk_id="chunk")
    with pytest.raises(ValidationError):
        EvidenceSpan(source=source, quote="")


def test_confidence_bounds_are_validated() -> None:
    source = ExtractionSource(document_id="doc", chunk_id="chunk")
    evidence = [EvidenceSpan(source=source, quote="ВТ6")]
    with pytest.raises(ValidationError):
        ExtractedEntity(
            entity_type="Material",
            raw_name="ВТ6",
            canonical_name="ВТ6",
            confidence=1.5,
            evidence=evidence,
        )


def test_bundle_serializes_and_deserializes() -> None:
    source = ExtractionSource(document_id="doc", chunk_id="chunk")
    bundle = ExtractionBundle(
        document_id="doc",
        source_name="demo.txt",
        extractor_version="test",
        entities=[
            ExtractedEntity(
                entity_type="Material",
                raw_name="ВТ6",
                canonical_name="ВТ6",
                confidence=0.9,
                evidence=[EvidenceSpan(source=source, quote="ВТ6")],
            )
        ],
    )

    restored = ExtractionBundle.model_validate(bundle.model_dump())

    assert restored.document_id == "doc"
    assert restored.entities[0].canonical_name == "ВТ6"


def test_rejected_item_preserves_reason() -> None:
    rejected = RejectedExtraction(item_type="measurement", reason="missing_property", raw_payload={"x": 1})
    assert rejected.reason == "missing_property"
    assert rejected.raw_payload == {"x": 1}

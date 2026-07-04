"""Optional structured LLM extractor for evidence-grounded extraction."""

from __future__ import annotations

import json

from ..llm.structured_llm import StructuredLLM
from .models import ExtractionBundle, ExtractionSource, RejectedExtraction


SYSTEM_PROMPT = """Extract only facts explicitly present in the source text.
Return JSON matching the provided schema.
Every extracted entity, experiment, measurement, conclusion and data gap must include direct evidence quote copied from the source.
Do not infer missing values.
Do not normalize by guessing; provide raw text and canonical only if obvious from aliases.
If the text does not contain experiments or measurements, return empty lists."""


class StructuredLLMExtractor:
    """LLM extractor wrapper. Disabled unless the existing LLM client is configured."""

    extractor_version = "llm_structured_v1"

    def __init__(self, client: StructuredLLM | None = None) -> None:
        self.client = client or StructuredLLM()

    @property
    def available(self) -> bool:
        return self.client.enabled

    def extract(self, text: str, source: ExtractionSource) -> ExtractionBundle:
        """Extract structured JSON with direct quotes or return rejected diagnostics."""
        if not self.available:
            raise RuntimeError("LLM structured extraction requested, but LLM is not configured")
        payload = {
            "schema": {
                "entities": [],
                "experiments": [],
                "data_gaps": [],
            },
            "source": source.model_dump(),
            "text": text[:6000],
            "instructions": SYSTEM_PROMPT,
        }
        raw = self.client._chat(SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))  # noqa: SLF001
        obj = self.client._parse_json_object(raw or "")  # noqa: SLF001
        if not obj:
            return ExtractionBundle(
                document_id=source.document_id,
                source_name=source.source_name,
                extractor_version=self.extractor_version,
                rejected_items=[
                    RejectedExtraction(
                        item_type="llm_result",
                        reason="invalid_json",
                        raw_payload=raw or self.client.last_error or "",
                    )
                ],
                diagnostics={"llm_error": self.client.last_error},
            )
        try:
            return ExtractionBundle(
                document_id=source.document_id,
                source_name=source.source_name,
                extractor_version=self.extractor_version,
                **{key: obj.get(key, []) for key in ["entities", "experiments", "data_gaps"]},
                diagnostics={"llm_used": True},
            )
        except Exception as exc:
            return ExtractionBundle(
                document_id=source.document_id,
                source_name=source.source_name,
                extractor_version=self.extractor_version,
                rejected_items=[RejectedExtraction(item_type="llm_result", reason="invalid_json", raw_payload=obj)],
                diagnostics={"llm_parse_error": str(exc)},
            )


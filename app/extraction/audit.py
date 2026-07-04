"""JSONL audit trail for extraction runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ExtractionBundle


class ExtractionAuditWriter:
    """Append accepted/rejected/diagnostic extraction records to JSONL files."""

    def __init__(self, audit_dir: str | Path) -> None:
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def write_bundle(self, bundle: ExtractionBundle) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document_id": bundle.document_id,
            "source_name": bundle.source_name,
            "extractor_version": bundle.extractor_version,
            "accepted_entities": len(bundle.entities),
            "accepted_experiments": len(bundle.experiments),
            "accepted_gaps": len(bundle.data_gaps),
            "rejected_items": len(bundle.rejected_items),
            "diagnostics": bundle.diagnostics,
        }
        self._append("diagnostics.jsonl", record)
        if bundle.experiments or bundle.entities or bundle.data_gaps:
            self._append(
                "accepted.jsonl",
                {
                    **record,
                    "entities": [item.model_dump() for item in bundle.entities],
                    "experiments": [item.model_dump() for item in bundle.experiments],
                    "data_gaps": [item.model_dump() for item in bundle.data_gaps],
                },
            )
        for rejected in bundle.rejected_items:
            self._append("rejected.jsonl", {**record, "rejected": rejected.model_dump()})

    def _append(self, name: str, payload: dict[str, Any]) -> None:
        path = self.audit_dir / name
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


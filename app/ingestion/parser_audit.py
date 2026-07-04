"""JSONL audit trail for document parsing diagnostics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .document_models import DocumentIntelligenceResult


class ParserAuditWriter:
    """Write parser diagnostics and errors as JSONL records."""

    def __init__(self, audit_dir: str | Path = "data/parser_audit") -> None:
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def write_parsed(self, result: DocumentIntelligenceResult) -> None:
        diagnostics = result.diagnostics or {}
        self._append(
            "parsed.jsonl",
            {
                "timestamp": _now(),
                "source_name": result.source_name,
                "parser_backend_requested": diagnostics.get("parser_backend_requested"),
                "parser_backend_used": diagnostics.get("parser_backend_used") or result.parser_name,
                "blocks_count": len(result.blocks),
                "tables_count": len(result.tables),
                "images_count": len(result.images),
                "chunks_count": len(result.chunks),
                "scanned_pdf_detected": diagnostics.get("scanned_pdf_detected", False),
                "warnings": diagnostics.get("warnings", []),
            },
        )

    def write_error(self, source_name: str, error: str, diagnostics: dict[str, Any] | None = None) -> None:
        payload = {
            "timestamp": _now(),
            "source_name": source_name,
            "error": error,
            **(diagnostics or {}),
        }
        self._append("errors.jsonl", payload)

    def _append(self, filename: str, payload: dict[str, Any]) -> None:
        with (self.audit_dir / filename).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


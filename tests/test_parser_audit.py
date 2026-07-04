from __future__ import annotations

import json

from app.ingestion.document_models import DocumentIntelligenceResult
from app.ingestion.parser_audit import ParserAuditWriter


def test_parser_audit_writes_parsed_jsonl(tmp_path) -> None:
    writer = ParserAuditWriter(tmp_path)
    result = DocumentIntelligenceResult(
        doc_id="doc",
        source_name="demo.txt",
        parser_name="plain",
        diagnostics={"parser_backend_requested": "fallback", "parser_backend_used": "plain"},
    )

    writer.write_parsed(result)

    rows = (tmp_path / "parsed.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    assert payload["parser_backend_used"] == "plain"
    assert payload["chunks_count"] == 0


def test_parser_audit_writes_errors_jsonl(tmp_path) -> None:
    writer = ParserAuditWriter(tmp_path)

    writer.write_error("bad.pdf", "parse failed", {"parser_backend_requested": "docling"})

    rows = (tmp_path / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    assert payload["source_name"] == "bad.pdf"
    assert payload["error"] == "parse failed"

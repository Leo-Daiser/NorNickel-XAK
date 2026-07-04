from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.ingestion.parser_audit import ParserAuditWriter
from app.ingestion.parser_router import ParserRouter


def _router(tmp_path: Path) -> ParserRouter:
    return ParserRouter(audit_writer=ParserAuditWriter(tmp_path / "audit"))


def test_fallback_backend_does_not_require_optional_parsers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "parser_backend", "fallback", raising=False)
    path = tmp_path / "demo.txt"
    path.write_text("ВТ6 после отжига показал прочность 1120 MPa.", encoding="utf-8")

    result = _router(tmp_path).parse_document_intelligence(str(path), doc_id="doc")

    assert result.parser_name == "plain"
    assert result.diagnostics["parser_backend_requested"] == "fallback"
    assert result.diagnostics["parser_backend_used"] == "plain"


def test_docling_backend_unavailable_fails_clearly(tmp_path: Path, monkeypatch) -> None:
    import app.ingestion.parser_router as parser_router_module

    monkeypatch.setattr(settings, "parser_backend", "docling", raising=False)
    monkeypatch.setattr(parser_router_module, "DocumentConverter", None)
    path = tmp_path / "demo.pdf"
    path.write_bytes(b"%PDF-1.4\n%%EOF")

    with pytest.raises(RuntimeError, match="Docling is not installed"):
        _router(tmp_path).parse_document_intelligence(str(path), doc_id="doc")


def test_markitdown_backend_unavailable_fails_clearly(tmp_path: Path, monkeypatch) -> None:
    import app.ingestion.parser_router as parser_router_module

    monkeypatch.setattr(settings, "parser_backend", "markitdown", raising=False)
    monkeypatch.setattr(parser_router_module, "MarkItDown", None)
    path = tmp_path / "demo.txt"
    path.write_text("text", encoding="utf-8")

    with pytest.raises(RuntimeError, match="MarkItDown is not installed"):
        _router(tmp_path).parse_document_intelligence(str(path), doc_id="doc")


def test_auto_backend_falls_back_gracefully(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "parser_backend", "auto", raising=False)
    path = tmp_path / "demo.txt"
    path.write_text("ВТ6\n\nотжиг прочность 1120 MPa", encoding="utf-8")

    result = _router(tmp_path).parse_document_intelligence(str(path), doc_id="doc")

    assert result.chunks
    assert result.diagnostics["parser_backend_requested"] == "auto"
    assert result.diagnostics["parser_backend_used"] in {"plain", "markitdown"}


def test_diagnostics_contains_requested_and_used_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "parser_backend", "fallback", raising=False)
    path = tmp_path / "demo.txt"
    path.write_text("text", encoding="utf-8")

    result = _router(tmp_path).parse_document_intelligence(str(path), doc_id="doc")

    assert "parser_backend_requested" in result.diagnostics
    assert "parser_backend_used" in result.diagnostics

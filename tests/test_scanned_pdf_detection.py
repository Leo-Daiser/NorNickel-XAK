from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.ingestion.parser_audit import ParserAuditWriter
from app.ingestion.parser_router import ParserRouter, detect_scanned_pdf


def _blank_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as fh:
        writer.write(fh)


def test_pdf_with_low_extracted_text_is_detected_as_scanned(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _blank_pdf(path)

    detection = detect_scanned_pdf(path, "")

    assert detection.is_probably_scanned is True
    assert detection.page_count == 1


def test_pdf_with_enough_text_is_not_scanned(tmp_path: Path) -> None:
    path = tmp_path / "text.pdf"
    _blank_pdf(path)

    detection = detect_scanned_pdf(path, "x" * 200)

    assert detection.is_probably_scanned is False


def test_non_pdf_is_not_scanned(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("text", encoding="utf-8")

    detection = detect_scanned_pdf(path, "")

    assert detection.is_probably_scanned is False
    assert detection.reason == "not_pdf"


def test_ocr_disabled_adds_warning_not_crash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "parser_backend", "fallback", raising=False)
    monkeypatch.setattr(settings, "enable_ocr", False, raising=False)
    path = tmp_path / "scan.pdf"
    _blank_pdf(path)

    result = ParserRouter(audit_writer=ParserAuditWriter(tmp_path / "audit")).parse_document_intelligence(str(path), doc_id="scan")

    assert result.diagnostics["scanned_pdf_detected"] is True
    assert "PDF appears scanned; OCR is disabled" in result.diagnostics["warnings"]

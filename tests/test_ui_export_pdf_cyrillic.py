from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

from pypdf import PdfReader

from app.ui_helpers import build_answer_markdown_export, build_answer_pdf_export, _find_pdf_font_path


def _payload() -> dict:
    return {
        "answer": "Краткий вывод: применяются холодильные установки, вентиляция и охлаждение воздуха.",
        "status": "partial",
        "sources": [
            {
                "source_name": "Глубокие рудники 2017.pdf",
                "filename": "Глубокие рудники 2017.pdf",
                "source_type": "file",
                "page_start": 10,
                "quote": "На глубоких рудниках применяются способы охлаждения воздуха.",
            }
        ],
        "evidence": [
            {
                "source_name": "Глубокие рудники 2017.pdf",
                "filename": "Глубокие рудники 2017.pdf",
                "source_type": "file",
                "page_start": 10,
                "quote": "На глубоких рудниках применяются способы охлаждения воздуха.",
            }
        ],
        "diagnostics": {},
    }


def _question() -> str:
    return "Какие способы охлаждения применяются для глубоких рудников?"


def _generated_at() -> datetime:
    return datetime(2026, 7, 4, 0, 0, tzinfo=timezone.utc)


def test_markdown_export_keeps_utf8_cyrillic() -> None:
    markdown = build_answer_markdown_export(_payload(), question=_question(), generated_at=_generated_at())

    assert "Ответ GraphRAG" in markdown
    assert "Дата генерации" in markdown
    assert _question() in markdown
    assert "Глубокие рудники 2017.pdf" in markdown
    assert "Otvet GraphRAG" not in markdown
    assert "Data generatsii" not in markdown
    assert "Kakie sposoby" not in markdown


def test_pdf_export_keeps_cyrillic_text_and_source_name() -> None:
    pdf_bytes = build_answer_pdf_export(_payload(), question=_question(), generated_at=_generated_at())
    reader = PdfReader(BytesIO(pdf_bytes))
    extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    metadata_text = " ".join(str(value) for value in (reader.metadata or {}).values())
    observable_text = f"{extracted_text}\n{metadata_text}"

    assert _find_pdf_font_path()
    assert "Ответ GraphRAG" in observable_text
    assert "Дата генерации" in observable_text
    assert _question() in observable_text
    assert "Глубокие рудники 2017.pdf" in observable_text
    assert "Otvet GraphRAG" not in observable_text
    assert "Data generatsii" not in observable_text
    assert "Kakie sposoby" not in observable_text

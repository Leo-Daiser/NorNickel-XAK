from __future__ import annotations

import json
from pathlib import Path

from scripts.ocr_large_pdf_report import build_ocr_large_pdf_report, render_markdown, safe_stem, write_report


def test_ocr_large_pdf_queue_reports_missing_tools(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    (root / "scan.gif").write_bytes(b"gif")
    (root / "large.pdf").write_bytes(b"x" * 4096)
    (root / "ready.txt").write_text("ВТ6 980 MPa", encoding="utf-8")

    report = build_ocr_large_pdf_report(
        root,
        max_file_mb=0.001,
        ocrmypdf_path=str(tmp_path / "missing-ocrmypdf"),
        tesseract_path=str(tmp_path / "missing-tesseract"),
        pdftotext_path=str(tmp_path / "missing-pdftotext"),
    )
    summary = report["summary"]

    assert summary["queue_count"] == 2
    assert summary["reason_counts"]["ocr_required"] == 1
    assert summary["reason_counts"]["file_too_large_for_batch"] == 1
    assert summary["blocking_reason_counts"]["ocr_tools_missing"] == 1
    assert summary["blocking_reason_counts"]["large_pdf_text_tool_missing"] == 1
    assert summary["ready_to_run_count"] == 0


def test_ocr_large_pdf_queue_marks_tools_ready(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    (root / "scan.gif").write_bytes(b"gif")
    (root / "large.pdf").write_bytes(b"x" * 4096)
    ocr = tmp_path / "ocrmypdf.exe"
    tess = tmp_path / "tesseract.exe"
    pdftext = tmp_path / "pdftotext.exe"
    for tool in [ocr, tess, pdftext]:
        tool.write_bytes(b"fake")

    report = build_ocr_large_pdf_report(
        root,
        max_file_mb=0.001,
        ocrmypdf_path=str(ocr),
        tesseract_path=str(tess),
        pdftotext_path=str(pdftext),
    )

    assert report["summary"]["ready_to_run_count"] == 2
    assert report["summary"]["blocked_count"] == 0
    assert all(row["tool_ready"] for row in report["rows"])


def test_ocr_large_pdf_provenance_policy_and_safe_stem(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    (root / "скан отчета.gif").write_bytes(b"gif")

    report = build_ocr_large_pdf_report(root)
    row = report["rows"][0]

    assert "provenance" in row["provenance_policy"].lower()
    assert "скан_отчета" in row["staged_output_path"]
    assert safe_stem(Path("bad:name?.pdf")) == "bad_name_"


def test_ocr_large_pdf_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    (root / "scan.gif").write_bytes(b"gif")
    json_path = tmp_path / "ocr.json"
    markdown_path = tmp_path / "ocr.md"

    report = build_ocr_large_pdf_report(root)
    write_report(report, json_path=json_path, markdown_path=markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["summary"]["queue_count"] == 1
    assert "OCR / Large PDF Queue Report" in markdown
    assert render_markdown(report).startswith("# OCR / Large PDF Queue Report")

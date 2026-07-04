from __future__ import annotations

from pathlib import Path

from app.ingestion.document_models import DocumentIntelligenceResult
from app.parsing.file_profile import profile_corpus, profile_file, render_markdown_report
from app.parsing.text_quality import normalize_dirty_scientific_text


def test_dirty_scientific_text_normalization_is_generic() -> None:
    text = "ВТ 6 после отжига: предел прочности 980 M Pa; твердость НV 240. 7075 Т6: 77 ksi."

    normalized = normalize_dirty_scientific_text(text)

    assert "ВТ6" in normalized
    assert "980 MPa" in normalized
    assert "HV 240" in normalized
    assert "7075-T6" in normalized


def test_profile_file_extracts_facts_with_evidence(tmp_path: Path) -> None:
    path = tmp_path / "vt6.txt"
    path.write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")

    profile = profile_file(path)

    assert profile["parse_status"] == "ok"
    assert profile["chunks_count"] > 0
    assert profile["canonical_facts"] >= 1
    assert profile["facts_without_evidence"] == 0
    assert profile["fact_preview"][0]["material"] == "ВТ6"


def test_profile_file_marks_scanned_pdf_as_ocr_required(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4\n% fake test file")

    class FakeParser:
        def parse_document_intelligence(self, *_args, **_kwargs):
            return DocumentIntelligenceResult(
                doc_id="doc_scan",
                source_name="scan.pdf",
                parser_name="fake_pdf",
                text="",
                chunks=[],
                diagnostics={
                    "scanned_pdf_detected": True,
                    "scanned_pdf_page_count": 3,
                    "warnings": ["PDF appears scanned; OCR is disabled"],
                },
            )

    profile = profile_file(path, parser=FakeParser())  # type: ignore[arg-type]

    assert profile["parse_status"] == "ocr_required"
    assert "ocr_required" in profile["warnings"]
    assert profile["canonical_facts"] == 0


def test_profile_corpus_and_markdown_report(tmp_path: Path) -> None:
    (tmp_path / "vt6.txt").write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")
    (tmp_path / "unsupported.zip").write_bytes(b"PK")

    report = profile_corpus(tmp_path)
    markdown = render_markdown_report(report)

    assert report["summary"]["documents_total"] == 2
    assert report["summary"]["facts_without_evidence"] == 0
    assert report["summary"]["parse_status_counts"]["unsupported"] == 1
    assert "Corpus Readiness Report" in markdown
    assert "economy_core" in markdown


def test_inventory_mode_does_not_parse_supported_file(tmp_path: Path) -> None:
    path = tmp_path / "article.pdf"
    path.write_bytes(b"%PDF-1.4\nnot a real pdf")

    profile = profile_file(path, profile_mode="inventory")

    assert profile["parse_status"] == "partial"
    assert profile["parser_backend"] == "inventory"
    assert "inventory_only_parse_not_run" in profile["warnings"]
    assert profile["facts_extracted"] == 0


def test_auto_mode_skips_large_file_with_controlled_warning(tmp_path: Path) -> None:
    path = tmp_path / "large.pdf"
    path.write_bytes(b"x" * 2048)

    profile = profile_file(path, profile_mode="auto", max_parse_mb=0.001)

    assert profile["parse_status"] == "partial"
    assert "parse_skipped_large_file" in profile["warnings"]
    assert "exceeds max_parse_mb" in profile["parser_error"]


def test_legacy_archive_and_image_formats_are_controlled(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.rar"
    legacy = tmp_path / "legacy.xls"
    image = tmp_path / "scan.gif"
    archive.write_bytes(b"rar")
    legacy.write_bytes(b"xls")
    image.write_bytes(b"gif")

    archive_profile = profile_file(archive)
    legacy_profile = profile_file(legacy)
    image_profile = profile_file(image)

    assert archive_profile["parse_status"] == "unsupported"
    assert "archive_needs_extraction" in archive_profile["warnings"]
    assert legacy_profile["parse_status"] == "unsupported"
    assert "legacy_format_needs_conversion" in legacy_profile["warnings"]
    assert image_profile["parse_status"] == "ocr_required"
    assert "image_ocr_required" in image_profile["warnings"]


def test_data_storage_source_group_is_reported(tmp_path: Path) -> None:
    group = tmp_path / "data_storage" / "Статьи"
    group.mkdir(parents=True)
    (group / "vt6.txt").write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")

    report = profile_corpus(tmp_path / "data_storage", profile_mode="inventory")

    assert report["summary"]["source_group_counts"]["Статьи"] == 1
    assert report["files"][0]["source_group"] == "Статьи"
    assert report["summary"]["inventory_only_count"] == 1

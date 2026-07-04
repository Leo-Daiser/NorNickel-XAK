from __future__ import annotations

import json
from pathlib import Path

from scripts.conversion_backlog_report import build_conversion_backlog, render_markdown, write_report


def test_conversion_backlog_groups_blocking_files(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    group = root / "Журналы"
    group.mkdir(parents=True)
    (group / "ready.txt").write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")
    (group / "archive.zip").write_bytes(b"zip")
    (group / "legacy.doc").write_bytes(b"doc")
    (group / "scan.gif").write_bytes(b"gif")
    (group / "huge.pdf").write_bytes(b"x" * 4096)

    report = build_conversion_backlog(root, max_file_mb=0.001)
    summary = report["summary"]

    assert summary["documents_total_found"] == 5
    assert summary["backlog_count"] == 4
    assert summary["ready_for_direct_ingest_count"] == 1
    assert summary["reason_counts"]["archive_needs_extraction"] == 1
    assert summary["reason_counts"]["legacy_format_needs_conversion"] == 1
    assert summary["reason_counts"]["ocr_required"] == 1
    assert summary["reason_counts"]["file_too_large_for_batch"] == 1
    assert summary["recommended_action_counts"]["extract_archive"] == 1
    assert summary["recommended_action_counts"]["convert_legacy_office"] == 1
    assert summary["recommended_action_counts"]["run_ocr_or_mark_unreadable"] == 1
    assert summary["recommended_action_counts"]["large_file_parse_queue"] == 1


def test_conversion_backlog_preserves_provenance_policy(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    (root / "legacy.xls").write_bytes(b"xls")

    report = build_conversion_backlog(root)

    row = report["rows"][0]
    assert row["recommended_action"] == "convert_legacy_office"
    assert "original" in row["provenance_policy"].lower()
    assert "provenance" in row["provenance_policy"].lower()


def test_conversion_backlog_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    (root / "bundle.rar").write_bytes(b"rar")
    json_path = tmp_path / "backlog.json"
    markdown_path = tmp_path / "backlog.md"

    report = build_conversion_backlog(root)
    write_report(report, json_path=json_path, markdown_path=markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["summary"]["backlog_count"] == 1
    assert "Conversion / OCR Backlog Report" in markdown
    assert "extract_archive" in markdown
    assert render_markdown(report).startswith("# Conversion / OCR Backlog Report")

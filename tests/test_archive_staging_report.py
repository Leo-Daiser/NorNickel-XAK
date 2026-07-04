from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.archive_staging_report import build_archive_staging_report, render_markdown, safe_zip_member, write_report


def _zip(path: Path, entries: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in entries.items():
            archive.writestr(name, text)


def test_safe_zip_member_blocks_traversal() -> None:
    assert safe_zip_member("reports/vt6.txt")
    assert not safe_zip_member("../evil.txt")
    assert not safe_zip_member("/abs/evil.txt")
    assert not safe_zip_member("C:/evil.txt")
    assert not safe_zip_member("nested/../../evil.txt")


def test_archive_staging_inventory_detects_supported_zip_members(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    _zip(root / "bundle.zip", {"reports/vt6.txt": "После отжига сплава ВТ6 предел прочности составил 980 MPa.", "image.bin": "x"})

    report = build_archive_staging_report(root)

    row = report["rows"][0]
    assert row["archive_status"] == "zip_inventory_ok"
    assert row["supported_members_count"] == 1
    assert row["members_count"] == 2
    assert report["summary"]["zip_supported_members_count"] == 1


def test_archive_staging_extracts_supported_zip_members_only_when_enabled(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    staging = tmp_path / "staging"
    root.mkdir()
    _zip(root / "bundle.zip", {"reports/vt6.txt": "ВТ6 980 MPa", "ignored.bin": "x"})

    report = build_archive_staging_report(root, output_dir=staging, extract_zip=True)

    row = report["rows"][0]
    assert row["archive_status"] == "zip_extracted"
    assert row["extracted_files_count"] == 1
    staged = Path(row["extracted_files"][0]["staged_path"])
    assert staged.exists()
    assert staged.read_text(encoding="utf-8") == "ВТ6 980 MPa"


def test_archive_staging_blocks_unsafe_zip_extraction(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    _zip(root / "bad.zip", {"../evil.txt": "bad", "safe.txt": "ok"})

    report = build_archive_staging_report(root, output_dir=tmp_path / "staging", extract_zip=True)

    row = report["rows"][0]
    assert row["archive_status"] == "zip_extraction_blocked"
    assert "unsafe_zip_members" in row["warnings"]
    assert not (tmp_path / "evil.txt").exists()


def test_archive_staging_reports_rar_as_external_extractor_required(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    (root / "bundle.rar").write_bytes(b"rar")

    report = build_archive_staging_report(root)

    row = report["rows"][0]
    assert row["archive_status"] == "external_extractor_required"
    assert row["recommended_action"] == "extract_with_controlled_external_tool"


def test_archive_staging_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    _zip(root / "bundle.zip", {"reports/vt6.txt": "ВТ6 980 MPa"})
    json_path = tmp_path / "archive.json"
    markdown_path = tmp_path / "archive.md"

    report = build_archive_staging_report(root)
    write_report(report, json_path=json_path, markdown_path=markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["summary"]["archives_selected"] == 1
    assert "Archive Staging Report" in markdown
    assert render_markdown(report).startswith("# Archive Staging Report")

from __future__ import annotations

import json
from pathlib import Path

from scripts.legacy_office_conversion_report import (
    build_legacy_conversion_report,
    conversion_command,
    conversion_target,
    render_markdown,
    write_report,
)


def test_conversion_target_mapping() -> None:
    assert conversion_target(".doc") == ("docx", ".docx")
    assert conversion_target(".docm") == ("docx", ".docx")
    assert conversion_target(".xls") == ("xlsx", ".xlsx")
    assert conversion_target(".ppt") == ("pptx", ".pptx")


def test_legacy_conversion_reports_missing_tool(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    group = root / "Материалы конференций"
    group.mkdir(parents=True)
    (group / "legacy.doc").write_bytes(b"doc")
    (group / "ready.txt").write_text("ВТ6 980 MPa", encoding="utf-8")

    report = build_legacy_conversion_report(root, soffice_path=str(tmp_path / "missing-soffice.exe"))

    summary = report["summary"]
    row = report["rows"][0]
    assert summary["legacy_files_selected"] == 1
    assert summary["soffice_available"] is False
    assert row["conversion_status"] == "conversion_tool_missing"
    assert row["recommended_action"] == "install_or_configure_libreoffice_soffice"


def test_legacy_conversion_plans_command_when_tool_exists(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    (root / "legacy.xls").write_bytes(b"xls")
    soffice = tmp_path / "soffice.exe"
    soffice.write_bytes(b"fake")

    report = build_legacy_conversion_report(root, soffice_path=str(soffice), convert=False)

    row = report["rows"][0]
    assert row["conversion_status"] == "planned"
    assert row["target_extension"] == ".xlsx"
    assert "--convert-to" in row["command"]
    assert "xlsx" in row["command"]
    assert str(soffice) in row["command"][0]


def test_conversion_command_shape(tmp_path: Path) -> None:
    source = tmp_path / "legacy.doc"
    target = tmp_path / "out"

    command = conversion_command("soffice", source, target, "docx")

    assert command == ["soffice", "--headless", "--convert-to", "docx", "--outdir", str(target), str(source)]


def test_legacy_conversion_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    (root / "legacy.docm").write_bytes(b"docm")
    json_path = tmp_path / "legacy.json"
    markdown_path = tmp_path / "legacy.md"

    report = build_legacy_conversion_report(root, soffice_path=str(tmp_path / "missing.exe"))
    write_report(report, json_path=json_path, markdown_path=markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["summary"]["legacy_files_selected"] == 1
    assert "Legacy Office Conversion Report" in markdown
    assert "conversion_tool_missing" in markdown
    assert render_markdown(report).startswith("# Legacy Office Conversion Report")

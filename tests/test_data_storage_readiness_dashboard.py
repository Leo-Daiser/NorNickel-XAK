from __future__ import annotations

import json
from pathlib import Path

from scripts.data_storage_readiness_dashboard import build_dashboard, render_markdown, write_dashboard


def test_data_storage_dashboard_combines_direct_ingest_and_backlogs(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    group = root / "Журналы"
    group.mkdir(parents=True)
    (group / "ready.txt").write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")
    (group / "archive.zip").write_bytes(b"zip")
    (group / "legacy.doc").write_bytes(b"doc")
    (group / "scan.gif").write_bytes(b"gif")
    (group / "huge.pdf").write_bytes(b"x" * 4096)

    report = build_dashboard(root, max_file_mb=0.001)
    summary = report["summary"]

    assert summary["status"] == "WARN"
    assert summary["documents_total_found"] == 5
    assert summary["direct_ingest_ready_count"] == 1
    assert summary["blocked_or_preprocessing_required_count"] == 4
    assert summary["planned_reason_counts"]["supported"] == 1
    assert summary["planned_reason_counts"]["archive_needs_extraction"] == 1
    assert summary["planned_reason_counts"]["legacy_format_needs_conversion"] == 1
    assert summary["planned_reason_counts"]["ocr_required"] == 1
    assert summary["planned_reason_counts"]["file_too_large_for_batch"] == 1
    assert summary["economy_core_compatible"] is True
    assert report["resource_profile"]["llm_required"] is False
    assert report["resource_profile"]["embeddings_required"] is False
    assert report["resource_profile"]["ocr_executed"] is False
    assert report["resource_profile"]["conversion_executed"] is False
    assert report["resource_profile"]["archive_extraction_executed"] is False


def test_data_storage_dashboard_recommends_action_order(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    (root / "ready.txt").write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")
    (root / "legacy.xls").write_bytes(b"xls")
    (root / "scan.gif").write_bytes(b"gif")

    report = build_dashboard(root)
    actions = report["recommended_next_actions"]

    assert actions[0]["action"] == "direct_batch_ingest"
    assert any(item["action"] == "install_libreoffice_for_legacy_office" for item in actions)
    assert any(item["action"] == "install_ocr_and_pdf_text_tools" for item in actions)


def test_data_storage_dashboard_writes_json_and_markdown(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    (root / "ready.txt").write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")
    json_path = tmp_path / "dashboard.json"
    markdown_path = tmp_path / "dashboard.md"

    report = build_dashboard(root)
    write_dashboard(report, json_path=json_path, markdown_path=markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["summary"]["direct_ingest_ready_count"] == 1
    assert "Data Storage Readiness Dashboard" in markdown
    assert "No LLM extraction" in markdown
    assert render_markdown(report).startswith("# Data Storage Readiness Dashboard")

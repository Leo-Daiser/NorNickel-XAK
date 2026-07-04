from __future__ import annotations

import json

from app.models.schemas import Chunk, Document
from app.storage.catalog import SQLiteCatalog
from scripts import extraction_quality_report


def _prepare_catalog(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.sqlite3"
    report_path = tmp_path / "report.json"
    catalog = SQLiteCatalog(catalog_path)
    document = Document(
        doc_id="doc",
        workspace_uid="test",
        title="gold.txt",
        parser="test",
        status="ready",
    )
    chunk = Chunk(
        chunk_id="chunk",
        doc_id="doc",
        workspace_uid="test",
        text="После отжига сплава ВТ6 предел прочности составил 980 MPa.",
        page_start=1,
        page_end=1,
        section_path="test",
        metadata={"filename": "gold.txt"},
    )
    catalog.upsert_document(document)
    catalog.replace_chunks("doc", [chunk])
    monkeypatch.setattr(extraction_quality_report.settings, "catalog_db_path", catalog_path, raising=False)
    monkeypatch.setattr(extraction_quality_report, "REPORT_PATH", report_path)
    return report_path


def test_extraction_quality_report_produces_json(tmp_path, monkeypatch) -> None:
    report_path = _prepare_catalog(tmp_path, monkeypatch)

    exit_code = extraction_quality_report.main(["--skip-neo4j"])
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["summary"]["total_documents"] == 1
    assert payload["summary"]["total_chunks"] == 1
    assert payload["summary"]["total_facts"] >= 1
    assert payload["summary"]["raw_facts_count"] >= 1
    assert payload["summary"]["canonical_facts_count"] >= 1
    assert "duplicate_groups_count" in payload["summary"]
    assert "duplicate_facts_count" in payload["summary"]
    assert "conflict_groups_count" in payload["summary"]
    assert "facts_without_evidence" in payload["summary"]
    assert payload["summary"]["normalized_measurements_count"] >= 1
    assert payload["summary"]["measurements_missing_normalized_fields"] == 0
    assert payload["summary"]["neo4j_scan_status"] == "skipped"
    assert "neo4j_scan_warning" in payload["summary"]
    assert "legacy_neo4j_records_missing_normalized_fields" in payload["summary"]
    assert payload["neo4j_scan"]["status"] == "skipped"
    assert payload["summary"]["facts_without_source_or_evidence"] == 0
    assert "conflict_summary" in payload


def test_extraction_quality_report_skip_neo4j_has_controlled_output(tmp_path, monkeypatch, capsys) -> None:
    report_path = _prepare_catalog(tmp_path, monkeypatch)

    exit_code = extraction_quality_report.main(["--skip-neo4j"])
    captured = capsys.readouterr()
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "ValueError" not in captured.out
    assert "Neo4j scan skipped by --skip-neo4j." in captured.out
    assert payload["summary"]["neo4j_scan_status"] == "skipped"


def test_extraction_quality_report_missing_neo4j_env_is_controlled(tmp_path, monkeypatch) -> None:
    _prepare_catalog(tmp_path, monkeypatch)
    for name in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE"]:
        monkeypatch.delenv(name, raising=False)

    report = extraction_quality_report.build_report(extraction_quality_report.Neo4jScanOptions())

    assert report["summary"]["neo4j_scan_status"] == "skipped"
    assert report["summary"]["neo4j_scan_warning"] == extraction_quality_report.NEO4J_NOT_CONFIGURED_WARNING
    assert "ValueError" not in str(report["warnings"])

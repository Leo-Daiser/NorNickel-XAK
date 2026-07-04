from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.knowledge.expansion import KnowledgeExpansionEngine
from app.models.schemas import Chunk, Document
from app.storage.catalog import SQLiteCatalog
from scripts import knowledge_expansion_report as report_script
from tests.strict_qa_helpers import reset_api


VT6_980 = "После отжига сплава ВТ6 предел прочности составил 980 MPa."
VT6_1120 = "Ti-6Al-4V was annealed, resulting in ultimate tensile strength of 1120 MPa."
AL_7075 = "The 7075-T6 aluminum alloy showed tensile strength of 77 ksi after aging treatment."
CORROSION_GAP = "Corrosion resistance after heat treatment was discussed, but no numerical corrosion data were reported."


def _catalog(tmp_path: Path) -> SQLiteCatalog:
    return SQLiteCatalog(tmp_path / "catalog.sqlite3")


def _add_document(
    catalog: SQLiteCatalog,
    doc_id: str,
    text: str,
    *,
    filename: str | None = None,
    version: int = 1,
    active: bool = True,
) -> None:
    filename = filename or f"{doc_id}.txt"
    document = Document(
        doc_id=doc_id,
        workspace_uid="test",
        title=filename,
        source_uid=f"source_{doc_id}",
        external_id=f"hash_{doc_id}",
        parser="text",
        status="ingested",
        version=version,
    )
    chunk = Chunk(
        chunk_id=f"chunk_{doc_id}_0",
        doc_id=doc_id,
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="",
        ordinal=0,
        metadata={"filename": filename, "source_type": "file"},
    )
    catalog.upsert_document(
        document,
        metadata={
            "filename": filename,
            "source_type": "file",
            "content_hash": f"hash_{doc_id}",
            "document_version": version,
            "active": active,
        },
    )
    catalog.replace_chunks(doc_id, [chunk])


def test_same_document_reingest_is_idempotent(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    _add_document(catalog, "doc_a", VT6_980)
    engine = KnowledgeExpansionEngine(catalog)
    before = engine.build_report()
    _add_document(catalog, "doc_a", VT6_980)
    after = engine.build_report()
    delta = engine.delta_from_reports(before, after, new_document_ids=["doc_a"])

    assert after["canonical_facts_count"] == before["canonical_facts_count"]
    assert delta["new_canonical_facts_count"] == 0
    assert delta["corroborated_facts_count"] == 0
    assert after["facts_without_evidence"] == 0


def test_changed_document_version_is_visible_in_report(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    _add_document(catalog, "doc_a_v1", VT6_980, filename="vt6.txt", version=1)
    _add_document(catalog, "doc_a_v2", VT6_1120, filename="vt6.txt", version=2)

    report = KnowledgeExpansionEngine(catalog).build_report(active_only=False)
    versions = sorted(item["document_version"] for item in report["documents"] if item["source_name"] == "vt6.txt")

    assert versions == [1, 2]
    assert report["conflict_groups_count"] >= 1


def test_deactivated_document_is_excluded_from_expansion_report(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    _add_document(catalog, "doc_vt6", VT6_980)
    _add_document(catalog, "doc_7075", AL_7075)
    engine = KnowledgeExpansionEngine(catalog)
    active = engine.build_report()
    assert "7075-T6" in active["materials"]

    assert catalog.set_document_active("doc_7075", False)
    inactive = engine.build_report()
    assert "7075-T6" not in inactive["materials"]

    assert catalog.set_document_active("doc_7075", True)
    reactivated = engine.build_report()
    assert "7075-T6" in reactivated["materials"]


def test_new_material_creates_comparison_opportunity(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    _add_document(catalog, "doc_vt6", VT6_980)
    engine = KnowledgeExpansionEngine(catalog)
    before = engine.build_report()
    _add_document(catalog, "doc_7075", AL_7075)
    delta = engine.build_delta_report(before, document_id="doc_7075")

    assert "7075-T6" in delta["new_materials"]
    assert delta["new_comparison_opportunities_count"] >= 1
    assert any(item.get("property") == "прочность" for item in delta["new_comparison_opportunities"])
    fact = next(item for item in delta["new_canonical_facts"] if item["material"] == "7075-T6")
    assert fact["unit_original"] == "ksi"
    assert fact["unit_normalized"] == "MPa"
    assert round(float(fact["value_normalized"]), 1) == 530.9


def test_new_conflict_group_and_data_gap_detected(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    _add_document(catalog, "doc_vt6_980", VT6_980)
    engine = KnowledgeExpansionEngine(catalog)
    before_conflict = engine.build_report()
    _add_document(catalog, "doc_vt6_1120", VT6_1120)
    conflict_delta = engine.build_delta_report(before_conflict, document_id="doc_vt6_1120")
    assert conflict_delta["conflict_groups_added_count"] >= 1

    before_gap = engine.build_report()
    _add_document(catalog, "doc_gap", CORROSION_GAP)
    gap_delta = engine.build_delta_report(before_gap, document_id="doc_gap")
    assert gap_delta["data_gaps_added_count"] >= 1
    assert any(item.get("property") == "коррозионная стойкость" for item in gap_delta["data_gaps_added"])


def test_new_evidence_for_existing_fact_is_merged(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    _add_document(catalog, "doc_a", VT6_980)
    engine = KnowledgeExpansionEngine(catalog)
    before = engine.build_report()
    _add_document(catalog, "doc_b", VT6_980)
    after = engine.build_report()
    delta = engine.delta_from_reports(before, after, new_document_ids=["doc_b"])

    assert after["canonical_facts_count"] == before["canonical_facts_count"]
    assert delta["corroborated_facts_count"] >= 1
    assert after["facts_without_evidence"] == 0


def test_knowledge_summary_api_and_ui_payload_are_clean(tmp_path: Path) -> None:
    api = reset_api(tmp_path)
    client = TestClient(api.app)
    response = client.post(
        "/ingest/documents",
        files=[("files", ("vt6.txt", VT6_980.encode("utf-8"), "text/plain"))],
    )
    assert response.status_code == 200, response.text
    payload = response.json()["ingested"][0]["knowledge_expansion"]
    assert payload["new_canonical_facts_count"] >= 1

    summary = client.get("/knowledge/summary")
    assert summary.status_code == 200
    text = json.dumps(summary.json(), ensure_ascii=False)
    assert "doc_" not in text
    assert "chunk_" not in text
    assert summary.json()["canonical_facts_count"] >= 1


def test_knowledge_expansion_report_script_writes_json(tmp_path: Path, monkeypatch) -> None:
    catalog_path = tmp_path / "catalog.sqlite3"
    catalog = SQLiteCatalog(catalog_path)
    _add_document(catalog, "doc_a", VT6_980)
    output_path = tmp_path / "knowledge_report.json"
    monkeypatch.setattr(report_script.settings, "catalog_db_path", str(catalog_path))
    args = type(
        "Args",
        (),
        {
            "document_id": None,
            "include_inactive": False,
            "since_last_ingest": False,
            "json_path": str(output_path),
        },
    )()

    report = report_script.build_report(args)
    output_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["canonical_facts_count"] >= 1
    assert payload["facts_without_evidence"] == 0

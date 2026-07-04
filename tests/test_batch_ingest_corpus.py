from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.batch_ingest_corpus import build_ingest_plan, run_batch, summarize


def test_batch_ingest_plan_classifies_realistic_formats(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    group = root / "Статьи"
    group.mkdir(parents=True)
    (group / "ok.txt").write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")
    (group / "archive.rar").write_bytes(b"rar")
    (group / "legacy.xls").write_bytes(b"xls")
    (group / "scan.gif").write_bytes(b"gif")
    (group / "large.pdf").write_bytes(b"x" * 2048)

    plan, selection = build_ingest_plan(root, max_file_mb=0.001)
    by_name = {Path(item.relative_path).name: item for item in plan}

    assert selection["total_files_found"] == 5
    assert by_name["ok.txt"].planned_status == "ready"
    assert by_name["archive.rar"].planned_reason == "archive_needs_extraction"
    assert by_name["legacy.xls"].planned_reason == "legacy_format_needs_conversion"
    assert by_name["scan.gif"].planned_reason == "ocr_required"
    assert by_name["large.pdf"].planned_reason == "file_too_large_for_batch"
    assert by_name["ok.txt"].source_group == "Статьи"


def test_batch_ingest_dry_run_writes_report_without_api(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    group = root / "Обзоры"
    group.mkdir(parents=True)
    (group / "review.txt").write_text("Ti-6Al-4V was annealed, resulting in ultimate tensile strength of 1120 MPa.", encoding="utf-8")
    report_path = tmp_path / "report.json"
    state_path = tmp_path / "state.json"
    args = argparse.Namespace(
        input=str(root),
        api_base="http://api.invalid",
        state=str(state_path),
        report=str(report_path),
        max_file_mb=25.0,
        max_files=None,
        sample_per_group=None,
        timeout=1,
        sync_graph=False,
        dry_run=True,
        force=False,
    )

    report, exit_code = run_batch(args)

    assert exit_code == 0
    assert report["summary"] == "PASS"
    assert report["status_counts"]["planned"] == 1
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["source_group"] == "Обзоры"


def test_batch_ingest_uses_resume_state_in_dry_run(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    root.mkdir()
    path = root / "done.txt"
    path.write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")
    plan, _ = build_ingest_plan(root, max_file_mb=25.0)
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "completed": {
                    plan[0].relative_path: {
                        "fingerprint": plan[0].fingerprint(),
                        "status": "ingested",
                        "doc_id": "doc_existing",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        input=str(root),
        api_base="http://api.invalid",
        state=str(state_path),
        report=str(tmp_path / "report.json"),
        max_file_mb=25.0,
        max_files=None,
        sample_per_group=None,
        timeout=1,
        sync_graph=False,
        dry_run=True,
        force=False,
    )

    report, exit_code = run_batch(args)

    assert exit_code == 0
    assert report["status_counts"]["already_ingested"] == 1
    assert report["rows"][0]["doc_id"] == "doc_existing"


def test_batch_ingest_summarize_warns_on_per_file_failure_by_default() -> None:
    report = summarize(
        [
            {"status": "ingested", "planned_reason": "supported", "knowledge_expansion": {"facts_without_evidence": 0}},
            {"status": "failed", "planned_reason": "supported", "error": "read_timeout"},
        ],
        selection={"total_files_found": 2, "selected_files": 2},
        dry_run=False,
        api_base="http://localhost:8000",
    )

    assert report["summary"] == "WARN"
    assert report["failed_files_count"] == 1
    assert report["read_timeout_count"] == 1
    assert "file_ingest_failures_present" in report["warnings"]
    assert "read_timeouts_present" in report["warnings"]


def test_batch_ingest_summarize_can_fail_strictly_on_per_file_failure() -> None:
    report = summarize(
        [{"status": "failed", "planned_reason": "supported", "error": "read_timeout"}],
        selection={"total_files_found": 1, "selected_files": 1},
        dry_run=False,
        api_base="http://localhost:8000",
        fail_on_file_error=True,
    )

    assert report["summary"] == "FAIL"
    assert report["fail_on_file_error"] is True


def test_batch_ingest_summarize_still_fails_on_facts_without_evidence() -> None:
    report = summarize(
        [{"status": "ingested", "planned_reason": "supported", "knowledge_expansion": {"facts_without_evidence": 1}}],
        selection={"total_files_found": 1, "selected_files": 1},
        dry_run=False,
        api_base="http://localhost:8000",
    )

    assert report["summary"] == "FAIL"
    assert report["facts_without_evidence"] == 1

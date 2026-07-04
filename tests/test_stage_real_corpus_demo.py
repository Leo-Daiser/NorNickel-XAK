from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.stage_real_corpus_demo import choose_stage_files, run
from scripts.batch_ingest_corpus import build_ingest_plan


def test_choose_stage_files_balances_groups_and_extensions(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    for group in ["Доклады", "Журналы", "Статьи"]:
        folder = root / group
        folder.mkdir(parents=True)
        (folder / f"{group}.txt").write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")
        (folder / f"{group}.html").write_text("<html><body>7075-T6 tensile strength 77 ksi</body></html>", encoding="utf-8")

    plan, _ = build_ingest_plan(root, max_file_mb=25.0)
    selected = choose_stage_files(plan, target_count=4)

    assert len(selected) == 4
    assert len({item.source_group for item in selected}) >= 2
    assert len({item.extension for item in selected}) >= 2


def test_stage_real_corpus_dry_run_writes_questions(tmp_path: Path) -> None:
    root = tmp_path / "data_storage"
    folder = root / "Обзоры"
    folder.mkdir(parents=True)
    (folder / "review.txt").write_text("Ti-6Al-4V was annealed, resulting in ultimate tensile strength of 1120 MPa.", encoding="utf-8")
    report = tmp_path / "stage.json"
    questions = tmp_path / "questions.md"
    args = argparse.Namespace(
        input=str(root),
        api_base="http://api.invalid",
        count=1,
        max_file_mb=25.0,
        max_files=None,
        sample_per_group=None,
        timeout=1,
        reset=False,
        sync_neo4j=False,
        dry_run=True,
        report=str(report),
        questions=str(questions),
    )

    result, code = run(args)

    assert code == 0
    assert result["summary"] == "PASS"
    assert report.exists()
    assert questions.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["selected_count"] == 1
    assert "Какие материалы" in questions.read_text(encoding="utf-8")

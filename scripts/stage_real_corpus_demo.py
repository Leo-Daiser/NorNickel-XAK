"""Stage a clean real-corpus demo set from data_storage through the API.

The script is intentionally conservative:
* it never resets the active corpus unless --reset is passed;
* reset requires the API endpoint confirmation token;
* files are uploaded one by one to avoid Streamlit/API batch timeouts;
* graph/Neo4j refresh is performed once after ingestion.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.batch_ingest_corpus import PlannedFile, build_ingest_plan, planned_row  # noqa: E402


DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_REPORT = ROOT / "artifacts" / "real_corpus_stage_report.json"
DEFAULT_QUESTIONS = ROOT / "artifacts" / "real_corpus_ui_questions.md"


REAL_CORPUS_QUESTIONS = [
    "Какие материалы, процессы и свойства встречаются в загруженных источниках?",
    "Какие методы обессоливания воды подходят для обогатительной фабрики при сульфатах и хлоридах 200-300 мг/л?",
    "Какие решения по циркуляции католита при электроэкстракции никеля описаны в источниках?",
    "Какие системы очистки газов и печи взвешенной плавки применяются для удаления SO2?",
    "Покажите эксперименты и публикации по распределению Au, Ag и МПГ между штейном и шлаком.",
    "Какие способы закачки шахтных вод в глубокие горизонты упоминаются в российских и зарубежных источниках?",
    "Какие пробелы в данных найдены в активном корпусе?",
    "Есть ли противоречия или неоднородные данные по численным параметрам?",
    "Какие источники подтверждают найденные выводы по гидрометаллургии?",
    "Какие лаборатории, команды или авторы упоминаются в загруженных документах?",
]


def choose_stage_files(plan: list[PlannedFile], *, target_count: int) -> list[PlannedFile]:
    """Pick ready files across source groups and extensions."""

    ready = [item for item in plan if item.planned_status == "ready"]
    groups: dict[tuple[str, str], list[PlannedFile]] = {}
    for item in ready:
        key = (item.source_group or "root", item.extension)
        groups.setdefault(key, []).append(item)
    for items in groups.values():
        items.sort(key=lambda item: (item.file_size_mb, item.relative_path))

    selected: list[PlannedFile] = []
    seen = set()
    while len(selected) < target_count:
        changed = False
        for key in sorted(groups):
            items = groups[key]
            while items and items[0].relative_path in seen:
                items.pop(0)
            if not items:
                continue
            item = items.pop(0)
            selected.append(item)
            seen.add(item.relative_path)
            changed = True
            if len(selected) >= target_count:
                break
        if not changed:
            break
    return selected


def request_json(method: str, api_base: str, path: str, *, timeout: int = 120, **kwargs) -> dict[str, Any]:
    response = requests.request(method, f"{api_base.rstrip('/')}{path}", timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


def ingest_file(api_base: str, item: PlannedFile, *, timeout: int, sync_graph: bool) -> dict[str, Any]:
    mime = mimetypes.guess_type(item.path.name)[0] or "application/octet-stream"
    started = time.perf_counter()
    try:
        with item.path.open("rb") as handle:
            response = requests.post(
                f"{api_base.rstrip('/')}/ingest/documents",
                params={"sync_graph": "true" if sync_graph else "false"},
                files=[("files", (item.path.name, handle, mime))],
                timeout=timeout,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"raw": response.text[:500]}
        if response.status_code != 200:
            return {"status": "failed", "http_status": response.status_code, "elapsed_ms": elapsed_ms, "error": payload}
        ingested = (payload.get("ingested") or [{}])[0]
        return {
            "status": ingested.get("parse_status") or ingested.get("status") or "ingested",
            "http_status": response.status_code,
            "elapsed_ms": elapsed_ms,
            "doc_id": ingested.get("doc_id"),
            "chunks": ingested.get("chunks", 0),
            "knowledge_expansion": ingested.get("knowledge_expansion") or {},
        }
    except requests.ReadTimeout:
        return {"status": "failed", "elapsed_ms": int((time.perf_counter() - started) * 1000), "error": "read_timeout"}
    except requests.RequestException as exc:
        return {"status": "failed", "elapsed_ms": int((time.perf_counter() - started) * 1000), "error": f"{type(exc).__name__}: {exc}"}


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    plan, selection = build_ingest_plan(
        args.input,
        max_file_mb=args.max_file_mb,
        max_files=args.max_files,
        sample_per_group=args.sample_per_group,
    )
    selected = choose_stage_files(plan, target_count=args.count)
    rows: list[dict[str, Any]] = []

    if args.dry_run:
        report = {
            "summary": "PASS" if selected else "WARN",
            "dry_run": True,
            "selection": selection,
            "selected_count": len(selected),
            "rows": [planned_row(item) | {"status": "planned"} for item in selected],
            "questions": REAL_CORPUS_QUESTIONS,
        }
        write_outputs(report, args.report, args.questions)
        return report, 0 if selected else 1

    if args.reset:
        reset = request_json(
            "POST",
            args.api_base,
            "/admin/reset-corpus",
            params={"confirm": "RESET_ACTIVE_CORPUS", "clear_neo4j": "true"},
            timeout=args.timeout,
        )
    else:
        reset = {"status": "skipped"}

    for item in selected:
        row = planned_row(item)
        row.update(ingest_file(args.api_base, item, timeout=args.timeout, sync_graph=False))
        rows.append(row)

    refresh = request_json(
        "POST",
        args.api_base,
        "/graph/refresh",
        params={"sync_neo4j": "true" if args.sync_neo4j else "false"},
        timeout=args.timeout,
    )
    failures = [row for row in rows if str(row.get("status")) in {"failed", "parse_failed"}]
    report = {
        "summary": "FAIL" if failures else "PASS",
        "dry_run": False,
        "reset": reset,
        "selection": selection,
        "selected_count": len(selected),
        "ingested_count": len(rows) - len(failures),
        "failed_count": len(failures),
        "refresh": refresh,
        "rows": rows,
        "questions": REAL_CORPUS_QUESTIONS,
    }
    write_outputs(report, args.report, args.questions)
    return report, 1 if failures else 0


def write_outputs(report: dict[str, Any], report_path: str | Path, questions_path: str | Path) -> None:
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    questions_file = Path(questions_path)
    questions_file.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# UI questions for staged real corpus", ""]
    for index, question in enumerate(report.get("questions") or [], start=1):
        lines.append(f"{index}. {question}")
    questions_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage a clean real-corpus demo set via the API.")
    parser.add_argument("--input", default="data_storage")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--max-file-mb", type=float, default=25.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sample-per-group", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--reset", action="store_true", help="Clear current catalog/runtime/Neo4j before staging.")
    parser.add_argument("--sync-neo4j", action="store_true", help="Sync Neo4j once after all files are uploaded.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result, code = run(args)
    print(f"SUMMARY: {result['summary']}")
    print(f"selected_count: {result.get('selected_count')}")
    print(f"report: {args.report}")
    print(f"questions: {args.questions}")
    raise SystemExit(code)

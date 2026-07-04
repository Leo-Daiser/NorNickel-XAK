from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.parser_audit import ParserAuditWriter  # noqa: E402
from app.ingestion.parser_router import ParserRouter  # noqa: E402


GOLD_PATH = Path(__file__).with_name("parser_gold.json")


def _score_case(router: ParserRouter, case: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    path = ROOT / case["path"]
    result = router.parse_document_intelligence(str(path), doc_id=f"parser_eval_{case['id']}")
    chunks = result.chunks
    table_rows = [chunk for chunk in chunks if chunk.metadata.get("chunk_kind") == "table_row"]
    paragraph_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_kind") in {"paragraph", "text_window", "section_heading"}]
    image_refs = result.images or [
        image
        for chunk in chunks
        for image in chunk.metadata.get("image_refs", [])
    ]
    metadata_ok = all(
        chunk.metadata.get("chunk_kind") and chunk.metadata.get("parser_name") and chunk.text_hash
        for chunk in chunks
    )
    diagnostics_ok = bool(result.diagnostics.get("parser_backend_requested") and result.diagnostics.get("parser_backend_used"))
    checks = {
        "parse_success": float(bool(chunks)),
        "chunks_nonempty": float(bool(chunks) and all(chunk.text.strip() for chunk in chunks)),
        "table_row_presence": float((not case.get("expect_table_rows")) or bool(table_rows)),
        "metadata_presence": float(metadata_ok),
        "diagnostics_presence": float(diagnostics_ok),
        "paragraph_presence": float((not case.get("expect_paragraph_chunks")) or bool(paragraph_chunks)),
        "image_presence": float((not case.get("expect_image_refs")) or bool(image_refs)),
        "section_path_presence": float((not case.get("expect_section_path")) or all(chunk.section_path for chunk in chunks)),
    }
    passed = all(value == 1.0 for value in checks.values())
    print(
        f"{'PASS' if passed else 'FAIL'} {case['id']}: "
        f"chunks={len(chunks)} blocks={len(result.blocks)} tables={len(result.tables)} images={len(result.images)} "
        f"backend={result.diagnostics.get('parser_backend_used')}"
    )
    return passed, checks


def main() -> int:
    cases = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    audit_dir = ROOT / "data" / "parser_audit_eval"
    router = ParserRouter(audit_writer=ParserAuditWriter(audit_dir))
    results = []
    all_checks: dict[str, list[float]] = {}
    for case in cases:
        passed, checks = _score_case(router, case)
        results.append(passed)
        for key, value in checks.items():
            all_checks.setdefault(key, []).append(value)

    metrics = {
        "documents_parsed": float(len(cases)),
        "parse_success_rate": mean(all_checks["parse_success"]),
        "chunks_nonempty_rate": mean(all_checks["chunks_nonempty"]),
        "table_row_chunk_presence": mean(all_checks["table_row_presence"]),
        "metadata_presence_rate": mean(all_checks["metadata_presence"]),
        "parser_diagnostics_presence": mean(all_checks["diagnostics_presence"]),
    }
    print("\nParser evaluation:")
    print(f"documents_parsed: {int(metrics['documents_parsed'])}")
    for key, value in metrics.items():
        if key == "documents_parsed":
            continue
        print(f"{key}: {value:.3f}")
    passed = all(results) and all(value >= 1.0 for key, value in metrics.items() if key != "documents_parsed")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

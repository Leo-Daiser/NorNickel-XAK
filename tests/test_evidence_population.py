from __future__ import annotations

from app.answering.human_answer import enhance_answer_payload
from app.ui_helpers import evidence_to_rows


def test_sources_are_promoted_to_evidence_rows() -> None:
    payload = enhance_answer_payload(
        {
            "answer": "legacy",
            "status": "ok",
            "constraints": {},
            "facts": [],
            "sources": [
                {
                    "doc_id": "doc1",
                    "chunk_id": "chunk1",
                    "title": "source.txt",
                    "quote": "ВТ6 отжиг прочность 1120 MPa",
                }
            ],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "diagnostics": {},
            "retrieval": {},
        },
        "expert_max",
    )
    assert payload["evidence"]
    row = evidence_to_rows(payload)[0]
    assert row["source_name"] == "source.txt"
    assert row["chunk_id"] == "chunk1"
    assert row["quote"]


def test_ui_no_longer_uses_evidence_na_text() -> None:
    from pathlib import Path

    text = Path("app/ui.py").read_text(encoding="utf-8")
    assert "Evidence: n/a" not in text

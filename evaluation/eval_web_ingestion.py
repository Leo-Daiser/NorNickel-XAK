"""Offline regression eval for URL web-page ingestion."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("KG_BACKEND", "fallback")
os.environ.setdefault("RUNTIME_PROFILE", "economy_core")
os.environ.setdefault("ENABLE_LLM", "false")
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("ENABLE_LOCAL_EMBEDDINGS", "false")
os.environ.setdefault("RETRIEVAL_MODE", "bm25")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from tests.strict_qa_helpers import reset_api  # noqa: E402


HTML = """
<html>
  <head><title>VT6 annealing study</title></head>
  <body>
    <h1>Исследование ВТ6 после отжига</h1>
    <p>После отжига сплава ВТ6 предел прочности составил 980 MPa.</p>
    <p>Источник указывает на необходимость дополнительных данных по коррозионной стойкости.</p>
  </body>
</html>
""".encode("utf-8")

RAW_MARKERS = ["doc_", "chunk_", "EXP-", "SCI-", "technical_answer", "PropertyValue", "SourceChunk"]


class FakeHtmlResponse:
    headers = {"content-type": "text/html; charset=utf-8"}
    content = HTML
    is_redirect = False
    is_permanent_redirect = False

    def raise_for_status(self) -> None:
        return None


def _check(rows: list[dict[str, Any]], condition: bool, name: str, reason: str) -> None:
    rows.append({"check": name, "status": "PASS" if condition else "FAIL", "reason": reason})


def _no_raw_main(payload: dict[str, Any]) -> bool:
    main = str(payload.get("answer") or "")
    return not any(marker in main for marker in RAW_MARKERS)


def run_eval() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        api = reset_api(Path(tmp))
        api.settings.kg_backend = "fallback"

        import app.security.url_safety as url_safety

        url_safety._resolve_host = lambda host, port: {url_safety.ipaddress.ip_address("93.184.216.34")}
        api.requests.get = lambda *args, **kwargs: FakeHtmlResponse()

        client = TestClient(api.app)
        url = "https://example.org/reports/vt6-annealing.html?utm_source=demo"
        ingest = client.post("/ingest/url", params={"url": url})
        _check(rows, ingest.status_code == 200, "url_ingested", f"status={ingest.status_code}")
        payload = ingest.json()["ingested"] if ingest.status_code == 200 else {}
        delta = payload.get("knowledge_expansion") or {}
        doc_id = payload.get("doc_id")

        _check(rows, payload.get("chunks", 0) > 0, "chunks_created", "HTML produced chunks")
        _check(rows, payload.get("source_name") == "VT6 annealing study", "readable_source_name", "HTML title is used as source name")
        metadata = api.catalog.get_document_metadata(doc_id) if doc_id else {}
        _check(rows, metadata.get("source_type") == "url", "source_type_url", "Document metadata source_type=url")
        _check(rows, metadata.get("source_url") == url, "source_url_saved", "Raw URL is preserved in metadata")
        _check(rows, bool(metadata.get("content_hash")), "content_hash_saved", "Content hash saved")
        _check(rows, bool(metadata.get("ingested_at")), "ingested_at_saved", "Ingest timestamp saved")

        report = client.get("/knowledge/expansion-report").json()
        facts_text = json.dumps(report.get("canonical_facts") or [], ensure_ascii=False)
        gaps_text = json.dumps(report.get("data_gaps") or [], ensure_ascii=False)
        _check(rows, "ВТ6" in report.get("materials", []), "material_extracted", "Material ВТ6 extracted")
        _check(rows, "отжиг" in report.get("regimes", []), "regime_extracted", "Regime отжиг extracted")
        _check(rows, "прочность" in report.get("properties", []), "property_extracted", "Property прочность extracted")
        _check(rows, "980.0" in facts_text and "MPa" in facts_text, "measurement_extracted", "Measurement 980 MPa extracted")
        _check(rows, report.get("facts_without_evidence") == 0, "evidence_exists", "Accepted facts have evidence")
        _check(rows, "коррозионная стойкость" in gaps_text, "data_gap_detected", "Corrosion resistance gap detected")
        _check(rows, delta.get("new_canonical_facts_count", 0) >= 1, "delta_new_facts", "Knowledge expansion delta has new facts")

        answer = client.post(
            "/ask",
            json={"question": "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?", "preset_id": "strict_audit"},
        )
        answer_payload = answer.json() if answer.status_code == 200 else {}
        answer_text = json.dumps(answer_payload, ensure_ascii=False)
        _check(rows, answer.status_code == 200 and answer_payload.get("status") == "ok", "answer_uses_web_source", "Answer uses URL-ingested source")
        _check(rows, _no_raw_main(answer_payload), "no_raw_main_answer", "Main answer has no raw IDs")
        _check(rows, "VT6 annealing study" in answer_text, "readable_source_in_payload", "Readable source title appears in payload")

    return {
        "summary": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "checks": rows,
    }


def main() -> int:
    result = run_eval()
    path = ROOT / "artifacts" / "eval_web_ingestion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY: {result['summary']}")
    for row in result["checks"]:
        print(f"[{row['status']}] {row['check']}: {row['reason']}")
    print(f"JSON report: {path}")
    return 0 if result["summary"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.strict_qa_helpers import STRICT_QA_SAMPLE, reset_api


def test_upload_then_refresh_flow_reports_active_counts(tmp_path) -> None:
    api = reset_api(tmp_path)
    client = TestClient(api.app)
    upload = client.post(
        "/ingest/documents",
        files=[("files", ("sample.txt", STRICT_QA_SAMPLE.encode("utf-8"), "text/plain"))],
    )
    assert upload.status_code == 200
    assert upload.json()["ingested"][0]["chunks"] >= 1
    refresh = client.post("/graph/refresh")
    assert refresh.status_code == 200
    payload = refresh.json()
    assert payload["status"] == "refreshed"
    assert payload["active_documents"] == 1
    assert payload["active_chunks"] >= 1

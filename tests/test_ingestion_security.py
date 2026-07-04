from __future__ import annotations

from fastapi.testclient import TestClient
import pytest


def test_url_safety_blocks_localhost_private_and_non_http() -> None:
    from app.security.url_safety import UnsafeUrlError, validate_ingest_url

    for url in [
        "http://localhost/page.html",
        "http://127.0.0.1/page.html",
        "http://10.0.0.5/page.html",
        "http://metadata.google.internal/page.html",
        "http://intranet/page.html",
        "file:///etc/passwd",
    ]:
        try:
            validate_ingest_url(url)
        except UnsafeUrlError:
            continue
        raise AssertionError(f"URL should be blocked: {url}")


def test_url_fetch_blocks_redirect_to_private(monkeypatch) -> None:
    from app.security import url_safety
    from app.security.url_safety import UnsafeUrlError, fetch_url_safely

    monkeypatch.setattr(url_safety, "_resolve_host", lambda host, port: {url_safety.ipaddress.ip_address("93.184.216.34")} if host == "example.org" else {url_safety.ipaddress.ip_address("127.0.0.1")})

    class RedirectResponse:
        headers = {"location": "http://127.0.0.1/private.html"}
        is_redirect = True
        is_permanent_redirect = False

        def close(self) -> None:
            return None

    with pytest.raises(UnsafeUrlError):
        fetch_url_safely("https://example.org/page.html", request_get=lambda *args, **kwargs: RedirectResponse())


def test_url_fetch_oversized_response_is_controlled(monkeypatch) -> None:
    from app.security import url_safety
    from app.security.url_safety import UnsafeUrlError, fetch_url_safely

    monkeypatch.setattr(url_safety, "_resolve_host", lambda host, port: {url_safety.ipaddress.ip_address("93.184.216.34")})

    class LargeResponse:
        headers = {"content-type": "text/html"}
        is_redirect = False
        is_permanent_redirect = False

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int):
            yield b"x" * 8
            yield b"x" * 8

        def close(self) -> None:
            return None

    with pytest.raises(UnsafeUrlError):
        fetch_url_safely("https://example.org/page.html", max_bytes=10, request_get=lambda *args, **kwargs: LargeResponse())


def test_url_ingest_rejects_non_html_response(tmp_path, monkeypatch) -> None:
    from tests.strict_qa_helpers import reset_api
    import app.security.url_safety as url_safety

    api = reset_api(tmp_path)
    monkeypatch.setattr(url_safety, "_resolve_host", lambda host, port: {url_safety.ipaddress.ip_address("93.184.216.34")})

    class JsonResponse:
        headers = {"content-type": "application/json"}
        content = b'{"ok": true}'
        is_redirect = False
        is_permanent_redirect = False

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(api.requests, "get", lambda *args, **kwargs: JsonResponse())
    response = TestClient(api.app).post("/ingest/url", params={"url": "https://example.org/data.json"})
    assert response.status_code == 400
    assert "not HTML" in response.text


def test_valid_html_url_ingest_is_accepted(tmp_path, monkeypatch) -> None:
    from tests.strict_qa_helpers import reset_api
    import app.security.url_safety as url_safety

    api = reset_api(tmp_path)
    monkeypatch.setattr(url_safety, "_resolve_host", lambda host, port: {url_safety.ipaddress.ip_address("93.184.216.34")})

    class HtmlResponse:
        headers = {"content-type": "text/html; charset=utf-8"}
        content = b"<html><head><title>VT6 annealing study</title></head><body><p>After annealing VT6 strength was 980 MPa.</p></body></html>"
        is_redirect = False
        is_permanent_redirect = False

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(api.requests, "get", lambda *args, **kwargs: HtmlResponse())
    response = TestClient(api.app).post("/ingest/url", params={"url": "https://example.org/reports/vt6.html"})
    assert response.status_code == 200, response.text
    item = response.json()["ingested"]
    assert item["source_name"] == "VT6 annealing study"


def test_upload_rejects_unsupported_extension() -> None:
    import app.api as api

    client = TestClient(api.app)
    response = client.post("/ingest/documents", files=[("files", ("payload.exe", b"bad", "application/octet-stream"))])
    assert response.status_code == 400
    assert "Unsupported file extension" in response.text


def test_upload_max_size_enforced(monkeypatch) -> None:
    import app.api as api

    monkeypatch.setattr(api.settings, "max_upload_mb", 0)
    client = TestClient(api.app)
    response = client.post("/ingest/documents", files=[("files", ("small.txt", b"x", "text/plain"))])
    assert response.status_code == 413

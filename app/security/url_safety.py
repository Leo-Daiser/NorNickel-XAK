"""URL ingestion safety checks against SSRF and oversized downloads."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests


class UnsafeUrlError(ValueError):
    """Raised when a URL is not safe for server-side ingestion."""


@dataclass(frozen=True)
class FetchedUrl:
    url: str
    content: bytes
    content_type: str


def validate_ingest_url(url: str, *, allow_private: bool = False) -> None:
    """Validate scheme, host and resolved addresses before fetching."""

    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError("Only http/https URLs are allowed")
    if not parsed.hostname:
        raise UnsafeUrlError("URL host is required")
    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"}:
        raise UnsafeUrlError("localhost URLs are blocked")
    if "." not in host:
        raise UnsafeUrlError("Internal hostnames are blocked")
    if host.endswith((".localhost", ".local", ".internal", ".lan", ".home", ".corp")):
        raise UnsafeUrlError("Internal hostnames are blocked")
    if allow_private:
        return
    for address in _resolve_host(host, parsed.port):
        if _is_blocked_address(address):
            raise UnsafeUrlError(f"Private or local address is blocked: {address}")


def fetch_url_safely(
    url: str,
    *,
    allow_private: bool = False,
    max_bytes: int = 10_485_760,
    timeout_seconds: int = 10,
    max_redirects: int = 3,
    headers: dict[str, str] | None = None,
    request_get=None,
) -> FetchedUrl:
    """Fetch a URL with redirect revalidation and response-size limit."""

    current = str(url or "").strip()
    request_headers = {"User-Agent": "ScientificKnowledgeGraphDemo/1.0", **(headers or {})}
    get = request_get or requests.get
    for _ in range(max_redirects + 1):
        validate_ingest_url(current, allow_private=allow_private)
        response = get(
            current,
            timeout=timeout_seconds,
            headers=request_headers,
            stream=True,
            allow_redirects=False,
        )
        if getattr(response, "is_redirect", False) or getattr(response, "is_permanent_redirect", False):
            location = response.headers.get("location")
            response.close()
            if not location:
                raise UnsafeUrlError("Redirect without Location header")
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        iterable = response.iter_content(chunk_size=65536) if hasattr(response, "iter_content") else [getattr(response, "content", b"")]
        for chunk in iterable:
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                response.close()
                raise UnsafeUrlError(f"URL response exceeds max size: {max_bytes} bytes")
            chunks.append(chunk)
        return FetchedUrl(
            url=current,
            content=b"".join(chunks),
            content_type=response.headers.get("content-type", ""),
        )
    raise UnsafeUrlError("Too many redirects")


def _resolve_host(host: str, port: int | None) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, port or 80, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return set()
    addresses = set()
    for info in infos:
        raw = info[4][0]
        try:
            addresses.add(ipaddress.ip_address(raw))
        except ValueError:
            continue
    return addresses


def _is_blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )

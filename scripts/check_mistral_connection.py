from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


def _load_local_env_if_needed() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _chat_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _safe_error(message: str, api_key: str) -> str:
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return message[:240]


def main() -> int:
    _load_local_env_if_needed()
    api_key = os.getenv("MISTRAL_API_KEY") or getattr(settings, "mistral_api_key", "")
    base_url = os.getenv("MISTRAL_BASE_URL") or getattr(settings, "mistral_base_url", "https://api.mistral.ai/v1")
    model = os.getenv("MISTRAL_MODEL") or getattr(settings, "mistral_model", "mistral-small-latest")
    timeout = int(os.getenv("MISTRAL_TIMEOUT_SECONDS") or getattr(settings, "mistral_timeout_seconds", 60))
    max_tokens = int(os.getenv("MISTRAL_MAX_TOKENS") or getattr(settings, "mistral_max_tokens", 1200))
    temperature = float(os.getenv("MISTRAL_TEMPERATURE") or getattr(settings, "mistral_temperature", 0.2))

    result: dict[str, Any] = {
        "provider": "mistral",
        "base_url": base_url,
        "model": model,
        "api_key_configured": bool(api_key),
        "result": "error",
        "error": "",
    }
    if not api_key:
        result["error"] = "MISTRAL_API_KEY is missing."
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    try:
        response = requests.post(
            _chat_url(base_url),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return a short plain response."},
                    {"role": "user", "content": "Say OK."},
                ],
                "temperature": temperature,
                "max_tokens": min(max_tokens, 32),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        choices = (response.json() or {}).get("choices") or []
        content = ((choices[0] if choices else {}).get("message") or {}).get("content") or ""
        if not content.strip():
            result["error"] = "Empty response content."
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 3
        result["result"] = "ok"
        result["error"] = ""
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        text = getattr(exc.response, "text", "") or ""
        result["error"] = _safe_error(f"http_{status}:{text}", api_key)
    except requests.Timeout:
        result["error"] = "timeout"
    except Exception as exc:
        result["error"] = _safe_error(f"{type(exc).__name__}:{exc}", api_key)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

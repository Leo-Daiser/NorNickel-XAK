"""Optional structured LLM layer for answer synthesis and extraction.

Supported providers:
- ``ollama``: POST ``/api/chat``.
- ``openai_compatible``: POST ``/v1/chat/completions``.
- ``openrouter``: OpenAI-compatible endpoint with OpenRouter headers.
- ``mistral``: Mistral La Plateforme chat completions endpoint.
- ``groq``: OpenAI-compatible endpoint at ``/openai/v1/chat/completions``.

The adapter intentionally depends only on ``requests`` and is disabled by
default.  It never raises to the API layer; failures return ``None`` so the
rule-based pipeline remains the canonical fallback.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

import requests

from ..config import settings


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class StructuredLLM:
    """Small provider-neutral wrapper around local/openai-compatible chat APIs."""

    def __init__(self) -> None:
        env = os.environ
        openrouter_key = _env_or_setting("OPENROUTER_API_KEY", "openrouter_api_key")
        mistral_key = _env_or_setting("MISTRAL_API_KEY", "mistral_api_key")
        explicit_provider = env.get("LLM_PROVIDER")
        configured_provider = str(explicit_provider or getattr(settings, "llm_provider", "none") or "none").lower()
        explicit_model = env.get("LLM_MODEL")
        openrouter_model = _env_or_setting("OPENROUTER_MODEL", "openrouter_model") or explicit_model
        mistral_model = (
            env.get("MISTRAL_MODEL")
            or (explicit_model if configured_provider in {"mistral", "auto"} else "")
            or getattr(settings, "mistral_model", "mistral-small-latest")
        )
        configured_key = _env_or_setting("LLM_API_KEY", "llm_api_key") or openrouter_key
        settings_model = getattr(settings, "llm_model", "") or ""
        model_hint = explicit_model or openrouter_model or settings_model
        looks_openrouter = bool(openrouter_key) or configured_key.startswith("sk-or-") or str(model_hint).startswith("openrouter/")
        openrouter_key_available = bool(openrouter_key or configured_key.startswith("sk-or-"))
        if (not explicit_provider or configured_provider == "none") and looks_openrouter:
            configured_provider = "openrouter"
        if configured_provider == "auto":
            if mistral_key:
                provider = "mistral"
            elif openrouter_key_available:
                provider = "openrouter"
            else:
                provider = "offline"
        else:
            provider = configured_provider
        if configured_provider == "none" and looks_openrouter:
            provider = "openrouter"
            configured_provider = "openrouter"
        self.provider_configured = configured_provider
        self.provider = str(provider).lower()
        self.provider_active = self.provider
        self.fallback_reason = ""

        self.openrouter_api_key = openrouter_key or (configured_key if self.provider in {"openrouter", "auto"} or configured_key.startswith("sk-or-") else "")
        if openrouter_model:
            self.openrouter_model = openrouter_model
        elif self.provider == "openrouter" and self.openrouter_api_key:
            self.openrouter_model = ""
        elif configured_key.startswith("sk-or-") and not settings_model:
            self.openrouter_model = ""
        else:
            self.openrouter_model = settings_model

        self.mistral_api_key = mistral_key
        self.mistral_model = mistral_model or "mistral-small-latest"
        self.mistral_base_url = str(
            env.get("MISTRAL_BASE_URL")
            or getattr(settings, "mistral_base_url", "https://api.mistral.ai/v1")
            or "https://api.mistral.ai/v1"
        ).rstrip("/")
        self.mistral_timeout = _env_int("MISTRAL_TIMEOUT_SECONDS", int(getattr(settings, "mistral_timeout_seconds", 60)))
        self.mistral_max_tokens = _env_int("MISTRAL_MAX_TOKENS", int(getattr(settings, "mistral_max_tokens", 1200)))
        self.mistral_temperature = _env_float("MISTRAL_TEMPERATURE", float(getattr(settings, "mistral_temperature", 0.2)))

        if self.provider == "mistral":
            self.model = self.mistral_model
            self.api_key = self.mistral_api_key
        elif self.provider == "openrouter":
            self.model = self.openrouter_model
            self.api_key = self.openrouter_api_key
        elif explicit_model:
            self.model = explicit_model
            self.api_key = configured_key
        elif self.provider == "openrouter" and openrouter_key:
            self.model = ""
            self.api_key = configured_key
        elif self.provider == "openrouter" and configured_key.startswith("sk-or-") and not settings_model:
            self.model = ""
            self.api_key = configured_key
        else:
            self.model = settings_model
            self.api_key = configured_key

        raw_base_url = str(
            env.get("LLM_BASE_URL")
            or _env_or_setting("OPENROUTER_BASE_URL", "openrouter_base_url")
            or getattr(settings, "llm_base_url", "")
        ).rstrip("/")
        default_ollama_urls = {"", "http://localhost:11434", "http://host.docker.internal:11434"}
        if self.provider == "openrouter" and raw_base_url in default_ollama_urls:
            raw_base_url = "https://openrouter.ai/api/v1"
        self.openrouter_base_url = str(_env_or_setting("OPENROUTER_BASE_URL", "openrouter_base_url") or raw_base_url or "https://openrouter.ai/api/v1").rstrip("/")
        if self.openrouter_base_url in default_ollama_urls:
            self.openrouter_base_url = "https://openrouter.ai/api/v1"
        if self.provider == "openrouter":
            raw_base_url = self.openrouter_base_url
        if self.provider == "mistral":
            raw_base_url = self.mistral_base_url
        if self.provider == "groq" and raw_base_url in default_ollama_urls:
            raw_base_url = "https://api.groq.com/openai/v1"
        self.base_url = raw_base_url
        self.referer = getattr(settings, "llm_referer", "http://localhost:8501") or "http://localhost:8501"
        self.app_title = getattr(settings, "llm_app_title", "Scientific Knowledge Graph Demo") or "Scientific Knowledge Graph Demo"
        self.timeout = self.mistral_timeout if self.provider == "mistral" else int(getattr(settings, "llm_timeout_seconds", 20))
        explicit_llm_enabled = _env_bool("LLM_ENABLED", None)
        legacy_enable_llm = _env_bool("ENABLE_LLM", None)
        if explicit_llm_enabled is not None:
            self.config_enabled = explicit_llm_enabled
        elif legacy_enable_llm is not None:
            self.config_enabled = legacy_enable_llm
        else:
            self.config_enabled = None
        if self.config_enabled is None:
            self.config_enabled = bool(getattr(settings, "enable_llm", False))
        if explicit_llm_enabled is None and legacy_enable_llm is None and self.provider in {"openrouter", "mistral"} and self.api_key and self.model:
            self.config_enabled = True
        if self.provider == "offline":
            self.config_enabled = False
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return self.ready

    @property
    def ready(self) -> bool:
        provider_ok = self.provider in {"ollama", "openai_compatible", "openrouter", "mistral", "groq"}
        key_ok = self.provider == "ollama" or bool(self.api_key) or self.provider == "openai_compatible"
        return bool(self.config_enabled) and provider_ok and bool(self.base_url and self.model) and key_ok and not _is_placeholder_model(self.model)

    def _configuration_error(self) -> str:
        if self.provider == "offline":
            return "LLM provider is offline; template fallback is active."
        if self.provider not in {"ollama", "openai_compatible", "openrouter", "mistral", "groq"}:
            return f"Unsupported or missing LLM provider: {self.provider}."
        if not self.base_url:
            return "MISTRAL_BASE_URL is missing." if self.provider == "mistral" else "LLM_BASE_URL is missing."
        if not self.model:
            if self.provider == "openrouter" and self.api_key:
                return "OpenRouter API key is configured, but LLM_MODEL/OPENROUTER_MODEL is missing."
            if self.provider == "mistral" and self.api_key:
                return "Mistral API key is configured, but MISTRAL_MODEL/LLM_MODEL is missing."
            return "LLM_API_KEY is configured, but LLM_MODEL is missing" if self.api_key else "LLM_MODEL is missing."
        if _is_placeholder_model(self.model):
            return "LLM_MODEL/OPENROUTER_MODEL still contains a placeholder; set a real model slug."
        if self.provider == "mistral" and not self.api_key:
            return "MISTRAL_API_KEY is missing."
        if self.provider != "ollama" and self.provider != "openai_compatible" and not self.api_key:
            return "LLM_API_KEY is missing."
        if not self.config_enabled:
            return "LLM is disabled; set LLM_ENABLED=true."
        return self.last_error

    def status(self) -> Dict[str, Any]:
        error = self.last_error if self.ready else self._configuration_error()
        return {
            "enabled": bool(self.config_enabled),
            "provider": self.provider,
            "provider_configured": self.provider_configured,
            "provider_active": self.provider_active,
            "base_url": self.base_url or None,
            "model": self.model or None,
            "api_key_configured": bool(self.api_key),
            "ready": self.ready,
            "last_error": error,
            "fallback_reason": self.fallback_reason,
            "llm_enabled": bool(self.config_enabled),
            "llm_provider_configured": self.provider_configured,
            "llm_provider_active": self.provider_active,
            "mistral_base_url": self.mistral_base_url or None,
            "mistral_model": self.mistral_model or None,
            "mistral_api_key_configured": bool(self.mistral_api_key),
            "openrouter_api_key_configured": bool(self.openrouter_api_key),
            "llm_ready": self.ready,
            "llm_last_error": error,
        }

    def _provider_ready(self, provider: str) -> bool:
        if provider == "mistral":
            return bool(self.config_enabled and self.mistral_api_key and self.mistral_base_url and self.mistral_model)
        if provider == "openrouter":
            return bool(self.config_enabled and self.openrouter_api_key and self.openrouter_base_url and self.openrouter_model)
        return self.ready if provider == self.provider else False

    def _provider_error(self, provider: str) -> str:
        if provider == "mistral":
            if not self.mistral_api_key:
                return "MISTRAL_API_KEY is missing."
            if not self.mistral_base_url:
                return "MISTRAL_BASE_URL is missing."
            if not self.mistral_model:
                return "MISTRAL_MODEL is missing."
        if provider == "openrouter":
            if not self.openrouter_api_key:
                return "OPENROUTER_API_KEY/LLM_API_KEY is missing."
            if not self.openrouter_model:
                return "OPENROUTER_MODEL/LLM_MODEL is missing."
            if not self.openrouter_base_url:
                return "OPENROUTER_BASE_URL/LLM_BASE_URL is missing."
        return self._configuration_error()

    def _openai_headers(self, provider: str | None = None) -> Dict[str, str]:
        active_provider = provider or self.provider
        api_key = self.mistral_api_key if active_provider == "mistral" else self.openrouter_api_key if active_provider == "openrouter" else self.api_key
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if active_provider == "openrouter":
            # OpenRouter recommends HTTP-Referer and X-Title for rankings/analytics.
            headers["HTTP-Referer"] = self.referer
            headers["X-Title"] = self.app_title
        return headers

    def _chat_completions_url(self, provider: str | None = None) -> str:
        active_provider = provider or self.provider
        base_url = self.base_url
        if active_provider == "mistral":
            base_url = self.mistral_base_url
        elif active_provider == "openrouter":
            base_url = self.openrouter_base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    def _parse_json_object(self, text: str) -> Dict[str, Any] | None:
        if not text:
            return None
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except Exception:
            pass
        match = _JSON_RE.search(text)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def _chat(self, system: str, user: str) -> str | None:
        if not self.ready:
            self.last_error = self._configuration_error()
            return None
        self.last_error = ""
        self.fallback_reason = ""
        self.provider_active = self.provider
        providers = [self.provider]
        if self.provider == "mistral" and self._provider_ready("openrouter"):
            providers.append("openrouter")
        errors: List[str] = []
        for provider in providers:
            content = self._chat_once(provider, system, user)
            if content:
                if provider != self.provider:
                    self.provider_active = provider
                    self.fallback_reason = f"{self.provider}_failed:{errors[0]}" if errors else f"fallback_provider:{provider}"
                    self.last_error = ""
                return content
            errors.append(self.last_error or self._provider_error(provider))
        deduped_errors = list(dict.fromkeys(errors))
        self.last_error = "; ".join(deduped_errors)
        return None

    def _chat_once(self, provider: str, system: str, user: str) -> str | None:
        try:
            if provider == "ollama":
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "options": {"temperature": 0.0},
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return (data.get("message") or {}).get("content")
            if provider in {"openai_compatible", "openrouter", "mistral", "groq"}:
                model = self.mistral_model if provider == "mistral" else self.openrouter_model if provider == "openrouter" else self.model
                temperature = self.mistral_temperature if provider == "mistral" else 0
                max_tokens = self.mistral_max_tokens if provider == "mistral" else None
                payload: Dict[str, Any] = {
                    "model": model,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                if provider != "mistral":
                    payload["response_format"] = {"type": "json_object"}
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                resp = requests.post(
                    self._chat_completions_url(provider),
                    headers=self._openai_headers(provider),
                    json=payload,
                    timeout=self.mistral_timeout if provider == "mistral" else self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    self.last_error = "llm_empty_choices"
                    return None
                return (choices[0].get("message") or {}).get("content")
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", "unknown")
            text = getattr(exc.response, "text", "") or ""
            self.last_error = _sanitize_error(f"http_{status}:{text[:160]}", self.api_key, self.openrouter_api_key, self.mistral_api_key)
            return None
        except requests.Timeout:
            self.last_error = "timeout"
            return None
        except Exception as exc:
            self.last_error = _sanitize_error(f"{type(exc).__name__}:{str(exc)[:160]}", self.api_key, self.openrouter_api_key, self.mistral_api_key)
            return None
        return None

    def test_connection(self) -> Dict[str, Any]:
        """Run a minimal provider call for diagnostics."""

        status = self.status()
        if not status.get("ready"):
            return {
                **status,
                "success": False,
                "latency_ms": None,
                "short_response": None,
                "response_preview": None,
                "error": status.get("last_error"),
            }
        started = time.perf_counter()
        content = self._chat(
            "Return JSON only with key 'answer'.",
            json.dumps({"task": "Reply with OK in Russian using JSON."}, ensure_ascii=False),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        obj = self._parse_json_object(content or "")
        short_response = str((obj or {}).get("answer") or content or "").strip()[:200]
        return {
            **self.status(),
            "success": bool(short_response and not self.last_error),
            "latency_ms": latency_ms,
            "short_response": short_response,
            "response_preview": short_response,
            "error": self.last_error,
        }

    def synthesize_answer(
        self,
        *,
        question: str,
        intent: str,
        answer_draft: str,
        facts: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        gaps: List[Dict[str, Any]],
    ) -> str | None:
        """Ask an optional LLM to polish the final answer using only grounded facts."""
        if not self.enabled:
            return None
        system = (
            "You are a strict technical-document assistant. "
            "Use only the provided JSON facts and sources. "
            "Do not invent details. Return JSON only with key 'answer'. "
            "Answer in Russian in normal human prose, not as a raw list of triples. "
            "If exact facts are missing, say that directly, then mention only the nearest partial evidence if it is present. "
            "Never answer with facts about a different material/object/process/property as if they answered the question. "
            "Mention uncertainty and missing data explicitly."
        )
        payload = {
            "question": question,
            "intent": intent,
            "rule_based_answer_draft": answer_draft,
            "facts": facts[:40],
            "sources": sources[:12],
            "gaps": gaps[:12],
        }
        content = self._chat(system, json.dumps(payload, ensure_ascii=False))
        obj = self._parse_json_object(content or "")
        if not obj:
            if not self.last_error:
                self.last_error = "llm_answer_json_parse_failed"
            return None
        answer = obj.get("answer")
        return str(answer).strip() if answer else None

    def repair_grounded_answer(self, repair_request: Dict[str, Any]) -> str | None:
        """Ask the LLM for one grounded repair of an unsafe polished answer."""

        if not self.enabled:
            return None
        system = (
            "You repair a Russian technical answer after a grounding guard rejected it. "
            "Return JSON only with key 'answer'. "
            "Use only the allowed_* fields from the JSON payload. "
            "Do not add new numbers, units, materials, regimes, properties, sources or conclusions. "
            "If the allowed context is insufficient, say that grounded data is missing. "
            "Do not include raw ids, doc ids, chunk ids, tracebacks or technical graph labels."
        )
        payload = {
            "task": "Repair unsafe LLM-polished answer using only allowed grounded claims.",
            "repair_request": repair_request,
        }
        content = self._chat(system, json.dumps(payload, ensure_ascii=False))
        obj = self._parse_json_object(content or "")
        if not obj:
            if not self.last_error:
                self.last_error = "llm_grounding_repair_json_parse_failed"
            return None
        answer = obj.get("answer")
        return str(answer).strip() if answer else None

    def rewrite_question_for_retrieval(self, question: str) -> Dict[str, Any] | None:
        """Optional LLM query rewrite for messy user questions.

        This does not answer the question and does not create facts.  It only
        produces a broader search query, preserving explicit named entities
        like material grades, article numbers, DN/PN codes and standards.
        """
        if not self.enabled:
            return None
        system = (
            "You rewrite Russian/English technical-document search questions. "
            "Return JSON only with keys: search_query, normalized_question, notes. "
            "Preserve all explicit identifiers exactly: material grades, standards, DN/PN, article numbers. "
            "Add likely synonyms, abbreviations and spelling variants. Do not answer. Do not invent facts."
        )
        payload = {"question": question}
        content = self._chat(system, json.dumps(payload, ensure_ascii=False))
        obj = self._parse_json_object(content or "")
        if not obj:
            if not self.last_error:
                self.last_error = "llm_rewrite_json_parse_failed"
            return None
        search_query = str(obj.get("search_query") or "").strip()
        if not search_query:
            return None
        return obj

    def extract_structured_facts(self, chunk_text: str) -> Dict[str, Any] | None:
        """Optional JSON extraction for future GPU/LLM demos.

        The API currently keeps deterministic extraction as source of truth;
        this method is exposed for experiments and can be wired in without
        changing the public contract.
        """
        if not self.enabled:
            return None
        system = (
            "Extract structured technical facts from the chunk. "
            "Return JSON only with arrays: technical_objects, parts, article_numbers, "
            "materials, standards, parameters, requirements, measurements, gaps. "
            "Do not infer facts not present in the text."
        )
        content = self._chat(system, chunk_text[:6000])
        return self._parse_json_object(content or "")


def _env_bool(name: str, default: bool | None) -> bool | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_or_setting(env_name: str, setting_name: str, default: str = "") -> str:
    if env_name in os.environ:
        return os.environ.get(env_name, "") or ""
    return str(getattr(settings, setting_name, default) or "")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _sanitize_error(message: str, *secrets: str) -> str:
    safe = message
    for secret in secrets:
        if secret:
            safe = safe.replace(secret, "[redacted]")
    return safe


def _is_placeholder_model(model: str) -> bool:
    lowered = (model or "").lower()
    return any(token in lowered for token in ["your_openrouter_model_slug_here", "replace-with", "<openrouter-model-slug>"])

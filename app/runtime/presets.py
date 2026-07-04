"""Curated runtime presets exposed to the demo UI and API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RuntimePresetId(str, Enum):
    """Stable identifiers for user-facing runtime modes."""

    EXPERT_MAX = "expert_max"
    STRICT_AUDIT = "strict_audit"
    OFFLINE_RELIABLE = "offline_reliable"


class RuntimePreset(BaseModel):
    """User-facing operating mode with explicit effective configuration."""

    preset_id: RuntimePresetId
    title: str
    description: str
    kg_backend: str
    parser_backend: str
    extraction_mode: str
    answer_synthesis_mode: str
    strict_audit_mode: bool = False
    enable_ocr: bool = False
    quality_notes: list[str] = Field(default_factory=list)
    expected_dependencies: list[str] = Field(default_factory=list)


_PRESETS: dict[RuntimePresetId, RuntimePreset] = {
    RuntimePresetId.EXPERT_MAX: RuntimePreset(
        preset_id=RuntimePresetId.EXPERT_MAX,
        title="Лучший ответ",
        description=(
            "Человекоориентированный ответ по validated graph facts, с кратким выводом, "
            "ключевыми фактами, evidence и графом."
        ),
        kg_backend="auto",
        parser_backend="auto",
        extraction_mode="hybrid",
        answer_synthesis_mode="hybrid",
        strict_audit_mode=False,
        enable_ocr=False,
        quality_notes=[
            "LLM polish is optional; template answer is used if LLM is unavailable.",
            "Neo4j is used when available; validated fallback graph is used otherwise.",
        ],
        expected_dependencies=["Neo4j optional", "LLM optional", "Docling optional"],
    ),
    RuntimePresetId.STRICT_AUDIT: RuntimePreset(
        preset_id=RuntimePresetId.STRICT_AUDIT,
        title="Строгая проверка",
        description=(
            "Сухой audit-формат: explicit status, graph path и strict no-hallucination."
        ),
        kg_backend="auto",
        parser_backend="auto",
        extraction_mode="deterministic",
        answer_synthesis_mode="template",
        strict_audit_mode=True,
        enable_ocr=False,
        quality_notes=[
            "Partial matches are reported separately.",
            "LLM answer polishing is disabled for auditability.",
        ],
        expected_dependencies=["Neo4j optional"],
    ),
    RuntimePresetId.OFFLINE_RELIABLE: RuntimePreset(
        preset_id=RuntimePresetId.OFFLINE_RELIABLE,
        title="Офлайн-режим",
        description=(
            "Локальный fallback без Neo4j, Qdrant, LLM, Docling и OCR, но с тем же "
            "validated extraction contract."
        ),
        kg_backend="fallback",
        parser_backend="fallback",
        extraction_mode="deterministic",
        answer_synthesis_mode="template",
        strict_audit_mode=False,
        enable_ocr=False,
        quality_notes=[
            "Uses validated fallback facts from ExtractionPipeline.",
            "External services are not required.",
        ],
        expected_dependencies=[],
    ),
}


def list_runtime_presets() -> list[RuntimePreset]:
    """Return presets in the order intended for the UI."""

    return [
        _PRESETS[RuntimePresetId.EXPERT_MAX],
        _PRESETS[RuntimePresetId.STRICT_AUDIT],
        _PRESETS[RuntimePresetId.OFFLINE_RELIABLE],
    ]


def get_runtime_preset(preset_id: RuntimePresetId | str | None) -> RuntimePreset:
    """Resolve a preset id, defaulting to expert maximum."""

    if preset_id is None or str(preset_id).strip() == "":
        return _PRESETS[RuntimePresetId.EXPERT_MAX]
    if isinstance(preset_id, RuntimePresetId):
        return _PRESETS[preset_id]
    try:
        resolved = RuntimePresetId(str(preset_id))
    except ValueError as exc:
        raise ValueError(f"Unknown runtime preset: {preset_id!r}") from exc
    return _PRESETS[resolved]


def preset_diagnostics(
    preset: RuntimePreset,
    *,
    active_backend: str | None = None,
    neo4j_available: bool | None = None,
    input_source: str | None = None,
    query_params_ignored: bool = False,
) -> dict[str, Any]:
    """Build a compact diagnostic payload for responses."""

    warnings: list[str] = []
    if preset.preset_id == RuntimePresetId.EXPERT_MAX and active_backend == "fallback":
        warnings.append("Neo4j unavailable, using validated fallback graph.")
    if preset.preset_id == RuntimePresetId.OFFLINE_RELIABLE and active_backend != "fallback":
        warnings.append("Offline reliable preset requested fallback backend, but active backend differs.")
    return {
        "preset_id": preset.preset_id.value,
        "preset_title": preset.title,
        "effective_runtime_mode": {
            "kg_backend": preset.kg_backend,
            "parser_backend": preset.parser_backend,
            "extraction_mode": preset.extraction_mode,
            "answer_synthesis_mode": preset.answer_synthesis_mode,
            "strict_audit_mode": preset.strict_audit_mode,
            "enable_ocr": preset.enable_ocr,
        },
        "input_source": input_source,
        "query_params_ignored": query_params_ignored,
        "neo4j_available": neo4j_available,
        "warnings": warnings,
    }

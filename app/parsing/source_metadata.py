"""Deterministic source metadata extraction for corpus readiness and filtering.

This module classifies document-level metadata only. It does not create domain
facts and does not use an LLM.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


YEAR_RE = re.compile(r"\b(?:19[5-9]\d|20[0-3]\d)\b")

GEOGRAPHY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Россия", re.compile(r"\b(?:росси[ияйюе]|россий\w+|рф|russia|russian federation|russian)\b|гост|ростех", re.IGNORECASE)),
    ("зарубежная практика", re.compile(r"зарубеж\w+|за\s+рубежом|foreign|international|worldwide|global|world\s+practice", re.IGNORECASE)),
    ("мировая практика", re.compile(r"миров\w+\s+практик\w+|world\s+practice|global\s+practice|international\s+practice", re.IGNORECASE)),
    ("Китай", re.compile(r"\b(?:кита[йяею]|china|chinese)\b", re.IGNORECASE)),
    ("США", re.compile(r"\b(?:сша|usa|u\.s\.a\.|united states)\b", re.IGNORECASE)),
    ("Канада", re.compile(r"\b(?:канад[аыеу]|canada|canadian)\b", re.IGNORECASE)),
    ("Европа", re.compile(r"\b(?:европ[аыеу]|europe|european|eu)\b", re.IGNORECASE)),
]

SOURCE_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("patent", re.compile(r"\b(?:патент|patent|ru\s*\d{6,}|us\s*\d{6,}|wo\s*\d{4})\b", re.IGNORECASE)),
    ("standard", re.compile(r"\b(?:гост|ост|iso|iec|standard|норматив\w+|регламент)\b", re.IGNORECASE)),
    ("dissertation", re.compile(r"\b(?:диссертац\w+|dissertation|thesis)\b", re.IGNORECASE)),
    ("internal_report", re.compile(r"\b(?:внутренн\w+\s+отчет|техническ\w+\s+отчет|отчет\s+нии|internal\s+report|technical\s+report|protocol|протокол)\b", re.IGNORECASE)),
    ("publication", re.compile(r"\b(?:стать[яьи]|публикац\w+|journal|article|review|литературн\w+\s+обзор|обзор)\b", re.IGNORECASE)),
    ("presentation", re.compile(r"\b(?:презентац\w+|conference|deck|slides|доклад)\b", re.IGNORECASE)),
    ("catalog", re.compile(r"\b(?:каталог|register|registry|справочник|catalog|реестр)\b", re.IGNORECASE)),
]

EXTENSION_SOURCE_TYPES = {
    ".ppt": "presentation",
    ".pptx": "presentation",
    ".csv": "catalog",
    ".xlsx": "catalog",
    ".xls": "catalog",
    ".html": "web_page",
    ".htm": "web_page",
}


def infer_source_metadata(
    *,
    source_name: str | None = None,
    source_url: str | None = None,
    source_type: str | None = None,
    parser_name: str | None = None,
    text: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return safe document-level metadata used by search/report diagnostics."""

    source_name = str(source_name or "").strip()
    source_url = str(source_url or "").strip()
    diagnostics = diagnostics or {}
    combined = _combined_text(source_name, source_url, text, diagnostics)
    extension = _extension(source_name, source_url)
    years = _extract_years(combined)
    geographies = _extract_geographies(combined)
    detected_type, type_basis = _detect_source_type(combined, extension, source_type)
    reliability_level, reliability_basis = _reliability_for(detected_type, combined)
    return {
        "source_metadata": {
            "source_type_detected": detected_type,
            "source_type_declared": source_type or None,
            "source_extension": extension or None,
            "publication_year": max(years) if years else None,
            "years_detected": years,
            "geographies": geographies,
            "practice_scope": _practice_scope(geographies),
            "reliability_level": reliability_level,
            "reliability_basis": reliability_basis,
            "type_basis": type_basis,
            "parser_name": parser_name or None,
        }
    }


def _combined_text(source_name: str, source_url: str, text: str | None, diagnostics: dict[str, Any]) -> str:
    parts = [
        source_name,
        source_url,
        str(diagnostics.get("title") or ""),
        str(diagnostics.get("source_title") or ""),
        str(text or "")[:25_000],
    ]
    return "\n".join(part for part in parts if part)


def _extension(source_name: str, source_url: str) -> str:
    candidate = source_name or urlparse(source_url).path
    return Path(candidate).suffix.lower()


def _extract_years(text: str) -> list[int]:
    years = sorted({int(match.group(0)) for match in YEAR_RE.finditer(text or "")})
    return [year for year in years if 1950 <= year <= 2035]


def _extract_geographies(text: str) -> list[str]:
    result: list[str] = []
    for label, pattern in GEOGRAPHY_PATTERNS:
        if pattern.search(text or "") and label not in result:
            result.append(label)
    return result


def _detect_source_type(text: str, extension: str, declared: str | None) -> tuple[str, str]:
    for label, pattern in SOURCE_TYPE_PATTERNS:
        if pattern.search(text or ""):
            return label, "text_pattern"
    if extension in EXTENSION_SOURCE_TYPES:
        return EXTENSION_SOURCE_TYPES[extension], "extension"
    if declared == "url":
        return "web_page", "declared_url"
    return "unknown", "not_detected"


def _reliability_for(source_type: str, text: str) -> tuple[str, str]:
    lowered = (text or "").lower()
    if source_type == "standard":
        return "high", "normative_source"
    if source_type == "publication" and any(marker in lowered for marker in ["peer reviewed", "journal", "doi", "реценз"]):
        return "high", "publication_peer_review_hint"
    if source_type in {"publication", "patent", "internal_report"}:
        return "medium", f"{source_type}_source"
    if source_type in {"catalog", "web_page", "presentation", "dissertation"}:
        return "medium", f"{source_type}_source"
    return "unknown", "insufficient_metadata"


def _practice_scope(geographies: list[str]) -> str | None:
    labels = set(geographies)
    if "Россия" in labels and labels.intersection({"зарубежная практика", "мировая практика", "Китай", "США", "Канада", "Европа"}):
        return "domestic_and_foreign"
    if "Россия" in labels:
        return "domestic"
    if labels.intersection({"зарубежная практика", "мировая практика", "Китай", "США", "Канада", "Европа"}):
        return "foreign_or_global"
    return None

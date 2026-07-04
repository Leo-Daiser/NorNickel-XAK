"""Rule-based document profile used to route deterministic extraction safely."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..models.schemas import Chunk


ALL_DOC_TYPES = {
    "experiment_report",
    "materials_article",
    "review_article",
    "market_capacity_reference",
    "presentation",
    "patent",
    "standard_or_normative",
    "directory_or_catalog",
    "unknown",
}


@dataclass(frozen=True)
class DocumentProfile:
    document_id: str
    source_name: str
    detected_type: str = "unknown"
    language: str | None = None
    source_family: str | None = None
    quality_flags: set[str] = field(default_factory=set)
    table_heavy: bool = False
    ocr_noise_score: float = 0.0
    text_density: float = 0.0


DOC_TYPE_FEATURES: dict[str, dict[str, list[str]]] = {
    "market_capacity_reference": {
        "positive_markers": [
            "production capacity",
            "capacity data",
            "mine capacity",
            "plant capacity",
            "directory of",
            "facility-by-facility",
            "summary country basis",
            "copper mines and plants",
            "provisional agenda",
        ],
        "negative_markers": [
            "experimental procedure",
            "sample preparation",
            "tensile test",
            "annealing temperature",
            "результаты испытаний",
        ],
    },
    "experiment_report": {
        "positive_markers": [
            "эксперимент",
            "методика эксперимента",
            "образец",
            "условия опыта",
            "результаты испытаний",
            "experimental procedure",
            "sample preparation",
            "test conditions",
            "showed tensile strength",
            "resulting in ultimate tensile strength",
        ],
        "negative_markers": [],
    },
    "review_article": {
        "positive_markers": [
            "обзор",
            "literature review",
            "world practice",
            "мировой практике",
            "патентный обзор",
        ],
        "negative_markers": [],
    },
    "standard_or_normative": {
        "positive_markers": ["гост", "iso ", "astm ", "standard", "норматив"],
        "negative_markers": [],
    },
    "patent": {
        "positive_markers": ["патент", "patent", "claims", "изобретение"],
        "negative_markers": [],
    },
    "directory_or_catalog": {
        "positive_markers": ["directory", "catalog", "каталог", "справочник", "register"],
        "negative_markers": [],
    },
}


def profile_chunk(chunk: Chunk) -> DocumentProfile:
    """Build a conservative profile from source metadata and local chunk text."""
    text = chunk.text or ""
    metadata: dict[str, Any] = chunk.metadata or {}
    source_name = str(metadata.get("source_name") or metadata.get("filename") or chunk.doc_id)
    sample = f"{source_name}\n{text[:6000]}"
    norm = _norm(sample)
    detected_type = classify_document_type(norm)
    quality_flags = set(str(item) for item in metadata.get("quality_flags", []) if item)
    if _looks_dirty_ocr(text):
        quality_flags.add("dirty_ocr_text")
    table_heavy = metadata.get("chunk_kind") == "table_row" or text.count(" | ") >= 4
    return DocumentProfile(
        document_id=chunk.doc_id,
        source_name=source_name,
        detected_type=detected_type,
        language=_detect_language(text),
        source_family=str(metadata.get("source_type") or metadata.get("parser") or "") or None,
        quality_flags=quality_flags,
        table_heavy=table_heavy,
        ocr_noise_score=_ocr_noise_score(text),
        text_density=_text_density(text),
    )


def classify_document_type(normalized_text: str) -> str:
    scores: dict[str, int] = {}
    for doc_type, features in DOC_TYPE_FEATURES.items():
        positives = sum(1 for marker in features.get("positive_markers", []) if marker in normalized_text)
        negatives = sum(1 for marker in features.get("negative_markers", []) if marker in normalized_text)
        if positives:
            scores[doc_type] = positives * 2 - negatives * 3
    if not scores:
        return "unknown"
    doc_type, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return "unknown"
    if doc_type == "directory_or_catalog" and scores.get("market_capacity_reference", 0) >= score:
        return "market_capacity_reference"
    return doc_type


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().replace("ё", "е"))


def _detect_language(text: str) -> str | None:
    cyr = len(re.findall(r"[А-Яа-яЁё]", text or ""))
    lat = len(re.findall(r"[A-Za-z]", text or ""))
    if cyr > lat * 1.5:
        return "ru"
    if lat > cyr * 1.5:
        return "en"
    if cyr or lat:
        return "mixed"
    return None


def _text_density(text: str) -> float:
    if not text:
        return 0.0
    return round(len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]", text)) / max(1, len(text)), 4)


def _ocr_noise_score(text: str) -> float:
    if not text:
        return 0.0
    suspicious = len(re.findall(r"[^\w\s.,;:!?%°/+=<>()\-\u0400-\u04FF]", text, flags=re.UNICODE))
    broken_spaces = len(re.findall(r"\b[А-ЯA-Z]\s+[а-яa-z]\b|\b[МM]\s*[РP]\s*[аa]\b", text))
    return round(min(1.0, (suspicious + broken_spaces * 3) / max(50, len(text))), 4)


def _looks_dirty_ocr(text: str) -> bool:
    return _ocr_noise_score(text) >= 0.08 or bool(re.search(r"\b[MМ]\s*[PР]\s*[aа]\b|\b[HН]\s*V\b", text or "", re.IGNORECASE))

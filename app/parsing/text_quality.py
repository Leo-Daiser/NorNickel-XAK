"""Text-layer quality and conservative OCR-noise normalization.

The helpers in this module are deliberately resource-light. They do not run OCR
or create facts; they only classify parser output and normalize common
scientific OCR artefacts before deterministic extraction sees the text.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


SUPPORTED_FILE_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".html", ".htm", ".txt", ".md"}


def normalize_dirty_scientific_text(text: str) -> str:
    """Return text with conservative materials-science OCR fixes applied.

    This function intentionally handles recurring notation damage, not
    document-specific strings. Provenance remains on the original chunk/source;
    only the chunk text used for deterministic extraction is cleaned.
    """

    if not text:
        return text
    result = str(text)
    result = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", result)
    result = re.sub(r"[ \t]*\n[ \t]*(?=(?:MPa|МПа|ksi|HV|HRC|%|°C|C)\b)", " ", result, flags=re.IGNORECASE)
    result = re.sub(r"\b[MМ]\s*[PР]\s*[aа]\b", "MPa", result, flags=re.IGNORECASE)
    result = re.sub(r"\b[MМ]\s*[PР]\s*[АA]\b", "MPa", result, flags=re.IGNORECASE)
    result = re.sub(r"\bМ\s*Па\b", "MPa", result, flags=re.IGNORECASE)
    result = re.sub(r"\bмпа\b", "MPa", result, flags=re.IGNORECASE)
    result = re.sub(r"\b[HН]\s*V\b", "HV", result, flags=re.IGNORECASE)
    result = re.sub(r"\bВТ\s*[- ]?\s*6\b", "ВТ6", result, flags=re.IGNORECASE)
    result = re.sub(r"\bBT\s*[- ]?\s*6\b", "VT6", result, flags=re.IGNORECASE)
    result = re.sub(r"\b7075\s*[–—-]?\s*[ТT]\s*6\b", "7075-T6", result, flags=re.IGNORECASE)
    result = re.sub(r"\bTi\s*[- ]?\s*6\s*Al\s*[- ]?\s*4\s*V\b", "Ti-6Al-4V", result, flags=re.IGNORECASE)
    result = re.sub(r"[ \t]{2,}", " ", result)
    return result


def dirty_ocr_signals(text: str) -> list[str]:
    """Return quality warnings for known OCR-like artefacts."""

    raw = str(text or "")
    signals: list[str] = []
    patterns = {
        "split_mpa_unit": r"\b[MМ]\s+[PР]\s*[aаАA]\b|\bМ\s*Па\b",
        "mixed_cyrillic_latin_unit": r"\b(?:MРa|MPа|МPa|МРа|НV)\b",
        "split_vt6": r"\b(?:ВТ|BT)\s*[- ]\s*6\b",
        "split_7075_t6": r"\b7075\s*[–—-]?\s*[ТT]\s*6\b",
        "hyphenated_linebreaks": r"(?<=\w)-\s*\n\s*(?=\w)",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, raw, flags=re.IGNORECASE):
            signals.append(name)
    if _short_line_ratio(raw) > 0.35 and len(raw) > 400:
        signals.append("many_short_ocr_lines")
    if _replacement_char_count(raw) > 0:
        signals.append("replacement_characters")
    return signals


def text_quality_metrics(text: str, *, pages_estimated: int | None = None) -> dict[str, Any]:
    """Return text-layer metrics independent of downstream extraction."""

    raw = str(text or "")
    chars = len(raw.strip())
    words = len(re.findall(r"\w+", raw, flags=re.UNICODE))
    pages = max(1, int(pages_estimated or 1))
    chars_per_page = chars / pages
    density = text_density_label(chars, pages_estimated=pages)
    signals = dirty_ocr_signals(raw)
    return {
        "text_chars": chars,
        "word_count": words,
        "pages_estimated": pages,
        "chars_per_page": chars_per_page,
        "text_density": density,
        "dirty_ocr_signals": signals,
        "dirty_ocr_score": len(signals),
        "line_count": len(raw.splitlines()),
        "short_line_ratio": _short_line_ratio(raw),
        "replacement_characters": _replacement_char_count(raw),
        "character_mix": _character_mix(raw),
    }


def text_density_label(text_chars: int, *, pages_estimated: int | None = None) -> str:
    pages = max(1, int(pages_estimated or 1))
    per_page = float(text_chars) / pages
    if text_chars <= 0:
        return "empty"
    if per_page < 80:
        return "very_low"
    if per_page < 400:
        return "low"
    if per_page < 1500:
        return "medium"
    return "high"


def _short_line_ratio(text: str) -> float:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return 0.0
    short = sum(1 for line in lines if len(line) <= 24)
    return short / len(lines)


def _replacement_char_count(text: str) -> int:
    return str(text or "").count("\ufffd") + str(text or "").count("?")


def _character_mix(text: str) -> dict[str, int]:
    counts = Counter()
    for char in str(text or ""):
        code = ord(char)
        if "A" <= char <= "Z" or "a" <= char <= "z":
            counts["latin"] += 1
        elif 0x0400 <= code <= 0x04FF:
            counts["cyrillic"] += 1
        elif char.isdigit():
            counts["digits"] += 1
    return dict(counts)

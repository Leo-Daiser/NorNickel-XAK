"""Canonical normalization for materials, regimes and properties."""

from __future__ import annotations

import re

from .aliases import MATERIAL_ALIASES, PROPERTY_ALIASES, REGIME_ALIASES


def normalize_text(value: str | None) -> str:
    """Normalize text for case-insensitive Russian/English matching."""
    text = str(value or "").strip().lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text


def _compact(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", normalize_text(value))


def canonical_from_alias(value: str | None, aliases: dict[str, str]) -> str:
    """Return a canonical value if an alias occurs in the provided text."""
    text = normalize_text(value)
    compact_text = _compact(text)
    if not text:
        return ""
    for alias, canonical in sorted(aliases.items(), key=lambda item: len(normalize_text(item[0])), reverse=True):
        alias_norm = normalize_text(alias)
        if alias_norm in text or _compact(alias_norm) in compact_text:
            return canonical
    return str(value or "").strip()


def canonical_material(value: str | None) -> str:
    """Canonical material name with conservative alias mapping."""
    return canonical_from_alias(value, MATERIAL_ALIASES)


def canonical_regime(value: str | None) -> str:
    """Canonical process regime name with conservative alias mapping."""
    return canonical_from_alias(value, REGIME_ALIASES)


def canonical_property(value: str | None) -> str:
    """Canonical property name with conservative alias mapping."""
    return canonical_from_alias(value, PROPERTY_ALIASES)


def matches_alias(value: str | None, requested: str | None, aliases: dict[str, str]) -> bool:
    """Compare two values after alias canonicalization."""
    left = normalize_text(canonical_from_alias(value, aliases))
    right = normalize_text(canonical_from_alias(requested, aliases))
    return bool(left and right and left == right)


def material_matches(value: str | None, requested: str | None) -> bool:
    return matches_alias(value, requested, MATERIAL_ALIASES)


def regime_matches(value: str | None, requested: str | None) -> bool:
    return matches_alias(value, requested, REGIME_ALIASES)


def property_matches(value: str | None, requested: str | None) -> bool:
    return matches_alias(value, requested, PROPERTY_ALIASES)

"""Canonical resolver for extracted raw names and units."""

from __future__ import annotations

import re

from ..domain.normalization import canonical_material, canonical_property, canonical_regime
from ..domain.reference_loader import aliases_for, resolve_reference


UNIT_ALIASES = {
    "mpa": "MPa",
    "мпа": "MPa",
    "мПа".lower(): "MPa",
    "gpa": "GPa",
    "гпа": "GPa",
    "ksi": "ksi",
    "hv": "HV",
    "hrc": "HRC",
    "%": "%",
    "°c": "C",
    "°с": "C",
    "c": "C",
    "с": "C",
    "celsius": "C",
    "сelsius": "C",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "ч": "h",
    "ч.": "h",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "мин": "min",
    "мин.": "min",
    "mg/l": "mg/L",
    "mg/dm3": "mg/L",
    "mg/dm³": "mg/L",
    "мг/л": "mg/L",
    "мг/дм3": "mg/L",
    "мг/дм³": "mg/L",
    "g/l": "g/L",
    "г/л": "g/L",
    "m/s": "m/s",
    "м/с": "m/s",
    "m3/h": "m3/h",
    "m³/h": "m3/h",
    "м3/ч": "m3/h",
    "м³/ч": "m3/h",
    "t/day": "t/day",
    "t/d": "t/day",
    "т/сут": "t/day",
    "т/сутки": "t/day",
    "t/y": "t/y",
    "t/year": "t/y",
    "tpa": "t/y",
    "т/год": "t/y",
    "т/г": "t/y",
    "kt/y": "kt/y",
    "kt/year": "kt/y",
    "ktpa": "kt/y",
    "кт/год": "kt/y",
    "тыс.т/год": "kt/y",
    "тыст/год": "kt/y",
    "mt/y": "Mt/y",
    "mt/year": "Mt/y",
    "mtpa": "Mt/y",
    "мт/год": "Mt/y",
    "млнт/год": "Mt/y",
    "млн.т/год": "Mt/y",
    "ppm": "ppm",
    "г/т": "г/т",
    "g/t": "g/t",
    "mg/kg": "mg/kg",
    "usd/t": "USD/t",
    "$/t": "USD/t",
    "eur/t": "EUR/t",
    "€/t": "EUR/t",
    "rub/t": "RUB/t",
    "руб/т": "RUB/t",
    "руб./т": "RUB/t",
    "usd/m3": "USD/m3",
    "rub/m3": "RUB/m3",
    "руб/м3": "RUB/m3",
    "руб/м³": "RUB/m3",
    "млнруб": "mln RUB",
    "млнруб/год": "mln RUB/year",
    "млнrub": "mln RUB",
    "млнrub/year": "mln RUB/year",
}
UNIT_ALIASES = {**aliases_for("units"), **UNIT_ALIASES}

EXPERIMENT_ID_RE = re.compile(r"\b(?:E\d+|EXP-[A-ZА-Я0-9_.-]+)\b", re.IGNORECASE)


def clean_raw(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" .;|\"'«»"))


def resolve_material(raw: str | None) -> str:
    value = clean_raw(raw)
    if EXPERIMENT_ID_RE.fullmatch(value):
        return ""
    canonical = canonical_material(value)
    if _should_preserve_specific_material_phrase(value, canonical):
        return value
    return canonical


def resolve_regime(raw: str | None) -> str:
    return canonical_regime(clean_raw(raw))


def resolve_property(raw: str | None) -> str:
    return canonical_property(clean_raw(raw))


def resolve_equipment(raw: str | None) -> str:
    value = clean_raw(raw)
    return resolve_reference("equipment", value) if value else ""


def resolve_unit(raw: str | None) -> str | None:
    value = clean_raw(raw).replace("°", "°").lower()
    value = value.replace(" ", "")
    return UNIT_ALIASES.get(value, clean_raw(raw) or None)


def _should_preserve_specific_material_phrase(raw: str, canonical: str) -> bool:
    """Keep descriptive material/media phrases instead of collapsing to a broad class."""
    raw_norm = clean_raw(raw).lower().replace("ё", "е")
    canonical_norm = clean_raw(canonical).lower().replace("ё", "е")
    if not raw_norm or not canonical_norm or raw_norm == canonical_norm:
        return False
    broad_media = {
        "концентрат",
        "руда",
        "шлак",
        "штейн",
        "раствор",
        "электролит",
        "вода",
        "шахтные воды",
    }
    if canonical_norm not in broad_media:
        return False
    return canonical_norm in raw_norm and len(raw_norm.split()) > 1

"""Small unit conversion helpers for user-facing comparison answers."""

from __future__ import annotations

from typing import Any


KSI_TO_MPA = 6.894757


def normalize_unit_label(unit: str | None) -> str:
    """Return a canonical display label for common technical units."""

    raw = str(unit or "").strip()
    key = raw.lower().replace("м", "m").replace("а", "a").replace("с", "c")
    compact = key.replace(" ", "").replace("³", "3")
    if key in {"mpa", "mпa", "мpa"} or raw in {"МПа", "мПа"}:
        return "MPa"
    if key == "ksi":
        return "ksi"
    if compact in {"mg/l", "mg/dm3", "mг/l", "mг/dm3"} or raw.lower().replace(" ", "") in {"мг/л", "мг/дм3", "мг/дм³"}:
        return "mg/L"
    if compact in {"g/l"} or raw.lower().replace(" ", "") in {"г/л"}:
        return "g/L"
    if compact in {"m/s"} or raw.lower().replace(" ", "") in {"м/с"}:
        return "m/s"
    if compact in {"m3/h"} or raw.lower().replace(" ", "") in {"м3/ч", "м³/ч"}:
        return "m3/h"
    if compact in {"t/day", "t/d"} or raw.lower().replace(" ", "") in {"т/сут"}:
        return "t/day"
    if compact in {"t/y", "t/year", "tpa", "tonnes/year", "tons/year"} or raw.lower().replace(" ", "") in {"т/год", "т/г"}:
        return "t/y"
    if compact in {"kt/y", "kt/year", "ktpa"} or raw.lower().replace(" ", "") in {"кт/год", "тыс.т/год", "тыст/год"}:
        return "kt/y"
    if compact in {"mt/y", "mt/year", "mtpa"} or raw.lower().replace(" ", "") in {"мт/год", "млнт/год", "млн.т/год"}:
        return "Mt/y"
    if compact in {"°c", "c"} or raw.lower().replace(" ", "") in {"°с", "с"}:
        return "C"
    if compact == "%":
        return "%"
    if compact == "ppm":
        return "ppm"
    if compact in {"usd/t", "$/t"}:
        return "USD/t"
    if compact in {"eur/t", "€/t"}:
        return "EUR/t"
    if compact in {"rub/t"} or raw.lower().replace(" ", "") in {"руб/т", "руб./т"}:
        return "RUB/t"
    if compact in {"usd/m3"}:
        return "USD/m3"
    if compact in {"rub/m3"} or raw.lower().replace(" ", "") in {"руб/м3", "руб/м³"}:
        return "RUB/m3"
    if raw.lower().replace(" ", "") in {"млнруб"}:
        return "mln RUB"
    if raw.lower().replace(" ", "") in {"млнруб/год"}:
        return "mln RUB/year"
    return raw


def normalize_strength_to_mpa(value: Any, unit: str | None) -> tuple[float | None, str | None]:
    """Convert a strength value to MPa when supported.

    Returns `(converted_value, note)`. `note` is populated only when the
    original value was converted from another unit.
    """

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, None
    normalized_unit = normalize_unit_label(unit)
    if normalized_unit == "MPa":
        return numeric, None
    if normalized_unit == "ksi":
        converted = numeric * KSI_TO_MPA
        return converted, f"{numeric:g} ksi ≈ {converted:.0f} MPa"
    return None, None

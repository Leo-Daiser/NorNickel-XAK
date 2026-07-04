from __future__ import annotations

from app.domain.unit_normalization import normalize_strength_to_mpa


def test_ksi_converts_to_mpa() -> None:
    converted, note = normalize_strength_to_mpa(77, "ksi")
    assert converted is not None
    assert round(converted) == 531
    assert note == "77 ksi ≈ 531 MPa"


def test_mpa_stays_mpa_without_conversion_note() -> None:
    converted, note = normalize_strength_to_mpa(520, "MPa")
    assert converted == 520
    assert note is None


def test_unknown_unit_is_not_silently_converted() -> None:
    converted, note = normalize_strength_to_mpa(10, "HV")
    assert converted is None
    assert note is None

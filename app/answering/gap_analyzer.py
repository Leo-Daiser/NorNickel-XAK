"""Gap filtering and inferred gap creation for strict graph QA."""

from __future__ import annotations

import hashlib

from ..domain.normalization import (
    canonical_material,
    canonical_property,
    canonical_regime,
    material_matches,
    property_matches,
    regime_matches,
)
from ..domain.ontology import DataGap
from ..domain.query_constraints import QueryConstraints


def filter_gaps(gaps: list[DataGap], constraints: QueryConstraints) -> list[DataGap]:
    """Return only gaps that match all explicitly provided constraints."""
    result: list[DataGap] = []
    for gap in gaps:
        if constraints.materials and not _gap_matches_any(gap.material, gap.reason, constraints.materials, material_matches):
            continue
        if constraints.regimes and not _gap_matches_any(gap.regime, gap.reason, constraints.regimes, regime_matches):
            continue
        if constraints.properties and not _gap_matches_any(gap.property, gap.reason, constraints.properties, property_matches):
            continue
        result.append(gap)
    return result


def inferred_gap_for_missing_exact(constraints: QueryConstraints) -> DataGap:
    """Create a non-persistent gap when no exact material+regime+property chain exists."""
    material = constraints.materials[0] if constraints.materials else None
    regime = constraints.regimes[0] if constraints.regimes else None
    property_name = constraints.properties[0] if constraints.properties else None
    reason = "В корпусе не найден эксперимент, связывающий указанный материал, режим и свойство."
    raw = "|".join([material or "", regime or "", property_name or "", reason])
    return DataGap(
        gap_id=f"inferred_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}",
        material=canonical_material(material) or material,
        regime=canonical_regime(regime) or regime,
        property=canonical_property(property_name) or property_name,
        reason=reason,
        evidence=[],
    )


def _gap_matches_any(value: str | None, reason: str, requested_values: list[str], matcher) -> bool:
    if value:
        return any(matcher(value, requested) for requested in requested_values)
    reason_norm = (reason or "").lower().replace("ё", "е")
    return any(str(requested).lower().replace("ё", "е") in reason_norm for requested in requested_values)


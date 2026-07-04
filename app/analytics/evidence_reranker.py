"""Heuristic evidence reranking for constrained analytical questions."""

from __future__ import annotations

from datetime import datetime

from ..domain.aliases import MATERIAL_ALIASES
from ..domain.normalization import normalize_text
from ..domain.query_constraints import QueryConstraints
from ..retrieval.metadata_filters import metadata_match_score
from .query_models import EvidenceItem


class EvidenceReranker:
    """Boost evidence that matches query constraints and penalize conflicts."""

    def rerank(self, items: list[EvidenceItem], constraints: QueryConstraints) -> list[EvidenceItem]:
        reranked: list[EvidenceItem] = []
        now_year = datetime.utcnow().year
        for item in items:
            score = item.score + self._constraint_score(item.quote, constraints)
            metadata_score, _ = metadata_match_score(
                item.metadata or {},
                constraints.geographies or [],
                constraints.time_filters or [],
                now_year=now_year,
            )
            score += min(metadata_score, 0.5)
            if (constraints.geographies or constraints.time_filters) and metadata_score <= 0 and _has_source_metadata(item.metadata):
                score -= 0.30
            if len(item.quote.strip()) < 40:
                score -= 0.20
            reranked.append(item.model_copy(update={"score": max(0.0, min(1.0, score))}))
        reranked.sort(key=lambda value: value.score, reverse=True)
        return reranked

    def _constraint_score(self, quote: str, constraints: QueryConstraints) -> float:
        text = normalize_text(quote)
        score = 0.0
        if constraints.materials and any(normalize_text(item) in text for item in constraints.materials):
            score += 0.35
        if constraints.regimes and any(normalize_text(item) in text for item in constraints.regimes):
            score += 0.25
        if constraints.properties and any(normalize_text(item) in text for item in constraints.properties):
            score += 0.25
        if constraints.equipment and any(normalize_text(item) in text for item in constraints.equipment):
            score += 0.10
        if constraints.laboratories and any(normalize_text(item) in text for item in constraints.laboratories):
            score += 0.10
        if constraints.materials and _has_conflicting_material(text, constraints.materials):
            score -= 0.30
        return score


def _has_conflicting_material(text: str, expected: list[str]) -> bool:
    expected_norm = {normalize_text(item) for item in expected}
    canonical_values = {normalize_text(value) for value in MATERIAL_ALIASES.values()}
    tokens = set(text.replace("-", " ").split())
    for value in canonical_values:
        if not value or value in expected_norm:
            continue
        if len(value) <= 2:
            if value in tokens:
                return True
            continue
        if value in text:
            return True
    return False


def _has_source_metadata(metadata: dict | None) -> bool:
    metadata = metadata or {}
    source_metadata = metadata.get("source_metadata") if isinstance(metadata.get("source_metadata"), dict) else {}
    return any(
        bool(value)
        for value in [
            metadata.get("publication_year"),
            metadata.get("geographies"),
            metadata.get("practice_scope"),
            source_metadata.get("publication_year"),
            source_metadata.get("geographies"),
            source_metadata.get("practice_scope"),
        ]
    )

"""Metadata-aware reranking for geography and publication-time constraints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..domain.query_constraints import QueryConstraints
from ..models.schemas import Chunk


FOREIGN_OR_GLOBAL = {"зарубежная практика", "мировая практика", "Китай", "США", "Канада", "Европа"}


def rerank_chunks_by_source_metadata(
    chunks: list[Chunk],
    constraints: QueryConstraints,
    *,
    now_year: int | None = None,
) -> tuple[list[Chunk], dict[str, Any]]:
    """Rank chunks matching source metadata constraints before other chunks.

    This is deliberately not a hard filter. If source metadata is absent or no
    chunk matches it, the original retrieval order is preserved and diagnostics
    explain the reason.
    """

    requested_geo = list(constraints.geographies or [])
    requested_time = list(constraints.time_filters or [])
    if not chunks or (not requested_geo and not requested_time):
        return chunks, {
            "applied": False,
            "reason": "no_metadata_constraints",
            "requested_geographies": requested_geo,
            "requested_time_filters": requested_time,
            "matched_chunks": 0,
            "total_chunks": len(chunks),
        }

    year = now_year or datetime.utcnow().year
    scored: list[tuple[float, int, Chunk, list[str]]] = []
    for idx, chunk in enumerate(chunks):
        score, reasons = metadata_match_score(chunk.metadata or {}, requested_geo, requested_time, now_year=year)
        scored.append((score, idx, chunk, reasons))

    matched = [item for item in scored if item[0] > 0]
    diagnostics = {
        "applied": True,
        "requested_geographies": requested_geo,
        "requested_time_filters": requested_time,
        "matched_chunks": len(matched),
        "total_chunks": len(chunks),
        "strict_filter": False,
        "reason": "" if matched else "no_chunks_matched_source_metadata",
    }
    if not matched:
        return chunks, diagnostics

    scored.sort(key=lambda item: (-item[0], item[1]))
    diagnostics["top_match_reasons"] = scored[0][3]
    return [chunk for _, _, chunk, _ in scored], diagnostics


def metadata_match_score(
    metadata: dict[str, Any],
    requested_geographies: list[str],
    requested_time_filters: list[dict],
    *,
    now_year: int,
) -> tuple[float, list[str]]:
    source_meta = metadata.get("source_metadata") if isinstance(metadata.get("source_metadata"), dict) else {}
    geographies = set(_as_list(metadata.get("geographies")) + _as_list(source_meta.get("geographies")))
    practice_scope = str(metadata.get("practice_scope") or source_meta.get("practice_scope") or "")
    publication_year = _year(metadata.get("publication_year") or source_meta.get("publication_year"))
    reasons: list[str] = []
    score = 0.0

    if requested_geographies:
        if any(_geo_matches(requested, geographies, practice_scope) for requested in requested_geographies):
            score += 0.55
            reasons.append("geography_match")
    if requested_time_filters:
        if publication_year is not None and any(_time_matches(publication_year, item, now_year=now_year) for item in requested_time_filters):
            score += 0.45
            reasons.append("time_match")
    return score, reasons


def _geo_matches(requested: str, geographies: set[str], practice_scope: str) -> bool:
    if requested == "Россия":
        return "Россия" in geographies or practice_scope in {"domestic", "domestic_and_foreign"}
    if requested in {"зарубежная практика", "мировая практика"}:
        return bool(geographies.intersection(FOREIGN_OR_GLOBAL)) or practice_scope in {"foreign_or_global", "domestic_and_foreign"}
    return requested in geographies


def _time_matches(publication_year: int, item: dict, *, now_year: int) -> bool:
    kind = item.get("type")
    if kind == "relative_years":
        years = int(item.get("years") or 0)
        if years <= 0:
            return False
        return publication_year >= now_year - years + 1
    if kind == "year_range":
        start = _year(item.get("start"))
        end = _year(item.get("end"))
        if start is None or end is None:
            return False
        return start <= publication_year <= end
    return False


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1950 <= year <= 2035 else None

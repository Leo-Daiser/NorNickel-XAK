"""Data-driven domain reference dictionaries.

The files under ``app/domain/reference`` are intentionally simple JSON arrays
with ``.yml`` extensions.  JSON is valid YAML, so the same files can later be
edited as regular YAML if PyYAML is available, while economy_core does not
need an extra dependency.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


@dataclass(frozen=True)
class ReferenceEntry:
    canonical_id: str
    entity_type: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    compatible_fact_types: tuple[str, ...] = field(default_factory=tuple)
    compatible_units: tuple[str, ...] = field(default_factory=tuple)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceRegistry:
    entries_by_domain: dict[str, tuple[ReferenceEntry, ...]]
    alias_maps: dict[str, dict[str, str]]
    entries_by_canonical: dict[str, ReferenceEntry]

    def alias_map(self, domain: str) -> dict[str, str]:
        return self.alias_maps.get(domain, {})

    def resolve(self, domain: str, value: str | None) -> str:
        text = normalize_text(value)
        if not text:
            return ""
        aliases = self.alias_map(domain)
        compact_text = _compact(text)
        for alias, canonical in sorted(aliases.items(), key=lambda item: len(normalize_text(item[0])), reverse=True):
            alias_norm = normalize_text(alias)
            if alias_norm == text or alias_norm in text or _compact(alias_norm) in compact_text:
                return canonical
        return str(value or "").strip()

    def entry(self, canonical_or_alias: str | None, *, domain: str | None = None) -> ReferenceEntry | None:
        if not canonical_or_alias:
            return None
        canonical = self.resolve(domain, canonical_or_alias) if domain else str(canonical_or_alias).strip()
        return self.entries_by_canonical.get(_entry_key(canonical))

    def entity_type(self, canonical_or_alias: str | None, *, domain: str | None = None) -> str | None:
        entry = self.entry(canonical_or_alias, domain=domain)
        return entry.entity_type if entry else None


@lru_cache(maxsize=1)
def get_reference_registry() -> ReferenceRegistry:
    entries_by_domain: dict[str, tuple[ReferenceEntry, ...]] = {}
    alias_maps: dict[str, dict[str, str]] = {}
    entries_by_canonical: dict[str, ReferenceEntry] = {}
    for domain in ("materials", "processes", "equipment", "properties", "units"):
        entries = tuple(_load_domain(domain))
        entries_by_domain[domain] = entries
        alias_map: dict[str, str] = {}
        for entry in entries:
            entries_by_canonical[_entry_key(entry.canonical_id)] = entry
            for alias in (entry.canonical_id, *entry.aliases):
                alias_norm = normalize_text(alias)
                if alias_norm:
                    alias_map[alias_norm] = entry.canonical_id
        alias_maps[domain] = alias_map
    return ReferenceRegistry(
        entries_by_domain=entries_by_domain,
        alias_maps=alias_maps,
        entries_by_canonical=entries_by_canonical,
    )


def aliases_for(domain: str) -> dict[str, str]:
    return dict(get_reference_registry().alias_map(domain))


def resolve_reference(domain: str, value: str | None) -> str:
    return get_reference_registry().resolve(domain, value)


def reference_entity_type(domain: str, value: str | None) -> str | None:
    return get_reference_registry().entity_type(value, domain=domain)


def _load_domain(domain: str) -> list[ReferenceEntry]:
    path = REFERENCE_DIR / f"{domain}.yml"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    data = _parse_reference_text(text)
    result: list[ReferenceEntry] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        canonical_id = str(item.get("canonical_id") or "").strip()
        if not canonical_id:
            continue
        aliases = []
        for key in ("ru", "en", "aliases"):
            values = item.get(key) or []
            if isinstance(values, str):
                values = [values]
            aliases.extend(str(value).strip() for value in values if str(value).strip())
        result.append(
            ReferenceEntry(
                canonical_id=canonical_id,
                entity_type=str(item.get("entity_type") or domain[:-1].title()),
                aliases=tuple(dict.fromkeys(aliases)),
                compatible_fact_types=tuple(str(value) for value in item.get("compatible_fact_types") or []),
                compatible_units=tuple(str(value) for value in item.get("compatible_units") or []),
                raw=dict(item),
            )
        )
    return result


def _parse_reference_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ValueError("Reference file is not JSON and PyYAML is not installed") from exc
        return yaml.safe_load(text)


def _compact(value: str) -> str:
    return "".join(ch for ch in normalize_text(value) if ch.isalnum())


def _entry_key(value: str) -> str:
    return normalize_text(value)


def normalize_text(value: str | None) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text)

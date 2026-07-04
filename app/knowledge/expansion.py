"""Deterministic knowledge expansion reporting.

This module does not create facts with an LLM. It reuses the existing
catalog, deterministic extraction pipeline and canonical fact layer to
explain how the knowledge base changes when documents are added,
updated, activated or deactivated.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from ..domain.fact_normalization import (
    build_conflict_summary,
    canonical_fact_key_from_row,
    dedupe_fact_rows,
    fact_rows_from_experiments,
)
from ..domain.normalization import canonical_material, canonical_property, canonical_regime
from ..extraction.pipeline import ExtractionPipeline
from ..extraction.to_graph_models import bundle_to_data_gaps, bundle_to_experiment_facts
from ..models.schemas import Chunk
from ..storage.catalog import SQLiteCatalog


def build_knowledge_expansion_report(
    catalog: SQLiteCatalog,
    *,
    pipeline: ExtractionPipeline | None = None,
    document_id: str | None = None,
    active_only: bool = True,
) -> dict[str, Any]:
    """Build a deterministic report for the current catalog state."""

    return KnowledgeExpansionEngine(catalog, pipeline=pipeline).build_report(
        document_id=document_id,
        active_only=active_only,
    )


def build_knowledge_delta(before: dict[str, Any], after: dict[str, Any], *, new_document_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Return graph/fact delta between two reports."""

    return KnowledgeExpansionEngine.delta_from_reports(before, after, new_document_ids=new_document_ids)


@dataclass
class KnowledgeExpansionEngine:
    """Report-only engine for KB growth, deduplication and connection discovery."""

    catalog: SQLiteCatalog
    pipeline: ExtractionPipeline | None = None

    def __post_init__(self) -> None:
        if self.pipeline is None:
            self.pipeline = ExtractionPipeline(mode="deterministic", enable_llm=False, audit_enabled=False)

    def build_report(self, *, document_id: str | None = None, active_only: bool = True) -> dict[str, Any]:
        documents = _document_records(self.catalog, document_id=document_id)
        chunks = self.catalog.list_chunks(document_id, active_only=active_only)
        fact_rows, data_gaps, rejected_count, quarantine_count = self._extract_rows(chunks)
        canonical_rows = dedupe_fact_rows(fact_rows)
        canonical_rows = [_row_with_key(row) for row in canonical_rows]
        raw_keys = [canonical_fact_key_from_row(row) for row in fact_rows if isinstance(row, dict)]
        raw_counts = Counter(raw_keys)
        duplicate_groups = {key: count for key, count in raw_counts.items() if key and count > 1}
        conflicts = build_conflict_summary(canonical_rows)
        materials = sorted({canonical_material(row.get("material")) for row in canonical_rows if canonical_material(row.get("material"))})
        regimes = sorted({canonical_regime(row.get("regime")) for row in canonical_rows if canonical_regime(row.get("regime"))})
        properties = sorted({canonical_property(row.get("property")) for row in canonical_rows if canonical_property(row.get("property"))})
        triples = _material_regime_property_triples(canonical_rows)
        opportunities = _comparison_opportunities(canonical_rows)
        gaps = _dedupe_gaps(data_gaps)
        facts_without_evidence = sum(1 for row in canonical_rows if not row.get("evidence"))
        normalized_count = sum(1 for row in canonical_rows if row.get("value_normalized") is not None and row.get("unit_normalized"))
        missing_normalized = sum(
            1
            for row in canonical_rows
            if row.get("value") is not None
            and (row.get("value_normalized") is None or not row.get("unit_normalized"))
        )
        counts = self.catalog.counts()
        filtered_documents = [item for item in documents if not document_id or item.get("doc_id") == document_id]
        report = {
            "status": "ok",
            "scope": {
                "document_id": document_id,
                "active_only": active_only,
                "since_last_ingest_supported": False,
            },
            "documents_count": len(filtered_documents),
            "active_documents_count": counts.get("active_documents", 0),
            "chunks_count": len(chunks),
            "active_chunks_count": counts.get("active_chunks", 0),
            "raw_facts_count": len(fact_rows),
            "canonical_facts_count": len(canonical_rows),
            "duplicate_groups_count": len(duplicate_groups),
            "duplicate_facts_count": max(0, len(fact_rows) - len(canonical_rows)),
            "facts_without_evidence": facts_without_evidence,
            "normalized_measurements_count": normalized_count,
            "measurements_missing_normalized_fields": missing_normalized,
            "conflict_groups_count": len(conflicts),
            "data_gaps_count": len(gaps),
            "rejected_or_low_confidence_candidates": rejected_count,
            "rejected_candidates_count": rejected_count,
            "quarantine_candidates_count": quarantine_count,
            "materials_count": len(materials),
            "regimes_count": len(regimes),
            "properties_count": len(properties),
            "documents": filtered_documents,
            "canonical_facts": canonical_rows,
            "canonical_fact_keys": [row.get("canonical_fact_key") for row in canonical_rows if row.get("canonical_fact_key")],
            "conflict_groups": conflicts,
            "data_gaps": gaps,
            "materials": materials,
            "regimes": regimes,
            "properties": properties,
            "material_regime_property_triples": triples,
            "new_cross_material_comparison_opportunities": [
                item for item in opportunities if item.get("type") == "same_property_different_materials"
            ],
            "comparison_opportunities": opportunities,
            "new_research_questions": _research_questions(canonical_rows, gaps, opportunities),
            "resources": {
                "llm_extraction_used": False,
                "embeddings_required": False,
                "qdrant_required": False,
                "source": "deterministic_extraction_catalog",
            },
        }
        return report

    def build_delta_report(
        self,
        before_report: dict[str, Any],
        *,
        document_id: str | None = None,
        active_only: bool = True,
    ) -> dict[str, Any]:
        after = self.build_report(document_id=None, active_only=active_only)
        return self.delta_from_reports(before_report, after, new_document_ids=[document_id] if document_id else None)

    @staticmethod
    def delta_from_reports(
        before: dict[str, Any],
        after: dict[str, Any],
        *,
        new_document_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        new_doc_ids = {str(item) for item in (new_document_ids or []) if item}
        before_keys = set(before.get("canonical_fact_keys") or [])
        after_keys = set(after.get("canonical_fact_keys") or [])
        new_keys = sorted(after_keys - before_keys)
        removed_keys = sorted(before_keys - after_keys)
        before_facts = _facts_by_key(before.get("canonical_facts") or [])
        after_facts = _facts_by_key(after.get("canonical_facts") or [])
        new_facts = [after_facts[key] for key in new_keys if key in after_facts]
        corroborated = _corroborated_facts(before_facts, after_facts, new_doc_ids)
        duplicate_facts = _duplicate_facts(after.get("canonical_facts") or [], before_keys, new_doc_ids)
        before_conflicts = {_conflict_identity(item) for item in before.get("conflict_groups") or []}
        conflict_groups_added = [
            item for item in after.get("conflict_groups") or []
            if _conflict_identity(item) not in before_conflicts
        ]
        before_gap_ids = {_gap_identity(item) for item in before.get("data_gaps") or []}
        data_gaps_added = [
            item for item in after.get("data_gaps") or []
            if _gap_identity(item) not in before_gap_ids
        ]
        before_opps = {_opportunity_identity(item) for item in before.get("comparison_opportunities") or []}
        new_opportunities = [
            item for item in after.get("comparison_opportunities") or []
            if _opportunity_identity(item) not in before_opps
        ]
        before_triples = {_triple_identity(item) for item in before.get("material_regime_property_triples") or []}
        new_triples = [
            item for item in after.get("material_regime_property_triples") or []
            if _triple_identity(item) not in before_triples
        ]
        new_materials = sorted(set(after.get("materials") or []) - set(before.get("materials") or []))
        new_regimes = sorted(set(after.get("regimes") or []) - set(before.get("regimes") or []))
        new_properties = sorted(set(after.get("properties") or []) - set(before.get("properties") or []))
        new_nodes_count = len(new_materials) + len(new_regimes) + len(new_properties) + len(new_facts) + len(data_gaps_added)
        new_edges_count = len(new_facts) * 4 + len(data_gaps_added) * 3 + len(new_opportunities)
        return {
            "status": "ok",
            "new_document_ids": sorted(new_doc_ids),
            "new_nodes_count": new_nodes_count,
            "new_edges_count": new_edges_count,
            "new_materials": new_materials,
            "new_regimes": new_regimes,
            "new_properties": new_properties,
            "new_canonical_facts": new_facts,
            "new_canonical_facts_count": len(new_facts),
            "removed_canonical_facts_count": len(removed_keys),
            "updated_facts": corroborated,
            "updated_facts_count": len(corroborated),
            "duplicate_facts": duplicate_facts,
            "duplicate_facts_count": len(duplicate_facts),
            "corroborated_facts": corroborated,
            "corroborated_facts_count": len(corroborated),
            "conflict_groups_added": conflict_groups_added,
            "conflict_groups_added_count": len(conflict_groups_added),
            "data_gaps_added": data_gaps_added,
            "data_gaps_added_count": len(data_gaps_added),
            "new_material_regime_property_triples": new_triples,
            "new_material_regime_property_triples_count": len(new_triples),
            "new_comparison_opportunities": new_opportunities,
            "new_comparison_opportunities_count": len(new_opportunities),
            "new_research_questions": _research_questions(
                after.get("canonical_facts") or [],
                data_gaps_added,
                new_opportunities,
            ),
            "facts_without_evidence": after.get("facts_without_evidence", 0),
        }

    def _extract_rows(self, chunks: list[Chunk]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
        fact_rows: list[dict[str, Any]] = []
        gaps: list[dict[str, Any]] = []
        rejected_count = 0
        quarantine_count = 0
        assert self.pipeline is not None
        for chunk in chunks:
            bundle = self.pipeline.extract_from_chunk(chunk)
            rejected_count += len(bundle.rejected_items)
            quarantine_count += len(getattr(bundle, "quarantined_items", []) or [])
            experiments = bundle_to_experiment_facts(bundle)
            for row in fact_rows_from_experiments(experiments):
                enriched = _row_with_key(_enrich_fact_row(row, chunk))
                fact_rows.append(enriched)
            for gap in bundle_to_data_gaps(bundle):
                gaps.append(_gap_to_dict(gap, chunk))
        return fact_rows, gaps, rejected_count, quarantine_count


def _document_records(catalog: SQLiteCatalog, *, document_id: str | None = None) -> list[dict[str, Any]]:
    records = catalog.list_document_records()
    if document_id:
        records = [item for item in records if item.get("doc_id") == document_id]
    result = []
    for item in records:
        metadata = item.get("metadata") or {}
        result.append(
            {
                "doc_id": item.get("doc_id"),
                "source_name": item.get("source_name") or item.get("filename") or item.get("title"),
                "source_type": item.get("source_type") or metadata.get("source_type"),
                "content_hash": metadata.get("content_hash"),
                "document_version": item.get("version") or metadata.get("document_version"),
                "active": bool(item.get("active", True)),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "chunks": item.get("chunks"),
            }
        )
    return result


def _enrich_fact_row(row: dict[str, Any], chunk: Chunk) -> dict[str, Any]:
    result = dict(row)
    evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
    result["evidence"] = evidence
    result["evidence_count"] = len(evidence)
    result.setdefault("doc_id", chunk.doc_id)
    result.setdefault("source_chunk_id", chunk.chunk_id)
    result.setdefault("source_name", chunk.metadata.get("source_name") or chunk.metadata.get("filename") or chunk.doc_id)
    result["document_ids"] = sorted({str(item.get("document_id") or item.get("doc_id")) for item in evidence if isinstance(item, dict) and (item.get("document_id") or item.get("doc_id"))} or {chunk.doc_id})
    result["chunk_ids"] = sorted({str(item.get("chunk_id")) for item in evidence if isinstance(item, dict) and item.get("chunk_id")} or {chunk.chunk_id})
    result["source_names"] = sorted({str(item.get("source_name")) for item in evidence if isinstance(item, dict) and item.get("source_name")} or {str(result["source_name"])})
    return result


def _row_with_key(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["canonical_fact_key"] = result.get("canonical_fact_key") or canonical_fact_key_from_row(result)
    result["evidence_count"] = len(result.get("evidence") or [])
    return result


def _gap_to_dict(gap: Any, chunk: Chunk) -> dict[str, Any]:
    evidence = [item.model_dump() for item in getattr(gap, "evidence", [])]
    return {
        "gap_id": getattr(gap, "gap_id", None),
        "material": canonical_material(getattr(gap, "material", None)),
        "regime": canonical_regime(getattr(gap, "regime", None)),
        "property": canonical_property(getattr(gap, "property", None)),
        "reason": getattr(gap, "reason", None),
        "evidence": evidence,
        "evidence_count": len(evidence),
        "document_ids": sorted({str(item.get("document_id") or item.get("doc_id")) for item in evidence if isinstance(item, dict) and (item.get("document_id") or item.get("doc_id"))} or {chunk.doc_id}),
    }


def _dedupe_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for gap in gaps:
        key = _gap_identity(gap)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = dict(gap)
            continue
        merged = dict(existing)
        merged["evidence"] = _merge_evidence(existing.get("evidence") or [], gap.get("evidence") or [])
        merged["evidence_count"] = len(merged["evidence"])
        merged["document_ids"] = sorted(set(existing.get("document_ids") or []) | set(gap.get("document_ids") or []))
        by_key[key] = merged
    return list(by_key.values())


def _merge_evidence(left: list[Any], right: list[Any]) -> list[dict[str, Any]]:
    seen = set()
    result: list[dict[str, Any]] = []
    for item in [*left, *right]:
        if not isinstance(item, dict):
            continue
        key = (item.get("document_id") or item.get("doc_id"), item.get("chunk_id"), item.get("quote"))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _facts_by_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row.get("canonical_fact_key"): row for row in rows if isinstance(row, dict) and row.get("canonical_fact_key")}


def _evidence_identities(row: dict[str, Any]) -> set[tuple[str, str, str]]:
    identities = set()
    for item in row.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        identities.add((str(item.get("document_id") or item.get("doc_id") or ""), str(item.get("chunk_id") or ""), str(item.get("quote") or "")))
    return identities


def _corroborated_facts(
    before_facts: dict[str, dict[str, Any]],
    after_facts: dict[str, dict[str, Any]],
    new_document_ids: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key, after in after_facts.items():
        before = before_facts.get(key)
        if before is None:
            continue
        before_evidence = _evidence_identities(before)
        after_evidence = _evidence_identities(after)
        added = after_evidence - before_evidence
        if not added:
            continue
        if new_document_ids:
            added_docs = {doc_id for doc_id, _, _ in added if doc_id}
            if not (added_docs & new_document_ids):
                continue
        item = dict(after)
        item["new_evidence_count"] = len(added)
        result.append(item)
    return result


def _duplicate_facts(rows: list[dict[str, Any]], before_keys: set[str], new_document_ids: set[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        key = row.get("canonical_fact_key")
        if key not in before_keys:
            continue
        if new_document_ids:
            docs = set(row.get("document_ids") or [])
            if not (docs & new_document_ids):
                continue
        result.append(row)
    return result


def _material_regime_property_triples(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    triples = set()
    for row in rows:
        material = canonical_material(row.get("material"))
        regime = canonical_regime(row.get("regime"))
        prop = canonical_property(row.get("property"))
        if material and regime and prop:
            triples.add((material, regime, prop))
    return [
        {"material": material, "regime": regime, "property": prop}
        for material, regime, prop in sorted(triples)
    ]


def _comparison_opportunities(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_property: dict[str, set[str]] = defaultdict(set)
    by_material_property: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        material = canonical_material(row.get("material"))
        regime = canonical_regime(row.get("regime"))
        prop = canonical_property(row.get("property"))
        if material and prop:
            by_property[prop].add(material)
        if material and prop and regime:
            by_material_property[(material, prop)].add(regime)
    opportunities: list[dict[str, Any]] = []
    for prop, materials in sorted(by_property.items()):
        if len(materials) < 2:
            continue
        opportunities.append(
            {
                "type": "same_property_different_materials",
                "property": prop,
                "materials": sorted(materials),
                "description": f"Можно сравнить материалы по свойству: {prop}.",
            }
        )
    for (material, prop), regimes in sorted(by_material_property.items()):
        if len(regimes) < 2:
            continue
        opportunities.append(
            {
                "type": "same_material_different_regime",
                "material": material,
                "property": prop,
                "regimes": sorted(regimes),
                "description": f"Можно сравнить режимы обработки материала {material} по свойству {prop}.",
            }
        )
    return opportunities


def _research_questions(rows: list[dict[str, Any]], gaps: list[dict[str, Any]], opportunities: list[dict[str, Any]]) -> list[str]:
    questions: list[str] = []
    for gap in gaps:
        material = gap.get("material") or "материала"
        prop = gap.get("property") or "свойства"
        regime = gap.get("regime") or "режима обработки"
        questions.append(f"Для {material} не хватает проверяемых данных по {prop} в контексте {regime}.")
    for opportunity in opportunities:
        if opportunity.get("type") == "same_property_different_materials":
            materials = ", ".join(opportunity.get("materials") or [])
            prop = opportunity.get("property")
            questions.append(f"Можно проверить сопоставимость данных по {prop} между материалами: {materials}.")
        elif opportunity.get("type") == "same_material_different_regime":
            regimes = ", ".join(opportunity.get("regimes") or [])
            questions.append(f"Для {opportunity.get('material')} есть несколько режимов ({regimes}); стоит сравнить их по {opportunity.get('property')}.")
    return list(dict.fromkeys(questions))[:10]


def _conflict_identity(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        canonical_material(item.get("material")),
        canonical_regime(item.get("regime")),
        canonical_property(item.get("property")),
    )


def _gap_identity(item: dict[str, Any]) -> str:
    parts = [
        canonical_material(item.get("material")),
        canonical_regime(item.get("regime")),
        canonical_property(item.get("property")),
        str(item.get("reason") or "").strip().lower(),
    ]
    return "|".join(parts)


def _opportunity_identity(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("type") or ""),
        str(item.get("material") or ""),
        str(item.get("property") or ""),
        ",".join(item.get("materials") or []),
        ",".join(item.get("regimes") or []),
    ]
    return "|".join(parts)


def _triple_identity(item: dict[str, Any]) -> str:
    return "|".join([str(item.get("material") or ""), str(item.get("regime") or ""), str(item.get("property") or "")])

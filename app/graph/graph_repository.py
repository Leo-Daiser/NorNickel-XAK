"""Graph/fact repository used by ontology-driven QA."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Iterable, Protocol

from ..config import settings
from ..domain.fact_normalization import dedupe_measurements
from ..domain.normalization import canonical_material, canonical_property, canonical_regime, material_matches, property_matches, regime_matches
from ..domain.ontology import DataGap, Evidence
from ..extraction.extraction import EntityRelationExtractor
from ..extraction.models import AcceptedFact
from ..extraction.pipeline import ExtractionPipeline
from ..extraction.to_graph_models import bundle_to_data_gaps, bundle_to_experiment_facts, bundle_to_structured_accepted_experiment_facts
from ..models.schemas import Chunk, Document
from ..storage.catalog import SQLiteCatalog
from .graph_db import GraphDB
from .graph_models import DecisionHistoryItem, EntityCard, EntitySummary, ExperimentFact, GraphStats, PartialMatches, SimilarExperiment
from .neo4j_repository import Neo4jGraphRepository
from .explorer import (
    build_entity_card,
    build_neighborhood_from_facts,
    decision_history_filtered,
    filter_gaps,
    graph_stats_from_facts,
    list_entities_from_facts,
    similar_experiments,
)


class GraphRepository(Protocol):
    def find_exact_material_regime_property(self, material: str, regime: str, property_name: str) -> list[ExperimentFact]:
        ...

    def find_partial_matches(self, material: str | None = None, regime: str | None = None, property_name: str | None = None) -> PartialMatches:
        ...

    def get_decision_history(self, material: str) -> list[DecisionHistoryItem]:
        ...

    def find_gaps(self, material: str | None = None, regime: str | None = None, property_name: str | None = None) -> list[DataGap]:
        ...

    def find_experiments(
        self,
        material: str | None = None,
        regime: str | None = None,
        property_name: str | None = None,
        limit: int = 50,
    ) -> list[ExperimentFact]:
        ...

    def find_accepted_facts(
        self,
        fact_types: list[str] | None = None,
        limit: int = 200,
    ) -> list[AcceptedFact]:
        ...

    def list_entities(self, entity_type: str | None = None, query: str | None = None, limit: int = 50) -> list[EntitySummary]:
        ...

    def get_entity_card(self, entity_type: str, entity_id: str) -> EntityCard:
        ...

    def get_neighborhood(self, entity_type: str, entity_id: str, depth: int = 1, limit_nodes: int = 50, limit_edges: int = 80) -> dict[str, Any]:
        ...

    def get_graph_stats(self) -> GraphStats:
        ...

    def get_gaps(self, material: str | None = None, regime: str | None = None, property_name: str | None = None, limit: int = 50) -> list[DataGap]:
        ...

    def get_decision_history_filtered(
        self,
        material: str | None = None,
        regime: str | None = None,
        property_name: str | None = None,
        limit: int = 50,
    ) -> list[DecisionHistoryItem]:
        ...

    def get_similar_experiments(
        self,
        material: str | None = None,
        regime: str | None = None,
        property_name: str | None = None,
        experiment_id: str | None = None,
        limit: int = 10,
    ) -> list[SimilarExperiment]:
        ...


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _gap_id(material: str | None, regime: str | None, property_name: str | None, reason: str) -> str:
    raw = "|".join([material or "", regime or "", property_name or "", reason])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class CatalogGraphRepository:
    """Fallback graph repository built from the durable SQLite catalog."""

    backend_name = "fallback"

    def __init__(
        self,
        catalog: SQLiteCatalog,
        extractor: EntityRelationExtractor | None = None,
        document_getter: Callable[[str], Document | None] | None = None,
        extraction_pipeline: ExtractionPipeline | None = None,
    ) -> None:
        self.catalog = catalog
        self.extractor = extractor
        self.extraction_pipeline = extraction_pipeline or ExtractionPipeline(audit_enabled=False)
        self.document_getter = document_getter
        self._experiments: list[ExperimentFact] | None = None
        self._gaps: list[DataGap] | None = None
        self._accepted_facts: list[AcceptedFact] | None = None
        self._cache_signature: tuple[tuple[str, str | None, str | None, int], ...] | None = None
        self._accepted_cache_signature: tuple[tuple[str, str | None, str | None, int], ...] | None = None

    def find_exact_material_regime_property(self, material: str, regime: str, property_name: str) -> list[ExperimentFact]:
        return [
            exp for exp in self._load_experiments()
            if self._has_material(exp, material)
            and self._has_regime(exp, regime)
            and self._has_property(exp, property_name)
        ]

    def list_experiments(self) -> list[ExperimentFact]:
        """Return all extracted experiment facts for materialization/sync."""
        return list(self._load_experiments())

    def find_experiments(
        self,
        material: str | None = None,
        regime: str | None = None,
        property_name: str | None = None,
        limit: int = 50,
    ) -> list[ExperimentFact]:
        """Return experiments matching any supplied analytical constraints."""
        result: list[ExperimentFact] = []
        for exp in self._load_experiments():
            if material and not self._has_material(exp, material):
                continue
            if regime and not self._has_regime(exp, regime):
                continue
            if property_name and not self._has_property(exp, property_name):
                continue
            result.append(exp)
            if len(result) >= limit:
                break
        return result

    def find_accepted_facts(
        self,
        fact_types: list[str] | None = None,
        limit: int = 200,
    ) -> list[AcceptedFact]:
        wanted = set(fact_types or [])
        result: list[AcceptedFact] = []
        for fact in self._load_accepted_facts():
            if wanted and fact.fact_type not in wanted:
                continue
            result.append(fact)
            if len(result) >= limit:
                break
        return result

    def list_gaps(self) -> list[DataGap]:
        """Return all extracted data gaps for materialization/sync."""
        return list(self._load_gaps())

    def find_partial_matches(self, material: str | None = None, regime: str | None = None, property_name: str | None = None) -> PartialMatches:
        experiments = self._load_experiments()
        same_material = [exp for exp in experiments if material and self._has_material(exp, material)]
        same_material_and_regime = [exp for exp in same_material if regime and self._has_regime(exp, regime)]
        same_material_and_property = [exp for exp in same_material if property_name and self._has_property(exp, property_name)]
        same_regime_and_property = [
            exp for exp in experiments
            if regime and property_name and self._has_regime(exp, regime) and self._has_property(exp, property_name)
        ]
        return PartialMatches(
            same_material=same_material,
            same_material_and_regime=same_material_and_regime,
            same_material_and_property=same_material_and_property,
            same_regime_and_property=same_regime_and_property,
        )

    def get_decision_history(self, material: str) -> list[DecisionHistoryItem]:
        history: list[DecisionHistoryItem] = []
        for exp in self._load_experiments():
            if not self._has_material(exp, material):
                continue
            history.append(
                DecisionHistoryItem(
                    experiment_id=exp.experiment_id,
                    material=canonical_material(material),
                    regime=exp.regimes[0] if exp.regimes else None,
                    equipment=exp.equipment,
                    laboratory=exp.laboratories[0] if exp.laboratories else None,
                    measurements=exp.measurements,
                    conclusions=exp.conclusions,
                    evidence=exp.evidence,
                )
            )
        return history

    def find_gaps(self, material: str | None = None, regime: str | None = None, property_name: str | None = None) -> list[DataGap]:
        result: list[DataGap] = []
        for gap in self._load_gaps():
            if material and gap.material and not material_matches(gap.material, material):
                continue
            if material and not gap.material and material not in (gap.reason or ""):
                continue
            if regime and gap.regime and not regime_matches(gap.regime, regime):
                continue
            if property_name and gap.property and not property_matches(gap.property, property_name):
                continue
            if property_name and not gap.property and canonical_property(property_name) not in (gap.reason or ""):
                continue
            result.append(gap)
        return result

    def list_entities(self, entity_type: str | None = None, query: str | None = None, limit: int = 50) -> list[EntitySummary]:
        return list_entities_from_facts(self._load_experiments(), self._load_gaps(), entity_type=entity_type, query=query, limit=limit)

    def get_entity_card(self, entity_type: str, entity_id: str) -> EntityCard:
        return build_entity_card(
            entity_type=entity_type,
            entity_id=entity_id,
            experiments=self._load_experiments(),
            gaps=self._load_gaps(),
            diagnostics={"kg_backend_active": self.backend_name},
        )

    def get_neighborhood(
        self,
        entity_type: str,
        entity_id: str,
        depth: int = 1,
        limit_nodes: int = 50,
        limit_edges: int = 80,
    ) -> dict[str, Any]:
        return build_neighborhood_from_facts(
            entity_type=entity_type,
            entity_id=entity_id,
            experiments=self._load_experiments(),
            gaps=self._load_gaps(),
            depth=depth,
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
        )

    def get_graph_stats(self) -> GraphStats:
        counts = self.catalog.counts()
        return graph_stats_from_facts(
            self._load_experiments(),
            self._load_gaps(),
            documents=counts.get("documents", 0),
            chunks=counts.get("chunks", 0),
            kg_backend_active=self.backend_name,
            diagnostics={"source": "fallback_catalog"},
        )

    def get_gaps(self, material: str | None = None, regime: str | None = None, property_name: str | None = None, limit: int = 50) -> list[DataGap]:
        return filter_gaps(self._load_gaps(), material=material, regime=regime, property_name=property_name, limit=limit)

    def get_decision_history_filtered(
        self,
        material: str | None = None,
        regime: str | None = None,
        property_name: str | None = None,
        limit: int = 50,
    ) -> list[DecisionHistoryItem]:
        return decision_history_filtered(
            self._load_experiments(),
            material=material,
            regime=regime,
            property_name=property_name,
            limit=limit,
        )

    def get_similar_experiments(
        self,
        material: str | None = None,
        regime: str | None = None,
        property_name: str | None = None,
        experiment_id: str | None = None,
        limit: int = 10,
    ) -> list[SimilarExperiment]:
        return similar_experiments(
            self._load_experiments(),
            material=material,
            regime=regime,
            property_name=property_name,
            experiment_id=experiment_id,
            limit=limit,
        )

    def _load_experiments(self) -> list[ExperimentFact]:
        stored_chunks = list(self.catalog.list_chunks())
        signature = tuple(
            (chunk.chunk_id, chunk.text_hash, chunk.updated_at, len(chunk.text or ""))
            for chunk in stored_chunks
        )
        if self._experiments is not None and signature == self._cache_signature:
            return self._experiments

        experiments: dict[str, dict] = {}
        gaps: list[DataGap] = []
        for stored_chunk in stored_chunks:
            for chunk in self._extraction_chunks(stored_chunk):
                self._extract_chunk_into_experiments(chunk, experiments, gaps)

        built: list[ExperimentFact] = []
        for item in experiments.values():
            if not item["materials"] and not item["regimes"] and not item["measurements"]:
                continue
            materials = _unique(item["materials"])
            regimes = _unique(item["regimes"])
            built.append(
                ExperimentFact(
                    experiment_id=item["experiment_id"],
                    materials=materials,
                    regimes=regimes,
                    measurements=dedupe_measurements(
                        item["measurements"],
                        material=materials[0] if materials else None,
                        regime=regimes[0] if regimes else None,
                    ),
                    equipment=_unique(item["equipment"]),
                    laboratories=_unique(item["laboratories"]),
                    conclusions=_unique(item["conclusions"]),
                    evidence=self._unique_evidence(item["evidence"]),
                    source_chunk_ids=_unique(item["source_chunk_ids"]),
                )
            )

        self._experiments = built
        self._gaps = self._dedupe_gaps(gaps)
        self._cache_signature = signature
        return self._experiments

    def _extract_chunk_into_experiments(self, chunk: Chunk, experiments: dict[str, dict], gaps: list[DataGap]) -> None:
        bundle = self.extraction_pipeline.extract_from_chunk(chunk)
        for fact in [*bundle_to_experiment_facts(bundle), *bundle_to_structured_accepted_experiment_facts(bundle)]:
            exp = experiments.setdefault(
                fact.experiment_id,
                {
                    "experiment_id": fact.experiment_id,
                    "materials": [],
                    "regimes": [],
                    "measurements": [],
                    "equipment": [],
                    "laboratories": [],
                    "conclusions": [],
                    "evidence": [],
                    "source_chunk_ids": [],
                },
            )
            exp["materials"].extend(fact.materials)
            exp["regimes"].extend(fact.regimes)
            exp["measurements"].extend(fact.measurements)
            exp["equipment"].extend(fact.equipment)
            exp["laboratories"].extend([*fact.laboratories, *fact.teams, *fact.employees])
            exp["conclusions"].extend(fact.conclusions)
            exp["evidence"].extend(fact.evidence)
            exp["source_chunk_ids"].extend(fact.source_chunk_ids or [chunk.chunk_id])
        gaps.extend(bundle_to_data_gaps(bundle))

    def _extraction_chunks(self, chunk: Chunk) -> list[Chunk]:
        """Split dense experiment lists before extraction to avoid cross-linking rows."""
        text = chunk.text or ""
        experiment_markers = re.findall(r"(?i)(?:experiment_id|experiment\s+id|id\s+эксперимента|experiment)\s*[:=]", text)
        if len(experiment_markers) <= 1:
            return [chunk]
        segments = [line.strip() for line in text.splitlines() if line.strip()]
        if len(segments) <= 1:
            segments = [part.strip(" .;\n\t") for part in re.split(r"(?=(?:experiment_id|experiment\s+id|id\s+эксперимента|experiment)\s*[:=])", text, flags=re.IGNORECASE) if part.strip()]
        result: list[Chunk] = []
        for idx, segment in enumerate(segments):
            if not segment:
                continue
            update = {
                "chunk_id": f"{chunk.chunk_id}:seg{idx}",
                "text": segment,
                "ordinal": (chunk.ordinal or 0) * 1000 + idx,
                "metadata": {**(chunk.metadata or {}), "parent_chunk_id": chunk.chunk_id, "segment_id": idx},
            }
            if hasattr(chunk, "model_copy"):
                result.append(chunk.model_copy(update=update))
            else:
                result.append(chunk.copy(update=update))
        return result or [chunk]

    def _load_gaps(self) -> list[DataGap]:
        if self._gaps is None:
            self._load_experiments()
        return self._gaps or []

    def _load_accepted_facts(self) -> list[AcceptedFact]:
        stored_chunks = list(self.catalog.list_chunks())
        signature = tuple(
            (chunk.chunk_id, chunk.text_hash, chunk.updated_at, len(chunk.text or ""))
            for chunk in stored_chunks
        )
        if self._accepted_facts is not None and signature == self._accepted_cache_signature:
            return self._accepted_facts

        facts: dict[str, AcceptedFact] = {}
        for stored_chunk in stored_chunks:
            for chunk in self._extraction_chunks(stored_chunk):
                bundle = self.extraction_pipeline.extract_from_chunk(chunk)
                for fact in bundle.accepted_facts:
                    if not fact.evidence:
                        continue
                    facts.setdefault(fact.candidate_id, fact)
        self._accepted_facts = list(facts.values())
        self._accepted_cache_signature = signature
        return self._accepted_facts

    def _source_name(self, chunk: Chunk) -> str:
        if chunk.metadata.get("source_name"):
            return str(chunk.metadata.get("source_name"))
        if chunk.metadata.get("filename"):
            return str(chunk.metadata.get("filename"))
        doc = self.document_getter(chunk.doc_id) if self.document_getter else None
        return doc.title if doc else chunk.doc_id

    @staticmethod
    def _has_material(exp: ExperimentFact, material: str) -> bool:
        return any(material_matches(value, material) for value in exp.materials)

    @staticmethod
    def _has_regime(exp: ExperimentFact, regime: str) -> bool:
        return any(regime_matches(value, regime) for value in exp.regimes)

    @staticmethod
    def _has_property(exp: ExperimentFact, property_name: str) -> bool:
        return any(property_matches(measurement.property_name, property_name) for measurement in exp.measurements)

    @staticmethod
    def _unique_evidence(items: list[Evidence]) -> list[Evidence]:
        seen = set()
        result: list[Evidence] = []
        for item in items:
            key = (item.document_id, item.chunk_id, item.quote)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _gap_from_text(self, text: str, missing_for: str | None, evidence: Evidence) -> DataGap:
        joined = f"{text} {missing_for or ''}"
        material = self._first_canonical(joined, canonical_material, ["ВТ6", "7075-T6", "12Х18Н10Т", "09Г2С"])
        regime = self._first_canonical(joined, canonical_regime, ["отжиг", "старение", "закалка", "криообработка"])
        property_name = self._first_canonical(joined, canonical_property, ["прочность", "твёрдость", "пластичность", "вязкость", "коррозионная стойкость"])
        reason = re.sub(r"\s+", " ", str(text or "").strip(" .;|"))
        return DataGap(
            gap_id=_gap_id(material, regime, property_name, reason),
            material=material,
            regime=regime,
            property=property_name,
            reason=reason,
            evidence=[evidence],
        )

    @staticmethod
    def _first_canonical(text: str, canonicalizer, candidates: list[str]) -> str | None:
        for candidate in candidates:
            canonical = canonicalizer(candidate)
            if canonical and canonical in text:
                return canonical
            if canonicalizer(text) == canonical:
                return canonical
        return None

    @staticmethod
    def _dedupe_gaps(gaps: list[DataGap]) -> list[DataGap]:
        seen = set()
        result: list[DataGap] = []
        for gap in gaps:
            key = (gap.material, gap.regime, gap.property, gap.reason)
            if key in seen:
                continue
            seen.add(key)
            result.append(gap)
        return result


class GraphRepositoryFactory:
    """Create a graph repository according to KG_BACKEND."""

    @staticmethod
    def create(
        catalog: SQLiteCatalog,
        extractor: EntityRelationExtractor,
        graph_db: GraphDB | None = None,
        document_getter: Callable[[str], Document | None] | None = None,
        configured_backend: str | None = None,
        extraction_pipeline: ExtractionPipeline | None = None,
    ) -> GraphRepository:
        mode = (configured_backend or getattr(settings, "kg_backend", "auto") or "auto").lower()
        if mode not in {"auto", "neo4j", "fallback"}:
            raise ValueError(f"Unsupported KG_BACKEND={mode!r}; expected auto, neo4j or fallback")
        if mode == "fallback":
            return CatalogGraphRepository(catalog=catalog, extractor=extractor, document_getter=document_getter, extraction_pipeline=extraction_pipeline)
        if mode == "neo4j":
            if graph_db is None:
                raise RuntimeError("KG_BACKEND=neo4j requested, but Neo4j is unavailable")
            return Neo4jGraphRepository(graph_db=graph_db)
        if graph_db is not None:
            return Neo4jGraphRepository(graph_db=graph_db)
        return CatalogGraphRepository(catalog=catalog, extractor=extractor, document_getter=document_getter, extraction_pipeline=extraction_pipeline)


def repository_backend_name(repository: GraphRepository) -> str:
    return str(getattr(repository, "backend_name", repository.__class__.__name__))

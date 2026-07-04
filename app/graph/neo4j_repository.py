"""Neo4j-backed strict graph repository."""

from __future__ import annotations

import json
from typing import Any

from ..domain.fact_normalization import measurement_normalization_fields
from ..domain.normalization import canonical_material, canonical_property, canonical_regime
from ..domain.ontology import DataGap, Evidence, Measurement
from ..extraction.models import AcceptedFact, EvidenceSpan, ExtractionSource
from .explorer import (
    build_entity_card,
    build_neighborhood_from_facts,
    decision_history_filtered,
    filter_gaps,
    graph_stats_from_facts,
    list_entities_from_facts,
    similar_experiments,
)
from .cypher_queries import (
    DECISION_HISTORY_BY_MATERIAL,
    EXACT_MATERIAL_REGIME_PROPERTY,
    FIND_EXPERIMENTS_BY_CONSTRAINTS,
    FIND_ACCEPTED_FACTS,
    FIND_GAPS,
)
from .graph_db import GraphDB
from .graph_models import DecisionHistoryItem, EntityCard, EntitySummary, ExperimentFact, GraphStats, PartialMatches, SimilarExperiment


class Neo4jGraphRepository:
    """GraphRepository implementation that reads strict facts from Neo4j."""

    backend_name = "neo4j"

    def __init__(self, graph_db: GraphDB) -> None:
        self.graph_db = graph_db

    def find_exact_material_regime_property(self, material: str, regime: str, property_name: str) -> list[ExperimentFact]:
        rows = self.graph_db.run(
            EXACT_MATERIAL_REGIME_PROPERTY,
            material=canonical_material(material),
            regime=canonical_regime(regime),
            property=canonical_property(property_name),
        )
        return [self._record_to_experiment_fact(row) for row in rows]

    def find_partial_matches(self, material: str | None = None, regime: str | None = None, property_name: str | None = None) -> PartialMatches:
        material_c = canonical_material(material) if material else None
        regime_c = canonical_regime(regime) if regime else None
        property_c = canonical_property(property_name) if property_name else None
        same_material = self._find_experiments(material=material_c, limit=5) if material_c else []
        same_material_and_regime = self._find_experiments(material=material_c, regime=regime_c, limit=5) if material_c and regime_c else []
        same_material_and_property = self._find_experiments(material=material_c, property_name=property_c, limit=5) if material_c and property_c else []
        same_regime_and_property = self._find_experiments(regime=regime_c, property_name=property_c, limit=5) if regime_c and property_c else []
        return PartialMatches(
            same_material=same_material,
            same_material_and_regime=same_material_and_regime,
            same_material_and_property=same_material_and_property,
            same_regime_and_property=same_regime_and_property,
        )

    def get_decision_history(self, material: str) -> list[DecisionHistoryItem]:
        rows = self.graph_db.run(DECISION_HISTORY_BY_MATERIAL, material=canonical_material(material))
        items: list[DecisionHistoryItem] = []
        for row in rows:
            fact = self._record_to_experiment_fact(row)
            items.append(
                DecisionHistoryItem(
                    experiment_id=fact.experiment_id,
                    material=canonical_material(material),
                    regime=fact.regimes[0] if fact.regimes else None,
                    equipment=fact.equipment,
                    laboratory=fact.laboratories[0] if fact.laboratories else None,
                    measurements=fact.measurements,
                    conclusions=fact.conclusions,
                    evidence=fact.evidence,
                )
            )
        return items

    def find_gaps(self, material: str | None = None, regime: str | None = None, property_name: str | None = None) -> list[DataGap]:
        rows = self.graph_db.run(
            FIND_GAPS,
            material=canonical_material(material) if material else None,
            regime=canonical_regime(regime) if regime else None,
            property=canonical_property(property_name) if property_name else None,
        )
        return [self._record_to_gap(row) for row in rows]

    def find_experiments(
        self,
        material: str | None = None,
        regime: str | None = None,
        property_name: str | None = None,
        limit: int = 50,
    ) -> list[ExperimentFact]:
        """Return experiments matching optional graph constraints."""
        return self._find_experiments(
            material=canonical_material(material) if material else None,
            regime=canonical_regime(regime) if regime else None,
            property_name=canonical_property(property_name) if property_name else None,
            limit=limit,
        )

    def find_accepted_facts(
        self,
        fact_types: list[str] | None = None,
        limit: int = 200,
    ) -> list[AcceptedFact]:
        rows = self.graph_db.run(
            FIND_ACCEPTED_FACTS,
            fact_types=fact_types or None,
            limit=limit,
        )
        return [self._record_to_accepted_fact(row) for row in rows]

    def list_entities(self, entity_type: str | None = None, query: str | None = None, limit: int = 50) -> list[EntitySummary]:
        experiments = self.find_experiments(limit=max(limit * 3, 50))
        gaps = self.find_gaps()
        return list_entities_from_facts(experiments, gaps, entity_type=entity_type, query=query, limit=limit)

    def get_entity_card(self, entity_type: str, entity_id: str) -> EntityCard:
        experiments = self.find_experiments(limit=200)
        gaps = self.find_gaps()
        return build_entity_card(
            entity_type=entity_type,
            entity_id=entity_id,
            experiments=experiments,
            gaps=gaps,
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
            experiments=self.find_experiments(limit=200),
            gaps=self.find_gaps(),
            depth=depth,
            limit_nodes=limit_nodes,
            limit_edges=limit_edges,
        )

    def get_graph_stats(self) -> GraphStats:
        experiments = self.find_experiments(limit=1000)
        gaps = self.find_gaps()
        stats = graph_stats_from_facts(
            experiments,
            gaps,
            kg_backend_active=self.backend_name,
            diagnostics={"source": "neo4j_repository"},
        )
        try:
            raw_stats = self.graph_db.stats()
        except Exception:
            raw_stats = {}
        stats.documents = int(raw_stats.get("nodes_Document", stats.documents))
        stats.chunks = int(raw_stats.get("nodes_DocumentChunk", raw_stats.get("nodes_Chunk", stats.chunks)))
        stats.relationships = sum(int(value) for key, value in raw_stats.items() if key.startswith("relationships_")) or stats.relationships
        stats.diagnostics["neo4j_raw_stats"] = raw_stats
        return stats

    def get_gaps(self, material: str | None = None, regime: str | None = None, property_name: str | None = None, limit: int = 50) -> list[DataGap]:
        return filter_gaps(self.find_gaps(material=material, regime=regime, property_name=property_name), material=material, regime=regime, property_name=property_name, limit=limit)

    def get_decision_history_filtered(
        self,
        material: str | None = None,
        regime: str | None = None,
        property_name: str | None = None,
        limit: int = 50,
    ) -> list[DecisionHistoryItem]:
        experiments = self.find_experiments(material=material, regime=regime, property_name=property_name, limit=max(limit, 50))
        return decision_history_filtered(experiments, material=material, regime=regime, property_name=property_name, limit=limit)

    def get_similar_experiments(
        self,
        material: str | None = None,
        regime: str | None = None,
        property_name: str | None = None,
        experiment_id: str | None = None,
        limit: int = 10,
    ) -> list[SimilarExperiment]:
        return similar_experiments(
            self.find_experiments(limit=300),
            material=material,
            regime=regime,
            property_name=property_name,
            experiment_id=experiment_id,
            limit=limit,
        )

    def _find_experiments(
        self,
        material: str | None = None,
        regime: str | None = None,
        property_name: str | None = None,
        limit: int = 20,
    ) -> list[ExperimentFact]:
        rows = self.graph_db.run(
            FIND_EXPERIMENTS_BY_CONSTRAINTS,
            material=material,
            regime=regime,
            property=property_name,
            limit=limit,
        )
        return [self._record_to_experiment_fact(row) for row in rows]

    def _record_to_experiment_fact(self, record: Any) -> ExperimentFact:
        experiment = _record_get(record, "e") or {}
        materials = [_node_prop(node, "canonical_name") for node in _record_get(record, "materials", [])]
        regimes = [_node_prop(node, "canonical_name") for node in _record_get(record, "regimes", [])]
        equipment = [_node_prop(node, "canonical_name") for node in _record_get(record, "equipment", [])]
        laboratories = [_node_prop(node, "canonical_name") for node in _record_get(record, "laboratories", [])]
        conclusions = [_node_prop(node, "text") for node in _record_get(record, "conclusions", [])]
        evidence = self._evidence_from_record(record)
        measurements: list[Measurement] = []
        for item in _record_get(record, "measurements", []):
            measurement_node = _map_get(item, "measurement")
            property_node = _map_get(item, "property")
            if measurement_node is None or property_node is None:
                continue
            property_name = canonical_property(_node_prop(property_node, "canonical_name"))
            fields = measurement_normalization_fields(
                property_name,
                _node_prop(measurement_node, "value"),
                _node_prop(measurement_node, "unit"),
            )
            measurements.append(
                Measurement(
                    property_name=property_name,
                    value=_node_prop(measurement_node, "value"),
                    value_min=_node_prop(measurement_node, "value_min"),
                    value_max=_node_prop(measurement_node, "value_max"),
                    raw_value=_node_prop(measurement_node, "raw_value"),
                    unit=_node_prop(measurement_node, "unit"),
                    value_original=_node_prop(measurement_node, "value_original") if _node_prop(measurement_node, "value_original") is not None else fields["value_original"],
                    unit_original=_node_prop(measurement_node, "unit_original") or fields["unit_original"],
                    value_normalized=_node_prop(measurement_node, "value_normalized") if _node_prop(measurement_node, "value_normalized") is not None else fields["value_normalized"],
                    unit_normalized=_node_prop(measurement_node, "unit_normalized") or fields["unit_normalized"],
                    normalization_family=_node_prop(measurement_node, "normalization_family") or fields["normalization_family"],
                    effect=_node_prop(measurement_node, "effect"),
                    baseline_value=_node_prop(measurement_node, "baseline_value"),
                    delta_abs=_node_prop(measurement_node, "delta_abs"),
                    delta_rel_percent=_node_prop(measurement_node, "delta_rel_percent"),
                    confidence=_node_prop(measurement_node, "confidence"),
                    analyte=_node_prop(measurement_node, "analyte"),
                    fact_type=_node_prop(measurement_node, "fact_type"),
                    source_adapter=_node_prop(measurement_node, "source_adapter"),
                    evidence=evidence,
                )
            )
        return ExperimentFact(
            experiment_id=str(_node_prop(experiment, "experiment_id") or ""),
            materials=[value for value in materials if value],
            regimes=[value for value in regimes if value],
            measurements=measurements,
            equipment=[value for value in equipment if value],
            laboratories=list(dict.fromkeys(value for value in laboratories if value)),
            teams=[_node_prop(node, "canonical_name") for node in _record_get(record, "teams", []) if _node_prop(node, "canonical_name")],
            conclusions=[value for value in conclusions if value],
            evidence=evidence,
            source_chunk_ids=[item.chunk_id for item in evidence if item.chunk_id],
        )

    def _record_to_gap(self, record: Any) -> DataGap:
        gap = _record_get(record, "g") or {}
        evidence = self._evidence_from_record(record)
        material = _node_prop(gap, "material") or _first_node_prop(_record_get(record, "materials", []), "canonical_name")
        regime = _node_prop(gap, "regime") or _first_node_prop(_record_get(record, "regimes", []), "canonical_name")
        property_name = _node_prop(gap, "property") or _first_node_prop(_record_get(record, "properties", []), "canonical_name")
        return DataGap(
            gap_id=str(_node_prop(gap, "gap_id") or ""),
            material=material,
            regime=regime,
            property=property_name,
            reason=str(_node_prop(gap, "reason") or ""),
            evidence=evidence,
        )

    def _record_to_accepted_fact(self, record: Any) -> AcceptedFact:
        node = _record_get(record, "f") or {}
        fact_type = str(_node_prop(node, "fact_type") or "UnknownFact")
        materials = [_node_prop(item, "canonical_name") for item in _record_get(record, "materials", []) if _node_prop(item, "canonical_name")]
        processes = [_node_prop(item, "canonical_name") for item in _record_get(record, "processes", []) if _node_prop(item, "canonical_name")]
        properties = [_node_prop(item, "canonical_name") for item in _record_get(record, "properties", []) if _node_prop(item, "canonical_name")]
        equipment = [_node_prop(item, "canonical_name") for item in _record_get(record, "equipment", []) if _node_prop(item, "canonical_name")]
        facilities = [_node_prop(item, "canonical_name") for item in _record_get(record, "facilities", []) if _node_prop(item, "canonical_name")]
        geographies = [_node_prop(item, "canonical_name") for item in _record_get(record, "geographies", []) if _node_prop(item, "canonical_name")]
        publications = [_node_prop(item, "title") or _node_prop(item, "publication_id") for item in _record_get(record, "publications", []) if _node_prop(item, "title") or _node_prop(item, "publication_id")]
        experts = [_node_prop(item, "canonical_name") for item in _record_get(record, "experts", []) if _node_prop(item, "canonical_name")]
        labs = [_node_prop(item, "canonical_name") for item in _record_get(record, "laboratories", []) if _node_prop(item, "canonical_name")]
        teams = [_node_prop(item, "canonical_name") for item in _record_get(record, "teams", []) if _node_prop(item, "canonical_name")]
        evidence = self._evidence_spans_from_record(record)
        stored_subject = _json_mapping(_node_prop(node, "subject_json"))
        stored_object = _json_mapping(_node_prop(node, "object_json"))
        subject = {
            "material": _first(materials),
            "materials": materials,
            "process": _first(processes),
            "processes": processes,
            "regimes": processes,
            "equipment": _first(equipment),
            "facility": _first(facilities),
            "geography": _first(geographies),
            "expert": _first(experts),
            "laboratory": _first(labs),
            "team": _first(teams),
        }
        subject = {**stored_subject, **{key: value for key, value in subject.items() if value}}
        obj = {
            "property": _first(properties),
            "properties": properties,
            "parameter": _first(properties),
            "source_note": _first(publications),
        }
        obj = {**stored_object, **{key: value for key, value in obj.items() if value}}
        return AcceptedFact(
            candidate_id=str(_node_prop(node, "candidate_id") or _node_prop(node, "fact_id") or ""),
            fact_type=fact_type,
            normalized_fact={
                "subject": {key: value for key, value in subject.items() if value},
                "predicate": _node_prop(node, "predicate") or "",
                "object": {key: value for key, value in obj.items() if value},
                "value": _node_prop(node, "value"),
                "unit": _node_prop(node, "unit"),
                "document_type": _node_prop(node, "document_type"),
                "source_adapter": _node_prop(node, "source_adapter"),
            },
            evidence=evidence,
            score=float(_node_prop(node, "confidence") or 0.0),
            validation_reasons=["loaded_from_neo4j_accepted_fact"],
        )

    @staticmethod
    def _evidence_from_record(record: Any) -> list[Evidence]:
        chunks = _record_get(record, "chunks", []) or []
        docs = _record_get(record, "documents", []) or []
        doc_by_id = {_node_prop(doc, "document_id"): doc for doc in docs if _node_prop(doc, "document_id")}
        evidence: list[Evidence] = []
        seen = set()
        for chunk in chunks:
            chunk_id = _node_prop(chunk, "chunk_id")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            document_id = _node_prop(chunk, "document_id")
            if not document_id and len(doc_by_id) == 1:
                document_id = next(iter(doc_by_id))
            doc = doc_by_id.get(document_id)
            source_name = _node_prop(chunk, "source_name") or _node_prop(doc, "source_name") or _node_prop(doc, "title")
            evidence.append(
                Evidence(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    source_name=source_name,
                    page=_node_prop(chunk, "page") or _node_prop(chunk, "page_start"),
                    quote=_node_prop(chunk, "text"),
                    confidence=0.9,
                )
            )
        return evidence

    @staticmethod
    def _evidence_spans_from_record(record: Any) -> list[EvidenceSpan]:
        chunks = _record_get(record, "chunks", []) or []
        docs = _record_get(record, "documents", []) or []
        doc_by_id = {_node_prop(doc, "document_id"): doc for doc in docs if _node_prop(doc, "document_id")}
        result: list[EvidenceSpan] = []
        seen = set()
        for chunk in chunks:
            chunk_id = _node_prop(chunk, "chunk_id")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            document_id = _node_prop(chunk, "document_id")
            if not document_id and len(doc_by_id) == 1:
                document_id = next(iter(doc_by_id))
            doc = doc_by_id.get(document_id)
            source_name = _node_prop(chunk, "source_name") or _node_prop(doc, "source_name") or _node_prop(doc, "title")
            result.append(
                EvidenceSpan(
                    source=ExtractionSource(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        source_name=source_name,
                        page=_node_prop(chunk, "page") or _node_prop(chunk, "page_start"),
                    ),
                    quote=str(_node_prop(chunk, "text") or ""),
                    confidence=0.9,
                )
            )
        return result


def _record_get(record: Any, key: str, default: Any = None) -> Any:
    try:
        return record[key]
    except Exception:
        if isinstance(record, dict):
            return record.get(key, default)
        return getattr(record, key, default)


def _map_get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    try:
        return value[key]
    except Exception:
        return getattr(value, key, default)


def _node_prop(node: Any, key: str) -> Any:
    if node is None:
        return None
    if isinstance(node, dict):
        if key in node:
            return node.get(key)
        props = node.get("properties")
        if isinstance(props, dict):
            return props.get(key)
    try:
        return dict(node).get(key)
    except Exception:
        return getattr(node, key, None)


def _first_node_prop(nodes: list[Any], key: str) -> Any:
    for node in nodes or []:
        value = _node_prop(node, key)
        if value:
            return value
    return None


def _first(values: list[Any]) -> Any:
    for value in values:
        if value:
            return value
    return None


def _json_mapping(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}

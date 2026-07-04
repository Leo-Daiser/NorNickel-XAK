"""Idempotent materialization of strict ontology facts into Neo4j."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from ..domain.aliases import MATERIAL_ALIASES, PROPERTY_ALIASES, REGIME_ALIASES
from ..domain.fact_normalization import measurement_normalization_fields, with_normalized_measurement_fields
from ..domain.ontology import DataGap, Evidence
from ..extraction.deterministic import DeterministicExtractor
from ..models.schemas import Chunk, Document
from ..storage.catalog import SQLiteCatalog
from ..extraction.extraction import EntityRelationExtractor
from ..extraction.pipeline import ExtractionPipeline
from ..extraction.to_graph_models import bundle_to_data_gaps, bundle_to_experiment_facts, bundle_to_structured_accepted_experiment_facts
from .graph_db import GraphDB
from .graph_models import ExperimentFact


def deterministic_measurement_id(
    experiment_id: str,
    material: str | None,
    regime: str | None,
    property_name: str,
    value: object,
    unit: str | None,
    source_chunk_id: str | None,
) -> str:
    """Return a stable measurement ID for idempotent Neo4j MERGE writes."""
    raw = "|".join(
        [
            experiment_id or "",
            material or "",
            regime or "",
            property_name or "",
            "" if value is None else str(value),
            unit or "",
            source_chunk_id or "",
        ]
    )
    return "measurement_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]


@dataclass
class GraphWriteStats:
    documents_processed: int = 0
    chunks_processed: int = 0
    tables_processed: int = 0
    accepted_entities: int = 0
    accepted_experiments: int = 0
    accepted_measurements: int = 0
    accepted_gaps: int = 0
    rejected_items: int = 0
    candidate_facts_count: int = 0
    accepted_facts_count: int = 0
    rejected_candidates_count: int = 0
    quarantine_candidates_count: int = 0
    facts_without_evidence: int = 0
    structured_accepted_facts_projected: int = 0
    structured_accepted_facts_skipped_graph_projection: int = 0
    accepted_by_fact_type: Counter = field(default_factory=Counter)
    rejected_by_reason: Counter = field(default_factory=Counter)
    quarantine_by_reason: Counter = field(default_factory=Counter)
    accepted_by_doc_type: Counter = field(default_factory=Counter)
    rejected_by_doc_type: Counter = field(default_factory=Counter)
    quarantine_by_doc_type: Counter = field(default_factory=Counter)
    facts_by_extractor: Counter = field(default_factory=Counter)
    suspicious_entities: Counter = field(default_factory=Counter)
    accepted_entities_counter: Counter = field(default_factory=Counter)
    accepted_properties_counter: Counter = field(default_factory=Counter)
    confidence_values: list[float] = field(default_factory=list)
    documents_written: set[str] = field(default_factory=set)
    chunks_written: set[str] = field(default_factory=set)
    experiments_written: set[str] = field(default_factory=set)
    materials_written: set[str] = field(default_factory=set)
    regimes_written: set[str] = field(default_factory=set)
    properties_written: set[str] = field(default_factory=set)
    measurements_written: set[str] = field(default_factory=set)
    accepted_facts_written: set[str] = field(default_factory=set)
    equipment_written: set[str] = field(default_factory=set)
    facilities_written: set[str] = field(default_factory=set)
    geographies_written: set[str] = field(default_factory=set)
    publications_written: set[str] = field(default_factory=set)
    laboratories_written: set[str] = field(default_factory=set)
    teams_written: set[str] = field(default_factory=set)
    employees_written: set[str] = field(default_factory=set)
    topic_tags_written: set[str] = field(default_factory=set)
    conclusions_written: set[str] = field(default_factory=set)
    gaps_written: set[str] = field(default_factory=set)
    relationships_written: int = 0
    normalized_measurements_backfilled: int = 0

    def to_dict(self) -> dict[str, Any]:
        mean_confidence = 0.0
        if self.confidence_values:
            mean_confidence = round(sum(self.confidence_values) / len(self.confidence_values), 4)
        acceptance_rate = round(self.accepted_facts_count / self.candidate_facts_count, 6) if self.candidate_facts_count else 0.0
        return {
            "documents_processed": self.documents_processed,
            "chunks_processed": self.chunks_processed,
            "tables_processed": self.tables_processed,
            "accepted_entities": self.accepted_entities,
            "accepted_experiments": self.accepted_experiments,
            "accepted_measurements": self.accepted_measurements,
            "accepted_gaps": self.accepted_gaps,
            "rejected_items": self.rejected_items,
            "candidate_facts_count": self.candidate_facts_count,
            "accepted_facts_count": self.accepted_facts_count,
            "rejected_candidates_count": self.rejected_candidates_count,
            "quarantine_candidates_count": self.quarantine_candidates_count,
            "acceptance_rate": acceptance_rate,
            "facts_without_evidence": self.facts_without_evidence,
            "structured_accepted_facts_projected": self.structured_accepted_facts_projected,
            "structured_accepted_facts_skipped_graph_projection": self.structured_accepted_facts_skipped_graph_projection,
            "accepted_by_fact_type": _counter_dict(self.accepted_by_fact_type),
            "rejected_by_reason": _counter_dict(self.rejected_by_reason),
            "quarantine_by_reason": _counter_dict(self.quarantine_by_reason),
            "accepted_by_doc_type": _counter_dict(self.accepted_by_doc_type),
            "rejected_by_doc_type": _counter_dict(self.rejected_by_doc_type),
            "quarantine_by_doc_type": _counter_dict(self.quarantine_by_doc_type),
            "facts_by_extractor": _counter_dict(self.facts_by_extractor),
            "top_suspicious_entities": _counter_dict(self.suspicious_entities, limit=20),
            "top_accepted_entities": _counter_dict(self.accepted_entities_counter, limit=20),
            "top_accepted_properties": _counter_dict(self.accepted_properties_counter, limit=20),
            "mean_confidence": mean_confidence,
            "documents_written": len(self.documents_written),
            "chunks_written": len(self.chunks_written),
            "experiments_written": len(self.experiments_written),
            "materials_written": len(self.materials_written),
            "regimes_written": len(self.regimes_written),
            "properties_written": len(self.properties_written),
            "measurements_written": len(self.measurements_written),
            "accepted_facts_written": len(self.accepted_facts_written),
            "equipment_written": len(self.equipment_written),
            "facilities_written": len(self.facilities_written),
            "geographies_written": len(self.geographies_written),
            "publications_written": len(self.publications_written),
            "laboratories_written": len(self.laboratories_written),
            "teams_written": len(self.teams_written),
            "employees_written": len(self.employees_written),
            "topic_tags_written": len(self.topic_tags_written),
            "conclusions_written": len(self.conclusions_written),
            "gaps_written": len(self.gaps_written),
            "relationships_written": self.relationships_written,
            "normalized_measurements_backfilled": self.normalized_measurements_backfilled,
        }


class GraphWriter:
    """Write strict ontology facts to Neo4j with idempotent MERGE statements."""

    def __init__(self, graph_db: GraphDB, pipeline: ExtractionPipeline | None = None) -> None:
        self.graph_db = graph_db
        self.pipeline = pipeline

    def sync_catalog(
        self,
        catalog: SQLiteCatalog,
        extractor: EntityRelationExtractor | None = None,
        document_getter: Callable[[str], Document | None] | None = None,
        pipeline: ExtractionPipeline | None = None,
    ) -> dict[str, Any]:
        """Run structured extraction over the catalog and materialize accepted facts."""
        _ = document_getter  # kept for backward-compatible call sites
        active_pipeline = pipeline or self.pipeline
        if active_pipeline is None:
            deterministic = DeterministicExtractor(extractor) if extractor is not None else None
            active_pipeline = ExtractionPipeline(deterministic_extractor=deterministic)
        stats = GraphWriteStats()
        with self.graph_db.session() as session:
            for document in catalog.list_documents():
                active = catalog.is_document_active(document.doc_id) if hasattr(catalog, "is_document_active") else True
                self.write_document(session, document, stats, active=active)
                if not active:
                    self.mark_document_chunks_active(session, document.doc_id, active=False)
                    continue
                stats.documents_processed += 1
                self.mark_document_chunks_active(session, document.doc_id, active=False)
                for chunk in catalog.list_chunks(document.doc_id):
                    stats.chunks_processed += 1
                    if chunk.metadata.get("chunk_kind") == "table_row":
                        stats.tables_processed += 1
                    self.write_chunk(session, document, chunk, stats, active=True)
                    bundle = active_pipeline.extract_from_chunk(chunk)
                    self.write_bundle(session, bundle, stats)
            self.backfill_normalized_measurements(session, stats)
        return stats.to_dict()

    def backfill_normalized_measurements(self, session, stats: GraphWriteStats) -> None:
        """Populate normalized fields on legacy Measurement nodes without deleting data."""

        rows = list(
            session.run(
                """
                MATCH (meas:Measurement)-[:OF_PROPERTY]->(p:Property)
                WHERE meas.value IS NOT NULL
                  AND (
                    meas.value_original IS NULL OR
                    meas.unit_original IS NULL OR
                    meas.value_normalized IS NULL OR
                    meas.unit_normalized IS NULL OR
                    meas.normalization_family IS NULL
                  )
                RETURN meas.measurement_id AS measurement_id,
                       meas.value AS value,
                       meas.raw_value AS raw_value,
                       meas.unit AS unit,
                       p.canonical_name AS property
                """
            )
        )
        for row in rows:
            measurement_id = _record_get(row, "measurement_id")
            if not measurement_id:
                continue
            value = _record_get(row, "value")
            raw_value = _record_get(row, "raw_value")
            fields = measurement_normalization_fields(
                _record_get(row, "property"),
                value if value is not None else raw_value,
                _record_get(row, "unit"),
            )
            session.run(
                """
                MATCH (meas:Measurement {measurement_id: $measurement_id})
                SET meas.value_original = $value_original,
                    meas.unit_original = $unit_original,
                    meas.value_normalized = $value_normalized,
                    meas.unit_normalized = $unit_normalized,
                    meas.normalization_family = $normalization_family
                """,
                measurement_id=measurement_id,
                **fields,
            )
            stats.normalized_measurements_backfilled += 1

    def write_bundle(self, session, bundle, stats: GraphWriteStats) -> None:
        """Write accepted extraction bundle facts. Rejected items are intentionally skipped."""
        document_type = (bundle.diagnostics.get("document_profile") or {}).get("detected_type") or "unknown"
        stats.accepted_entities += len(bundle.entities)
        stats.accepted_experiments += len(bundle.experiments)
        stats.accepted_gaps += len(bundle.data_gaps)
        stats.rejected_items += len(bundle.rejected_items)
        stats.candidate_facts_count += len(getattr(bundle, "candidate_facts", []) or [])
        stats.accepted_facts_count += len(getattr(bundle, "accepted_facts", []) or [])
        stats.rejected_candidates_count += len(bundle.rejected_items)
        stats.quarantine_candidates_count += len(getattr(bundle, "quarantined_items", []) or [])
        for accepted in getattr(bundle, "accepted_facts", []) or []:
            stats.accepted_by_fact_type[accepted.fact_type] += 1
            stats.accepted_by_doc_type[document_type] += 1
            _count_accepted_fact_observability(accepted, stats)
            if not accepted.evidence:
                stats.facts_without_evidence += 1
            else:
                self.write_accepted_fact_node(session, accepted, stats)
        for candidate in getattr(bundle, "candidate_facts", []) or []:
            stats.facts_by_extractor[candidate.extractor_name] += 1
        for item in bundle.rejected_items:
            stats.rejected_by_reason[item.reason] += 1
            stats.rejected_by_doc_type[document_type] += 1
            suspicious = _candidate_name_from_payload(item.raw_payload)
            if suspicious:
                stats.suspicious_entities[str(suspicious)] += 1
        for item in getattr(bundle, "quarantined_items", []) or []:
            for reason in item.reasons:
                stats.quarantine_by_reason[reason] += 1
            stats.quarantine_by_doc_type[item.candidate.document_type or document_type] += 1
            suspicious = _candidate_name_from_payload(item.candidate.subject) or item.candidate.raw_span
            if suspicious:
                stats.suspicious_entities[str(suspicious)] += 1
        for entity in bundle.entities:
            stats.accepted_entities_counter[f"{entity.entity_type}:{entity.canonical_name}"] += 1
        for experiment in bundle.experiments:
            stats.accepted_measurements += len(experiment.measurements)
            stats.confidence_values.append(float(experiment.confidence))
            stats.confidence_values.extend(float(item.confidence) for item in experiment.measurements)
            for measurement in experiment.measurements:
                stats.accepted_properties_counter[measurement.property_canonical] += 1
        for gap in bundle.data_gaps:
            stats.confidence_values.append(float(gap.confidence))
        for experiment_fact in bundle_to_experiment_facts(bundle):
            self.write_experiment(session, experiment_fact, stats)
        structured_accepted_count = sum(
            1
            for accepted in getattr(bundle, "accepted_facts", []) or []
            if _is_structured_adapter_fact(accepted)
        )
        structured_experiment_facts = bundle_to_structured_accepted_experiment_facts(bundle)
        stats.structured_accepted_facts_projected += len(structured_experiment_facts)
        stats.structured_accepted_facts_skipped_graph_projection += max(0, structured_accepted_count - len(structured_experiment_facts))
        for experiment_fact in structured_experiment_facts:
            self.write_experiment(session, experiment_fact, stats)
        for gap in bundle_to_data_gaps(bundle):
            self.write_gap(session, gap, stats)

    def write_accepted_fact_node(self, session, accepted, stats: GraphWriteStats) -> None:
        """Write a first-class typed AcceptedFact node and its evidence links."""
        normalized = getattr(accepted, "normalized_fact", {}) or {}
        if not isinstance(normalized, dict) or not getattr(accepted, "evidence", None):
            return
        subject = normalized.get("subject") if isinstance(normalized.get("subject"), dict) else {}
        obj = normalized.get("object") if isinstance(normalized.get("object"), dict) else {}
        evidence = accepted.evidence[0]
        fact_id = _stable_id("accepted_fact", accepted.candidate_id)
        evidence_hash = _stable_id(
            "evidence",
            getattr(evidence.source, "document_id", None),
            getattr(evidence.source, "chunk_id", None),
            evidence.quote,
        )
        session.run(
            """
            MERGE (f:AcceptedFact {fact_id: $fact_id})
            SET f.candidate_id = $candidate_id,
                f.fact_type = $fact_type,
                f.predicate = $predicate,
                f.value = $value,
                f.unit = $unit,
                f.confidence = $confidence,
                f.validation_status = 'accepted',
                f.source_adapter = $source_adapter,
                f.document_type = $document_type,
                f.source_doc_id = $source_doc_id,
                f.source_chunk_id = $source_chunk_id,
                f.evidence_hash = $evidence_hash,
                f.subject_json = $subject_json,
                f.object_json = $object_json,
                f.updated_at = datetime()
            """,
            fact_id=fact_id,
            candidate_id=accepted.candidate_id,
            fact_type=accepted.fact_type,
            predicate=normalized.get("predicate"),
            value=normalized.get("value"),
            unit=normalized.get("unit"),
            confidence=accepted.score,
            source_adapter=normalized.get("source_adapter"),
            document_type=normalized.get("document_type"),
            source_doc_id=getattr(evidence.source, "document_id", None),
            source_chunk_id=getattr(evidence.source, "chunk_id", None),
            evidence_hash=evidence_hash,
            subject_json=json.dumps(subject, ensure_ascii=False, sort_keys=True),
            object_json=json.dumps(obj, ensure_ascii=False, sort_keys=True),
        )
        stats.accepted_facts_written.add(fact_id)
        for evidence_span in accepted.evidence:
            evidence_obj = _evidence_from_span(evidence_span)
            self._write_evidence_chunk(session, evidence_obj, stats)
            if evidence_obj.chunk_id:
                session.run(
                    """
                    MATCH (f:AcceptedFact {fact_id: $fact_id})
                    MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                    MERGE (f)-[:SUPPORTED_BY]->(c)
                    """,
                    fact_id=fact_id,
                    chunk_id=evidence_obj.chunk_id,
                )
                stats.relationships_written += 1

        self._link_fact_material(session, fact_id, subject.get("material") or subject.get("material_raw"), stats)
        self._link_fact_process(session, fact_id, subject.get("process") or obj.get("process"), stats)
        self._link_fact_property(session, fact_id, obj.get("property") or obj.get("parameter") or obj.get("topic"), stats)
        self._link_fact_equipment(session, fact_id, subject.get("equipment"), stats)
        self._link_fact_facility(session, fact_id, subject.get("facility"), stats)
        self._link_fact_geography(session, fact_id, subject.get("geography") or obj.get("geography"), stats)
        self._link_fact_publication(session, fact_id, obj.get("source_note") or obj.get("publication") or obj.get("doi") or obj.get("url"), stats)
        self._link_fact_expertise(session, fact_id, subject, stats)

    def _link_fact_material(self, session, fact_id: str, value: Any, stats: GraphWriteStats) -> None:
        material = _clean_optional(value)
        if not material:
            return
        session.run(
            """
            MATCH (f:AcceptedFact {fact_id: $fact_id})
            MERGE (m:Material {canonical_name: $material})
            SET m.aliases = $aliases
            MERGE (f)-[:FACT_SUBJECT]->(m)
            MERGE (f)-[:USES_MATERIAL]->(m)
            """,
            fact_id=fact_id,
            material=material,
            aliases=_aliases_for(material, MATERIAL_ALIASES),
        )
        stats.materials_written.add(material)
        stats.relationships_written += 2

    def _link_fact_process(self, session, fact_id: str, value: Any, stats: GraphWriteStats) -> None:
        process = _clean_optional(value)
        if not process:
            return
        session.run(
            """
            MATCH (f:AcceptedFact {fact_id: $fact_id})
            MERGE (p:ProcessRegime {canonical_name: $process})
            SET p.aliases = $aliases
            MERGE (f)-[:FACT_PROCESS]->(p)
            """,
            fact_id=fact_id,
            process=process,
            aliases=_aliases_for(process, REGIME_ALIASES),
        )
        stats.regimes_written.add(process)
        stats.relationships_written += 1

    def _link_fact_property(self, session, fact_id: str, value: Any, stats: GraphWriteStats) -> None:
        prop = _clean_optional(value)
        if not prop:
            return
        session.run(
            """
            MATCH (f:AcceptedFact {fact_id: $fact_id})
            MERGE (p:Property {canonical_name: $property})
            SET p.aliases = $aliases
            MERGE (f)-[:FACT_PARAMETER]->(p)
            """,
            fact_id=fact_id,
            property=prop,
            aliases=_aliases_for(prop, PROPERTY_ALIASES),
        )
        stats.properties_written.add(prop)
        stats.relationships_written += 1

    def _link_fact_equipment(self, session, fact_id: str, value: Any, stats: GraphWriteStats) -> None:
        equipment = _clean_optional(value)
        if not equipment:
            return
        session.run(
            """
            MATCH (f:AcceptedFact {fact_id: $fact_id})
            MERGE (eq:Equipment {canonical_name: $equipment})
            MERGE (f)-[:USES_EQUIPMENT]->(eq)
            """,
            fact_id=fact_id,
            equipment=equipment,
        )
        stats.equipment_written.add(equipment)
        stats.relationships_written += 1

    def _link_fact_facility(self, session, fact_id: str, value: Any, stats: GraphWriteStats) -> None:
        facility = _clean_optional(value)
        if not facility:
            return
        session.run(
            """
            MATCH (f:AcceptedFact {fact_id: $fact_id})
            MERGE (facility:Facility {canonical_name: $facility})
            MERGE (f)-[:FACT_SUBJECT]->(facility)
            """,
            fact_id=fact_id,
            facility=facility,
        )
        stats.facilities_written.add(facility)
        stats.relationships_written += 1

    def _link_fact_geography(self, session, fact_id: str, value: Any, stats: GraphWriteStats) -> None:
        geography = _clean_optional(value)
        if not geography:
            return
        session.run(
            """
            MATCH (f:AcceptedFact {fact_id: $fact_id})
            MERGE (g:Geography {canonical_name: $geography})
            MERGE (f)-[:HAS_GEOGRAPHY]->(g)
            """,
            fact_id=fact_id,
            geography=geography,
        )
        stats.geographies_written.add(geography)
        stats.relationships_written += 1

    def _link_fact_publication(self, session, fact_id: str, value: Any, stats: GraphWriteStats) -> None:
        source = _clean_optional(value)
        if not source:
            return
        publication_id = _stable_id("publication", source)
        session.run(
            """
            MATCH (f:AcceptedFact {fact_id: $fact_id})
            MERGE (p:Publication {publication_id: $publication_id})
            SET p.title = coalesce(p.title, $source)
            MERGE (f)-[:DESCRIBED_IN]->(p)
            """,
            fact_id=fact_id,
            publication_id=publication_id,
            source=source,
        )
        stats.publications_written.add(publication_id)
        stats.relationships_written += 1

    def _link_fact_expertise(self, session, fact_id: str, subject: dict[str, Any], stats: GraphWriteStats) -> None:
        expert = _clean_optional(subject.get("expert") or subject.get("expert_or_lab"))
        laboratory = _clean_optional(subject.get("laboratory"))
        team = _clean_optional(subject.get("team"))
        if expert:
            session.run(
                """
                MATCH (f:AcceptedFact {fact_id: $fact_id})
                MERGE (e:Employee {canonical_name: $expert})
                MERGE (f)-[:HAS_EXPERT]->(e)
                """,
                fact_id=fact_id,
                expert=expert,
            )
            stats.employees_written.add(expert)
            stats.relationships_written += 1
        if laboratory:
            session.run(
                """
                MATCH (f:AcceptedFact {fact_id: $fact_id})
                MERGE (lab:Laboratory {canonical_name: $laboratory})
                MERGE (f)-[:PERFORMED_AT]->(lab)
                """,
                fact_id=fact_id,
                laboratory=laboratory,
            )
            stats.laboratories_written.add(laboratory)
            stats.relationships_written += 1
        if team:
            session.run(
                """
                MATCH (f:AcceptedFact {fact_id: $fact_id})
                MERGE (team:ResearchTeam {canonical_name: $team})
                MERGE (f)-[:PERFORMED_BY]->(team)
                """,
                fact_id=fact_id,
                team=team,
            )
            stats.teams_written.add(team)
            stats.relationships_written += 1

    def write_document(self, session, document: Document, stats: GraphWriteStats, *, active: bool = True) -> None:
        query = """
        MERGE (d:Document {document_id: $document_id})
        SET d.source_name = $source_name,
            d.title = $title,
            d.parser = $parser,
            d.status = $status,
            d.version = $version,
            d.active = $active,
            d.updated_at = datetime()
        """
        session.run(
            query,
            document_id=document.doc_id,
            source_name=document.title,
            title=document.title,
            parser=document.parser,
            status=document.status,
            version=document.version,
            active=bool(active),
        )
        stats.documents_written.add(document.doc_id)

    def mark_document_chunks_active(self, session, document_id: str, *, active: bool) -> None:
        session.run(
            """
            MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:DocumentChunk)
            SET c.active = $active
            """,
            document_id=document_id,
            active=bool(active),
        )

    def write_chunk(self, session, document: Document, chunk: Chunk, stats: GraphWriteStats, *, active: bool = True) -> None:
        query = """
        MERGE (d:Document {document_id: $document_id})
        SET d.source_name = coalesce(d.source_name, $source_name),
            d.title = coalesce(d.title, $source_name),
            d.active = $active,
            d.updated_at = datetime()
        MERGE (c:DocumentChunk {chunk_id: $chunk_id})
        SET c.text = $text,
            c.text_hash = $text_hash,
            c.page = $page,
            c.page_start = $page_start,
            c.page_end = $page_end,
            c.section_path = $section_path,
            c.source_name = $source_name,
            c.document_id = $document_id,
            c.active = $active,
            c.updated_at = datetime()
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        session.run(
            query,
            document_id=document.doc_id,
            source_name=document.title,
            active=bool(active),
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            text_hash=chunk.text_hash,
            page=chunk.page_start,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_path=chunk.section_path,
        )
        stats.chunks_written.add(chunk.chunk_id)
        stats.relationships_written += 1

    def write_experiment(self, session, experiment: ExperimentFact, stats: GraphWriteStats) -> None:
        session.run(
            """
            MERGE (e:Experiment {experiment_id: $experiment_id})
                SET e.updated_at = datetime(),
                e.validation_status = 'accepted',
                e.fact_type = coalesce($fact_type, e.fact_type),
                e.source_adapter = coalesce($source_adapter, e.source_adapter)
            """,
            experiment_id=experiment.experiment_id,
            fact_type=_first_measurement_attr(experiment, "fact_type"),
            source_adapter=_first_measurement_attr(experiment, "source_adapter"),
        )
        stats.experiments_written.add(experiment.experiment_id)

        for evidence in experiment.evidence:
            self._write_evidence_chunk(session, evidence, stats)
            if evidence.chunk_id:
                session.run(
                    """
                    MATCH (e:Experiment {experiment_id: $experiment_id})
                    MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                    MERGE (e)-[:SUPPORTED_BY]->(c)
                    """,
                    experiment_id=experiment.experiment_id,
                    chunk_id=evidence.chunk_id,
                )
                stats.relationships_written += 1

        for material in experiment.materials:
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (m:Material {canonical_name: $material})
                SET m.aliases = $aliases
                MERGE (e)-[:USES_MATERIAL]->(m)
                """,
                experiment_id=experiment.experiment_id,
                material=material,
                aliases=_aliases_for(material, MATERIAL_ALIASES),
            )
            stats.materials_written.add(material)
            stats.relationships_written += 1

        for regime in experiment.regimes:
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (r:ProcessRegime {canonical_name: $regime})
                SET r.temperature = $temperature,
                    r.temperature_unit = $temperature_unit,
                    r.duration = $duration,
                    r.duration_unit = $duration_unit,
                    r.medium = $medium
                MERGE (e)-[:HAS_REGIME]->(r)
                """,
                experiment_id=experiment.experiment_id,
                regime=regime,
                temperature=None,
                temperature_unit=None,
                duration=None,
                duration_unit=None,
                medium=None,
            )
            stats.regimes_written.add(regime)
            stats.relationships_written += 1

        for raw_measurement in experiment.measurements:
            measurement = with_normalized_measurement_fields(raw_measurement)
            material = experiment.materials[0] if experiment.materials else None
            regime = experiment.regimes[0] if experiment.regimes else None
            source_chunk_id = None
            if measurement.evidence:
                source_chunk_id = measurement.evidence[0].chunk_id
            elif experiment.evidence:
                source_chunk_id = experiment.evidence[0].chunk_id
            measurement_id = deterministic_measurement_id(
                experiment.experiment_id,
                material,
                regime,
                measurement.property_name,
                measurement.value_normalized if measurement.value_normalized is not None else measurement.value if measurement.value is not None else measurement.raw_value,
                measurement.unit_normalized or measurement.unit,
                source_chunk_id,
            )
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (p:Property {canonical_name: $property})
                SET p.aliases = $property_aliases,
                    p.unit_family = $unit_family
                MERGE (meas:Measurement {measurement_id: $measurement_id})
                SET meas.value = $value,
                    meas.value_min = $value_min,
                    meas.value_max = $value_max,
                    meas.raw_value = $raw_value,
                    meas.unit = $unit,
                    meas.value_original = $value_original,
                    meas.unit_original = $unit_original,
                    meas.value_normalized = $value_normalized,
                    meas.unit_normalized = $unit_normalized,
                    meas.normalization_family = $normalization_family,
                    meas.effect = $effect,
                    meas.baseline_value = $baseline_value,
                    meas.delta_abs = $delta_abs,
                    meas.delta_rel_percent = $delta_rel_percent,
                    meas.confidence = $confidence,
                    meas.analyte = $analyte,
                    meas.fact_type = $fact_type,
                    meas.source_adapter = $source_adapter,
                    meas.validation_status = 'accepted'
                MERGE (e)-[:MEASURED]->(meas)
                MERGE (meas)-[:OF_PROPERTY]->(p)
                """,
                experiment_id=experiment.experiment_id,
                property=measurement.property_name,
                property_aliases=_aliases_for(measurement.property_name, PROPERTY_ALIASES),
                unit_family=None,
                measurement_id=measurement_id,
                value=measurement.value,
                value_min=measurement.value_min,
                value_max=measurement.value_max,
                raw_value=measurement.raw_value,
                unit=measurement.unit,
                value_original=measurement.value_original,
                unit_original=measurement.unit_original,
                value_normalized=measurement.value_normalized,
                unit_normalized=measurement.unit_normalized,
                normalization_family=measurement.normalization_family,
                effect=measurement.effect,
                baseline_value=measurement.baseline_value,
                delta_abs=measurement.delta_abs,
                delta_rel_percent=measurement.delta_rel_percent,
                confidence=measurement.confidence,
                analyte=measurement.analyte,
                fact_type=measurement.fact_type,
                source_adapter=measurement.source_adapter,
            )
            stats.properties_written.add(measurement.property_name)
            stats.measurements_written.add(measurement_id)
            stats.relationships_written += 2
            for evidence in measurement.evidence:
                self._write_evidence_chunk(session, evidence, stats)
                if evidence.chunk_id:
                    session.run(
                        """
                        MATCH (meas:Measurement {measurement_id: $measurement_id})
                        MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                        MERGE (meas)-[:SUPPORTED_BY]->(c)
                        """,
                        measurement_id=measurement_id,
                        chunk_id=evidence.chunk_id,
                    )
                    stats.relationships_written += 1

        for equipment in experiment.equipment:
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (eq:Equipment {canonical_name: $equipment})
                MERGE (e)-[:USED_EQUIPMENT]->(eq)
                """,
                experiment_id=experiment.experiment_id,
                equipment=equipment,
            )
            stats.equipment_written.add(equipment)
            stats.relationships_written += 1

        for laboratory in experiment.laboratories:
            team = laboratory
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (lab:Laboratory {canonical_name: $laboratory})
                MERGE (team:ResearchTeam {canonical_name: $team})
                MERGE (team)-[:BELONGS_TO]->(lab)
                MERGE (e)-[:PERFORMED_BY]->(team)
                MERGE (e)-[:PERFORMED_AT]->(lab)
                """,
                experiment_id=experiment.experiment_id,
                laboratory=laboratory,
                team=team,
            )
            stats.laboratories_written.add(laboratory)
            stats.teams_written.add(team)
            stats.relationships_written += 3

        for team in experiment.teams:
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (team:ResearchTeam {canonical_name: $team})
                MERGE (e)-[:PERFORMED_BY]->(team)
                """,
                experiment_id=experiment.experiment_id,
                team=team,
            )
            stats.teams_written.add(team)
            stats.relationships_written += 1

        for employee in experiment.employees:
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (employee:Employee {canonical_name: $employee})
                MERGE (e)-[:HAS_EXECUTOR]->(employee)
                """,
                experiment_id=experiment.experiment_id,
                employee=employee,
            )
            stats.employees_written.add(employee)
            stats.relationships_written += 1

        for topic in experiment.topic_tags:
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (topic:TopicTag {canonical_name: $topic})
                MERGE (e)-[:TAGGED_WITH]->(topic)
                """,
                experiment_id=experiment.experiment_id,
                topic=topic,
            )
            stats.topic_tags_written.add(topic)
            stats.relationships_written += 1

        for conclusion in experiment.conclusions:
            conclusion_id = _stable_id("conclusion", experiment.experiment_id, conclusion)
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (concl:Conclusion {conclusion_id: $conclusion_id})
                SET concl.text = $text
                MERGE (e)-[:LED_TO]->(concl)
                """,
                experiment_id=experiment.experiment_id,
                conclusion_id=conclusion_id,
                text=conclusion,
            )
            stats.conclusions_written.add(conclusion_id)
            stats.relationships_written += 1

    def write_gap(self, session, gap: DataGap, stats: GraphWriteStats) -> None:
        for evidence in gap.evidence:
            self._write_evidence_chunk(session, evidence, stats)
        session.run(
            """
            MERGE (g:DataGap {gap_id: $gap_id})
            SET g.material = $material,
                g.regime = $regime,
                g.property = $property,
                g.reason = $reason,
                g.validation_status = 'accepted',
                g.updated_at = datetime()
            """,
            gap_id=gap.gap_id,
            material=gap.material,
            regime=gap.regime,
            property=gap.property,
            reason=gap.reason,
        )
        stats.gaps_written.add(gap.gap_id)
        if gap.material:
            session.run(
                """
                MATCH (g:DataGap {gap_id: $gap_id})
                MERGE (m:Material {canonical_name: $material})
                SET m.aliases = $aliases
                MERGE (g)-[:GAP_FOR_ENTITY]->(m)
                """,
                gap_id=gap.gap_id,
                material=gap.material,
                aliases=_aliases_for(gap.material, MATERIAL_ALIASES),
            )
            stats.materials_written.add(gap.material)
            stats.relationships_written += 1
        if gap.regime:
            session.run(
                """
                MATCH (g:DataGap {gap_id: $gap_id})
                MERGE (r:ProcessRegime {canonical_name: $regime})
                SET r.aliases = $aliases
                MERGE (g)-[:GAP_FOR_REGIME]->(r)
                """,
                gap_id=gap.gap_id,
                regime=gap.regime,
                aliases=_aliases_for(gap.regime, REGIME_ALIASES),
            )
            stats.regimes_written.add(gap.regime)
            stats.relationships_written += 1
        if gap.property:
            session.run(
                """
                MATCH (g:DataGap {gap_id: $gap_id})
                MERGE (p:Property {canonical_name: $property})
                SET p.aliases = $aliases
                MERGE (g)-[:GAP_FOR_PROPERTY]->(p)
                """,
                gap_id=gap.gap_id,
                property=gap.property,
                aliases=_aliases_for(gap.property, PROPERTY_ALIASES),
            )
            stats.properties_written.add(gap.property)
            stats.relationships_written += 1
        for evidence in gap.evidence:
            if evidence.chunk_id:
                session.run(
                    """
                    MATCH (g:DataGap {gap_id: $gap_id})
                    MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                    MERGE (g)-[:SUPPORTED_BY]->(c)
                    """,
                    gap_id=gap.gap_id,
                    chunk_id=evidence.chunk_id,
                )
                stats.relationships_written += 1

    def _write_evidence_chunk(self, session, evidence: Evidence, stats: GraphWriteStats) -> None:
        if not evidence.chunk_id:
            return
        session.run(
            """
            MERGE (d:Document {document_id: $document_id})
            SET d.source_name = coalesce(d.source_name, $source_name),
                d.title = coalesce(d.title, $source_name),
                d.updated_at = datetime()
            MERGE (c:DocumentChunk {chunk_id: $chunk_id})
            SET c.text = coalesce($quote, c.text),
                c.text_hash = coalesce(c.text_hash, $text_hash),
                c.page = $page,
                c.source_name = $source_name,
                c.document_id = $document_id,
                c.active = coalesce(c.active, true),
                c.updated_at = datetime()
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            document_id=evidence.document_id,
            source_name=evidence.source_name,
            chunk_id=evidence.chunk_id,
            quote=evidence.quote,
            text_hash=_stable_id("chunk_text", evidence.quote or ""),
            page=evidence.page,
        )
        if evidence.document_id:
            stats.documents_written.add(evidence.document_id)
        stats.chunks_written.add(evidence.chunk_id)
        stats.relationships_written += 1


def sync_catalog_to_neo4j(
    graph_db: GraphDB,
    catalog: SQLiteCatalog,
    extractor: EntityRelationExtractor | None = None,
    document_getter: Callable[[str], Document | None] | None = None,
    pipeline: ExtractionPipeline | None = None,
) -> dict[str, Any]:
    """Convenience wrapper used by API and CLI scripts."""
    return GraphWriter(graph_db, pipeline=pipeline).sync_catalog(
        catalog=catalog,
        extractor=extractor,
        document_getter=document_getter,
        pipeline=pipeline,
    )


def _aliases_for(canonical: str, aliases: dict[str, str]) -> list[str]:
    values = [alias for alias, value in aliases.items() if value == canonical]
    return list(dict.fromkeys([canonical, *values]))


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:32]}"


def _counter_dict(counter: Counter, limit: int = 50) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common(limit)}


def _candidate_name_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return str(payload)[:120] if payload else None
    for key in ("canonical_name", "raw_name", "property_canonical", "property_raw", "name", "material"):
        value = payload.get(key)
        if value:
            return str(value)[:120]
    materials = payload.get("materials")
    if isinstance(materials, list) and materials:
        first = materials[0]
        if isinstance(first, dict):
            return _candidate_name_from_payload(first)
        return str(first)[:120]
    measurements = payload.get("measurements")
    if isinstance(measurements, list) and measurements:
        first = measurements[0]
        if isinstance(first, dict):
            return _candidate_name_from_payload(first)
    subject = payload.get("subject")
    if isinstance(subject, dict):
        return _candidate_name_from_payload(subject)
    return None


def _record_get(record, key: str):
    try:
        return record[key]
    except Exception:
        if isinstance(record, dict):
            return record.get(key)
        return getattr(record, key, None)


def _first_measurement_attr(experiment: ExperimentFact, attr: str) -> Any:
    for measurement in experiment.measurements:
        value = getattr(measurement, attr, None)
        if value:
            return value
    return None


def _is_structured_adapter_fact(accepted: Any) -> bool:
    normalized = getattr(accepted, "normalized_fact", {}) or {}
    return isinstance(normalized, dict) and bool(normalized.get("source_adapter"))


def _evidence_from_span(span: Any) -> Evidence:
    source = getattr(span, "source", None)
    return Evidence(
        document_id=getattr(source, "document_id", None),
        chunk_id=getattr(source, "chunk_id", None),
        source_name=getattr(source, "source_name", None),
        page=getattr(source, "page", None),
        quote=getattr(span, "quote", None),
        confidence=getattr(span, "confidence", None),
    )


def _clean_optional(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"none", "null", "unknown"} else text


def _count_accepted_fact_observability(accepted: Any, stats: GraphWriteStats) -> None:
    normalized = getattr(accepted, "normalized_fact", {}) or {}
    if not isinstance(normalized, dict):
        return
    subject = normalized.get("subject") or {}
    obj = normalized.get("object") or {}
    if isinstance(subject, dict):
        for key in ("material", "material_raw", "process", "facility", "geography"):
            value = subject.get(key)
            if value:
                stats.accepted_entities_counter[f"{key}:{value}"] += 1
    if isinstance(obj, dict):
        prop = obj.get("property") or obj.get("parameter")
        if prop:
            stats.accepted_properties_counter[str(prop)] += 1
        analyte = obj.get("analyte")
        if analyte:
            stats.accepted_entities_counter[f"analyte:{analyte}"] += 1

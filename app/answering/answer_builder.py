"""Answer builder for strict ontology-driven graph QA."""

from __future__ import annotations

from typing import Any

from ..domain.fact_normalization import (
    build_conflict_summary,
    canonical_fact_key_from_row,
    dedupe_fact_rows,
    with_normalized_measurement_fields,
)
from ..domain.ontology import DataGap, Evidence
from ..domain.query_constraints import QueryConstraints
from ..graph.graph_models import DecisionHistoryItem, ExperimentFact, PartialMatches
from .decision_history import build_decision_history_payload
from .gap_analyzer import filter_gaps
from .no_match import build_no_match_payload


class AnswerBuilder:
    """Build API-compatible responses from structured graph facts."""

    def exact_match_response(
        self,
        constraints: QueryConstraints,
        facts: list[ExperimentFact],
        partial_matches: PartialMatches | None,
        gaps: list[DataGap] | None,
        retrieval: dict[str, Any],
        llm: dict[str, Any],
    ) -> dict[str, Any]:
        fact_rows = self._fact_rows(facts, constraints)
        sources = self._sources_from_experiments(facts)
        subgraph = self._subgraph_from_experiments(facts)
        data_gaps = [gap.model_dump() for gap in filter_gaps(gaps or [], constraints)]
        response = self._base_response(
            answer=self._exact_answer_text(constraints, facts),
            status="ok",
            answer_mode="graph_exact",
            intent=constraints.intent.value,
            constraints=constraints,
            retrieval=retrieval,
            llm=llm,
        )
        response.update(
            {
                "facts": fact_rows,
                "experiments": [fact.summary() for fact in facts],
                "materials": [{"name": material} for material in self._unique(v for fact in facts for v in fact.materials)],
                "equipment": [{"name": item} for item in self._unique(v for fact in facts for v in fact.equipment)],
                "laboratories": [{"name": item} for item in self._unique(v for fact in facts for v in fact.laboratories)],
                "teams": [{"name": item} for item in self._unique(v for fact in facts for v in fact.teams)],
                "employees": [{"name": item} for item in self._unique(v for fact in facts for v in fact.employees)],
                "topic_tags": [{"name": item} for item in self._unique(v for fact in facts for v in fact.topic_tags)],
                "sources": sources,
                "gaps": data_gaps,
                "data_gaps": data_gaps,
                "subgraph": subgraph,
                "partial_matches": partial_matches.to_response() if partial_matches else {},
                "graph_context": _graph_context_stats(fact_rows, sources, subgraph),
            }
        )
        response["diagnostics"]["fact_conflicts"] = build_conflict_summary(fact_rows)
        return response

    def no_match_response(
        self,
        constraints: QueryConstraints,
        partial_matches: PartialMatches,
        gaps: list[DataGap] | None,
        retrieval: dict[str, Any],
        llm: dict[str, Any],
    ) -> dict[str, Any]:
        payload = build_no_match_payload(constraints, partial_matches, filter_gaps(gaps or [], constraints))
        response = self._base_response(
            answer=payload["answer"],
            status="no_exact_match",
            answer_mode="graph_no_exact_match",
            intent=constraints.intent.value,
            constraints=constraints,
            retrieval=retrieval,
            llm=llm,
        )
        response.update(payload)
        return response

    def decision_history_response(
        self,
        constraints: QueryConstraints,
        history: list[DecisionHistoryItem],
        retrieval: dict[str, Any],
        llm: dict[str, Any],
    ) -> dict[str, Any]:
        payload = build_decision_history_payload(constraints, history)
        response = self._base_response(
            answer=payload["answer"],
            status=payload["status"],
            answer_mode="graph_decision_history",
            intent=constraints.intent.value,
            constraints=constraints,
            retrieval=retrieval,
            llm=llm,
        )
        response.update(payload)
        return response

    def gap_response(
        self,
        constraints: QueryConstraints,
        gaps: list[DataGap],
        retrieval: dict[str, Any],
        llm: dict[str, Any],
    ) -> dict[str, Any]:
        relevant = filter_gaps(gaps, constraints)
        if relevant:
            material = ", ".join(constraints.materials) if constraints.materials else "заданным ограничениям"
            answer = "Найдены пробелы в данных по " + material + ": " + "; ".join(gap.reason for gap in relevant[:6])
            status = "ok"
        else:
            answer = "По указанным ограничениям явных пробелов в данных не найдено."
            status = "no_exact_match"
        response = self._base_response(
            answer=answer,
            status=status,
            answer_mode="graph_gap_analysis",
            intent=constraints.intent.value,
            constraints=constraints,
            retrieval=retrieval,
            llm=llm,
        )
        response.update(
            {
                "gaps": [gap.model_dump() for gap in relevant],
                "data_gaps": [gap.model_dump() for gap in relevant],
                "sources": self._sources_from_gaps(relevant),
                "subgraph": self._subgraph_from_gaps(relevant),
            }
        )
        response["graph_context"] = _graph_context_stats(
            response["gaps"],
            response["sources"],
            response["subgraph"],
        )
        return response

    def _base_response(
        self,
        answer: str,
        status: str,
        answer_mode: str,
        intent: str,
        constraints: QueryConstraints,
        retrieval: dict[str, Any],
        llm: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "status": status,
            "answer_mode": answer_mode,
            "analytical_intent": "strict_material_regime_property"
            if intent == "material_regime_property_effect"
            else intent,
            "intent": intent,
            "constraints": constraints.model_dump(),
            "partial_matches": {},
            "decision_history": [],
            "data_gaps": [],
            "diagnostics": {
                key: retrieval.get(key)
                for key in ["kg_backend_configured", "kg_backend_active", "neo4j_available", "neo4j_error"]
                if key in retrieval
            },
            "facts": [],
            "experiments": [],
            "technical_objects": [],
            "parts": [],
            "parameters": [],
            "standards": [],
            "materials": [],
            "requirements": [],
            "equipment": [],
            "laboratories": [],
            "sources": [],
            "evidence": [],
            "gaps": [],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {
                "facts_count": 0,
                "sources_count": 0,
                "evidence_count": 0,
                "subgraph_nodes": 0,
                "subgraph_edges": 0,
            },
            "retrieval": retrieval,
            "llm": llm,
        }

    def _exact_answer_text(self, constraints: QueryConstraints, facts: list[ExperimentFact]) -> str:
        material = constraints.materials[0] if constraints.materials else "материал"
        regime = constraints.regimes[0] if constraints.regimes else "режим"
        property_name = constraints.properties[0] if constraints.properties else "свойство"
        fragments: list[str] = []
        for fact in facts[:4]:
            measurements = [
                measurement
                for measurement in fact.measurements
                if not constraints.properties or measurement.property_name in constraints.properties
            ]
            measurement_text = "; ".join(self._measurement_text(item) for item in measurements) or "измерения без численного значения"
            source = fact.evidence[0].source_name if fact.evidence else "источник не указан"
            fragments.append(f"{fact.experiment_id}: {measurement_text}. Источник: {source}.")
        return f"Найдены точные данные по сочетанию {material} + {regime} + {property_name}. " + " ".join(fragments)

    @staticmethod
    def _measurement_text(measurement) -> str:
        label_map = {
            "прочность": "Прочность",
            "твёрдость": "Твёрдость",
            "пластичность": "Пластичность",
            "вязкость": "Вязкость",
            "коррозионная стойкость": "Коррозионная стойкость",
        }
        label = label_map.get(measurement.property_name, measurement.property_name)
        value = measurement.value if measurement.value is not None else measurement.raw_value
        unit = measurement.unit or ""
        effect = f", эффект: {measurement.effect}" if measurement.effect else ""
        return f"{label}: {value} {unit}{effect}".strip()

    def _fact_rows(self, facts: list[ExperimentFact], constraints: QueryConstraints) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fact in facts:
            for measurement in fact.measurements:
                if constraints.properties and measurement.property_name not in constraints.properties:
                    continue
                normalized = with_normalized_measurement_fields(measurement)
                row = {
                    "experiment_id": fact.experiment_id,
                    "material": ", ".join(fact.materials),
                    "regime": ", ".join(fact.regimes),
                    "property": normalized.property_name,
                    "value": normalized.value,
                    "raw_value": normalized.raw_value,
                    "unit": normalized.unit,
                    "value_original": normalized.value_original,
                    "unit_original": normalized.unit_original,
                    "value_normalized": normalized.value_normalized,
                    "unit_normalized": normalized.unit_normalized,
                    "normalization_family": normalized.normalization_family,
                    "effect": normalized.effect,
                    "equipment": ", ".join(fact.equipment),
                    "laboratory": ", ".join(fact.laboratories),
                    "teams": list(fact.teams),
                    "employees": list(fact.employees),
                    "topic_tags": list(fact.topic_tags),
                    "source_chunk_id": fact.source_chunk_ids[0] if fact.source_chunk_ids else None,
                    "doc_id": fact.evidence[0].document_id if fact.evidence else None,
                    "evidence": [evidence.model_dump() for evidence in normalized.evidence or fact.evidence],
                }
                row["canonical_fact_key"] = canonical_fact_key_from_row(row)
                rows.append(row)
        return dedupe_fact_rows(rows)

    def _sources_from_experiments(self, facts: list[ExperimentFact]) -> list[dict[str, Any]]:
        return self._sources_from_evidence([evidence for fact in facts for evidence in fact.evidence])

    def _sources_from_gaps(self, gaps: list[DataGap]) -> list[dict[str, Any]]:
        return self._sources_from_evidence([evidence for gap in gaps for evidence in gap.evidence])

    @staticmethod
    def _sources_from_evidence(items: list[Evidence]) -> list[dict[str, Any]]:
        seen = set()
        sources: list[dict[str, Any]] = []
        for evidence in items:
            key = (evidence.document_id, evidence.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "doc_id": evidence.document_id,
                    "chunk_id": evidence.chunk_id,
                    "title": evidence.source_name,
                    "filename": evidence.source_name,
                    "page_start": evidence.page,
                    "page_end": evidence.page,
                    "quote": evidence.quote,
                }
            )
        return sources

    def _subgraph_from_experiments(self, facts: list[ExperimentFact]) -> dict[str, list[dict[str, Any]]]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        for fact in facts:
            exp_id = f"experiment:{fact.experiment_id}"
            nodes[exp_id] = {"id": exp_id, "label": fact.experiment_id, "type": "Experiment", "properties": {}}
            for material in fact.materials:
                material_id = f"material:{material}"
                nodes[material_id] = {"id": material_id, "label": material, "type": "Material", "properties": {}}
                edges[f"{exp_id}:STUDIES:{material_id}"] = {"id": f"{exp_id}:STUDIES:{material_id}", "source": exp_id, "target": material_id, "label": "STUDIES", "properties": {}}
            for regime in fact.regimes:
                regime_id = f"regime:{regime}"
                nodes[regime_id] = {"id": regime_id, "label": regime, "type": "ProcessRegime", "properties": {}}
                edges[f"{exp_id}:USES_REGIME:{regime_id}"] = {"id": f"{exp_id}:USES_REGIME:{regime_id}", "source": exp_id, "target": regime_id, "label": "USES_REGIME", "properties": {}}
            for measurement in fact.measurements:
                value_id = f"measurement:{fact.experiment_id}:{measurement.property_name}:{measurement.value or measurement.raw_value}"
                nodes[value_id] = {"id": value_id, "label": self._measurement_text(measurement), "type": "PropertyValue", "properties": measurement.model_dump()}
                edges[f"{exp_id}:MEASURES:{value_id}"] = {"id": f"{exp_id}:MEASURES:{value_id}", "source": exp_id, "target": value_id, "label": "MEASURES", "properties": {}}
                prop_id = f"property:{measurement.property_name}"
                nodes[prop_id] = {"id": prop_id, "label": measurement.property_name, "type": "Property", "properties": {}}
                edges[f"{value_id}:OF_PROPERTY:{prop_id}"] = {"id": f"{value_id}:OF_PROPERTY:{prop_id}", "source": value_id, "target": prop_id, "label": "OF_PROPERTY", "properties": {}}
            for evidence in fact.evidence:
                if not evidence.chunk_id:
                    continue
                chunk_id = f"chunk:{evidence.chunk_id}"
                nodes[chunk_id] = {"id": chunk_id, "label": evidence.source_name or evidence.chunk_id, "type": "SourceChunk", "properties": {"quote": evidence.quote}}
                edges[f"{exp_id}:FACT_SUPPORTED_BY_CHUNK:{chunk_id}"] = {"id": f"{exp_id}:FACT_SUPPORTED_BY_CHUNK:{chunk_id}", "source": exp_id, "target": chunk_id, "label": "FACT_SUPPORTED_BY_CHUNK", "properties": {}}
        return {"nodes": list(nodes.values()), "edges": list(edges.values())}

    def _subgraph_from_gaps(self, gaps: list[DataGap]) -> dict[str, list[dict[str, Any]]]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        for gap in gaps:
            gap_id = f"gap:{gap.gap_id}"
            nodes[gap_id] = {"id": gap_id, "label": gap.reason, "type": "DataGap", "properties": gap.model_dump()}
            for label, node_type, value in [
                ("GAP_FOR_ENTITY", "Material", gap.material),
                ("GAP_FOR_REGIME", "ProcessRegime", gap.regime),
                ("GAP_FOR_PROPERTY", "Property", gap.property),
            ]:
                if not value:
                    continue
                node_id = f"{node_type.lower()}:{value}"
                nodes[node_id] = {"id": node_id, "label": value, "type": node_type, "properties": {}}
                edges[f"{gap_id}:{label}:{node_id}"] = {"id": f"{gap_id}:{label}:{node_id}", "source": gap_id, "target": node_id, "label": label, "properties": {}}
        return {"nodes": list(nodes.values()), "edges": list(edges.values())}

    @staticmethod
    def _unique(values) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))


def _graph_context_stats(
    facts: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    subgraph: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    return {
        "facts_count": len(facts),
        "sources_count": len(sources),
        "evidence_count": len(sources),
        "subgraph_nodes": len(subgraph.get("nodes") or []),
        "subgraph_edges": len(subgraph.get("edges") or []),
    }

"""Build compact graph context for analytical answers."""

from __future__ import annotations

from typing import Any

from ..config import settings
from ..domain.fact_normalization import canonical_fact_key_from_row, dedupe_fact_rows, with_normalized_measurement_fields
from ..domain.normalization import material_matches, property_matches, regime_matches
from ..domain.ontology import DataGap, Evidence, Measurement
from ..graph.graph_models import DecisionHistoryItem, ExperimentFact
from .query_models import AnalyticalQueryPlan, EvidenceItem, GraphContext


class GraphContextBuilder:
    """Normalize, deduplicate and limit graph facts for answer synthesis."""

    def __init__(self, max_facts: int | None = None, max_sources: int | None = None, max_nodes: int | None = None, max_edges: int | None = None) -> None:
        self.max_facts = int(max_facts or getattr(settings, "analytics_max_facts", 30))
        self.max_sources = int(max_sources or getattr(settings, "analytics_max_sources", 12))
        self.max_nodes = int(max_nodes or getattr(settings, "analytics_max_graph_nodes", 50))
        self.max_edges = int(max_edges or getattr(settings, "analytics_max_graph_edges", 80))

    def from_experiments(
        self,
        plan: AnalyticalQueryPlan,
        experiments: list[ExperimentFact],
        gaps: list[DataGap] | None = None,
        evidence: list[EvidenceItem] | None = None,
        decision_history: list[DecisionHistoryItem] | None = None,
        partial_matches: dict[str, Any] | None = None,
    ) -> GraphContext:
        filtered = [exp for exp in experiments if self._matches_constraints(exp, plan)]
        facts = dedupe_fact_rows([row for exp in filtered for row in _fact_rows(exp)])
        facts = [row for row in facts if self._row_matches(row, plan)][: self.max_facts]
        sources = self._sources_from_experiments(filtered)
        gap_rows = [gap.model_dump() for gap in (gaps or []) if self._gap_matches(gap, plan)][: self.max_facts]
        subgraph = self._subgraph(filtered, gaps or [], plan)
        entities = self._filter_entities(self._entities(filtered, gaps or []), plan)
        return GraphContext(
            intent=plan.intent,
            constraints=plan.constraints,
            facts=facts,
            grouped_facts=self._grouped_facts(filtered),
            decision_history=[item.model_dump() for item in (decision_history or [])],
            gaps=gap_rows,
            entities=entities,
            sources=sources[: self.max_sources],
            evidence=(evidence or [])[: self.max_sources],
            subgraph=subgraph,
            partial_matches=partial_matches or {},
            diagnostics={
                "max_facts": self.max_facts,
                "max_sources": self.max_sources,
                "input_experiments": len(experiments),
                "filtered_experiments": len(filtered),
            },
        )

    def _matches_constraints(self, exp: ExperimentFact, plan: AnalyticalQueryPlan) -> bool:
        c = plan.constraints
        if c.materials and not any(any(material_matches(value, material) for value in exp.materials) for material in c.materials):
            return False
        if c.regimes and not any(any(regime_matches(value, regime) for value in exp.regimes) for regime in c.regimes):
            return False
        if c.properties and not any(any(property_matches(m.property_name, prop) for m in exp.measurements) for prop in c.properties):
            return False
        return True

    def _row_matches(self, row: dict[str, Any], plan: AnalyticalQueryPlan) -> bool:
        c = plan.constraints
        if c.materials and not any(material_matches(str(row.get("material") or ""), item) for item in c.materials):
            return False
        if c.regimes and not any(regime_matches(str(row.get("regime") or ""), item) for item in c.regimes):
            return False
        if c.properties and not any(property_matches(str(row.get("property") or ""), item) for item in c.properties):
            return False
        return True

    def _gap_matches(self, gap: DataGap, plan: AnalyticalQueryPlan) -> bool:
        c = plan.constraints
        if c.materials and gap.material and not any(material_matches(gap.material, material) for material in c.materials):
            return False
        if c.regimes and gap.regime and not any(regime_matches(gap.regime, regime) for regime in c.regimes):
            return False
        if c.properties and gap.property and not any(property_matches(gap.property, prop) for prop in c.properties):
            return False
        return True

    def _sources_from_experiments(self, experiments: list[ExperimentFact]) -> list[dict[str, Any]]:
        evidence = [item for exp in experiments for item in exp.evidence]
        return _sources_from_evidence(evidence, self.max_sources)

    def _subgraph(self, experiments: list[ExperimentFact], gaps: list[DataGap], plan: AnalyticalQueryPlan) -> dict[str, list[dict[str, Any]]]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}

        def node(node_id: str, label: str, node_type: str, **props) -> None:
            if len(nodes) >= self.max_nodes and node_id not in nodes:
                return
            nodes.setdefault(node_id, {"id": node_id, "label": label, "type": node_type, "properties": props})

        def edge(source: str, label: str, target: str, **props) -> None:
            if len(edges) >= self.max_edges:
                return
            edge_id = f"{source}:{label}:{target}"
            if source in nodes and target in nodes:
                edges.setdefault(edge_id, {"id": edge_id, "source": source, "target": target, "label": label, "type": label, "properties": props})

        for exp in experiments:
            exp_id = f"Experiment:{exp.experiment_id}"
            node(exp_id, exp.experiment_id, "Experiment")
            for material in exp.materials:
                if plan.constraints.materials and not any(material_matches(material, item) for item in plan.constraints.materials):
                    continue
                mid = f"Material:{material}"
                node(mid, material, "Material")
                edge(exp_id, "USES_MATERIAL", mid)
            for regime in exp.regimes:
                if plan.constraints.regimes and not any(regime_matches(regime, item) for item in plan.constraints.regimes):
                    continue
                rid = f"ProcessRegime:{regime}"
                node(rid, regime, "ProcessRegime")
                edge(exp_id, "HAS_REGIME", rid)
            for measurement in exp.measurements:
                if plan.constraints.properties and not any(property_matches(measurement.property_name, item) for item in plan.constraints.properties):
                    continue
                pid = f"Property:{measurement.property_name}"
                mid = f"Measurement:{exp.experiment_id}:{measurement.property_name}:{measurement.value or measurement.raw_value}"
                node(pid, measurement.property_name, "Property")
                node(mid, _measurement_label(measurement), "Measurement", **measurement.model_dump())
                edge(exp_id, "MEASURED", mid)
                edge(mid, "OF_PROPERTY", pid)
            for equipment in exp.equipment:
                eqid = f"Equipment:{equipment}"
                node(eqid, equipment, "Equipment")
                edge(exp_id, "USED_EQUIPMENT", eqid)
            for lab in exp.laboratories:
                if not _is_useful_name(lab):
                    continue
                labid = f"Laboratory:{lab}"
                node(labid, lab, "Laboratory")
                edge(exp_id, "PERFORMED_AT", labid)
            for team in exp.teams:
                if not _is_useful_name(team):
                    continue
                team_id = f"ResearchTeam:{team}"
                node(team_id, team, "ResearchTeam")
                edge(exp_id, "PERFORMED_BY", team_id)
            for employee in exp.employees:
                if not _is_useful_name(employee):
                    continue
                employee_id = f"Employee:{employee}"
                node(employee_id, employee, "Employee")
                edge(exp_id, "HAS_EXECUTOR", employee_id)
            for topic in exp.topic_tags:
                if not _is_useful_name(topic):
                    continue
                topic_id = f"TopicTag:{topic}"
                node(topic_id, topic, "TopicTag")
                edge(exp_id, "TAGGED_WITH", topic_id)
            for source in exp.evidence[:2]:
                if source.chunk_id:
                    cid = f"SourceChunk:{source.chunk_id}"
                    node(cid, source.source_name or source.chunk_id, "SourceChunk", quote=source.quote)
                    edge(exp_id, "SUPPORTED_BY", cid)
        for gap in gaps:
            gid = f"DataGap:{gap.gap_id}"
            node(gid, gap.reason, "DataGap", **gap.model_dump())
            for label, node_type, value in [
                ("GAP_FOR_ENTITY", "Material", gap.material),
                ("GAP_FOR_REGIME", "ProcessRegime", gap.regime),
                ("GAP_FOR_PROPERTY", "Property", gap.property),
            ]:
                if not value:
                    continue
                if node_type == "Material" and plan.constraints.materials:
                    if not any(material_matches(value, item) for item in plan.constraints.materials):
                        continue
                if node_type == "ProcessRegime" and plan.constraints.regimes:
                    if not any(regime_matches(value, item) for item in plan.constraints.regimes):
                        continue
                if node_type == "Property" and plan.constraints.properties:
                    if not any(property_matches(value, item) for item in plan.constraints.properties):
                        continue
                related_id = f"{node_type}:{value}"
                node(related_id, value, node_type)
                edge(gid, label, related_id)
        return {"nodes": list(nodes.values()), "edges": list(edges.values())}

    def _filter_entities(self, entities: list[dict[str, Any]], plan: AnalyticalQueryPlan) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in entities:
            item_type = item.get("type")
            name = str(item.get("name") or "")
            if item_type == "Material" and plan.constraints.materials:
                if not any(material_matches(name, value) for value in plan.constraints.materials):
                    continue
            if item_type == "ProcessRegime" and plan.constraints.regimes:
                if not any(regime_matches(name, value) for value in plan.constraints.regimes):
                    continue
            if item_type == "Property" and plan.constraints.properties:
                if not any(property_matches(name, value) for value in plan.constraints.properties):
                    continue
            if item_type == "Laboratory" and not _is_useful_name(name):
                continue
            result.append(item)
        return result

    @staticmethod
    def _grouped_facts(experiments: list[ExperimentFact]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for exp in experiments:
            for material in exp.materials or [""]:
                for regime in exp.regimes or [""]:
                    key = (material, regime)
                    row = grouped.setdefault(key, {"material": material, "regime": regime, "experiment_count": 0, "properties": set(), "effects": set()})
                    row["experiment_count"] += 1
                    for measurement in exp.measurements:
                        row["properties"].add(measurement.property_name)
                        if measurement.effect:
                            row["effects"].add(measurement.effect)
        return [
            {**row, "properties": sorted(row["properties"]), "effects": sorted(row["effects"])}
            for row in grouped.values()
        ]

    @staticmethod
    def _entities(experiments: list[ExperimentFact], gaps: list[DataGap]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for exp in experiments:
            rows.extend({"type": "Material", "name": item} for item in exp.materials)
            rows.extend({"type": "ProcessRegime", "name": item} for item in exp.regimes)
            rows.extend({"type": "Property", "name": item.property_name} for item in exp.measurements)
            rows.extend({"type": "Equipment", "name": item} for item in exp.equipment)
            rows.extend({"type": "Laboratory", "name": item} for item in exp.laboratories if _is_useful_name(item))
            rows.extend({"type": "ResearchTeam", "name": item} for item in exp.teams if _is_useful_name(item))
            rows.extend({"type": "Employee", "name": item} for item in exp.employees if _is_useful_name(item))
            rows.extend({"type": "TopicTag", "name": item} for item in exp.topic_tags if _is_useful_name(item))
        for gap in gaps:
            if gap.material:
                rows.append({"type": "Material", "name": gap.material})
            if gap.property:
                rows.append({"type": "Property", "name": gap.property})
        return _dedupe_dicts(rows, ["type", "name"])


def _fact_rows(exp: ExperimentFact) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    measurements = exp.measurements or [Measurement(property_name="", raw_value="")]
    for material in exp.materials or [""]:
        for regime in exp.regimes or [""]:
            for measurement in measurements:
                normalized = with_normalized_measurement_fields(measurement)
                row = {
                    "experiment_id": exp.experiment_id,
                    "material": material,
                    "regime": regime,
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
                    "equipment": exp.equipment,
                    "laboratories": [item for item in exp.laboratories if _is_useful_name(item)],
                    "teams": [item for item in exp.teams if _is_useful_name(item)],
                    "employees": [item for item in exp.employees if _is_useful_name(item)],
                    "topic_tags": [item for item in exp.topic_tags if _is_useful_name(item)],
                    "evidence": [item.model_dump() for item in normalized.evidence or exp.evidence],
                }
                row["canonical_fact_key"] = canonical_fact_key_from_row(row)
                rows.append(row)
    return rows


def _sources_from_evidence(items: list[Evidence], limit: int) -> list[dict[str, Any]]:
    seen = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = (item.document_id, item.chunk_id)
        if key in seen or not item.quote:
            continue
        seen.add(key)
        result.append(
            {
                "doc_id": item.document_id,
                "chunk_id": item.chunk_id,
                "title": item.source_name,
                "filename": item.source_name,
                "page_start": item.page,
                "page_end": item.page,
                "quote": item.quote,
            }
        )
        if len(result) >= limit:
            break
    return result


def _dedupe_dicts(items: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = tuple(_hashable(item.get(k)) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(val)) for key, val in value.items()))
    return value


def _measurement_label(measurement: Measurement) -> str:
    value = measurement.value if measurement.value is not None else measurement.raw_value
    unit = f" {measurement.unit}" if measurement.unit else ""
    return f"{measurement.property_name}: {value}{unit}".strip()


def _is_useful_name(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized not in {"oratory", "oratories", "laboratory", "laboratories", "lab", "labs"}

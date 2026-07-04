"""Repository-agnostic graph explorer helpers.

The cockpit API should work both with Neo4j and the local fallback catalog.
These helpers build explorer payloads from normalized ExperimentFact/DataGap
objects, so the UI can inspect the same semantic graph in either backend mode.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..domain.aliases import MATERIAL_ALIASES, PROPERTY_ALIASES, REGIME_ALIASES
from ..domain.normalization import material_matches, normalize_text, property_matches, regime_matches
from ..domain.ontology import DataGap, Evidence
from .graph_models import DecisionHistoryItem, EntityCard, EntitySummary, ExperimentFact, GraphStats, SimilarExperiment


ENTITY_TYPES = {
    "Material",
    "ProcessRegime",
    "Property",
    "Equipment",
    "Laboratory",
    "ResearchTeam",
    "Employee",
    "TopicTag",
    "DataGap",
    "Experiment",
    "Document",
    "DocumentChunk",
}


def validate_entity_type(entity_type: str | None) -> str | None:
    """Return whitelisted entity type or raise ValueError."""
    if entity_type is None or entity_type == "":
        return None
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"Unsupported entity_type={entity_type!r}")
    return entity_type


def list_entities_from_facts(
    experiments: list[ExperimentFact],
    gaps: list[DataGap],
    entity_type: str | None = None,
    query: str | None = None,
    limit: int = 50,
) -> list[EntitySummary]:
    """List graph entities extracted from experiments and gaps."""
    entity_type = validate_entity_type(entity_type)
    query_norm = normalize_text(query or "")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}

    def add(type_name: str, label: str, fact_id: str | None = None, degree: int = 1) -> None:
        if not label or (entity_type and type_name != entity_type):
            return
        if query_norm and query_norm not in normalize_text(label):
            return
        key = (type_name, label)
        row = by_key.setdefault(
            key,
            {
                "id": _entity_node_id(type_name, label),
                "type": type_name,
                "label": label,
                "canonical_name": label,
                "aliases": _aliases_for(type_name, label),
                "degree": 0,
                "facts": set(),
            },
        )
        row["degree"] += degree
        if fact_id:
            row["facts"].add(fact_id)

    for exp in experiments:
        add("Experiment", exp.experiment_id, exp.experiment_id, degree=len(exp.materials) + len(exp.regimes) + len(exp.measurements))
        for material in exp.materials:
            add("Material", material, exp.experiment_id)
        for regime in exp.regimes:
            add("ProcessRegime", regime, exp.experiment_id)
        for measurement in exp.measurements:
            add("Property", measurement.property_name, exp.experiment_id)
        for equipment in exp.equipment:
            add("Equipment", equipment, exp.experiment_id)
        for laboratory in exp.laboratories:
            add("Laboratory", laboratory, exp.experiment_id)
        for team in exp.teams:
            add("ResearchTeam", team, exp.experiment_id)
        for evidence in exp.evidence:
            if evidence.document_id:
                add("Document", evidence.document_id, exp.experiment_id)
            if evidence.chunk_id:
                add("DocumentChunk", evidence.chunk_id, exp.experiment_id)

    for gap in gaps:
        add("DataGap", gap.gap_id, gap.gap_id)
        if gap.material:
            add("Material", gap.material, gap.gap_id)
        if gap.regime:
            add("ProcessRegime", gap.regime, gap.gap_id)
        if gap.property:
            add("Property", gap.property, gap.gap_id)

    rows = [
        EntitySummary(
            id=row["id"],
            type=row["type"],
            label=row["label"],
            canonical_name=row["canonical_name"],
            aliases=row["aliases"],
            degree=int(row["degree"]),
            facts_count=len(row["facts"]),
        )
        for row in by_key.values()
    ]
    rows.sort(key=lambda item: (-item.degree, item.type, item.label))
    return rows[: max(1, limit)]


def build_entity_card(
    entity_type: str,
    entity_id: str,
    experiments: list[ExperimentFact],
    gaps: list[DataGap],
    diagnostics: dict[str, Any] | None = None,
) -> EntityCard:
    """Build a drill-down card for one entity."""
    validate_entity_type(entity_type)
    matching_experiments = [exp for exp in experiments if _experiment_has_entity(exp, entity_type, entity_id)]
    matching_gaps = [gap for gap in gaps if _gap_has_entity(gap, entity_type, entity_id)]
    summaries = list_entities_from_facts(matching_experiments, matching_gaps, entity_type=entity_type, query=entity_id, limit=1)
    entity = summaries[0].model_dump() if summaries else {
        "id": _entity_node_id(entity_type, entity_id),
        "type": entity_type,
        "label": entity_id,
        "canonical_name": entity_id,
        "aliases": _aliases_for(entity_type, entity_id),
        "degree": 0,
        "facts_count": 0,
    }
    related = _related_payload(matching_experiments, matching_gaps)
    sources = _sources_from_evidence([evidence for exp in matching_experiments for evidence in exp.evidence])
    subgraph = build_neighborhood_from_facts(
        entity_type=entity_type,
        entity_id=entity_id,
        experiments=matching_experiments,
        gaps=matching_gaps,
        depth=1,
        limit_nodes=50,
        limit_edges=80,
    )
    return EntityCard(
        entity=entity,
        related=related,
        sources=sources,
        subgraph=subgraph,
        diagnostics=diagnostics or {},
    )


def build_neighborhood_from_facts(
    entity_type: str,
    entity_id: str,
    experiments: list[ExperimentFact],
    gaps: list[DataGap],
    depth: int = 1,
    limit_nodes: int = 50,
    limit_edges: int = 80,
) -> dict[str, list[dict[str, Any]]]:
    """Build compact neighborhood subgraph around an entity."""
    validate_entity_type(entity_type)
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    def node(type_name: str, label: str, **props: Any) -> str:
        node_id = _entity_node_id(type_name, label)
        if len(nodes) < limit_nodes or node_id in nodes:
            nodes.setdefault(node_id, {"id": node_id, "label": label, "type": type_name, "properties": props})
        return node_id

    def edge(source: str, edge_type: str, target: str, **props: Any) -> None:
        if len(edges) >= limit_edges:
            return
        if source not in nodes or target not in nodes:
            return
        edge_id = f"{source}:{edge_type}:{target}"
        edges.setdefault(edge_id, {"id": edge_id, "source": source, "target": target, "label": edge_type, "type": edge_type, "properties": props})

    center_id = node(entity_type, entity_id, center=True)
    for exp in experiments:
        if not _experiment_has_entity(exp, entity_type, entity_id) and depth <= 1:
            continue
        exp_id = node("Experiment", exp.experiment_id)
        if entity_type != "Experiment":
            edge(center_id, "RELATED_TO_EXPERIMENT", exp_id)
        for material in exp.materials:
            mid = node("Material", material)
            edge(exp_id, "USES_MATERIAL", mid)
        for regime in exp.regimes:
            rid = node("ProcessRegime", regime)
            edge(exp_id, "HAS_REGIME", rid)
        for measurement in exp.measurements:
            pid = node("Property", measurement.property_name)
            meas_id = node("Measurement", f"{exp.experiment_id}:{measurement.property_name}:{measurement.value or measurement.raw_value}", **measurement.model_dump())
            edge(exp_id, "MEASURED", meas_id)
            edge(meas_id, "OF_PROPERTY", pid)
        for equipment in exp.equipment:
            eqid = node("Equipment", equipment)
            edge(exp_id, "USED_EQUIPMENT", eqid)
        for lab in exp.laboratories:
            labid = node("Laboratory", lab)
            edge(exp_id, "PERFORMED_AT", labid)
        for evidence in exp.evidence[:2]:
            if evidence.chunk_id:
                cid = node("DocumentChunk", evidence.chunk_id, source_name=evidence.source_name, quote=evidence.quote)
                edge(exp_id, "SUPPORTED_BY", cid)

    for gap in gaps:
        if not _gap_has_entity(gap, entity_type, entity_id) and depth <= 1:
            continue
        gap_id = node("DataGap", gap.gap_id, reason=gap.reason)
        edge(center_id, "HAS_RELATED_GAP", gap_id)
        if gap.material:
            mid = node("Material", gap.material)
            edge(gap_id, "GAP_FOR_ENTITY", mid)
        if gap.regime:
            rid = node("ProcessRegime", gap.regime)
            edge(gap_id, "GAP_FOR_REGIME", rid)
        if gap.property:
            pid = node("Property", gap.property)
            edge(gap_id, "GAP_FOR_PROPERTY", pid)

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def graph_stats_from_facts(
    experiments: list[ExperimentFact],
    gaps: list[DataGap],
    documents: int = 0,
    chunks: int = 0,
    kg_backend_active: str = "unknown",
    diagnostics: dict[str, Any] | None = None,
) -> GraphStats:
    """Build graph stats from fact objects."""
    materials = {item for exp in experiments for item in exp.materials}
    regimes = {item for exp in experiments for item in exp.regimes}
    properties = {measurement.property_name for exp in experiments for measurement in exp.measurements}
    equipment = {item for exp in experiments for item in exp.equipment}
    labs = {item for exp in experiments for item in exp.laboratories}
    teams = {item for exp in experiments for item in exp.teams}
    measurements = sum(len(exp.measurements) for exp in experiments)
    relationships = sum(
        len(exp.materials)
        + len(exp.regimes)
        + len(exp.measurements) * 2
        + len(exp.equipment)
        + len(exp.laboratories)
        + len(exp.evidence)
        for exp in experiments
    ) + len(gaps) * 3
    return GraphStats(
        documents=documents,
        chunks=chunks,
        experiments=len(experiments),
        materials=len(materials),
        regimes=len(regimes),
        properties=len(properties),
        measurements=measurements,
        equipment=len(equipment),
        laboratories=len(labs),
        teams=len(teams),
        data_gaps=len(gaps),
        relationships=relationships,
        kg_backend_active=kg_backend_active,
        diagnostics=diagnostics or {},
    )


def filter_gaps(gaps: list[DataGap], material: str | None = None, regime: str | None = None, property_name: str | None = None, limit: int = 50) -> list[DataGap]:
    """Filter gaps by canonical constraints."""
    result: list[DataGap] = []
    for gap in gaps:
        if material and gap.material and not material_matches(gap.material, material):
            continue
        if material and not gap.material:
            continue
        if regime and gap.regime and not regime_matches(gap.regime, regime):
            continue
        if regime and not gap.regime:
            continue
        if property_name and gap.property and not property_matches(gap.property, property_name):
            continue
        if property_name and not gap.property:
            continue
        result.append(gap)
        if len(result) >= limit:
            break
    return result


def decision_history_filtered(
    experiments: list[ExperimentFact],
    material: str | None = None,
    regime: str | None = None,
    property_name: str | None = None,
    limit: int = 50,
) -> list[DecisionHistoryItem]:
    """Build decision history rows with optional filters."""
    rows: list[DecisionHistoryItem] = []
    for exp in experiments:
        if material and not any(material_matches(value, material) for value in exp.materials):
            continue
        if regime and not any(regime_matches(value, regime) for value in exp.regimes):
            continue
        if property_name and not any(property_matches(measurement.property_name, property_name) for measurement in exp.measurements):
            continue
        rows.append(
            DecisionHistoryItem(
                experiment_id=exp.experiment_id,
                material=exp.materials[0] if exp.materials else material or "",
                regime=exp.regimes[0] if exp.regimes else None,
                equipment=exp.equipment,
                laboratory=exp.laboratories[0] if exp.laboratories else None,
                measurements=exp.measurements,
                conclusions=exp.conclusions,
                evidence=exp.evidence,
            )
        )
        if len(rows) >= limit:
            break
    return rows


def similar_experiments(
    experiments: list[ExperimentFact],
    material: str | None = None,
    regime: str | None = None,
    property_name: str | None = None,
    experiment_id: str | None = None,
    limit: int = 10,
) -> list[SimilarExperiment]:
    """Return transparent graph-similarity matches."""
    anchor = next((exp for exp in experiments if experiment_id and exp.experiment_id == experiment_id), None)
    anchor_materials = anchor.materials if anchor else ([material] if material else [])
    anchor_regimes = anchor.regimes if anchor else ([regime] if regime else [])
    anchor_properties = [m.property_name for m in anchor.measurements] if anchor else ([property_name] if property_name else [])
    anchor_equipment = set(anchor.equipment if anchor else [])
    anchor_labs = set(anchor.laboratories if anchor else [])

    scored: list[SimilarExperiment] = []
    for exp in experiments:
        if experiment_id and exp.experiment_id == experiment_id:
            continue
        score = 0.0
        reasons: list[str] = []
        if anchor_materials and any(any(material_matches(value, target) for target in anchor_materials) for value in exp.materials):
            score += 0.35
            reasons.append("same material")
        if anchor_regimes and any(any(regime_matches(value, target) for target in anchor_regimes) for value in exp.regimes):
            score += 0.25
            reasons.append("same regime")
        if anchor_properties and any(any(property_matches(m.property_name, target) for target in anchor_properties) for m in exp.measurements):
            score += 0.20
            reasons.append("same property")
        if anchor_equipment and anchor_equipment.intersection(exp.equipment):
            score += 0.10
            reasons.append("same equipment")
        if anchor_labs and anchor_labs.intersection(exp.laboratories):
            score += 0.05
            reasons.append("same lab")
        if score <= 0:
            continue
        first_measurement = exp.measurements[0] if exp.measurements else None
        scored.append(
            SimilarExperiment(
                experiment_id=exp.experiment_id,
                score=round(min(score, 1.0), 3),
                explanation=", ".join(reasons),
                material=exp.materials[0] if exp.materials else None,
                regime=exp.regimes[0] if exp.regimes else None,
                property=first_measurement.property_name if first_measurement else None,
                source=exp.evidence[0].source_name if exp.evidence else None,
                experiment=exp.summary(),
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]


def _related_payload(experiments: list[ExperimentFact], gaps: list[DataGap]) -> dict[str, list[dict[str, Any]]]:
    materials = sorted({item for exp in experiments for item in exp.materials})
    regimes = sorted({item for exp in experiments for item in exp.regimes})
    properties = sorted({m.property_name for exp in experiments for m in exp.measurements})
    equipment = sorted({item for exp in experiments for item in exp.equipment})
    labs = sorted({item for exp in experiments for item in exp.laboratories})
    teams = sorted({item for exp in experiments for item in exp.teams})
    return {
        "experiments": [exp.summary() for exp in experiments],
        "materials": [{"name": item} for item in materials],
        "regimes": [{"name": item} for item in regimes],
        "properties": [{"name": item} for item in properties],
        "equipment": [{"name": item} for item in equipment],
        "laboratories": [{"name": item} for item in labs],
        "teams": [{"name": item} for item in teams],
        "documents": _document_rows(experiments),
        "gaps": [gap.model_dump() for gap in gaps],
    }


def _document_rows(experiments: list[ExperimentFact]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for evidence in [item for exp in experiments for item in exp.evidence]:
        if not evidence.document_id:
            continue
        rows.setdefault(
            evidence.document_id,
            {"document_id": evidence.document_id, "source_name": evidence.source_name, "chunks": set()},
        )
        if evidence.chunk_id:
            rows[evidence.document_id]["chunks"].add(evidence.chunk_id)
    return [{**row, "chunks": sorted(row["chunks"])} for row in rows.values()]


def _experiment_has_entity(exp: ExperimentFact, entity_type: str, entity_id: str) -> bool:
    if entity_type == "Experiment":
        return exp.experiment_id == entity_id
    if entity_type == "Material":
        return any(material_matches(value, entity_id) for value in exp.materials)
    if entity_type == "ProcessRegime":
        return any(regime_matches(value, entity_id) for value in exp.regimes)
    if entity_type == "Property":
        return any(property_matches(measurement.property_name, entity_id) for measurement in exp.measurements)
    if entity_type == "Equipment":
        return any(_text_matches(value, entity_id) for value in exp.equipment)
    if entity_type == "Laboratory":
        return any(_text_matches(value, entity_id) for value in exp.laboratories)
    if entity_type == "ResearchTeam":
        return any(_text_matches(value, entity_id) for value in exp.teams)
    if entity_type == "Document":
        return any(evidence.document_id == entity_id for evidence in exp.evidence)
    if entity_type == "DocumentChunk":
        return any(evidence.chunk_id == entity_id for evidence in exp.evidence)
    return False


def _gap_has_entity(gap: DataGap, entity_type: str, entity_id: str) -> bool:
    if entity_type == "DataGap":
        return gap.gap_id == entity_id
    if entity_type == "Material" and gap.material:
        return material_matches(gap.material, entity_id)
    if entity_type == "ProcessRegime" and gap.regime:
        return regime_matches(gap.regime, entity_id)
    if entity_type == "Property" and gap.property:
        return property_matches(gap.property, entity_id)
    return False


def _sources_from_evidence(items: Iterable[Evidence], limit: int = 25) -> list[dict[str, Any]]:
    seen = set()
    rows: list[dict[str, Any]] = []
    for item in items:
        key = (item.document_id, item.chunk_id)
        if key in seen or not item.quote:
            continue
        seen.add(key)
        rows.append(
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
        if len(rows) >= limit:
            break
    return rows


def _entity_node_id(entity_type: str, label: str) -> str:
    return f"{entity_type}:{label}"


def _aliases_for(entity_type: str, label: str) -> list[str]:
    if entity_type == "Material":
        return _reverse_aliases(MATERIAL_ALIASES, label)
    if entity_type == "ProcessRegime":
        return _reverse_aliases(REGIME_ALIASES, label)
    if entity_type == "Property":
        return _reverse_aliases(PROPERTY_ALIASES, label)
    return []


def _reverse_aliases(mapping: dict[str, str], canonical: str) -> list[str]:
    return sorted({alias for alias, value in mapping.items() if value == canonical and alias != canonical})


def _text_matches(value: str, query: str) -> bool:
    left = normalize_text(value)
    right = normalize_text(query)
    return bool(right and (left == right or right in left or left in right))

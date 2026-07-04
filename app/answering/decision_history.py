"""Decision-history answer helpers."""

from __future__ import annotations

from typing import Any

from ..domain.query_constraints import QueryConstraints
from ..graph.graph_models import DecisionHistoryItem


def build_decision_history_payload(constraints: QueryConstraints, history: list[DecisionHistoryItem]) -> dict[str, Any]:
    """Build a structured decision-history response for one canonical material."""
    material = constraints.materials[0] if constraints.materials else "указанный материал"
    rows = [_history_item_to_dict(item) for item in history]
    if not rows:
        answer = f"История решений по {material} в загруженном корпусе не найдена."
        status = "no_exact_match"
    else:
        fragments = []
        for idx, item in enumerate(rows[:5], start=1):
            measurements = item.get("measurements") or []
            measured = "; ".join(
                f"{m.get('property_name')}: {m.get('value') or m.get('raw_value') or 'значение не указано'} {m.get('unit') or ''}".strip()
                for m in measurements
            ) or "измерения не указаны"
            fragments.append(
                f"{idx}. {item.get('experiment_id')}: режим {item.get('regime') or 'не указан'}, "
                f"оборудование {', '.join(item.get('equipment') or []) or 'не указано'}, измерения: {measured}"
            )
        answer = f"История решений по {material}: " + " ".join(fragments)
        status = "ok"
    return {
        "answer": answer,
        "status": status,
        "constraints": constraints.model_dump(),
        "decision_history": rows,
        "facts": [],
        "sources": _history_sources(history),
        "subgraph": _history_subgraph(rows),
    }


def _history_item_to_dict(item: DecisionHistoryItem) -> dict[str, Any]:
    return {
        "experiment_id": item.experiment_id,
        "material": item.material,
        "regime": item.regime,
        "equipment": item.equipment,
        "laboratory": item.laboratory,
        "measurements": [measurement.model_dump() for measurement in item.measurements],
        "conclusions": item.conclusions,
        "evidence": [evidence.model_dump() for evidence in item.evidence],
    }


def _history_sources(history: list[DecisionHistoryItem]) -> list[dict[str, Any]]:
    seen = set()
    sources: list[dict[str, Any]] = []
    for item in history:
        for evidence in item.evidence:
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


def _history_subgraph(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    for item in rows:
        exp_id = f"experiment:{item['experiment_id']}"
        nodes[exp_id] = {"id": exp_id, "label": item["experiment_id"], "type": "Experiment", "properties": {}}
        material_id = f"material:{item['material']}"
        nodes[material_id] = {"id": material_id, "label": item["material"], "type": "Material", "properties": {}}
        edges[f"{exp_id}:STUDIES:{material_id}"] = {"id": f"{exp_id}:STUDIES:{material_id}", "source": exp_id, "target": material_id, "label": "STUDIES", "properties": {}}
        if item.get("regime"):
            regime_id = f"regime:{item['regime']}"
            nodes[regime_id] = {"id": regime_id, "label": item["regime"], "type": "ProcessRegime", "properties": {}}
            edges[f"{exp_id}:USES_REGIME:{regime_id}"] = {"id": f"{exp_id}:USES_REGIME:{regime_id}", "source": exp_id, "target": regime_id, "label": "USES_REGIME", "properties": {}}
        for measurement in item.get("measurements") or []:
            prop = measurement.get("property_name")
            if not prop:
                continue
            prop_id = f"property:{prop}"
            nodes[prop_id] = {"id": prop_id, "label": prop, "type": "Property", "properties": {}}
            edges[f"{exp_id}:MEASURES:{prop_id}"] = {"id": f"{exp_id}:MEASURES:{prop_id}", "source": exp_id, "target": prop_id, "label": "MEASURES", "properties": measurement}
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


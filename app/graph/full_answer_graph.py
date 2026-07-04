"""Readable expanded answer graph for the Streamlit UI.

This renderer is intentionally separate from the compact answer graph.  It can
show more context around the answer, but it must not expose raw provenance IDs
or technical relation names on the canvas.
"""

from __future__ import annotations

import html
import json
import math
import re
from typing import Any

from pydantic import BaseModel, Field

from ..domain.fact_normalization import canonical_fact_key_from_row
from ..domain.unit_normalization import normalize_strength_to_mpa
from ..ui_helpers import friendly_source_name


EXPLANATION_GRAPH_MAX_NODES = 18
EXPLANATION_GRAPH_MAX_EDGES = 24
FULL_GRAPH_MAX_NODES = EXPLANATION_GRAPH_MAX_NODES
FULL_GRAPH_MAX_EDGES = EXPLANATION_GRAPH_MAX_EDGES

RAW_NODE_RE = re.compile(
    r"\b(?:doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|EXP-[A-Za-z0-9_-]+|SCI-[A-Za-z0-9_-]+|"
    r"DataGap:[A-Za-z0-9_:-]+|Material:[A-Za-z0-9_:-]+|ProcessRegime:[A-Za-z0-9_:-]+|"
    r"Experiment:[A-Za-z0-9_:-]+|PropertyValue|SourceChunk)\b",
    re.IGNORECASE,
)
RAW_RELATION_RE = re.compile(
    r"\b(?:MEASURES|OF_PROPERTY|GAP_FOR_PROPERTY|GAP_FOR_ENTITY|SUPPORTED_BY|HAS_REGIME|USES_MATERIAL|"
    r"FACT_SUPPORTED_BY_CHUNK|STUDIES|USES_REGIME|CHUNK_MENTIONS_ENTITY)\b",
    re.IGNORECASE,
)
TECH_PREFIX_RE = re.compile(r"^(?:Material|ProcessRegime|Property|PropertyValue|Experiment|DataGap|SourceChunk|DocumentChunk):", re.IGNORECASE)


class FullGraphNode(BaseModel):
    id: str
    label: str
    type: str
    title: str = ""
    raw_id: str = ""
    display_label: str = ""
    source_name: str | None = None
    value_original: Any = None
    value_normalized: Any = None
    unit_original: str | None = None
    unit_normalized: str | None = None
    score: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class FullGraphEdge(BaseModel):
    source: str
    target: str
    relation_raw: str = ""
    relation_readable: str = ""
    title: str = ""


class FullAnswerGraph(BaseModel):
    nodes: list[FullGraphNode]
    edges: list[FullGraphEdge]
    stats: dict[str, Any] = Field(default_factory=dict)


def build_full_answer_graph(payload: dict[str, Any], *, filters: dict[str, bool] | None = None) -> FullAnswerGraph:
    """Build an aggregated provenance graph explaining how the answer was formed."""

    filters = _normalize_filters(filters)
    facts = _canonical_fact_groups(payload)
    materials = _materials_for_graph(payload, facts)
    regimes = _regimes_for_graph(payload, facts)
    properties = _properties_for_graph(payload, facts)
    conflicts = _conflict_groups(payload) if filters.get("show_conflicts", True) else []
    gaps = _canonical_gaps(payload) if filters.get("show_gaps", True) else []
    intent = _detect_intent(payload, facts, conflicts, gaps)

    nodes: list[FullGraphNode] = []
    edges: list[FullGraphEdge] = []
    node_index: dict[tuple[str, str], str] = {}

    def add_node(
        node_id: str,
        label: str,
        node_type: str,
        *,
        title: str = "",
        raw_id: str = "",
        source_name: str | None = None,
        value_original: Any = None,
        value_normalized: Any = None,
        unit_original: str | None = None,
        unit_normalized: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> str:
        key = (node_type, _label_key(label))
        if key in node_index:
            return node_index[key]
        if len(nodes) >= EXPLANATION_GRAPH_MAX_NODES:
            return ""
        safe_label = _clean_canvas_label(label)
        node = FullGraphNode(
            id=node_id,
            label=safe_label,
            display_label=safe_label,
            type=node_type,
            title=title or safe_label.replace("\n", " · "),
            raw_id=raw_id,
            source_name=source_name,
            value_original=value_original,
            value_normalized=value_normalized,
            unit_original=unit_original,
            unit_normalized=unit_normalized,
            score=100.0 - len(nodes),
            details=details or {},
        )
        nodes.append(node)
        node_index[key] = node.id
        return node.id

    def add_edge(source: str, target: str, label: str) -> None:
        if source and target and source != target and len(edges) < EXPLANATION_GRAPH_MAX_EDGES:
            edges.append(FullGraphEdge(source=source, target=target, relation_raw=label, relation_readable=label, title=label))

    query_id = add_node("query", _query_label(payload, intent, materials, regimes, properties), "QueryNode", title=_question_text(payload))
    focus_id = add_node(
        "focus",
        _focus_label(intent, materials, regimes, properties),
        "AnswerFocusNode",
        title="Фокус ответа: извлеченные ограничения и найденные сущности",
    )
    add_edge(query_id, focus_id, "фокус")

    material_ids: dict[str, str] = {}
    for idx, material in enumerate(materials[:4]):
        material_ids[_norm_key(material)] = add_node(f"material_{idx}", _material_label(material), "Material", title=_material_title(material))
        add_edge(focus_id, material_ids[_norm_key(material)], "материал")

    property_ids: dict[str, str] = {}
    for idx, prop in enumerate(properties[:3]):
        property_ids[_norm_key(prop)] = add_node(f"property_{idx}", _property_label(prop, {}), "Property", title=f"Свойство: {prop}")

    regime_ids: dict[str, str] = {}
    for idx, regime in enumerate(regimes[:3]):
        regime_ids[_norm_key(regime)] = add_node(f"regime_{idx}", _regime_label(regime), "ProcessRegime", title=f"Режим: {regime}")

    for material_id in material_ids.values():
        for regime_id in regime_ids.values():
            add_edge(material_id, regime_id, "режим")
        for prop_id in property_ids.values():
            add_edge(material_id, prop_id, "свойство")
    for regime_id in regime_ids.values():
        for prop_id in property_ids.values():
            add_edge(regime_id, prop_id, "измеряет")

    fact_groups = _select_fact_groups_for_canvas(facts, intent)
    fact_ids: list[str] = []
    if filters.get("show_measurements", True):
        for idx, group in enumerate(fact_groups[:5]):
            fact_id = add_node(
                f"fact_{idx}",
                _fact_group_label(group),
                "CanonicalFactNode",
                title=_fact_group_title(group),
                raw_id=group.get("canonical_fact_key", ""),
                value_original=group.get("value_original"),
                value_normalized=group.get("value_normalized"),
                unit_original=group.get("unit_original"),
                unit_normalized=group.get("unit_normalized"),
                details=group,
            )
            fact_ids.append(fact_id)
            material_id = material_ids.get(_norm_key(str(group.get("material") or ""))) or next(iter(material_ids.values()), "")
            regime_id = regime_ids.get(_norm_key(str(group.get("regime") or "")))
            prop_id = property_ids.get(_norm_key(str(group.get("property") or ""))) or next(iter(property_ids.values()), "")
            add_edge(material_id, fact_id, "факт")
            if regime_id:
                add_edge(regime_id, fact_id, "режим")
            add_edge(prop_id, fact_id, "значение")

    evidence_id = ""
    if filters.get("show_sources", True):
        evidence = _evidence_groups(payload, facts)
        if evidence:
            evidence_id = add_node(
                "evidence_group",
                _evidence_group_label(evidence),
                "EvidenceGroupNode",
                title=_evidence_group_title(evidence),
                details={"sources": evidence},
            )
            for fact_id in fact_ids[:5]:
                add_edge(fact_id, evidence_id, "источник")

    if conflicts:
        for idx, conflict in enumerate(conflicts[:2]):
            conflict_id = add_node(
                f"conflict_{idx}",
                _conflict_group_label(conflict),
                "ConflictGroupNode",
                title=_conflict_group_title(conflict),
                details=conflict,
            )
            target = _nearest_fact_or_property(conflict, fact_ids, property_ids)
            add_edge(target or focus_id, conflict_id, "конфликт")

    if gaps:
        for idx, gap in enumerate(gaps[:2]):
            gap_id = add_node(f"gap_{idx}", _data_gap_group_label(gap), "DataGapNode", title=_data_gap_title(gap), details=gap)
            prop_id = property_ids.get(_norm_key(str(gap.get("property") or ""))) or focus_id
            add_edge(prop_id, gap_id, "пробел")

    conclusion_id = add_node("conclusion", _conclusion_label(payload, intent, facts, conflicts, gaps), "ConclusionNode", title="Итоговый статус ответа")
    if conflicts:
        add_edge(next((node.id for node in nodes if node.type == "ConflictGroupNode"), focus_id), conclusion_id, "учтено")
    elif gaps and not facts:
        add_edge(next((node.id for node in nodes if node.type == "DataGapNode"), focus_id), conclusion_id, "учтено")
    elif fact_ids:
        add_edge(fact_ids[0], conclusion_id, "вывод")
    else:
        add_edge(focus_id, conclusion_id, "вывод")

    edges = _dedupe_edges(edges)[:EXPLANATION_GRAPH_MAX_EDGES]
    raw_nodes = _raw_nodes(payload)
    raw_edges = _raw_edges(payload)
    total_nodes = max(len(nodes), len(raw_nodes) + len(facts) + len(conflicts) + len(gaps) + 3)
    total_edges = max(len(edges), len(raw_edges) + len(facts) * 4 + len(conflicts) + len(gaps))
    stats = {
        "total_nodes": total_nodes,
        "shown_nodes": len(nodes),
        "total_edges": total_edges,
        "shown_edges": len(edges),
        "truncated": total_nodes > len(nodes)
        or total_edges > len(edges)
        or len(nodes) >= EXPLANATION_GRAPH_MAX_NODES
        or len(edges) >= EXPLANATION_GRAPH_MAX_EDGES,
        "max_nodes": EXPLANATION_GRAPH_MAX_NODES,
        "max_edges": EXPLANATION_GRAPH_MAX_EDGES,
        "graph_kind": "answer_provenance",
        "intent": intent,
        "raw_nodes_available": len(raw_nodes),
        "raw_edges_available": len(raw_edges),
    }
    return FullAnswerGraph(nodes=nodes, edges=edges, stats=stats)


def full_answer_graph_to_html(
    graph: FullAnswerGraph,
    *,
    render_height: int = 620,
    render_width: int = 1200,
    container_id: str = "fullAnswerGraph",
    show_edge_labels: bool = False,
) -> str:
    """Render the full readable graph as fixed-layout interactive SVG HTML."""

    if not graph.nodes:
        return "<div style='padding:16px;color:#64748b'>Для ответа нет расширенной карты связей.</div>"
    node_width, node_height = 178, 72
    svg_id = f"{container_id}Svg"
    viewport_id = f"{container_id}Viewport"
    arrow_id = f"{container_id}Arrow"
    positions = _full_graph_positions(graph.nodes, render_width, render_height)
    edge_lines = []
    for edge in graph.edges:
        source = positions.get(edge.source)
        target = positions.get(edge.target)
        if not source or not target:
            continue
        edge_lines.append(
            f"<path class='fg-edge' d='M {source['x'] + node_width / 2:.1f} {source['y']:.1f} C "
            f"{source['x'] + 90:.1f} {source['y']:.1f}, {target['x'] - 90:.1f} {target['y']:.1f}, "
            f"{target['x'] - node_width / 2:.1f} {target['y']:.1f}' />"
        )
        if show_edge_labels and edge.relation_readable:
            edge_lines.append(
                f"<text class='fg-edge-label' x='{(source['x'] + target['x']) / 2:.1f}' y='{(source['y'] + target['y']) / 2 - 7:.1f}'>"
                f"{html.escape(_shorten(edge.relation_readable, 18))}</text>"
            )
    node_items = []
    for idx, node in enumerate(graph.nodes):
        pos = positions[node.id]
        label_lines = _label_lines(node.label, 22)[:3]
        start_y = -14 if len(label_lines) == 3 else -7 if len(label_lines) == 2 else 0
        text_spans = "".join(
            f"<tspan x='0' dy='{0 if line_idx == 0 else 16}'>{html.escape(line)}</tspan>"
            for line_idx, line in enumerate(label_lines)
        )
        node_items.append(
            f"<g class='fg-node' data-node-id='node-{idx}' transform='translate({pos['x']:.1f},{pos['y']:.1f})'>"
            f"<title>{html.escape(node.title or node.label)}</title>"
            f"<rect x='-{node_width / 2:.0f}' y='-{node_height / 2:.0f}' width='{node_width}' height='{node_height}' "
            f"rx='10' class='{_node_css_class(node.type)}' data-fixed='true' />"
            f"<text text-anchor='middle' y='{start_y}'>{text_spans}</text></g>"
        )
    stats_json = html.escape(json.dumps(graph.stats, ensure_ascii=False))
    return f"""
<div class="full-graph-wrap" data-stats="{stats_json}">
  <div class="full-graph-help">Колесо — масштаб · фон — перемещение карты · узлы зафиксированы</div>
  <svg id="{svg_id}" viewBox="0 0 {render_width} {render_height}" width="100%" height="{render_height}" role="img">
    <defs><marker id="{arrow_id}" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#64748b"></path></marker></defs>
    <g id="{viewport_id}">{"".join(edge_lines)}{"".join(node_items)}</g>
  </svg>
</div>
<style>
.full-graph-wrap {{ border:1px solid #d8dee9; border-radius:10px; background:#fff; position:relative; overflow:hidden; }}
.full-graph-help {{ position:absolute; right:10px; top:8px; z-index:2; font:12px Arial; color:#64748b; background:rgba(255,255,255,.88); padding:2px 6px; border-radius:4px; }}
#{svg_id} {{ cursor:grab; touch-action:none; }}
.fg-edge {{ fill:none; stroke:#64748b; stroke-width:1.35; marker-end:url(#{arrow_id}); opacity:.62; }}
.fg-edge-label {{ fill:#475569; font:10px Arial; paint-order:stroke; stroke:#fff; stroke-width:3px; }}
.fg-node rect {{ stroke:#334155; stroke-width:1.1; filter: drop-shadow(0 2px 4px rgba(15,23,42,.14)); }}
.fg-node text {{ fill:#0f172a; font:12px Arial; pointer-events:none; }}
.fg-node {{ cursor:default; }}
.fg-query {{ fill:#e0f2fe; }}
.fg-focus {{ fill:#f8fafc; }}
.fg-material {{ fill:#bfdbfe; }}
.fg-regime {{ fill:#bbf7d0; }}
.fg-property {{ fill:#fde68a; }}
.fg-measurement, .fg-fact, .fg-experiment {{ fill:#ddd6fe; }}
.fg-source, .fg-evidence {{ fill:#e2e8f0; }}
.fg-gap, .fg-conflict {{ fill:#fecaca; }}
.fg-conclusion {{ fill:#dcfce7; }}
.fg-equipment {{ fill:#fed7aa; }}
.fg-laboratory, .fg-team {{ fill:#ccfbf1; }}
.fg-entity {{ fill:#f1f5f9; }}
</style>
<script>
(function() {{
 const graphOptions = {{ layout: {{ hierarchical: {{ enabled: false }} }}, physics: {{ enabled: false }}, interaction: {{ zoomView: true, dragView: true, dragNodes: false }} }};
 const svg = document.getElementById('{svg_id}');
 const viewport = document.getElementById('{viewport_id}');
 let state = {{x:0, y:0, scale:1}};
 let drag = null;
 function apply() {{ viewport.setAttribute('transform', `translate(${{state.x}},${{state.y}}) scale(${{state.scale}})`); }}
 function isNodeHit(ev) {{
   return Array.from(svg.querySelectorAll('.fg-node')).some(function(node) {{
     const box = node.getBoundingClientRect();
     return ev.clientX >= box.left && ev.clientX <= box.right && ev.clientY >= box.top && ev.clientY <= box.bottom;
   }});
 }}
 svg.addEventListener('wheel', function(ev) {{
   ev.preventDefault();
   const delta = ev.deltaY < 0 ? 1.12 : 0.89;
   state.scale = Math.max(0.30, Math.min(3.4, state.scale * delta));
   apply();
 }}, {{passive:false}});
 svg.addEventListener('pointerdown', function(ev) {{
   const node = ev.target.closest && ev.target.closest('.fg-node');
   drag = (node || isNodeHit(ev)) ? null : {{startX: ev.clientX, startY: ev.clientY, x: state.x, y: state.y}};
   svg.setPointerCapture(ev.pointerId);
 }});
 svg.addEventListener('pointermove', function(ev) {{
   if (!drag) return;
   state.x = drag.x + ev.clientX - drag.startX;
   state.y = drag.y + ev.clientY - drag.startY;
   apply();
 }});
 svg.addEventListener('pointerup', function(ev) {{ drag = null; try {{ svg.releasePointerCapture(ev.pointerId); }} catch(e) {{}} }});
 apply();
}})();
</script>
"""


def full_graph_audit_tables(payload: dict[str, Any], graph: FullAnswerGraph | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return audit tables with readable columns and raw technical identifiers."""

    graph = graph or build_full_answer_graph(payload)
    raw_nodes = _raw_nodes(payload)
    raw_edges = _raw_edges(payload)
    raw_node_by_id = {str(node.get("id") or ""): node for node in raw_nodes if isinstance(node, dict)}
    readable_by_raw = {node.raw_id: node.label for node in graph.nodes if node.raw_id}
    node_rows = []
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            continue
        raw_id = str(raw.get("id") or "")
        node_type = _canonical_type(raw.get("type") or raw.get("label_type"))
        label = readable_by_raw.get(raw_id) or _readable_node_label(raw, node_type, payload)
        node_rows.append(
            {
                "label": label,
                "type": node_type,
                "readable_source": _source_from_node(raw),
                "value": _first_present(raw, "value", "value_original", "raw_value"),
                "unit": _first_present(raw, "unit", "unit_original"),
                "raw_id": raw_id,
            }
        )
    edge_rows = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        source_raw = str(edge.get("source") or edge.get("from") or "")
        target_raw = str(edge.get("target") or edge.get("to") or "")
        relation_raw = str(edge.get("label") or edge.get("type") or edge.get("predicate") or "")
        edge_rows.append(
            {
                "source_label": readable_by_raw.get(source_raw) or _readable_node_label(raw_node_by_id.get(source_raw, {}), _canonical_type((raw_node_by_id.get(source_raw, {}) or {}).get("type")), payload),
                "relation_readable": _readable_relation(relation_raw),
                "target_label": readable_by_raw.get(target_raw) or _readable_node_label(raw_node_by_id.get(target_raw, {}), _canonical_type((raw_node_by_id.get(target_raw, {}) or {}).get("type")), payload),
                "relation_raw": relation_raw,
                "source_raw_id": source_raw,
                "target_raw_id": target_raw,
            }
        )
    return node_rows, edge_rows


def _canonical_fact_groups(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deduplicated fact groups for canvas-level provenance."""

    groups: dict[str, dict[str, Any]] = {}
    for row in _payload_facts(payload):
        normalized = dict(row)
        key = str(normalized.get("canonical_fact_key") or canonical_fact_key_from_row(normalized))
        existing = groups.get(key)
        evidence = _fact_evidence_items(normalized)
        if existing is None:
            normalized["canonical_fact_key"] = key
            normalized["evidence"] = evidence
            normalized["evidence_count"] = max(_safe_int(normalized.get("evidence_count")), len(evidence), 1 if _source_from_fact(normalized) else 0)
            groups[key] = normalized
            continue
        existing["evidence"] = _merge_evidence(existing.get("evidence") or [], evidence)
        existing["evidence_count"] = max(
            _safe_int(existing.get("evidence_count")),
            _safe_int(normalized.get("evidence_count")),
            len(existing["evidence"]),
        )
        existing.setdefault("duplicates", 0)
        existing["duplicates"] += 1
        for field in ["source_name", "source_url", "quote", "evidence_quote"]:
            if not existing.get(field) and normalized.get(field):
                existing[field] = normalized.get(field)
    return list(groups.values())


def _materials_for_graph(payload: dict[str, Any], facts: list[dict[str, Any]]) -> list[str]:
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    values = [*(constraints.get("materials") or []), *(fact.get("material") for fact in facts if fact.get("material"))]
    if not values:
        values.extend(_materials_from_question(_question_text(payload)))
    return _unique_clean(values) or ["материал"]


def _regimes_for_graph(payload: dict[str, Any], facts: list[dict[str, Any]]) -> list[str]:
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    values = [*(constraints.get("regimes") or []), *(fact.get("regime") for fact in facts if fact.get("regime"))]
    if not values:
        values.extend(_regimes_from_question(_question_text(payload)))
    return _unique_clean(values)


def _properties_for_graph(payload: dict[str, Any], facts: list[dict[str, Any]]) -> list[str]:
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    values = [*(constraints.get("properties") or []), *(fact.get("property") for fact in facts if fact.get("property"))]
    gaps = payload.get("data_gaps") or payload.get("gaps") or []
    values.extend(gap.get("property") for gap in gaps if isinstance(gap, dict) and gap.get("property"))
    if not values:
        values.extend(_properties_from_question(_question_text(payload)))
    return _unique_clean(values) or ["свойство"]


def _detect_intent(payload: dict[str, Any], facts: list[dict[str, Any]], conflicts: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> str:
    text = (_question_text(payload) + " " + str(payload.get("answer") or "")).lower()
    materials = _materials_for_graph(payload, facts)
    if "сравн" in text or "compare" in text or len(materials) >= 2:
        return "comparison"
    if "противореч" in text or "неоднород" in text or conflicts:
        return "conflict"
    if "пробел" in text or ("gap" in text and gaps):
        return "data_gap"
    if not facts and (gaps or str(payload.get("status") or "") in {"no_exact_match", "partial"}):
        return "no_data"
    if "нет данных" in text:
        return "data_gap"
    if "evidence" in text or "источник" in text or "основан" in text:
        return "provenance"
    return "exact"


def _question_text(payload: dict[str, Any]) -> str:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    candidates = [
        payload.get("question"),
        payload.get("query"),
        payload.get("user_query"),
        payload.get("request_query"),
        diagnostics.get("query"),
        diagnostics.get("question"),
    ]
    return _clean_node_text(next((item for item in candidates if item), "")) or "вопрос"


def _materials_from_question(question: str) -> list[str]:
    result: list[str] = []
    patterns = [
        r"\bX\d{2,}\b",
        r"\b7075[-\s]?T6\b",
        r"\bTi[-\s]?6Al[-\s]?4V\b",
        r"\bVT6\b",
        r"\bBT6\b",
        r"\bВТ[-\s]?6\b",
        r"\b12[ХX]18Н10Т\b",
    ]
    for pattern in patterns:
        result.extend(match.group(0).replace(" ", "-") for match in re.finditer(pattern, question, flags=re.IGNORECASE))
    return result


def _regimes_from_question(question: str) -> list[str]:
    low = question.lower().replace("ё", "е")
    result: list[str] = []
    for needle, label in [
        ("отжиг", "отжиг"),
        ("anneal", "отжиг"),
        ("старен", "старение"),
        ("aging", "старение"),
        ("закал", "закалка"),
        ("лазер", "лазерная обработка"),
        ("laser", "лазерная обработка"),
        ("термо", "термообработка"),
        ("heat treatment", "термообработка"),
    ]:
        if needle in low:
            result.append(label)
    return result


def _properties_from_question(question: str) -> list[str]:
    low = question.lower().replace("ё", "е")
    result: list[str] = []
    for needle, label in [
        ("прочн", "прочность"),
        ("strength", "прочность"),
        ("тверд", "твердость"),
        ("hardness", "твердость"),
        ("корроз", "коррозионная стойкость"),
        ("corrosion", "коррозионная стойкость"),
    ]:
        if needle in low:
            result.append(label)
    return result


def _query_label(payload: dict[str, Any], intent: str, materials: list[str], regimes: list[str], properties: list[str]) -> str:
    name = {
        "comparison": "Вопрос: сравнение",
        "conflict": "Вопрос: противоречия",
        "data_gap": "Вопрос: пробелы",
        "no_data": "Вопрос: данных нет",
        "provenance": "Вопрос: evidence",
    }.get(intent, "Вопрос")
    parts = [*(materials[:2] if materials != ["материал"] else []), *regimes[:1], *properties[:1]]
    return _two_line(name, " · ".join(parts) or _shorten(_question_text(payload), 42))


def _focus_label(intent: str, materials: list[str], regimes: list[str], properties: list[str]) -> str:
    if intent == "comparison":
        return _two_line("Фокус ответа", f"{_short_join(materials[:2])} · {_short_join(properties[:1])}")
    if intent == "conflict":
        return _two_line("Фокус ответа", "неоднородные факты")
    if intent in {"data_gap", "no_data"}:
        return _two_line("Фокус ответа", "проверка наличия данных")
    parts = [materials[0] if materials else "", regimes[0] if regimes else "", properties[0] if properties else ""]
    return _two_line("Фокус ответа", " + ".join(item for item in parts if item) or "связанные факты")


def _select_fact_groups_for_canvas(facts: list[dict[str, Any]], intent: str) -> list[dict[str, Any]]:
    if not facts:
        return []
    if intent == "comparison":
        selected: list[dict[str, Any]] = []
        seen_materials: set[str] = set()
        for fact in facts:
            key = _norm_key(str(fact.get("material") or ""))
            if key and key not in seen_materials:
                selected.append(_aggregate_material_property_facts([item for item in facts if _norm_key(str(item.get("material") or "")) == key]))
                seen_materials.add(key)
            if len(selected) >= 3:
                break
        return selected
    if intent == "conflict":
        return facts[:5]
    return facts[:5]


def _aggregate_material_property_facts(facts: list[dict[str, Any]]) -> dict[str, Any]:
    base = dict(facts[0]) if facts else {}
    if len(facts) <= 1:
        base["facts_count"] = len(facts)
        base["evidence"] = _fact_evidence_items(base)
        base["evidence_count"] = max(_safe_int(base.get("evidence_count")), len(base["evidence"]), 1 if _source_from_fact(base) else 0)
        return base
    values = [_normalized_numeric(item) for item in facts]
    values = [item for item in values if item is not None]
    if values:
        base["value_normalized_min"] = min(values)
        base["value_normalized_max"] = max(values)
        base["unit_normalized"] = next((item.get("unit_normalized") for item in facts if item.get("unit_normalized")), "MPa")
    base["facts_count"] = len(facts)
    evidence: list[dict[str, Any]] = []
    for item in facts:
        evidence = _merge_evidence(evidence, _fact_evidence_items(item))
    base["evidence"] = evidence
    base["evidence_count"] = max(sum(max(_safe_int(item.get("evidence_count")), 1) for item in facts), len(evidence))
    base["canonical_fact_key"] = "aggregate:" + "|".join(
        [str(base.get("material") or ""), str(base.get("property") or ""), str(base.get("unit_normalized") or "")]
    )
    return base


def _aggregate_fact_range(all_facts: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    return _aggregate_material_property_facts(all_facts) if len(all_facts) > 5 else selected[0]


def _fact_group_label(group: dict[str, Any]) -> str:
    material = _clean_node_text(group.get("material"))
    prop = _property_public_name(group.get("property"))
    evidence_count = max(_safe_int(group.get("evidence_count")), len(group.get("evidence") or []))
    if group.get("value_normalized_min") is not None and group.get("value_normalized_max") is not None:
        low = _format_number(group.get("value_normalized_min"))
        high = _format_number(group.get("value_normalized_max"))
        unit = group.get("unit_normalized") or "MPa"
        facts_count = _safe_int(group.get("facts_count")) or 1
        value_text = f"{low} {unit}" if low == high else f"{low}–{high} {unit}"
        return f"{prop} {material}\n{value_text}\n{facts_count} фактов"
    value = _first_present(group, "value_original", "value", "raw_value")
    unit = _first_present(group, "unit_original", "unit")
    norm_value = _first_present(group, "value_normalized")
    norm_unit = _first_present(group, "unit_normalized")
    if value is not None and unit:
        value_text = f"{_format_number(value)} {unit}"
        if norm_value is not None and norm_unit and not _same_value_unit(value, unit, norm_value, norm_unit):
            value_text = f"{value_text} ≈ {_format_number(norm_value)} {norm_unit}"
    else:
        value_text = _effect_label(group.get("effect")) or "качественный факт"
    evidence_text = f"{evidence_count} источника" if evidence_count and evidence_count < 5 else f"{evidence_count} источников" if evidence_count else "источник указан"
    effect = _effect_label(group.get("effect"))
    if effect:
        evidence_text = f"{effect} · {evidence_text}"
    return f"{prop}\n{value_text}\n{evidence_text}"


def _fact_group_title(group: dict[str, Any]) -> str:
    parts = [
        f"Материал: {_clean_node_text(group.get('material'))}",
        f"Режим: {_clean_node_text(group.get('regime'))}",
        f"Свойство: {_clean_node_text(group.get('property'))}",
        f"Ключ: {group.get('canonical_fact_key')}",
    ]
    sources = [_source_name_from_evidence(item) for item in group.get("evidence") or []]
    if sources:
        parts.append("Источники: " + ", ".join(_unique_clean(sources)[:5]))
    return "; ".join(item for item in parts if item and not item.endswith(": "))


def _evidence_groups(payload: dict[str, Any], facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fact in facts:
        result = _merge_evidence(result, _fact_evidence_items(fact))
    for row in [*(payload.get("sources") or []), *(payload.get("evidence") or [])]:
        if isinstance(row, dict):
            result = _merge_evidence(result, [row])
    return result


def _evidence_group_label(evidence: list[dict[str, Any]]) -> str:
    count = len(evidence)
    names = _unique_clean(_source_name_from_evidence(item) for item in evidence)
    if count <= 3 and names:
        return "Источники\n" + "\n".join(_shorten(name, 26) for name in names[:2])
    return f"Источники\n{count} подтверждений"


def _evidence_group_title(evidence: list[dict[str, Any]]) -> str:
    names = _unique_clean(_source_name_from_evidence(item) for item in evidence)
    quotes = _unique_clean(_clean_node_text(item.get("quote") or item.get("evidence_quote") or "") for item in evidence if isinstance(item, dict))
    parts = ["Источники: " + ", ".join(names[:6])] if names else []
    if quotes:
        parts.append("Фрагменты: " + " | ".join(_shorten(quote, 80) for quote in quotes[:3]))
    return "; ".join(parts) or "Группа подтверждений"


def _conflict_groups(payload: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    rows = diagnostics.get("fact_conflicts") or diagnostics.get("conflict_groups") or payload.get("fact_conflicts") or []
    result: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (
            _norm_key(str(row.get("material") or "")),
            _norm_key(str(row.get("regime") or "")),
            _norm_key(str(row.get("property") or "")),
            _conflict_values_label(row),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _conflict_group_label(conflict: dict[str, Any]) -> str:
    values = _conflict_values_label(conflict)
    return f"Неоднородность\n{_shorten(values or 'разные значения', 32)}"


def _conflict_group_title(conflict: dict[str, Any]) -> str:
    material = _clean_node_text(conflict.get("material"))
    regime = _clean_node_text(conflict.get("regime"))
    prop = _clean_node_text(conflict.get("property"))
    return "; ".join(item for item in [material, regime, prop, _conflict_values_label(conflict)] if item)


def _canonical_gaps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [*(payload.get("data_gaps") or []), *(payload.get("gaps") or [])]
    result: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (_norm_key(str(row.get("material") or "")), _norm_key(str(row.get("property") or row.get("reason") or "")))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _data_gap_group_label(gap: dict[str, Any]) -> str:
    prop = _property_public_name(gap.get("property") or gap.get("gap") or gap.get("reason") or "нет численных данных")
    return f"Пробел\n{_shorten(prop, 34)}"


def _data_gap_title(gap: dict[str, Any]) -> str:
    return "; ".join(str(item) for item in [gap.get("material"), gap.get("regime"), gap.get("property"), gap.get("reason")] if item)


def _conclusion_label(
    payload: dict[str, Any],
    intent: str,
    facts: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> str:
    if not facts and gaps:
        return "Вывод\nточных данных нет"
    if not facts:
        return "Вывод\nданные не найдены"
    if conflicts:
        return "Вывод\nусловия неоднородны"
    if intent == "comparison":
        return "Вывод\nсравнение с caveat"
    return "Вывод\nоснован на фактах"


def _nearest_fact_or_property(conflict: dict[str, Any], fact_ids: list[str], property_ids: dict[str, str]) -> str:
    prop_id = property_ids.get(_norm_key(str(conflict.get("property") or "")))
    return prop_id or (fact_ids[0] if fact_ids else "")


def _fact_evidence_items(fact: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = fact.get("evidence")
    if isinstance(evidence, list):
        rows = [row for row in evidence if isinstance(row, dict)]
    else:
        rows = []
    source = _source_from_fact(fact)
    if source:
        rows.append(
            {
                "source_name": source,
                "source_url": fact.get("source_url"),
                "source_type": fact.get("source_type"),
                "title": fact.get("title"),
                "quote": fact.get("quote") or fact.get("evidence_quote"),
                "doc_id": fact.get("doc_id"),
                "chunk_id": fact.get("chunk_id") or fact.get("source_chunk_id"),
            }
        )
    return rows


def _merge_evidence(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = list(left)
    seen = {_evidence_key(item) for item in result}
    for item in right:
        key = _evidence_key(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _evidence_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("source_name") or item.get("title") or item.get("filename") or item.get("source_url") or item.get("doc_id") or ""),
        str(item.get("chunk_id") or item.get("source_chunk_id") or ""),
        str(item.get("quote") or item.get("evidence_quote") or "")[:80],
    )


def _source_from_fact(fact: dict[str, Any]) -> str | None:
    return fact.get("source_name") or fact.get("title") or fact.get("filename") or fact.get("source_url") or fact.get("doc_id")


def _source_name_from_evidence(item: dict[str, Any]) -> str:
    return friendly_source_name(
        item.get("source_name") or item.get("title") or item.get("filename") or item.get("doc_id"),
        source_url=item.get("source_url"),
        source_type=item.get("source_type"),
        title=item.get("title"),
    )


def _normalized_numeric(fact: dict[str, Any]) -> float | None:
    value = fact.get("value_normalized")
    if value is None:
        value = fact.get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _conflict_values_label(conflict: dict[str, Any]) -> str:
    values = conflict.get("values") or conflict.get("normalized_values") or []
    labels: list[str] = []
    if isinstance(values, list):
        for value in values:
            if isinstance(value, dict):
                amount = value.get("value_normalized") if value.get("value_normalized") is not None else value.get("value")
                unit = value.get("unit_normalized") or value.get("unit") or ""
                if amount is not None:
                    labels.append(f"{_format_number(amount)} {unit}".strip())
            elif value is not None:
                labels.append(str(value))
    effects = conflict.get("effects") or []
    if isinstance(effects, list):
        labels.extend(_effect_label(effect) for effect in effects if effect)
    if not labels:
        for field in ["value", "effect"]:
            if conflict.get(field):
                labels.append(str(conflict.get(field)))
    return " ↔ ".join(_unique_clean(labels)[:4])


def _effect_label(value: Any) -> str:
    mapping = {"increase": "рост", "decrease": "снижение", "unknown": "не указано"}
    return mapping.get(str(value or "").lower(), _clean_node_text(value))


def _property_public_name(value: Any) -> str:
    text = _clean_node_text(value)
    low = text.lower().replace("ё", "е")
    if "strength" in low or "прочн" in low:
        return "Прочность"
    if "hardness" in low or "тверд" in low:
        return "Твердость"
    if "corrosion" in low or "корроз" in low:
        return "Коррозионная стойкость"
    return text or "Свойство"


def _material_title(material: str) -> str:
    if "вт6" in material.lower() or "vt6" in material.lower():
        return "Тип: материал; ВТ6; aliases: Ti-6Al-4V, VT6"
    if "ti-6al-4v" in material.lower():
        return "Тип: материал; Ti-6Al-4V; aliases: ВТ6, VT6"
    return f"Тип: материал; {material}"


def _clean_canvas_label(label: str) -> str:
    lines = [_clean_node_text(line) for line in str(label).splitlines()]
    return "\n".join(line for line in lines if line) or "узел"


def _unique_clean(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _clean_node_text(value)
        if not text:
            continue
        key = _norm_key(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _short_join(values: list[str]) -> str:
    return " · ".join(_shorten(item, 20) for item in values if item) or "данные"


def _norm_key(value: str) -> str:
    text = _clean_node_text(value).lower().replace("ё", "е").replace("−", "-")
    text = text.replace("ti6al4v", "ti-6al-4v").replace("вт-6", "вт6")
    if text in {"vt6", "bt6"}:
        text = "вт6"
    if "ti-6al-4v" in text:
        text = "вт6"
    return re.sub(r"\s+", " ", text).strip()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_filters(filters: dict[str, bool] | None) -> dict[str, bool]:
    result = {
        "show_sources": True,
        "show_gaps": True,
        "show_conflicts": True,
        "show_measurements": True,
        "active_only": True,
    }
    if filters:
        result.update({key: bool(value) for key, value in filters.items()})
    return result


def _raw_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    subgraph = payload.get("subgraph")
    nodes = subgraph.get("nodes") if isinstance(subgraph, dict) else []
    return [node for node in nodes or [] if isinstance(node, dict)]


def _raw_edges(payload: dict[str, Any]) -> list[dict[str, Any]]:
    subgraph = payload.get("subgraph")
    edges = subgraph.get("edges") if isinstance(subgraph, dict) else []
    return [edge for edge in edges or [] if isinstance(edge, dict)]


def _canonical_type(value: Any) -> str:
    raw = str(value or "Entity")
    mapping = {
        "Document": "Source",
        "DocumentChunk": "Evidence",
        "SourceChunk": "Evidence",
        "PropertyValue": "Measurement",
        "ProcessRegime": "ProcessRegime",
        "Regime": "ProcessRegime",
        "ResearchTeam": "Team",
    }
    return mapping.get(raw, raw)


def _node_allowed(node_type: str, raw: dict[str, Any], filters: dict[str, bool]) -> bool:
    if filters.get("active_only", True) and _explicitly_inactive(raw):
        return False
    if node_type in {"Source", "Evidence"} and not filters.get("show_sources", True):
        return False
    if node_type == "DataGap" and not filters.get("show_gaps", True):
        return False
    if node_type == "Conflict" and not filters.get("show_conflicts", True):
        return False
    if node_type in {"Measurement", "Fact", "Experiment"} and not filters.get("show_measurements", True):
        return False
    return True


def _explicitly_inactive(raw: dict[str, Any]) -> bool:
    props = raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
    values = [raw.get("active"), props.get("active"), raw.get("is_active"), props.get("is_active")]
    return any(value is False or str(value).lower() == "false" for value in values)


def _synthetic_nodes_from_payload(payload: dict[str, Any], filters: dict[str, bool]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for fact_idx, fact in enumerate(_payload_facts(payload)):
        material = fact.get("material")
        regime = fact.get("regime")
        prop = fact.get("property")
        if material:
            entries.append(_synthetic_node("Material", f"fact:{fact_idx}:material:{material}", {"label": material}, 95))
        if regime:
            entries.append(_synthetic_node("ProcessRegime", f"fact:{fact_idx}:regime:{regime}", {"label": regime}, 85))
        if prop:
            entries.append(_synthetic_node("Property", f"fact:{fact_idx}:property:{prop}", {"label": prop, "unit": fact.get("unit_normalized") or fact.get("unit")}, 80))
        if filters.get("show_measurements", True):
            entries.append(_synthetic_node("Measurement", f"fact:{fact_idx}:measurement", fact, 88))
    if filters.get("show_sources", True):
        for idx, source in enumerate(payload.get("sources") or payload.get("evidence") or []):
            if isinstance(source, dict):
                entries.append(_synthetic_node("Source", f"source:{idx}", source, 45))
    if filters.get("show_gaps", True):
        for idx, gap in enumerate([*(payload.get("data_gaps") or []), *(payload.get("gaps") or [])]):
            if isinstance(gap, dict):
                entries.append(_synthetic_node("DataGap", f"gap:{idx}", gap, 70))
    if filters.get("show_conflicts", True):
        diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
        for idx, conflict in enumerate(diagnostics.get("fact_conflicts") or []):
            if isinstance(conflict, dict):
                entries.append(_synthetic_node("Conflict", f"conflict:{idx}", conflict, 75))
    return entries


def _synthetic_node(node_type: str, raw_id: str, raw: dict[str, Any], score: float) -> dict[str, Any]:
    return {"raw_id": raw_id, "type": node_type, "raw": {"id": raw_id, "type": node_type, **raw}, "score": score}


def _payload_facts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [*(payload.get("primary_facts") or []), *(payload.get("facts") or [])]
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        result.append(row)
    return result


def _select_nodes(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_entries = sorted(entries, key=lambda item: (-float(item.get("score") or 0), _type_order(str(item.get("type") or ""))))
    return sorted_entries[:FULL_GRAPH_MAX_NODES]


def _node_score(raw: dict[str, Any], node_type: str, payload: dict[str, Any]) -> float:
    text = json.dumps(raw, ensure_ascii=False, default=str).lower()
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    score = 10.0
    for values, weight in [
        (constraints.get("materials") or [], 60),
        (constraints.get("regimes") or [], 45),
        (constraints.get("properties") or [], 45),
    ]:
        if any(str(value).lower() in text for value in values if value):
            score += weight
    score += {
        "Material": 35,
        "ProcessRegime": 32,
        "Property": 32,
        "Measurement": 30,
        "Fact": 30,
        "Experiment": 26,
        "Conflict": 24,
        "DataGap": 20,
        "Source": 12,
        "Evidence": 12,
    }.get(node_type, 5)
    return score


def _readable_node_label(raw: dict[str, Any], node_type: str, payload: dict[str, Any]) -> str:
    props = raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
    base = _clean_node_text(
        raw.get("display_label")
        or raw.get("label")
        or raw.get("name")
        or raw.get("canonical_name")
        or props.get("canonical_name")
        or props.get("name")
        or raw.get("id")
    )
    if node_type == "Material":
        return _material_label(base)
    if node_type == "ProcessRegime":
        return _regime_label(base)
    if node_type == "Property":
        return _property_label(base, raw)
    if node_type in {"Measurement", "Fact"}:
        return _measurement_label(raw)
    if node_type == "Experiment":
        return _experiment_label(raw, payload)
    if node_type in {"Source", "Evidence"}:
        return _source_label(raw)
    if node_type == "DataGap":
        return _gap_label(raw)
    if node_type == "Conflict":
        return _conflict_label(raw)
    if node_type == "Equipment":
        return _two_line(base or "Оборудование", "оборудование")
    if node_type == "Laboratory":
        return _two_line(base or "Лаборатория", "лаборатория")
    if node_type == "Team":
        return _two_line(base or "Команда", "команда")
    return _shorten(base or "Связанная сущность", 54)


def _material_label(value: str) -> str:
    low = value.lower().replace("ё", "е")
    if "7075" in low:
        return "7075-T6\nалюминиевый сплав\nсостояние T6"
    if "ti-6al-4v" in low or "ti6al4v" in low:
        return "Ti-6Al-4V\nтитановый сплав\nаналог ВТ6"
    if "вт6" in low or "vt6" in low:
        return "ВТ6\nтитановый сплав"
    if "12х18н10т" in low or "12x18" in low:
        return "12Х18Н10Т\nнержавеющая сталь"
    return _two_line(value or "Материал", "материал из корпуса")


def _regime_label(value: str) -> str:
    low = value.lower().replace("ё", "е")
    if "anneal" in low or "отжиг" in low:
        return "Отжиг\nтермообработка"
    if "aging" in low or "стар" in low:
        return "Старение\nтермообработка"
    if "закал" in low:
        return "Закалка\nтермообработка"
    if "heat treatment" in low or "термо" in low:
        return "Термообработка\nрежим обработки"
    return _two_line(value or "Режим", "режим обработки")


def _property_label(value: str, raw: dict[str, Any]) -> str:
    low = value.lower().replace("ё", "е")
    unit = _first_present(raw, "unit_normalized", "unit") or ""
    if "strength" in low or "прочн" in low:
        return "Прочность\nMPa"
    if "hardness" in low or "тверд" in low:
        return "Твердость\nHV"
    if "corrosion" in low or "корроз" in low:
        return "Коррозионная стойкость\nкачественное свойство"
    return _two_line(value or "Свойство", str(unit or "измеряемое свойство"))


def _measurement_label(raw: dict[str, Any]) -> str:
    prop = _clean_node_text(raw.get("property") or raw.get("label") or raw.get("name") or "Значение")
    prop_label = "Прочность" if "проч" in prop.lower() or "strength" in prop.lower() else "Твердость" if "hard" in prop.lower() or "тверд" in prop.lower() else "Значение"
    value = _first_present(raw, "value_original", "value", "raw_value")
    unit = _first_present(raw, "unit_original", "unit")
    normalized_value = _first_present(raw, "value_normalized")
    normalized_unit = _first_present(raw, "unit_normalized")
    if value is None:
        parsed = _parse_value_unit(str(raw.get("label") or ""))
        value, unit = parsed if parsed else (None, None)
    if value is not None and unit:
        norm_value, note = normalize_strength_to_mpa(value, unit)
        if normalized_value is None and norm_value is not None:
            normalized_value, normalized_unit = norm_value, "MPa"
        original = f"{_format_number(value)} {unit}"
        if normalized_value is not None and normalized_unit and not _same_value_unit(value, unit, normalized_value, normalized_unit):
            return f"{prop_label}\n{original} ≈ {_format_number(normalized_value)} {normalized_unit}"
        return f"{prop_label}\n{original}"
    return _two_line(prop_label, "значение свойства")


def _experiment_label(raw: dict[str, Any], payload: dict[str, Any]) -> str:
    material = _first_present(raw, "material") or _first_query_value(payload, "materials") or ""
    regime = _first_present(raw, "regime") or _first_query_value(payload, "regimes") or ""
    suffix = " · ".join(str(item) for item in [material, regime] if item)
    return _two_line("Эксперимент", suffix or "связанный факт")


def _source_label(raw: dict[str, Any]) -> str:
    source = friendly_source_name(
        raw.get("source_name") or raw.get("title") or raw.get("filename") or raw.get("label"),
        source_url=raw.get("source_url"),
        source_type=raw.get("source_type"),
        title=raw.get("title"),
    )
    return _two_line(source or "Источник", "подтверждает данные")


def _gap_label(raw: dict[str, Any]) -> str:
    prop = _clean_node_text(raw.get("property") or raw.get("gap") or raw.get("reason") or raw.get("label") or "")
    if not prop:
        prop = "недостающие данные"
    return _two_line("Пробел в данных", prop)


def _conflict_label(raw: dict[str, Any]) -> str:
    prop = _clean_node_text(raw.get("property") or raw.get("label") or "разные значения")
    return _two_line("Неоднородность", prop)


def _synthetic_edges(nodes: list[FullGraphNode]) -> list[FullGraphEdge]:
    by_type: dict[str, list[FullGraphNode]] = {}
    for node in nodes:
        by_type.setdefault(node.type, []).append(node)
    edges: list[FullGraphEdge] = []
    materials = by_type.get("Material") or []
    regimes = by_type.get("ProcessRegime") or []
    properties = by_type.get("Property") or []
    measurements = by_type.get("Measurement") or by_type.get("Fact") or []
    sources = (by_type.get("Source") or []) + (by_type.get("Evidence") or [])
    gaps = by_type.get("DataGap") or []
    conflicts = by_type.get("Conflict") or []
    for material in materials[:3]:
        for regime in regimes[:3]:
            edges.append(_edge(material, regime, "режим"))
        for prop in properties[:3]:
            edges.append(_edge(material, prop, "свойство"))
    for prop in properties[:3]:
        for measurement in measurements[:5]:
            edges.append(_edge(prop, measurement, "значение"))
        for gap in gaps[:3]:
            edges.append(_edge(prop, gap, "пробел"))
        for conflict in conflicts[:3]:
            edges.append(_edge(prop, conflict, "конфликт"))
    for measurement in measurements[:5]:
        for source in sources[:4]:
            edges.append(_edge(measurement, source, "источник"))
    return edges


def _edge(source: FullGraphNode, target: FullGraphNode, label: str) -> FullGraphEdge:
    return FullGraphEdge(source=source.id, target=target.id, relation_raw=label, relation_readable=label, title=label)


def _dedupe_edges(edges: list[FullGraphEdge]) -> list[FullGraphEdge]:
    seen = set()
    result: list[FullGraphEdge] = []
    for edge in edges:
        key = (edge.source, edge.target, edge.relation_readable or edge.relation_raw)
        if key in seen:
            continue
        seen.add(key)
        result.append(edge)
    return result


def _full_graph_positions(nodes: list[FullGraphNode], width: int, height: int) -> dict[str, dict[str, float]]:
    columns: dict[str, int] = {
        "QueryNode": 0,
        "AnswerFocusNode": 1,
        "Material": 2,
        "ProcessRegime": 3,
        "Property": 3,
        "Measurement": 4,
        "Fact": 4,
        "CanonicalFactNode": 4,
        "Experiment": 4,
        "EvidenceGroupNode": 5,
        "Conflict": 5,
        "ConflictGroupNode": 5,
        "DataGap": 5,
        "DataGapNode": 5,
        "ConclusionNode": 6,
        "Source": 5,
        "Evidence": 5,
        "Equipment": 3,
        "Laboratory": 5,
        "Team": 5,
    }
    grouped: dict[int, list[FullGraphNode]] = {}
    for node in nodes:
        grouped.setdefault(columns.get(node.type, 2), []).append(node)
    max_col = max(grouped, default=0)
    positions: dict[str, dict[str, float]] = {}
    for col, col_nodes in grouped.items():
        x = 96 + (width - 192) * (col / max(max_col, 1))
        spacing = height / (len(col_nodes) + 1)
        for idx, node in enumerate(col_nodes, start=1):
            positions[node.id] = {"x": x, "y": spacing * idx}
    return positions


def _readable_relation(value: str) -> str:
    raw = str(value or "")
    mapping = {
        "MEASURES": "значение",
        "OF_PROPERTY": "свойство",
        "GAP_FOR_PROPERTY": "пробел",
        "GAP_FOR_ENTITY": "пробел",
        "SUPPORTED_BY": "подтверждает",
        "FACT_SUPPORTED_BY_CHUNK": "источник",
        "HAS_REGIME": "режим",
        "USES_REGIME": "режим",
        "USES_MATERIAL": "материал",
        "STUDIES": "материал",
    }
    return mapping.get(raw, "") if RAW_RELATION_RE.search(raw) else _shorten(_clean_node_text(raw), 18)


def _node_title(raw: dict[str, Any], node_type: str, label: str) -> str:
    parts = [label.replace("\n", " · "), _type_ru(node_type)]
    value = _measurement_label(raw) if node_type in {"Measurement", "Fact"} else ""
    if value and value not in parts:
        parts.append(value.replace("\n", " · "))
    source = _source_from_node(raw)
    if source:
        parts.append(f"Источник: {source}")
    return "; ".join(part for part in parts if part)


def _source_from_node(raw: dict[str, Any]) -> str | None:
    return friendly_source_name(
        raw.get("source_name") or raw.get("title") or raw.get("filename"),
        source_url=raw.get("source_url"),
        source_type=raw.get("source_type"),
        title=raw.get("title"),
    ) if any(raw.get(key) for key in ["source_name", "title", "filename", "source_url"]) else None


def _safe_details(raw: dict[str, Any]) -> dict[str, Any]:
    props = raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
    return {key: value for key, value in {**props, **raw}.items() if key not in {"id", "label", "properties"}}


def _clean_node_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\bMaterial\s*:", "Материал:", text, flags=re.IGNORECASE)
    text = re.sub(r"\bProcessRegime\s*:", "Режим:", text, flags=re.IGNORECASE)
    text = re.sub(r"\bProcess\s*:", "Режим:", text, flags=re.IGNORECASE)
    text = TECH_PREFIX_RE.sub("", text)
    text = RAW_NODE_RE.sub("", text)
    text = RAW_RELATION_RE.sub("", text)
    replacements = {
        "PropertyValue": "",
        "SourceChunk": "Источник",
        "Experiment": "Эксперимент",
        "effect: increase": "эффект: рост",
        "effect: decrease": "эффект: снижение",
        "effect: unknown": "эффект не указан",
        "increase": "рост",
        "decrease": "снижение",
        "unknown": "не указано",
    }
    for raw, replacement in replacements.items():
        text = re.sub(re.escape(raw), replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" :;,.|-_")


def _first_present(raw: dict[str, Any], *keys: str) -> Any:
    props = raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
    for key in keys:
        value = raw.get(key)
        if value is not None and value != "":
            return value
        prop_value = props.get(key)
        if prop_value is not None and prop_value != "":
            return prop_value
    return None


def _first_query_value(payload: dict[str, Any], key: str) -> str | None:
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    values = constraints.get(key) or []
    return str(values[0]) if values else None


def _parse_value_unit(text: str) -> tuple[float, str] | None:
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(MPa|МПа|ksi|HV|HRC)", text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace(",", ".")), match.group(2)


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(number) and abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.1f}".rstrip("0").rstrip(".")


def _same_value_unit(left_value: Any, left_unit: Any, right_value: Any, right_unit: Any) -> bool:
    try:
        return abs(float(left_value) - float(right_value)) < 1e-9 and str(left_unit) == str(right_unit)
    except (TypeError, ValueError):
        return False


def _two_line(first: str, second: str) -> str:
    return f"{_shorten(first, 32)}\n{_shorten(second, 38)}" if second else _shorten(first, 54)


def _label_key(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip().lower()


def _type_order(node_type: str) -> int:
    order = ["Material", "ProcessRegime", "Property", "Measurement", "Fact", "Experiment", "Conflict", "DataGap", "Source", "Evidence"]
    return order.index(node_type) if node_type in order else len(order)


def _type_ru(node_type: str) -> str:
    return {
        "QueryNode": "вопрос",
        "AnswerFocusNode": "фокус ответа",
        "CanonicalFactNode": "канонический факт",
        "EvidenceGroupNode": "группа источников",
        "ConflictGroupNode": "группа конфликтов",
        "DataGapNode": "пробел данных",
        "ConclusionNode": "вывод",
        "Material": "материал",
        "ProcessRegime": "режим",
        "Property": "свойство",
        "Measurement": "измерение",
        "Fact": "факт",
        "Experiment": "эксперимент",
        "Source": "источник",
        "Evidence": "evidence",
        "DataGap": "пробел",
        "Conflict": "конфликт",
    }.get(node_type, node_type)


def _node_css_class(node_type: str) -> str:
    return {
        "QueryNode": "fg-query",
        "AnswerFocusNode": "fg-focus",
        "Material": "fg-material",
        "ProcessRegime": "fg-regime",
        "Property": "fg-property",
        "CanonicalFactNode": "fg-measurement",
        "EvidenceGroupNode": "fg-source",
        "ConflictGroupNode": "fg-conflict",
        "DataGapNode": "fg-gap",
        "ConclusionNode": "fg-conclusion",
        "Measurement": "fg-measurement",
        "Fact": "fg-fact",
        "Experiment": "fg-experiment",
        "Source": "fg-source",
        "Evidence": "fg-evidence",
        "DataGap": "fg-gap",
        "Conflict": "fg-conflict",
        "Equipment": "fg-equipment",
        "Laboratory": "fg-laboratory",
        "Team": "fg-team",
    }.get(node_type, "fg-entity")


def _label_lines(label: str, width: int) -> list[str]:
    result: list[str] = []
    for raw_line in str(label).replace("\\n", "\n").splitlines():
        result.extend(_wrap_label(raw_line, width))
    return result or _wrap_label(str(label), width)


def _wrap_label(label: str, width: int) -> list[str]:
    words = str(label).replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [label]


def _shorten(value: Any, limit: int) -> str:
    text = _clean_node_text(value)
    return text[: limit - 1] + "…" if len(text) > limit else text

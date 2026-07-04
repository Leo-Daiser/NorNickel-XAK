"""User-facing answer evidence map.

This module builds a compact semantic graph from an /ask payload. It is
intentionally different from the raw technical subgraph: the main UI should
show the answer path, not database labels and internal ids.
"""

from __future__ import annotations

import html
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..domain.unit_normalization import normalize_strength_to_mpa
from .answer_graph_labels import enrich_answer_graph_labels


FORBIDDEN_LABEL_RE = re.compile(
    r"\b(?:doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|EXP-[A-Za-z0-9_-]+|SCI-[A-Za-z0-9_-]+|"
    r"Experiment|PropertyValue|SourceChunk|FACT_SUPPORTED_BY_CHUNK|OF_PROPERTY|STUDIES|MEASURES|USES_REGIME|"
    r"increase|decrease|unknown)\b"
)


class AnswerGraphNode(BaseModel):
    """A compact user-facing answer graph node."""

    id: str
    label: str
    type: Literal[
        "material",
        "regime",
        "property",
        "measurement_summary",
        "fact",
        "source_summary",
        "gap",
        "conclusion",
    ]
    tooltip: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AnswerGraphEdge(BaseModel):
    """A compact user-facing answer graph edge."""

    source: str
    target: str
    label: str | None = None


class AnswerGraph(BaseModel):
    """Semantic graph rendered as the main UI answer map."""

    nodes: list[AnswerGraphNode]
    edges: list[AnswerGraphEdge]
    title: str
    summary: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def translate_effect(effect: str | None) -> str:
    """Translate internal effect enum to Russian user-facing text."""

    if effect is None:
        return "эффект не указан"
    return {
        "increase": "рост",
        "decrease": "снижение",
        "no_change": "без заметного изменения",
        "unchanged": "без заметного изменения",
        "mixed": "смешанный эффект",
        "unknown": "эффект не указан",
        "": "эффект не указан",
    }.get(str(effect), str(effect))


def make_human_graph_label(node_or_fact: dict[str, Any]) -> str:
    """Return a safe short label for the answer graph."""

    label = str(
        node_or_fact.get("label")
        or node_or_fact.get("name")
        or node_or_fact.get("canonical_name")
        or node_or_fact.get("property")
        or node_or_fact.get("material")
        or node_or_fact.get("regime")
        or ""
    ).strip()
    if not label or FORBIDDEN_LABEL_RE.search(label):
        label = _fallback_label(node_or_fact)
    label = re.sub(r"\s+", " ", label.replace("\\n", " ").replace("\n", " ")).strip()
    label = label.replace("effect: increase", "эффект: рост")
    label = label.replace("effect: decrease", "эффект: снижение")
    label = label.replace("effect: unknown", "эффект не указан")
    label = FORBIDDEN_LABEL_RE.sub("", label).strip(" :-")
    return _shorten(label or "факт", 40)


def build_answer_graph(payload: dict[str, Any]) -> AnswerGraph:
    """Build a compact semantic answer graph from an /ask response payload."""

    if payload.get("status") == "no_exact_match":
        return enrich_answer_graph_labels(_build_no_match_graph(payload), payload)
    if _is_comparison(payload):
        return enrich_answer_graph_labels(_build_comparison_graph(payload), payload)
    if _is_overview(payload):
        return enrich_answer_graph_labels(_build_overview_graph(payload), payload)
    return enrich_answer_graph_labels(_build_strict_or_generic_graph(payload), payload)


def answer_graph_to_html(
    graph: AnswerGraph,
    *,
    render_height: int | None = None,
    render_width: int | None = None,
    container_id: str = "answerGraph",
) -> str:
    """Render AnswerGraph as layered interactive SVG HTML."""

    if not graph.nodes:
        return "<div style='padding:16px;color:#64748b'>Для ответа не удалось построить компактный граф.</div>"
    nodes = graph.nodes[:10]
    allowed = {node.id for node in nodes}
    edges = [edge for edge in graph.edges if edge.source in allowed and edge.target in allowed][:12]
    node_width, node_height = 176, 78
    height = render_height or max(420, 130 + 98 * max(_max_level_count(nodes), 1))
    width = render_width or (1500 if height >= 700 else 940)
    svg_id = f"{container_id}Svg"
    viewport_id = f"{container_id}Viewport"
    arrow_id = f"{container_id}Arrow"
    positions = _layer_positions(nodes, width, height)
    edge_lines = []
    for edge in edges:
        source = positions.get(edge.source)
        target = positions.get(edge.target)
        if not source or not target:
            continue
        edge_lines.append(
            f"<path class='ag-edge' d='M {source['x'] + node_width / 2:.1f} {source['y']:.1f} C "
            f"{source['x'] + 126:.1f} {source['y']:.1f}, {target['x'] - 126:.1f} {target['y']:.1f}, "
            f"{target['x'] - node_width / 2:.1f} {target['y']:.1f}' />"
        )
        if edge.label:
            edge_lines.append(
                f"<text class='ag-edge-label' x='{(source['x'] + target['x']) / 2:.1f}' y='{(source['y'] + target['y']) / 2 - 8:.1f}'>"
                f"{html.escape(_shorten(edge.label, 22))}</text>"
            )
    node_items = []
    for idx, node in enumerate(nodes):
        pos = positions[node.id]
        label_lines = _label_lines(node.label, 20)[:3]
        start_y = -14 if len(label_lines) == 3 else -7 if len(label_lines) == 2 else 0
        text_spans = "".join(
            f"<tspan x='0' dy='{0 if line_idx == 0 else 16}'>{html.escape(line)}</tspan>"
            for line_idx, line in enumerate(label_lines)
        )
        tooltip = html.escape(node.tooltip or node.label)
        node_items.append(
            f"<g class='ag-node' data-node-id='node-{idx}' transform='translate({pos['x']:.1f},{pos['y']:.1f})'>"
            f"<title>{tooltip}</title><rect x='-{node_width / 2:.0f}' y='-{node_height / 2:.0f}' "
            f"width='{node_width}' height='{node_height}' rx='10' class='ag-{node.type}' data-fixed='true' />"
            f"<text text-anchor='middle' y='{start_y}'>{text_spans}</text></g>"
        )
    return f"""
<div class="answer-graph-wrap">
  <div class="answer-graph-title">{html.escape(graph.title)}</div>
  <div class="answer-graph-help">Колесо — масштаб · фон — перемещение карты · узлы зафиксированы</div>
  <svg id="{svg_id}" viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">
    <defs><marker id="{arrow_id}" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#64748b"></path></marker></defs>
    <g id="{viewport_id}">{"".join(edge_lines)}{"".join(node_items)}</g>
  </svg>
</div>
<style>
.answer-graph-wrap {{ border:1px solid #d8dee9; border-radius:10px; background:#fff; position:relative; overflow:hidden; }}
.answer-graph-title {{ position:absolute; left:12px; top:8px; z-index:2; font:600 13px Arial; color:#334155; background:rgba(255,255,255,.9); padding:2px 6px; border-radius:4px; }}
.answer-graph-help {{ position:absolute; right:10px; top:8px; z-index:2; font:12px Arial; color:#64748b; background:rgba(255,255,255,.86); padding:2px 6px; border-radius:4px; }}
#{svg_id} {{ cursor:grab; touch-action:none; }}
.ag-edge {{ fill:none; stroke:#64748b; stroke-width:1.5; marker-end:url(#{arrow_id}); opacity:.72; }}
.ag-edge-label {{ fill:#475569; font:10px Arial; paint-order:stroke; stroke:#fff; stroke-width:3px; }}
.ag-node rect {{ stroke:#334155; stroke-width:1.1; filter: drop-shadow(0 2px 4px rgba(15,23,42,.14)); }}
.ag-node text {{ fill:#0f172a; font:12px Arial; pointer-events:none; }}
.ag-node {{ cursor:default; }}
.ag-material {{ fill:#bfdbfe; }}
.ag-regime {{ fill:#bbf7d0; }}
.ag-property {{ fill:#fde68a; }}
.ag-measurement_summary, .ag-fact {{ fill:#ddd6fe; }}
.ag-source_summary {{ fill:#e2e8f0; }}
.ag-gap {{ fill:#fecaca; }}
.ag-conclusion {{ fill:#ccfbf1; }}
</style>
<script>
(function() {{
 const graphOptions = {{ layout: {{ hierarchical: {{ enabled: true, direction: "LR" }} }}, physics: {{ enabled: false }}, interaction: {{ zoomView: true, dragView: true, dragNodes: false }} }};
 const svg = document.getElementById('{svg_id}');
 const viewport = document.getElementById('{viewport_id}');
 let state = {{x:0, y:0, scale:1}};
 let drag = null;
 function apply() {{ viewport.setAttribute('transform', `translate(${{state.x}},${{state.y}}) scale(${{state.scale}})`); }}
 function isNodeHit(ev) {{
   return Array.from(document.querySelectorAll('.ag-node')).some(function(node) {{
     const box = node.getBoundingClientRect();
     return ev.clientX >= box.left && ev.clientX <= box.right && ev.clientY >= box.top && ev.clientY <= box.bottom;
   }});
 }}
 svg.addEventListener('wheel', function(ev) {{
   ev.preventDefault();
   const delta = ev.deltaY < 0 ? 1.12 : 0.89;
   state.scale = Math.max(0.35, Math.min(3.2, state.scale * delta));
   apply();
 }}, {{passive:false}});
 svg.addEventListener('pointerdown', function(ev) {{
   const node = ev.target.closest && ev.target.closest('.ag-node');
   drag = (node || isNodeHit(ev)) ? null : {{kind: 'pan', startX: ev.clientX, startY: ev.clientY, x: state.x, y: state.y}};
   svg.setPointerCapture(ev.pointerId);
 }});
 svg.addEventListener('pointermove', function(ev) {{
   if (!drag) return;
   const dx = ev.clientX - drag.startX, dy = ev.clientY - drag.startY;
   state.x = drag.x + dx; state.y = drag.y + dy; apply();
 }});
 svg.addEventListener('pointerup', function(ev) {{ drag = null; try {{ svg.releasePointerCapture(ev.pointerId); }} catch(e) {{}} }});
 apply();
}})();
</script>
"""


def _build_strict_or_generic_graph(payload: dict[str, Any]) -> AnswerGraph:
    constraints = payload.get("constraints") or {}
    facts = _primary_facts(payload)
    materials = constraints.get("materials") or _unique(fact.get("material") for fact in facts)
    regimes = constraints.get("regimes") or _unique(fact.get("regime") for fact in facts)
    properties = constraints.get("properties") or _unique(fact.get("property") for fact in facts)
    material = _first(materials, "материал")
    regime = _first(regimes, "режим")
    prop = _first(properties, "свойство")
    nodes = [
        AnswerGraphNode(id="material", label=make_human_graph_label({"label": material}), type="material"),
        AnswerGraphNode(id="regime", label=make_human_graph_label({"label": regime}), type="regime"),
        AnswerGraphNode(id="property", label=make_human_graph_label({"label": prop}), type="property"),
    ]
    edges = [AnswerGraphEdge(source="material", target="regime"), AnswerGraphEdge(source="regime", target="property")]
    summary = _measurement_summary_label(facts)
    nodes.append(AnswerGraphNode(id="measurement_summary", label=summary, type="measurement_summary", tooltip=_facts_tooltip(facts)))
    edges.append(AnswerGraphEdge(source="property", target="measurement_summary"))
    for idx, fact in enumerate(facts[:3]):
        node_id = f"fact_{idx}"
        nodes.append(AnswerGraphNode(id=node_id, label=_fact_label(fact), type="fact", tooltip=_fact_tooltip(fact)))
        edges.append(AnswerGraphEdge(source="property", target=node_id))
    source_count = len(payload.get("evidence") or payload.get("sources") or [])
    nodes.append(AnswerGraphNode(id="sources", label=f"{source_count} источников" if source_count else "источники не указаны", type="source_summary"))
    edges.append(AnswerGraphEdge(source="measurement_summary", target="sources"))
    return _limit_graph(AnswerGraph(nodes=nodes, edges=edges, title="Карта ответа", summary=summary))


def _build_no_match_graph(payload: dict[str, Any]) -> AnswerGraph:
    constraints = payload.get("constraints") or {}
    material = _first(constraints.get("materials") or [], "материал")
    regime = _first(constraints.get("regimes") or [], "режим")
    prop = _first(constraints.get("properties") or [], "свойство")
    nodes = [
        AnswerGraphNode(id="material", label=material, type="material"),
        AnswerGraphNode(id="regime", label=regime, type="regime"),
        AnswerGraphNode(id="property", label=prop, type="property"),
        AnswerGraphNode(id="gap", label="точных данных нет", type="gap", tooltip="Exact path material + regime + property не найден."),
        AnswerGraphNode(id="conclusion", label="пробел в данных", type="conclusion"),
    ]
    edges = [
        AnswerGraphEdge(source="material", target="regime"),
        AnswerGraphEdge(source="regime", target="property"),
        AnswerGraphEdge(source="property", target="gap"),
        AnswerGraphEdge(source="gap", target="conclusion"),
    ]
    return AnswerGraph(nodes=nodes, edges=edges, title="Карта отсутствующего exact-факта")


def _build_overview_graph(payload: dict[str, Any]) -> AnswerGraph:
    facts = _primary_facts(payload) or payload.get("facts") or []
    constraints = payload.get("constraints") or {}
    material = _first(constraints.get("materials") or _unique(fact.get("material") for fact in facts), "запрос")
    regimes = _unique(fact.get("regime") for fact in facts if fact.get("regime"))
    properties = _unique(fact.get("property") for fact in facts if fact.get("property"))
    source_count = len(payload.get("evidence") or payload.get("sources") or [])
    gaps_count = len(payload.get("data_gaps") or payload.get("gaps") or [])
    nodes = [
        AnswerGraphNode(id="material", label=material, type="material"),
        AnswerGraphNode(id="regimes", label="режимы: " + _join_short(regimes), type="regime"),
        AnswerGraphNode(id="properties", label="свойства: " + _join_short(properties), type="property"),
        AnswerGraphNode(id="sources", label=f"источники: {source_count}", type="source_summary"),
    ]
    edges = [
        AnswerGraphEdge(source="material", target="regimes"),
        AnswerGraphEdge(source="material", target="properties"),
        AnswerGraphEdge(source="material", target="sources"),
    ]
    if gaps_count:
        nodes.append(AnswerGraphNode(id="gaps", label=f"пробелы: {gaps_count}", type="gap"))
        edges.append(AnswerGraphEdge(source="material", target="gaps"))
    return _limit_graph(AnswerGraph(nodes=nodes, edges=edges, title="Обзорная карта"))


def _build_comparison_graph(payload: dict[str, Any]) -> AnswerGraph:
    facts = payload.get("facts") or payload.get("primary_facts") or []
    constraints = payload.get("constraints") or {}
    materials = constraints.get("materials") or _unique(fact.get("material") for fact in facts)[:2]
    prop = _first(constraints.get("properties") or _unique(fact.get("property") for fact in facts), "сравниваемое свойство")
    nodes: list[AnswerGraphNode] = []
    edges: list[AnswerGraphEdge] = []
    for idx, material in enumerate(materials[:2]):
        node_id = f"material_{idx}"
        summary_id = f"summary_{idx}"
        material_facts = [
            fact for fact in facts
            if _same_text(str(fact.get("material") or ""), str(material))
            and (not constraints.get("properties") or _same_text(str(fact.get("property") or ""), str(prop)))
        ]
        nodes.append(AnswerGraphNode(id=node_id, label=make_human_graph_label({"label": material}), type="material"))
        nodes.append(AnswerGraphNode(id=summary_id, label=_comparison_range_label(material_facts), type="measurement_summary"))
        edges.append(AnswerGraphEdge(source=node_id, target=summary_id))
        edges.append(AnswerGraphEdge(source=summary_id, target="property"))
    nodes.append(AnswerGraphNode(id="property", label=prop, type="property"))
    nodes.append(AnswerGraphNode(id="conclusion", label="сравнение ограничено", type="conclusion", tooltip="Проверьте режимы, единицы и источники."))
    edges.append(AnswerGraphEdge(source="property", target="conclusion"))
    return _limit_graph(AnswerGraph(nodes=nodes, edges=edges, title="Карта сравнения"))


def _primary_facts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("primary_facts") or payload.get("facts") or []
    return [row for row in rows if isinstance(row, dict)]


def _measurement_summary_label(facts: list[dict[str, Any]]) -> str:
    values = [float(fact.get("value")) for fact in facts if isinstance(fact.get("value"), int | float)]
    unit = next((str(fact.get("unit")) for fact in facts if fact.get("unit")), "")
    if values:
        low, high = min(values), max(values)
        range_text = f"{low:g} {unit}" if low == high else f"{low:g}–{high:g} {unit}".strip()
        return f"{range_text}\n{len(facts)} фактов"
    return f"{len(facts)} подтверждённых фактов" if facts else "факты не найдены"


def _fact_label(fact: dict[str, Any]) -> str:
    value = fact.get("value") if fact.get("value") is not None else fact.get("raw_value")
    unit = fact.get("unit") or ""
    prop = fact.get("property") or "свойство"
    effect = translate_effect(fact.get("effect"))
    if value is None or value == "":
        return _shorten(f"{prop}\n{effect}", 40)
    return _shorten(f"{value:g} {unit}\n{effect}" if isinstance(value, float) else f"{value} {unit}\n{effect}", 40)


def _fact_tooltip(fact: dict[str, Any]) -> str:
    return "; ".join(str(item) for item in [fact.get("material"), fact.get("regime"), fact.get("property"), _fact_label(fact)] if item)


def _facts_tooltip(facts: list[dict[str, Any]]) -> str:
    return "\n".join(_fact_tooltip(fact) for fact in facts[:6])


def _layer_positions(nodes: list[AnswerGraphNode], width: int, height: int) -> dict[str, dict[str, float]]:
    levels: dict[str, int] = {
        "material": 0,
        "regime": 1,
        "property": 2,
        "measurement_summary": 3,
        "fact": 3,
        "gap": 3,
        "source_summary": 4,
        "conclusion": 4,
    }
    grouped: dict[int, list[AnswerGraphNode]] = {}
    for node in nodes:
        grouped.setdefault(levels.get(node.type, 3), []).append(node)
    positions: dict[str, dict[str, float]] = {}
    max_level = max(grouped) if grouped else 0
    for level, level_nodes in grouped.items():
        x = 90 + (width - 180) * (level / max(max_level, 1))
        spacing = height / (len(level_nodes) + 1)
        for idx, node in enumerate(level_nodes, start=1):
            positions[node.id] = {"x": x, "y": spacing * idx}
    return positions


def _max_level_count(nodes: list[AnswerGraphNode]) -> int:
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.type] = counts.get(node.type, 0) + 1
    return max(counts.values(), default=1)


def _limit_graph(graph: AnswerGraph) -> AnswerGraph:
    graph.nodes = graph.nodes[:10]
    allowed = {node.id for node in graph.nodes}
    graph.edges = [edge for edge in graph.edges if edge.source in allowed and edge.target in allowed][:12]
    graph.diagnostics.update({"main_graph_nodes": len(graph.nodes), "main_graph_edges": len(graph.edges)})
    return graph


def _is_overview(payload: dict[str, Any]) -> bool:
    answer_mode = str(payload.get("answer_mode") or "")
    intent = str(payload.get("analytical_intent") or payload.get("intent") or "")
    return answer_mode == "overview" or intent.endswith("_overview") or intent in {"topic_search", "graph_neighborhood", "equipment_usage", "lab_activity"}


def _is_comparison(payload: dict[str, Any]) -> bool:
    return str(payload.get("answer_mode") or "") == "comparison" or "comparison" in str(payload.get("analytical_intent") or "")


def _fallback_label(item: dict[str, Any]) -> str:
    item_type = str(item.get("type") or "")
    if item_type == "Document" or item_type == "DocumentChunk":
        return "источник"
    if item_type == "Experiment":
        return "эксперимент"
    if item_type == "PropertyValue":
        return "значение свойства"
    return str(item.get("type") or "узел")


def _comparison_range_label(facts: list[dict[str, Any]]) -> str:
    values: list[float] = []
    converted = False
    for fact in facts:
        value = fact.get("value") if fact.get("value") is not None else fact.get("raw_value")
        normalized, note = normalize_strength_to_mpa(value, fact.get("unit"))
        if normalized is None:
            continue
        values.append(normalized)
        converted = converted or bool(note)
    if not values:
        return "нет численного диапазона"
    low, high = min(values), max(values)
    label = f"{_format_mpa(low)} MPa" if low == high else f"{_format_mpa(low)}–{_format_mpa(high)} MPa"
    return f"{label}\nпересчёт" if converted else label


def _same_text(left: str, right: str) -> bool:
    left_norm = left.strip().lower().replace("ё", "е")
    right_norm = right.strip().lower().replace("ё", "е")
    return bool(left_norm and right_norm and (left_norm == right_norm or left_norm in right_norm or right_norm in left_norm))


def _format_mpa(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:g}"


def _unique(values: Any) -> list[str]:
    seen = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        text = str(value)
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _first(values: list[str], default: str) -> str:
    return values[0] if values else default


def _join_short(values: list[str]) -> str:
    if not values:
        return "нет данных"
    text = ", ".join(values[:4])
    return text + (f" +{len(values) - 4}" if len(values) > 4 else "")


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


def _shorten(value: str, limit: int) -> str:
    value = FORBIDDEN_LABEL_RE.sub("", str(value)).replace("\n", " ").strip()
    return value[: limit - 1] + "…" if len(value) > limit else value

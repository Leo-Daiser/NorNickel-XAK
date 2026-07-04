from __future__ import annotations

from app.graph.full_answer_graph import (
    EXPLANATION_GRAPH_MAX_EDGES,
    EXPLANATION_GRAPH_MAX_NODES,
    FULL_GRAPH_MAX_EDGES,
    FULL_GRAPH_MAX_NODES,
    build_full_answer_graph,
    full_answer_graph_to_html,
    full_graph_audit_tables,
)


FORBIDDEN_CANVAS_TOKENS = [
    "doc_",
    "chunk_",
    "EXP-",
    "SCI-",
    "DataGap:",
    "Material:",
    "ProcessRegime:",
    "PropertyValue",
    "SourceChunk",
    "Experiment:",
    "MEASURES",
    "OF_PROPERTY",
    "GAP_FOR_PROPERTY",
    "GAP_FOR_ENTITY",
    "SUPPORTED_BY",
    "HAS_REGIME",
    "USES_MATERIAL",
]


def _payload_with_raw_subgraph(extra_nodes: int = 0) -> dict:
    nodes = [
        {"id": "Experiment:SCI-VT6-AN-900", "label": "Experiment:SCI-VT6-AN-900", "type": "Experiment"},
        {"id": "Material:ВТ6", "label": "Material:ВТ6", "type": "Material"},
        {"id": "ProcessRegime:отжиг", "label": "ProcessRegime:отжиг", "type": "ProcessRegime"},
        {"id": "Property:прочность", "label": "Property:прочность", "type": "Property"},
        {
            "id": "PropertyValue:pv1",
            "label": "прочность: 77.0 ksi",
            "type": "PropertyValue",
            "value_original": 77.0,
            "unit_original": "ksi",
            "value_normalized": 530.9,
            "unit_normalized": "MPa",
            "property": "прочность",
        },
        {"id": "SourceChunk:chunk_abc", "label": "chunk_abc", "type": "SourceChunk", "source_name": "article_vt6.txt"},
        {"id": "DataGap:12b274", "label": "DataGap:12b274", "type": "DataGap", "property": "класс герметичности DN50"},
    ]
    for idx in range(extra_nodes):
        nodes.append({"id": f"doc_extra_{idx}", "label": f"doc_extra_{idx}", "type": "SourceChunk"})
    edges = [
        {"source": "Experiment:SCI-VT6-AN-900", "target": "Material:ВТ6", "label": "USES_MATERIAL"},
        {"source": "Experiment:SCI-VT6-AN-900", "target": "ProcessRegime:отжиг", "label": "HAS_REGIME"},
        {"source": "Experiment:SCI-VT6-AN-900", "target": "PropertyValue:pv1", "label": "MEASURES"},
        {"source": "PropertyValue:pv1", "target": "Property:прочность", "label": "OF_PROPERTY"},
        {"source": "PropertyValue:pv1", "target": "SourceChunk:chunk_abc", "label": "SUPPORTED_BY"},
        {"source": "DataGap:12b274", "target": "Property:прочность", "label": "GAP_FOR_PROPERTY"},
    ]
    for idx in range(extra_nodes):
        edges.append({"source": "Material:ВТ6", "target": f"doc_extra_{idx}", "label": "SUPPORTED_BY"})
    return {
        "constraints": {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]},
        "facts": [
            {
                "material": "7075-T6",
                "regime": "старение",
                "property": "прочность",
                "value_original": 77.0,
                "unit_original": "ksi",
                "value_normalized": 530.9,
                "unit_normalized": "MPa",
                "value": 77.0,
                "unit": "ksi",
            }
        ],
        "data_gaps": [{"property": "класс герметичности DN50", "reason": "not specified"}],
        "diagnostics": {
            "fact_conflicts": [
                {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "values": [{"value": 980, "unit": "MPa"}]}
            ]
        },
        "subgraph": {"nodes": nodes, "edges": edges},
    }


def test_full_graph_canvas_uses_readable_labels_without_raw_ids() -> None:
    graph = build_full_answer_graph(_payload_with_raw_subgraph())
    html = full_answer_graph_to_html(graph, container_id="testFullGraph")
    visible = html.split("<style>", 1)[0]
    labels = "\n".join(node.label for node in graph.nodes)

    assert any(node.type == "QueryNode" for node in graph.nodes)
    assert any(node.type == "AnswerFocusNode" for node in graph.nodes)
    assert any(node.type == "CanonicalFactNode" for node in graph.nodes)
    assert any(node.type == "ConclusionNode" for node in graph.nodes)
    assert "ВТ6\nтитановый сплав" in labels
    assert "Отжиг\nтермообработка" in labels
    assert "Прочность\n77 ksi ≈ 530.9 MPa" in labels
    assert "Пробел\nкласс герметичности DN50" in labels
    for token in FORBIDDEN_CANVAS_TOKENS:
        assert token not in visible


def test_explanation_graph_aggregates_duplicate_material_and_measurement_nodes() -> None:
    payload = _payload_with_raw_subgraph()
    payload["facts"] = [
        {
            "material": "ВТ6",
            "regime": "отжиг",
            "property": "прочность",
            "value": 980,
            "unit": "MPa",
            "value_original": 980,
            "unit_original": "MPa",
            "value_normalized": 980,
            "unit_normalized": "MPa",
            "source_name": "article_vt6.txt",
        },
        {
            "material": "ВТ6",
            "regime": "отжиг",
            "property": "прочность",
            "value": 980,
            "unit": "MPa",
            "value_original": 980,
            "unit_original": "MPa",
            "value_normalized": 980,
            "unit_normalized": "MPa",
            "source_name": "article_vt6_copy.txt",
        },
    ]

    graph = build_full_answer_graph(payload)
    material_nodes = [node for node in graph.nodes if node.type == "Material" and node.label.startswith("ВТ6")]
    fact_nodes = [node for node in graph.nodes if node.type == "CanonicalFactNode" and "980 MPa" in node.label]

    assert len(material_nodes) == 1
    assert len(fact_nodes) == 1
    assert "2 источника" in fact_nodes[0].label or "2 источников" in fact_nodes[0].label


def test_full_graph_is_bounded_and_reports_truncation() -> None:
    graph = build_full_answer_graph(_payload_with_raw_subgraph(extra_nodes=60))

    assert len(graph.nodes) <= EXPLANATION_GRAPH_MAX_NODES
    assert len(graph.edges) <= EXPLANATION_GRAPH_MAX_EDGES
    assert FULL_GRAPH_MAX_NODES == EXPLANATION_GRAPH_MAX_NODES
    assert FULL_GRAPH_MAX_EDGES == EXPLANATION_GRAPH_MAX_EDGES
    assert graph.stats["truncated"] is True
    assert graph.stats["total_nodes"] > graph.stats["shown_nodes"]


def test_full_graph_interaction_settings_disable_node_drag_and_keep_zoom_pan() -> None:
    graph = build_full_answer_graph(_payload_with_raw_subgraph())
    html = full_answer_graph_to_html(graph, container_id="interactionGraph")

    assert "dragNodes: false" in html
    assert "zoomView: true" in html
    assert "dragView: true" in html
    assert "physics: { enabled: false }" in html
    assert "Колесо — масштаб" in html
    assert "узлы зафиксированы" in html
    assert all("x" in node.details or node.type for node in graph.nodes)


def test_full_graph_filters_hide_source_gap_conflict_layers() -> None:
    graph = build_full_answer_graph(
        _payload_with_raw_subgraph(),
        filters={"show_sources": False, "show_gaps": False, "show_conflicts": False, "show_measurements": True},
    )
    labels = "\n".join(node.label for node in graph.nodes)

    assert "Пробел в данных" not in labels
    assert "Пробел\n" not in labels
    assert "Неоднородность" not in labels
    assert "Статья" not in labels


def test_comparison_explanation_graph_has_branches_and_caveat() -> None:
    graph = build_full_answer_graph(_payload_with_raw_subgraph())
    labels = "\n".join(node.label for node in graph.nodes)

    assert "ВТ6\nтитановый сплав" in labels
    assert "7075-T6\nалюминиевый сплав\nсостояние T6" in labels
    assert "77 ksi ≈ 530.9 MPa" in labels
    assert "Вывод\nусловия неоднородны" in labels
    assert len(graph.nodes) <= EXPLANATION_GRAPH_MAX_NODES
    assert len(graph.edges) <= EXPLANATION_GRAPH_MAX_EDGES


def test_negative_or_no_data_graph_does_not_invent_numbers() -> None:
    graph = build_full_answer_graph(
        {
            "status": "no_exact_match",
            "question": "Что известно о X999 при лазерной обработке?",
            "constraints": {"materials": ["X999"], "regimes": ["лазерная обработка"], "properties": ["прочность"]},
            "facts": [],
            "data_gaps": [{"material": "X999", "property": "прочность", "reason": "нет exact данных"}],
        }
    )
    labels = "\n".join(node.label for node in graph.nodes)

    assert "Вывод\nточных данных нет" in labels
    assert not any(char.isdigit() for char in labels if char not in "999")


def test_full_graph_audit_tables_preserve_raw_ids_and_relations() -> None:
    payload = _payload_with_raw_subgraph()
    graph = build_full_answer_graph(payload)
    node_rows, edge_rows = full_graph_audit_tables(payload, graph)

    raw_node_ids = "\n".join(str(row.get("raw_id")) for row in node_rows)
    raw_relations = "\n".join(str(row.get("relation_raw")) for row in edge_rows)
    assert "Experiment:SCI-VT6-AN-900" in raw_node_ids
    assert "SourceChunk:chunk_abc" in raw_node_ids
    assert "MEASURES" in raw_relations
    assert "OF_PROPERTY" in raw_relations
    assert any(row.get("relation_readable") == "значение" for row in edge_rows)

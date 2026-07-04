from __future__ import annotations

from app.graph.answer_graph import answer_graph_to_html, build_answer_graph


def test_answer_graph_renderer_uses_layered_interactive_layout() -> None:
    graph = build_answer_graph(
        {
            "status": "ok",
            "constraints": {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]},
            "facts": [{"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 1120.0, "unit": "MPa"}],
            "sources": [{"source_name": "synthetic_vt6_heat_treatment.csv"}],
        }
    )
    html = answer_graph_to_html(graph)
    assert "<svg" in html
    assert "hierarchical" in html
    assert "zoomView" in html
    assert "dragView" in html
    assert "dragNodes" in html
    assert "dragNodes: false" in html
    assert "dragNodes: true" not in html
    assert "physics" in html
    assert "enabled: false" in html
    assert "wheel" in html
    assert "pointerdown" in html
    assert "isNodeHit" in html
    assert "Колесо — масштаб" in html
    assert "узлы зафиксированы" in html
    assert "Wheel: zoom" not in html
    assert "drag node: move" not in html
    assert "cursor:move" not in html
    assert "setAttribute('transform', `translate(${baseX" not in html


def test_answer_graph_renderer_keeps_labels_clean_in_visible_html() -> None:
    graph = build_answer_graph(
        {
            "status": "ok",
            "constraints": {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]},
            "facts": [
                {
                    "experiment_id": "EXP-123",
                    "material": "ВТ6",
                    "regime": "отжиг",
                    "property": "прочность",
                    "value": 1120.0,
                    "unit": "MPa",
                    "effect": "unknown",
                }
            ],
            "sources": [{"document_id": "doc_abc", "chunk_id": "chunk_def"}],
        }
    )
    html = answer_graph_to_html(graph)
    visible = html.split("<style>", 1)[0]
    for token in ["doc_", "chunk_", "EXP-", "SCI-", "Experiment", "PropertyValue", "SourceChunk", "effect: unknown", "unknown"]:
        assert token not in visible
    assert "эффект не указан" in visible


def test_answer_graph_renderer_supports_large_fixed_layout_height() -> None:
    graph = build_answer_graph(
        {
            "status": "ok",
            "answer_mode": "comparison",
            "analytical_intent": "material_comparison",
            "constraints": {"materials": ["ВТ6", "7075-T6"], "properties": ["прочность"]},
            "facts": [
                {"material": "ВТ6", "property": "прочность", "value": 1120.0, "unit": "MPa"},
                {"material": "7075-T6", "property": "прочность", "value": 77.0, "unit": "ksi"},
            ],
        }
    )
    html = answer_graph_to_html(graph, render_height=820, render_width=1500, container_id="answerGraphExpanded_test")
    assert 'height="820"' in html
    assert 'viewBox="0 0 1500 820"' in html
    assert "answerGraphExpanded_testSvg" in html
    assert "dragNodes: false" in html
    assert "алюминиевый сплав" in html

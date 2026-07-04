from __future__ import annotations

from app.ui_helpers import clean_graph_label, graph_to_interactive_html


def test_graph_label_cleanup_hides_internal_ids() -> None:
    labels = [
        clean_graph_label({"id": "doc_123", "label": "doc_29765440445babcdef", "type": "Document"}),
        clean_graph_label({"id": "chunk_123", "label": "SourceChunk doc_abc chunk_def", "type": "DocumentChunk"}),
        clean_graph_label({"id": "EXP-123", "label": "Experiment SCI-VT6-AN-900", "type": "Experiment"}),
        clean_graph_label({"id": "m1", "label": "Прочность: 1120 MPa, effect: increase", "type": "Measurement"}),
    ]
    joined = " ".join(labels)
    assert "doc_" not in joined
    assert "chunk_" not in joined
    assert "EXP-" not in joined
    assert "SCI-" not in joined
    assert "effect: increase" not in joined
    assert "рост" in joined
    assert "\n" not in clean_graph_label({"id": "e1", "label": "Experiment\nSCI-VT6-AN-900", "type": "Experiment"})


def test_interactive_graph_html_uses_clean_labels() -> None:
    html = graph_to_interactive_html(
        {
            "nodes": [
                {"id": "doc_123", "label": "doc_29765440445babcdef", "type": "Document"},
                {"id": "m1", "label": "Прочность: 1120 MPa, effect: unknown", "type": "Measurement"},
            ],
            "edges": [{"source": "doc_123", "target": "m1", "type": "SUPPORTED_BY"}],
        }
    )
    visible_part = html.split("<style>", 1)[0]
    assert "doc_29765440445babcdef" not in visible_part
    assert "effect: unknown" not in visible_part
    assert "эффект не указан" in visible_part

from __future__ import annotations

from app.graph.answer_graph import build_answer_graph, make_human_graph_label, translate_effect


FORBIDDEN = [
    "doc_",
    "chunk_",
    "EXP-",
    "SCI-",
    "Experiment",
    "PropertyValue",
    "SourceChunk",
    "OF_PROPERTY",
    "MEASURES",
    "STUDIES",
    "USES_REGIME",
    "effect: increase",
    "effect: decrease",
    "effect: unknown",
    "increase",
    "decrease",
    "unknown",
]


def test_effect_values_are_translated_for_answer_graph() -> None:
    assert translate_effect("increase") == "рост"
    assert translate_effect("decrease") == "снижение"
    assert translate_effect("no_change") == "без заметного изменения"
    assert translate_effect("unknown") == "эффект не указан"
    assert translate_effect(None) == "эффект не указан"


def test_make_human_graph_label_removes_internal_tokens() -> None:
    label = make_human_graph_label(
        {
            "type": "PropertyValue",
            "label": "PropertyValue\nSCI-VT6-AN-900 doc_abc chunk_def effect: increase",
        }
    )
    assert "рост" in label or label == "значение свойства"
    assert "\n" not in label
    for token in FORBIDDEN:
        assert token not in label


def test_answer_graph_labels_do_not_expose_raw_database_terms() -> None:
    graph = build_answer_graph(
        {
            "status": "ok",
            "constraints": {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]},
            "primary_facts": [
                {
                    "experiment_id": "SCI-VT6-AN-900",
                    "material": "ВТ6",
                    "regime": "отжиг",
                    "property": "прочность",
                    "value": 1120.0,
                    "unit": "MPa",
                    "effect": "increase",
                    "source_chunk_id": "chunk_abc",
                }
            ],
            "sources": [{"chunk_id": "chunk_abc", "document_id": "doc_abc"}],
        }
    )
    visible_labels = " ".join(node.label for node in graph.nodes)
    for token in FORBIDDEN:
        assert token not in visible_labels
    assert "рост" in visible_labels

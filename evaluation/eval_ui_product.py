from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.answer_graph import answer_graph_to_html, build_answer_graph  # noqa: E402
from app.graph.full_answer_graph import (  # noqa: E402
    FULL_GRAPH_MAX_EDGES,
    FULL_GRAPH_MAX_NODES,
    build_full_answer_graph,
    full_answer_graph_to_html,
)
from app.ui_helpers import graph_to_interactive_html  # noqa: E402


FORBIDDEN_NAV_LABELS = [
    "Ask / GraphRAG",
    "Graph Explorer",
    "Entity Explorer",
    "Decision History",
    "Data Gaps",
    "Similar Experiments",
    "Evidence & Sources",
    "Demo Scenarios",
]


def main() -> int:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    answer_graph = build_answer_graph(
        {
            "status": "ok",
            "constraints": {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]},
            "primary_facts": [
                {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 1120.0, "unit": "MPa", "effect": "increase"},
                {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 980.0, "unit": "MPa", "effect": "decrease"},
                {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 862.0, "unit": "MPa", "effect": "unknown"},
            ],
            "sources": [
                {"source_name": "synthetic_vt6_heat_treatment.csv"},
                {"source_name": "article_vt6.txt"},
                {"source_name": "catalog.csv"},
            ],
        }
    )
    answer_graph_html = answer_graph_to_html(answer_graph)
    answer_graph_labels = " ".join(node.label for node in answer_graph.nodes)
    answer_graph_labels_lower = answer_graph_labels.lower()
    comparison_graph = build_answer_graph(
        {
            "status": "ok",
            "answer_mode": "comparison",
            "analytical_intent": "material_comparison",
            "constraints": {"materials": ["ВТ6", "7075-T6"], "properties": ["прочность"]},
            "facts": [
                {"material": "ВТ6", "property": "прочность", "value": 1120.0, "unit": "MPa", "effect": "increase"},
                {"material": "ВТ6", "property": "прочность", "value": 980.0, "unit": "MPa", "effect": "decrease"},
                {"material": "7075-T6", "property": "прочность", "value": 77.0, "unit": "ksi", "effect": "unknown"},
                {"material": "7075-T6", "property": "прочность", "value": 66.0, "unit": "ksi", "effect": "unknown"},
            ],
        }
    )
    comparison_labels = " ".join(node.label for node in comparison_graph.nodes)
    forbidden_graph_tokens = [
        "doc_",
        "chunk_",
        "EXP-",
        "SCI-",
        "Experiment",
        "PropertyValue",
        "SourceChunk",
        "FACT_SUPPORTED_BY_CHUNK",
        "OF_PROPERTY",
        "STUDIES",
        "MEASURES",
        "USES_REGIME",
        "effect: increase",
        "effect: decrease",
        "effect: unknown",
        "increase",
        "decrease",
        "unknown",
    ]
    graph_html = graph_to_interactive_html(
        {
            "nodes": [
                {"id": "doc_123", "label": "doc_29765440445babcdef", "type": "Document"},
                {"id": "m1", "label": "ВТ6", "type": "Material"},
                {"id": "p1", "label": "Прочность: 1120 MPa, effect: increase", "type": "Measurement"},
            ],
            "edges": [{"source": "m1", "target": "p1", "type": "MEASURED"}],
        }
    )
    visible_html = graph_html.split("<style>", 1)[0]
    full_graph = build_full_answer_graph(
        {
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
            "diagnostics": {"fact_conflicts": [{"material": "ВТ6", "regime": "отжиг", "property": "прочность"}]},
            "subgraph": {
                "nodes": [
                    {"id": "Experiment:SCI-VT6-AN-900", "label": "Experiment:SCI-VT6-AN-900", "type": "Experiment"},
                    {"id": "Material:ВТ6", "label": "Material:ВТ6", "type": "Material"},
                    {"id": "ProcessRegime:отжиг", "label": "ProcessRegime:отжиг", "type": "ProcessRegime"},
                    {"id": "Property:прочность", "label": "Property:прочность", "type": "Property"},
                    {"id": "PropertyValue:pv1", "label": "прочность: 77.0 ksi", "type": "PropertyValue"},
                    {"id": "SourceChunk:chunk_abc", "label": "chunk_abc", "type": "SourceChunk", "source_name": "article_vt6.txt"},
                    {"id": "DataGap:12b274", "label": "DataGap:12b274", "type": "DataGap", "property": "класс герметичности DN50"},
                ],
                "edges": [
                    {"source": "Experiment:SCI-VT6-AN-900", "target": "Material:ВТ6", "label": "USES_MATERIAL"},
                    {"source": "Experiment:SCI-VT6-AN-900", "target": "PropertyValue:pv1", "label": "MEASURES"},
                    {"source": "PropertyValue:pv1", "target": "Property:прочность", "label": "OF_PROPERTY"},
                    {"source": "DataGap:12b274", "target": "Property:прочность", "label": "GAP_FOR_PROPERTY"},
                ],
            },
        }
    )
    full_graph_html = full_answer_graph_to_html(full_graph)
    full_graph_visible_html = full_graph_html.split("<style>", 1)[0]
    checks = {
        "document_upload_available": "Загрузить в базу" in ui_text and "file_uploader" in ui_text,
        "large_upload_guard_available": "UI_UPLOAD_MAX_FILES" in ui_text
        and "UI_UPLOAD_MAX_TOTAL_MB" in ui_text
        and "Выбран слишком большой batch для Streamlit upload" in ui_text
        and "scripts/batch_ingest_corpus.py --input data_storage" in ui_text
        and "disabled=upload_blocked" in ui_text,
        "web_page_upload_available": "Добавить веб-страницу" in ui_text and "/ingest/url" in ui_text and "Загрузить страницу" in ui_text,
        "document_management_available": "/documents" in ui_text and "/active" in ui_text,
        "document_management_editable_active_column": "st.data_editor" in ui_text and "CheckboxColumn" in ui_text and "Активен" in ui_text,
        "no_document_toggle_selectbox": "Документ для включения/выключения" not in ui_text and "Выключить документ" not in ui_text,
        "no_nested_expanders": "_render_document_controls" in ui_text and 'st.expander("Metadata выбранного документа")' not in ui_text,
        "interactive_graph_available": "answer_graph_to_html(" in ui_text and "components.html" in ui_text,
        "no_sidebar_page_navigation": '"Раздел"' not in ui_text and all(label not in ui_text for label in FORBIDDEN_NAV_LABELS),
        "graph_labels_clean": all(token not in visible_html for token in ["doc_", "chunk_", "EXP-", "SCI-", "effect: increase"]),
        "answer_graph_available": "build_answer_graph(payload)" in ui_text
        and "answer_graph_to_html(" in ui_text
        and "answerGraphCompact_" in ui_text
        and "answerGraphExpanded_" in ui_text,
        "answer_graph_large_mode_available": "Развернуть карту" in ui_text
        and "Открыть крупно" not in ui_text
        and "render_height=820" in ui_text
        and "render_width=1500" in ui_text
        and "min(85vw, 1500px)" in ui_text
        and 'data-testid="stModal"' in ui_text
        and 'button[aria-label="Close"]' in ui_text
        and "answer_graph_modal_open" in ui_text
        and 'getattr(st, "dialog", None)' in ui_text,
        "answer_graph_node_limit_ok": len(answer_graph.nodes) <= 10 and len(answer_graph.edges) <= 12,
        "answer_graph_labels_clean": all(token not in answer_graph_labels for token in forbidden_graph_tokens),
        "answer_graph_has_semantic_path": "ВТ6" in answer_graph_labels
        and "титановый сплав" in answer_graph_labels
        and all(token in answer_graph_labels_lower for token in ["отжиг", "прочность"])
        and "862–1120 MPa" in answer_graph_labels,
        "comparison_answer_graph_compact_clean": len(comparison_graph.nodes) <= 10
        and len(comparison_graph.edges) <= 12
        and all(token not in comparison_labels for token in forbidden_graph_tokens)
        and all(token in comparison_labels for token in ["ВТ6", "титановый сплав", "7075-T6", "алюминиевый сплав", "состояние T6", "980–1120 MPa", "455–531 MPa", "сравнение ограничено"]),
        "answer_provenance_map_available": "Карта происхождения ответа" in ui_text
        and "Развернуть карту происхождения" in ui_text
        and "build_full_answer_graph(payload" in ui_text
        and "full_answer_graph_to_html(" in ui_text
        and "fullAnswerGraphCompact_" in ui_text
        and "fullAnswerGraphExpanded_" in ui_text,
        "source_metadata_summary_available": "Структура источников" in ui_text
        and "answer_source_metadata_rows(payload)" in ui_text,
        "full_graph_filters_available": all(
            token in ui_text
            for token in ["Источники/evidence", "Пробелы данных", "Конфликты", "Измерения", "Только активные"]
        ),
        "full_graph_modal_available": "full_graph_modal_open" in ui_text
        and "open_full_graph_modal" in ui_text
        and "close_full_graph_modal" in ui_text,
        "full_graph_canvas_clean": all(
            token not in full_graph_visible_html
            for token in [
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
        ),
        "full_graph_bounded": len(full_graph.nodes) <= FULL_GRAPH_MAX_NODES and len(full_graph.edges) <= FULL_GRAPH_MAX_EDGES,
        "provenance_graph_has_explanation_nodes": all(
            node_type in {node.type for node in full_graph.nodes}
            for node_type in ["QueryNode", "AnswerFocusNode", "CanonicalFactNode", "ConclusionNode"]
        ),
        "audit_graph_tables_available": 'with st.expander("Аудит графа")' in ui_text
        and "full_graph_audit_tables" in ui_text
        and "Аудит узлов" in ui_text
        and "Аудит связей" in ui_text,
        "no_noisy_technical_canvas_in_ui": "graph_to_interactive_html(" not in ui_text,
        "main_graph_not_raw_subgraph": "graph_to_interactive_html(" not in ui_text.split('with st.expander("Аудит графа")', 1)[0],
        "old_technical_subgraph_removed": "Технический подграф" not in ui_text
        and "Raw subgraph" not in ui_text
        and "debug graph" not in ui_text.lower(),
        "answer_graph_hierarchical_renderer": all(
            token in answer_graph_html
            for token in ["hierarchical", "physics", "enabled: false", "zoomView", "dragView", "dragNodes: false", "Колесо — масштаб", "узлы зафиксированы"]
        ),
        "answer_graph_no_drag_node_hint": "Wheel: zoom" not in answer_graph_html
        and "drag node: move" not in answer_graph_html
        and "dragNodes: true" not in answer_graph_html,
        "llm_diagnostics_available": "/system/test-llm" in ui_text and "LLM:" in ui_text,
        "examples_not_as_long_top_buttons": "Подставить пример" in ui_text and "EXAMPLE_QUESTIONS):" not in ui_text,
    }
    print("UI product evaluation:")
    for key, value in checks.items():
        print(f"{key}: {1 if value else 0}")
    passed = all(checks.values())
    print("PASS" if passed else "FAIL")
    print("SUMMARY", json.dumps(checks, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

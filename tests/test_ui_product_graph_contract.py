from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_ui_uses_answer_graph_for_primary_graph_area() -> None:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    assert "build_answer_graph(payload)" in ui_text
    assert "answer_graph_to_html(" in ui_text
    assert "answerGraphCompact_" in ui_text
    assert "answerGraphExpanded_" in ui_text
    assert "render_height=820" in ui_text
    assert "render_width=1500" in ui_text
    assert "Развернуть карту" in ui_text
    assert "Открыть крупно" not in ui_text
    assert "Карта происхождения ответа" in ui_text
    assert "Развернуть карту происхождения" in ui_text
    assert "Аудит графа" in ui_text
    assert "Технический подграф" not in ui_text
    assert "Raw subgraph" not in ui_text
    assert ui_text.index("Развернуть карту") < ui_text.index("def _render_interactive_graph")

    primary_section = ui_text.split('with st.expander("Аудит графа")', 1)[0]
    assert "graph_to_interactive_html(" not in primary_section


def test_answer_provenance_map_replaces_noisy_technical_canvas_and_audit_tables_remain() -> None:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    assert "build_full_answer_graph(payload" in ui_text
    assert "full_answer_graph_to_html(" in ui_text
    assert "fullAnswerGraphCompact_" in ui_text
    assert "fullAnswerGraphExpanded_" in ui_text
    assert "Показаны ключевые связи ответа" in ui_text
    assert "Дубли объединены" in ui_text
    assert 'with st.expander("Аудит графа")' in ui_text
    audit_section = ui_text.split('with st.expander("Аудит графа")', 1)[1]
    assert "full_graph_audit_tables" in audit_section
    assert "Аудит узлов" in audit_section
    assert "Аудит связей" in audit_section
    assert "graph_to_interactive_html" not in audit_section


def test_large_answer_graph_uses_dialog_or_inline_block_without_backend_request() -> None:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    assert "def _render_large_answer_graph" in ui_text
    assert 'getattr(st, "dialog", None)' in ui_text
    assert "answer_graph_modal_open" in ui_text
    assert "open_answer_graph_modal" in ui_text
    assert "close_answer_graph_modal" in ui_text
    assert "_close_answer_graph_modal(answer_key)" in ui_text
    assert "min(85vw, 1500px)" in ui_text
    assert 'data-testid="stModal"' in ui_text
    assert 'button[aria-label="Close"]' in ui_text
    assert "ask_api(" not in ui_text.split("def _render_large_answer_graph", 1)[1].split("def _render_answer", 1)[0]


def test_large_full_graph_uses_distinct_modal_state_without_backend_request() -> None:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    assert "def _render_large_full_graph" in ui_text
    assert "full_graph_modal_open" in ui_text
    assert "open_full_graph_modal" in ui_text
    assert "close_full_graph_modal" in ui_text
    assert "_close_full_graph_modal(answer_key)" in ui_text
    assert "fullAnswerGraphCompact_" in ui_text
    assert "fullAnswerGraphExpanded_" in ui_text
    full_graph_section = ui_text.split("def _render_large_full_graph", 1)[1].split("def _render_answer_graph_modal_css", 1)[0]
    assert "ask_api(" not in full_graph_section
    assert "api_post(\"/ask\"" not in full_graph_section

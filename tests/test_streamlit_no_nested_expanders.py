from __future__ import annotations

import inspect

import app.ui as ui


def test_document_controls_has_no_nested_expanders() -> None:
    source = inspect.getsource(ui._render_document_controls)
    assert source.count("st.expander(") == 1
    assert 'st.expander("Metadata выбранного документа")' not in source


def test_document_toggle_selectbox_workflow_removed() -> None:
    source = inspect.getsource(ui._render_document_controls)
    assert "Документ для включения/выключения" not in source
    assert "Выключить документ" not in source
    assert "Включить документ" not in source
    assert "st.data_editor" in source
    assert "CheckboxColumn" in source

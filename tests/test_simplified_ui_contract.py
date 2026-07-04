from __future__ import annotations

from pathlib import Path


def test_ui_has_no_sidebar_page_navigation() -> None:
    text = Path("app/ui.py").read_text(encoding="utf-8")
    assert '"Раздел"' not in text
    for forbidden in [
        "Ask / GraphRAG",
        "Graph Explorer",
        "Entity Explorer",
        "Decision History",
        "Data Gaps",
        "Similar Experiments",
        "Evidence & Sources",
        "Demo Scenarios",
    ]:
        assert forbidden not in text


def test_ui_contains_product_controls() -> None:
    text = Path("app/ui.py").read_text(encoding="utf-8")
    assert "Режим работы" in text
    assert "Документы" in text
    assert "Загрузить в базу" in text
    assert "Обновить граф по активным документам" in text
    assert "Введите исследовательский вопрос" in text
    assert "Интерактивный связанный граф" in text
    assert "Проверенные факты" in text
    assert "Источники и evidence" in text
    assert "Диагностика" in text


def test_ui_uses_interactive_graph_renderer() -> None:
    text = Path("app/ui.py").read_text(encoding="utf-8")
    assert "answer_graph_to_html" in text
    assert "components.html" in text
    assert "Аудит графа" in text
    assert "graph_to_interactive_html" not in text

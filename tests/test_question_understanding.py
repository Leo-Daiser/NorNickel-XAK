from __future__ import annotations

from app.api import _question_understanding


def test_tz_mine_water_query_is_not_rejected_before_planner() -> None:
    result = _question_understanding(
        "Какие способы закачки шахтных вод в глубокие горизонты применялись в России "
        "и за рубежом, и каковы их технико-экономические показатели?"
    )

    assert result["known_terms"] is True
    assert result["needs_clarification"] is False


def test_deep_mine_cooling_query_is_domain_question() -> None:
    result = _question_understanding("Какие способы охлаждения применяются для глубоких рудников?")

    assert result["has_domain_marker"] is True
    assert result["needs_clarification"] is False

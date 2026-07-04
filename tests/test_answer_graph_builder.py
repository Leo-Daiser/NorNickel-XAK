from __future__ import annotations

from app.graph.answer_graph import build_answer_graph


def _strict_payload() -> dict:
    return {
        "status": "ok",
        "constraints": {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]},
        "primary_facts": [
            {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 1120.0, "unit": "MPa", "effect": "increase"},
            {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 980.0, "unit": "MPa", "effect": "decrease"},
            {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 862.0, "unit": "MPa", "effect": "unknown"},
        ],
        "sources": [{"source_name": "a"}, {"source_name": "b"}, {"source_name": "c"}],
    }


def test_strict_positive_answer_graph_has_semantic_path() -> None:
    graph = build_answer_graph(_strict_payload())
    labels = [node.label for node in graph.nodes]
    joined = " ".join(labels)
    joined_lower = joined.lower()
    assert "ВТ6" in joined
    assert "титановый сплав" in joined
    assert "отжиг" in joined_lower
    assert "прочность" in joined_lower
    assert "862–1120 MPa" in joined
    assert "источников" in joined
    assert len(graph.nodes) <= 10
    assert len(graph.edges) <= 12


def test_no_match_answer_graph_marks_gap_not_partial_answer() -> None:
    graph = build_answer_graph(
        {
            "status": "no_exact_match",
            "constraints": {"materials": ["ВТ6"], "regimes": ["криообработка"], "properties": ["вязкость"]},
            "partial_matches": {"same_material": [{"material": "ВТ6"}]},
        }
    )
    labels = " ".join(node.label for node in graph.nodes)
    assert "ВТ6" in labels
    assert "титановый сплав" in labels
    assert "криообработка" in labels.lower()
    assert "вязкость" in labels.lower()
    assert "точных данных нет" in labels
    assert "пробел в данных" in labels


def test_overview_answer_graph_is_aggregated() -> None:
    graph = build_answer_graph(
        {
            "status": "ok",
            "answer_mode": "overview",
            "constraints": {"materials": ["ВТ6"]},
            "facts": [
                {"material": "ВТ6", "regime": "отжиг", "property": "прочность"},
                {"material": "ВТ6", "regime": "старение", "property": "твёрдость"},
            ],
            "sources": [{}, {}],
        }
    )
    labels = " ".join(node.label for node in graph.nodes)
    assert "режимы:" in labels.lower()
    assert "свойства:" in labels.lower()
    assert len(graph.nodes) <= 10


def test_comparison_answer_graph_shows_compact_converted_ranges() -> None:
    graph = build_answer_graph(
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
    labels = " ".join(node.label for node in graph.nodes)
    assert "ВТ6" in labels
    assert "титановый сплав" in labels
    assert "7075-T6" in labels
    assert "алюминиевый сплав" in labels
    assert "состояние T6" in labels
    assert "980–1120 MPa" in labels
    assert "455–531 MPa" in labels
    assert "сравнение ограничено" in labels
    assert len(graph.nodes) <= 10
    assert len(graph.edges) <= 12


def test_unknown_material_gets_neutral_enrichment_without_hallucination() -> None:
    graph = build_answer_graph(
        {
            "status": "ok",
            "constraints": {"materials": ["Материал-X"], "regimes": ["режим-X"], "properties": ["свойство-X"]},
            "facts": [{"material": "Материал-X", "regime": "режим-X", "property": "свойство-X"}],
        }
    )
    labels = " ".join(node.label for node in graph.nodes)
    assert "Материал-X" in labels
    assert "материал из корпуса" in labels
    assert "титановый сплав" not in labels
    assert "алюминиевый сплав" not in labels

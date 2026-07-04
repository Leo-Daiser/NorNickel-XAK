from __future__ import annotations

from app.answering.human_answer import enhance_answer_payload
from app.domain.fact_normalization import fact_rows_from_experiments
from app.graph.neo4j_repository import Neo4jGraphRepository


def _node(**properties):
    return properties


def _experiment_record(
    experiment_id: str = "EXP-VT6-AN",
    *,
    material: str | None = "ВТ6",
    regime: str = "отжиг",
    property_name: str = "прочность",
    measurement: dict | None = None,
):
    measurement_props = {
        "measurement_id": "m1",
        "value": 1120.0,
        "raw_value": "1120",
        "unit": "MPa",
        "effect": "increase",
        "confidence": 0.9,
    }
    measurement_props.update(measurement or {})
    chunk = _node(chunk_id="chunk-1", document_id="doc-1", source_name="source.txt", page=1, text="evidence quote")
    doc = _node(document_id="doc-1", source_name="source.txt")
    return {
        "e": _node(experiment_id=experiment_id),
        "materials": [_node(canonical_name=material)] if material else [],
        "regimes": [_node(canonical_name=regime)],
        "measurements": [
            {
                "measurement": _node(**measurement_props),
                "property": _node(canonical_name=property_name),
            }
        ],
        "equipment": [_node(canonical_name="Вакуумная печь")],
        "teams": [_node(canonical_name="Лаборатория легких сплавов")],
        "laboratories": [_node(canonical_name="Лаборатория легких сплавов")],
        "conclusions": [_node(text="отжиг повысил прочность")],
        "chunks": [chunk],
        "documents": [doc],
    }


class FakeGraphDB:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.rows = rows

    def run(self, query: str, **params):
        self.calls.append((query, params))
        if "MATCH (g:DataGap)" in query:
            return [
                {
                    "g": _node(gap_id="gap-1", material="7075-T6", regime="старение", property="коррозионная стойкость", reason="нет данных"),
                    "materials": [_node(canonical_name="7075-T6")],
                    "regimes": [_node(canonical_name="старение")],
                    "properties": [_node(canonical_name="коррозионная стойкость")],
                    "chunks": [],
                    "documents": [],
                }
            ]
        return self.rows if self.rows is not None else [_experiment_record()]


def test_exact_query_uses_material_regime_property_params() -> None:
    graph_db = FakeGraphDB()
    repo = Neo4jGraphRepository(graph_db)  # type: ignore[arg-type]
    facts = repo.find_exact_material_regime_property("vt6", "annealing", "tensile strength")

    assert facts[0].experiment_id == "EXP-VT6-AN"
    assert facts[0].materials == ["ВТ6"]
    assert facts[0].regimes == ["отжиг"]
    assert facts[0].measurements[0].property_name == "прочность"
    _, params = graph_db.calls[0]
    assert params == {"material": "ВТ6", "regime": "отжиг", "property": "прочность"}


def test_partial_matches_are_separate_from_exact_facts() -> None:
    graph_db = FakeGraphDB()
    repo = Neo4jGraphRepository(graph_db)  # type: ignore[arg-type]
    exact = repo.find_exact_material_regime_property("ВТ6", "отжиг", "прочность")
    partial = repo.find_partial_matches(material="ВТ6", regime="криообработка", property_name="вязкость")

    assert exact
    assert partial.same_material
    assert len(exact) == 1
    assert graph_db.calls[0][1]["property"] == "прочность"
    assert any(call[1].get("property") == "вязкость" for call in graph_db.calls[1:])


def test_decision_history_filters_by_material_param() -> None:
    graph_db = FakeGraphDB()
    repo = Neo4jGraphRepository(graph_db)  # type: ignore[arg-type]
    history = repo.get_decision_history("ВТ6")
    assert history
    assert history[0].material == "ВТ6"
    assert graph_db.calls[0][1]["material"] == "ВТ6"


def test_gaps_query_filters_by_constraints() -> None:
    graph_db = FakeGraphDB()
    repo = Neo4jGraphRepository(graph_db)  # type: ignore[arg-type]
    gaps = repo.find_gaps(material="7075", regime="aging", property_name="corrosion resistance")
    assert gaps[0].material == "7075-T6"
    assert gaps[0].property == "коррозионная стойкость"
    assert graph_db.calls[0][1] == {"material": "7075-T6", "regime": "старение", "property": "коррозионная стойкость"}


def test_neo4j_read_preserves_persisted_normalized_fields() -> None:
    graph_db = FakeGraphDB(
        rows=[
            _experiment_record(
                measurement={
                    "value": 77.0,
                    "raw_value": "77",
                    "unit": "ksi",
                    "value_original": 77.0,
                    "unit_original": "ksi",
                    "value_normalized": 530.896289,
                    "unit_normalized": "MPa",
                    "normalization_family": "strength",
                    "effect": "unknown",
                }
            )
        ]
    )
    repo = Neo4jGraphRepository(graph_db)  # type: ignore[arg-type]
    fact = repo.find_exact_material_regime_property("ВТ6", "отжиг", "прочность")[0]
    measurement = fact.measurements[0]

    assert measurement.value == 77.0
    assert measurement.unit == "ksi"
    assert abs(measurement.value_normalized - 530.896289) < 0.001
    assert measurement.unit_normalized == "MPa"
    assert measurement.normalization_family == "strength"


def test_legacy_neo4j_record_without_normalized_fields_is_readable() -> None:
    repo = Neo4jGraphRepository(FakeGraphDB())  # type: ignore[arg-type]
    fact = repo.find_exact_material_regime_property("ВТ6", "отжиг", "прочность")[0]
    measurement = fact.measurements[0]

    assert measurement.value == 1120.0
    assert measurement.value_normalized == 1120.0
    assert measurement.unit_normalized == "MPa"
    assert measurement.normalization_family == "strength"


def test_comparison_answer_uses_normalized_mpa_after_neo4j_read() -> None:
    rows = [
        _experiment_record(
            experiment_id="EXP-VT6-AN",
            material="ВТ6",
            regime="отжиг",
            measurement={"value": 1120.0, "raw_value": "1120", "unit": "MPa", "effect": "increase"},
        ),
        _experiment_record(
            experiment_id="EXP-7075-AG",
            material="7075-T6",
            regime="старение",
            measurement={"value": 77.0, "raw_value": "77", "unit": "ksi", "effect": "unknown"},
        ),
    ]
    repo = Neo4jGraphRepository(FakeGraphDB(rows=rows))  # type: ignore[arg-type]
    experiments = repo.find_experiments(limit=10)
    payload = enhance_answer_payload(
        {
            "answer": "draft",
            "status": "ok",
            "answer_mode": "comparison",
            "analytical_intent": "material_comparison",
            "constraints": {"materials": ["ВТ6", "7075-T6"], "regimes": [], "properties": ["прочность"]},
            "facts": fact_rows_from_experiments(experiments),
            "sources": [{"source_name": "source.txt", "quote": "evidence quote"}],
            "evidence": [],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "retrieval": {},
            "diagnostics": {},
        },
        "expert_max",
    )

    assert "77 ksi ≈ 531 MPa" in payload["answer"]
    assert "1120 MPa" in payload["answer"]


def test_find_experiments_allows_process_only_facts_when_material_filter_absent() -> None:
    rows = [
        _experiment_record(
            experiment_id="accepted-process-flow",
            material=None,
            regime="циркуляция католита",
            property_name="скорость потока",
            measurement={
                "value": 0.5,
                "raw_value": "0.5",
                "unit": "m/s",
                "fact_type": "ProcessParameterFact",
                "source_adapter": "structured_table_adapter",
            },
        )
    ]
    graph_db = FakeGraphDB(rows=rows)
    repo = Neo4jGraphRepository(graph_db)  # type: ignore[arg-type]

    experiments = repo.find_experiments(regime="циркуляция католита", property_name="скорость потока", limit=10)

    assert len(experiments) == 1
    assert experiments[0].materials == []
    assert experiments[0].regimes == ["циркуляция католита"]
    assert experiments[0].measurements[0].property_name == "скорость потока"
    assert experiments[0].measurements[0].source_adapter == "structured_table_adapter"
    query, params = graph_db.calls[0]
    assert "OPTIONAL MATCH (e)-[:USES_MATERIAL]->(m:Material)" in query
    assert params["material"] is None


def test_find_experiments_allows_regime_less_material_property_facts() -> None:
    rows = [
        _experiment_record(
            experiment_id="accepted-assay-ni",
            material="Пирротиновый концентрат",
            regime="",
            property_name="содержание",
            measurement={
                "value": 0.5,
                "value_min": 0.5,
                "value_max": 1.0,
                "raw_value": "0,5-1",
                "unit": "%",
                "analyte": "ni",
                "fact_type": "ProcessParameterFact",
                "source_adapter": "structured_table_adapter",
            },
        )
    ]
    rows[0]["regimes"] = []
    graph_db = FakeGraphDB(rows=rows)
    repo = Neo4jGraphRepository(graph_db)  # type: ignore[arg-type]

    experiments = repo.find_experiments(material="Пирротиновый концентрат", property_name="содержание", limit=10)

    assert len(experiments) == 1
    assert experiments[0].materials == ["Пирротиновый концентрат"]
    assert experiments[0].regimes == []
    measurement = experiments[0].measurements[0]
    assert measurement.property_name == "содержание"
    assert measurement.analyte == "ni"
    assert measurement.value_min == 0.5
    assert measurement.value_max == 1.0
    query, params = graph_db.calls[0]
    assert "OPTIONAL MATCH (e)-[:HAS_REGIME]->(r:ProcessRegime)" in query
    assert params["regime"] is None

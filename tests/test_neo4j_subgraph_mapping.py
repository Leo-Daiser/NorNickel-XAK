from __future__ import annotations

from app.answering.answer_builder import AnswerBuilder
from app.domain.ontology import Evidence, Measurement
from app.domain.query_constraints import QueryConstraints, QueryIntent
from app.graph.graph_models import ExperimentFact
from app.graph.neo4j_repository import Neo4jGraphRepository

from tests.test_neo4j_repository_queries import FakeGraphDB


def test_neo4j_fact_maps_to_ui_compatible_subgraph_without_duplicates() -> None:
    repo = Neo4jGraphRepository(FakeGraphDB())  # type: ignore[arg-type]
    fact = repo.find_exact_material_regime_property("ВТ6", "отжиг", "прочность")[0]
    constraints = QueryConstraints(
        intent=QueryIntent.MATERIAL_REGIME_PROPERTY_EFFECT,
        raw_question="Что делали по ВТ6 при отжиге и как изменилась прочность?",
        materials=["ВТ6"],
        regimes=["отжиг"],
        properties=["прочность"],
        require_exact_match=True,
    )
    payload = AnswerBuilder().exact_match_response(constraints, [fact, fact], None, [], {}, {})
    nodes = payload["subgraph"]["nodes"]
    edges = payload["subgraph"]["edges"]

    node_ids = [node["id"] for node in nodes]
    edge_ids = [edge["id"] for edge in edges]
    assert len(node_ids) == len(set(node_ids))
    assert len(edge_ids) == len(set(edge_ids))
    assert any(node["type"] == "Experiment" for node in nodes)
    assert any(edge["label"] == "USES_REGIME" for edge in edges)


def test_answer_builder_subgraph_ids_are_stable() -> None:
    evidence = Evidence(document_id="doc-1", chunk_id="chunk-1", source_name="source.txt", page=1, quote="quote")
    fact = ExperimentFact(
        experiment_id="EXP-STABLE",
        materials=["ВТ6"],
        regimes=["отжиг"],
        measurements=[Measurement(property_name="прочность", value=1120.0, raw_value="1120", unit="MPa", evidence=[evidence])],
        evidence=[evidence],
        source_chunk_ids=["chunk-1"],
    )
    constraints = QueryConstraints(
        intent=QueryIntent.MATERIAL_REGIME_PROPERTY_EFFECT,
        raw_question="q",
        materials=["ВТ6"],
        regimes=["отжиг"],
        properties=["прочность"],
        require_exact_match=True,
    )
    first = AnswerBuilder().exact_match_response(constraints, [fact], None, [], {}, {})["subgraph"]
    second = AnswerBuilder().exact_match_response(constraints, [fact], None, [], {}, {})["subgraph"]
    assert first == second


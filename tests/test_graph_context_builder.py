from __future__ import annotations

from app.analytics.graph_context import GraphContextBuilder
from app.analytics.router import AnalyticalQueryRouter
from app.domain.ontology import Evidence, Measurement
from app.graph.graph_models import ExperimentFact
from app.retrieval.query_planner import QueryPlanner


def _experiment(exp_id: str, material: str) -> ExperimentFact:
    evidence = Evidence(
        document_id="doc1",
        chunk_id=f"chunk-{exp_id}",
        source_name="demo.txt",
        quote=f"{material} отжиг прочность 1120 MPa",
    )
    return ExperimentFact(
        experiment_id=exp_id,
        materials=[material],
        regimes=["отжиг"],
        measurements=[Measurement(property_name="прочность", value=1120, unit="MPa", evidence=[evidence])],
        equipment=["Печь SNOL"],
        laboratories=["Лаборатория A"],
        evidence=[evidence],
        source_chunk_ids=[f"chunk-{exp_id}"],
    )


def _plan(question: str):
    constraints = QueryPlanner().parse(question)
    return AnalyticalQueryRouter().build_plan(question, constraints)


def test_graph_context_deduplicates_facts_sources_and_subgraph() -> None:
    context = GraphContextBuilder().from_experiments(
        plan=_plan("Что уже делали по ВТ6?"),
        experiments=[_experiment("EXP-1", "ВТ6"), _experiment("EXP-1", "ВТ6")],
    )
    assert len(context.facts) == 1
    assert len(context.sources) == 1
    node_ids = [node["id"] for node in context.subgraph["nodes"]]
    edge_ids = [edge["id"] for edge in context.subgraph["edges"]]
    assert len(node_ids) == len(set(node_ids))
    assert len(edge_ids) == len(set(edge_ids))


def test_graph_context_limits_are_respected() -> None:
    experiments = [_experiment(f"EXP-{idx}", "ВТ6") for idx in range(10)]
    context = GraphContextBuilder(max_facts=3, max_sources=2, max_nodes=8, max_edges=8).from_experiments(
        plan=_plan("Что уже делали по ВТ6?"),
        experiments=experiments,
    )
    assert len(context.facts) <= 3
    assert len(context.sources) <= 2
    assert len(context.subgraph["nodes"]) <= 8
    assert len(context.subgraph["edges"]) <= 8


def test_graph_context_filters_unrelated_material() -> None:
    context = GraphContextBuilder().from_experiments(
        plan=_plan("Что уже делали по ВТ6?"),
        experiments=[_experiment("EXP-1", "ВТ6"), _experiment("EXP-2", "7075-T6")],
    )
    assert context.facts
    assert all(row["material"] == "ВТ6" for row in context.facts)

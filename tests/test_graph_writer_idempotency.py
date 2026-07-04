from __future__ import annotations

from app.domain.ontology import Evidence, Measurement
from app.extraction.pipeline import ExtractionPipeline
from app.graph.graph_models import ExperimentFact
from app.graph.graph_writer import GraphWriteStats, GraphWriter, deterministic_measurement_id
from app.models.schemas import Chunk


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params):
        self.calls.append((query, params))
        return []


class BackfillSession(FakeSession):
    def run(self, query: str, **params):
        self.calls.append((query, params))
        if "RETURN meas.measurement_id AS measurement_id" in query:
            return [
                {
                    "measurement_id": "legacy-m1",
                    "value": 77.0,
                    "raw_value": "77",
                    "unit": "ksi",
                    "property": "прочность",
                }
            ]
        return []


class FakeGraphDB:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    def session(self):
        session = self._session

        class Context:
            def __enter__(self):
                return session

            def __exit__(self, exc_type, exc, tb):
                return False

        return Context()


def _fact() -> ExperimentFact:
    evidence = Evidence(document_id="doc-1", chunk_id="chunk-1", source_name="source.txt", page=1, quote="quote")
    return ExperimentFact(
        experiment_id="EXP-VT6-AN",
        materials=["ВТ6"],
        regimes=["отжиг"],
        measurements=[
            Measurement(property_name="прочность", value=1120.0, raw_value="1120", unit="MPa", effect="increase", confidence=0.9, evidence=[evidence])
        ],
        equipment=["Вакуумная печь"],
        laboratories=["Лаборатория легких сплавов"],
        teams=["Команда легких сплавов"],
        employees=["Иванов И.И."],
        topic_tags=["термообработка"],
        conclusions=["отжиг повысил прочность"],
        evidence=[evidence],
        source_chunk_ids=["chunk-1"],
    )


def test_measurement_id_is_deterministic() -> None:
    first = deterministic_measurement_id("EXP", "ВТ6", "отжиг", "прочность", 1120.0, "MPa", "chunk-1")
    second = deterministic_measurement_id("EXP", "ВТ6", "отжиг", "прочность", 1120.0, "MPa", "chunk-1")
    assert first == second
    assert first.startswith("measurement_")


def test_writer_uses_merge_and_stable_ids() -> None:
    session = FakeSession()
    writer = GraphWriter(FakeGraphDB(session))  # type: ignore[arg-type]
    stats = GraphWriteStats()

    writer.write_experiment(session, _fact(), stats)
    writer.write_experiment(session, _fact(), stats)

    query_text = "\n".join(query for query, _ in session.calls)
    assert "MERGE (e:Experiment" in query_text
    assert "MERGE (m:Material" in query_text
    assert "MERGE (meas:Measurement" in query_text
    assert "MERGE (team:ResearchTeam" in query_text
    assert "MERGE (employee:Employee" in query_text
    assert "MERGE (topic:TopicTag" in query_text
    assert "CREATE " not in query_text
    measurement_ids = [params["measurement_id"] for _, params in session.calls if "measurement_id" in params]
    assert len(set(measurement_ids)) == 1


def test_writer_persists_normalized_measurement_fields() -> None:
    session = FakeSession()
    writer = GraphWriter(FakeGraphDB(session))  # type: ignore[arg-type]
    stats = GraphWriteStats()

    writer.write_experiment(session, _fact(), stats)

    measurement_params = next(params for _, params in session.calls if "measurement_id" in params and "value_normalized" in params)
    assert measurement_params["value_original"] == 1120.0
    assert measurement_params["unit_original"] == "MPa"
    assert measurement_params["value_normalized"] == 1120.0
    assert measurement_params["unit_normalized"] == "MPa"
    assert measurement_params["normalization_family"] == "strength"


def test_writer_reports_team_employee_and_topic_stats() -> None:
    session = FakeSession()
    writer = GraphWriter(FakeGraphDB(session))  # type: ignore[arg-type]
    stats = GraphWriteStats()

    writer.write_experiment(session, _fact(), stats)
    payload = stats.to_dict()

    assert payload["teams_written"] == 2
    assert payload["employees_written"] == 1
    assert payload["topic_tags_written"] == 1


def test_writer_backfills_legacy_normalized_measurement_fields() -> None:
    session = BackfillSession()
    writer = GraphWriter(FakeGraphDB(session))  # type: ignore[arg-type]
    stats = GraphWriteStats()

    writer.backfill_normalized_measurements(session, stats)

    update_params = next(params for query, params in session.calls if "MATCH (meas:Measurement {measurement_id: $measurement_id})" in query)
    assert update_params["measurement_id"] == "legacy-m1"
    assert update_params["value_original"] == 77.0
    assert update_params["unit_original"] == "ksi"
    assert abs(update_params["value_normalized"] - 530.896289) < 0.001
    assert update_params["unit_normalized"] == "MPa"
    assert update_params["normalization_family"] == "strength"
    assert stats.normalized_measurements_backfilled == 1


def test_writer_projects_structured_accepted_table_fact() -> None:
    chunk = Chunk(
        chunk_id="row-assay",
        doc_id="doc-table",
        workspace_uid="test",
        text=(
            "Материал: Пирротиновый концентрат | Технология: автоклавное выщелачивание | "
            "Ni, %: 0,5-1"
        ),
        page_start=1,
        page_end=1,
        section_path="table",
        metadata={"filename": "assay.csv", "chunk_kind": "table_row", "row_id": 3},
    )
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(chunk)
    assert any(
        item.normalized_fact.get("source_adapter") == "structured_table_adapter"
        and item.normalized_fact["object"].get("property") == "содержание"
        for item in bundle.accepted_facts
    )

    session = FakeSession()
    writer = GraphWriter(FakeGraphDB(session))  # type: ignore[arg-type]
    stats = GraphWriteStats()

    writer.write_bundle(session, bundle, stats)

    query_text = "\n".join(query for query, _ in session.calls)
    assert "MERGE (e:Experiment" in query_text
    assert "MERGE (m:Material" in query_text
    assert "MERGE (r:ProcessRegime" in query_text
    assert "MERGE (meas:Measurement" in query_text
    assert "MERGE (meas)-[:OF_PROPERTY]->(p)" in query_text
    assert "MERGE (meas)-[:SUPPORTED_BY]->(c)" in query_text

    experiment_params = next(
        params
        for query, params in session.calls
        if "MERGE (e:Experiment" in query and params.get("source_adapter") == "structured_table_adapter"
    )
    assert experiment_params["fact_type"] == "ProcessParameterFact"
    assert experiment_params["source_adapter"] == "structured_table_adapter"

    measurement_params = next(
        params
        for query, params in session.calls
        if "MERGE (meas:Measurement" in query and params.get("analyte") == "ni"
    )
    assert measurement_params["property"] == "содержание"
    assert measurement_params["unit"] == "%"
    assert measurement_params["value"] == 0.5
    assert measurement_params["value_min"] == 0.5
    assert measurement_params["value_max"] == 1.0
    assert measurement_params["fact_type"] == "ProcessParameterFact"
    assert measurement_params["source_adapter"] == "structured_table_adapter"
    assert stats.measurements_written
    assert stats.structured_accepted_facts_projected == 1
    assert stats.structured_accepted_facts_skipped_graph_projection == 0


def test_writer_projects_process_only_structured_fact_without_synthetic_material() -> None:
    chunk = Chunk(
        chunk_id="row-flow",
        doc_id="doc-table",
        workspace_uid="test",
        text="process: catholyte circulation | parameter: flow rate | value: 0.5 | unit: m/s",
        page_start=1,
        page_end=1,
        section_path="table",
        metadata={"filename": "process.csv", "chunk_kind": "table_row", "row_id": 4},
    )
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(chunk)
    assert any(
        item.fact_type == "ProcessParameterFact"
        and item.normalized_fact["subject"].get("process") == "циркуляция католита"
        for item in bundle.accepted_facts
    )

    session = FakeSession()
    writer = GraphWriter(FakeGraphDB(session))  # type: ignore[arg-type]
    stats = GraphWriteStats()

    writer.write_bundle(session, bundle, stats)

    query_text = "\n".join(query for query, _ in session.calls)
    assert "MERGE (r:ProcessRegime" in query_text
    assert "MERGE (meas:Measurement" in query_text
    assert "MERGE (meas)-[:SUPPORTED_BY]->(c)" in query_text
    assert "MERGE (m:Material" not in query_text
    measurement_params = next(
        params
        for query, params in session.calls
        if "MERGE (meas:Measurement" in query and params.get("property") == "скорость потока"
    )
    assert measurement_params["value"] == 0.5
    assert measurement_params["unit"] == "m/s"
    assert stats.structured_accepted_facts_projected == 1
    assert stats.structured_accepted_facts_skipped_graph_projection == 0


def test_writer_projects_material_content_fact_without_synthetic_regime() -> None:
    chunk = Chunk(
        chunk_id="row-assay-no-process",
        doc_id="doc-table",
        workspace_uid="test",
        text="Материал: Пирротиновый концентрат | Ni, %: 0,5-1",
        page_start=1,
        page_end=1,
        section_path="table",
        metadata={"filename": "assay.csv", "chunk_kind": "table_row", "row_id": 5},
    )
    bundle = ExtractionPipeline(audit_enabled=False).extract_from_chunk(chunk)
    assert any(
        item.fact_type == "ProcessParameterFact"
        and item.normalized_fact["object"].get("property") == "содержание"
        for item in bundle.accepted_facts
    )

    session = FakeSession()
    writer = GraphWriter(FakeGraphDB(session))  # type: ignore[arg-type]
    stats = GraphWriteStats()

    writer.write_bundle(session, bundle, stats)

    query_text = "\n".join(query for query, _ in session.calls)
    assert "MERGE (m:Material" in query_text
    assert "MERGE (meas:Measurement" in query_text
    assert "MERGE (r:ProcessRegime" not in query_text
    measurement_params = next(
        params
        for query, params in session.calls
        if "MERGE (meas:Measurement" in query and params.get("analyte") == "ni"
    )
    assert measurement_params["property"] == "содержание"
    assert measurement_params["value_min"] == 0.5
    assert measurement_params["value_max"] == 1.0
    assert stats.structured_accepted_facts_projected == 1
    assert stats.structured_accepted_facts_skipped_graph_projection == 0

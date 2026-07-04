from __future__ import annotations

from app.api import _build_local_subgraph
from app.extraction.extraction import EntityRelationExtractor
from app.models.schemas import Chunk
from app.retrieval.retrieval import expand_query


def test_query_expansion_adds_technical_synonyms():
    expanded = expand_query("Какие параметры у клапана DN50?")
    assert "Ду 50" in expanded or "DN 50" in expanded
    assert "valve" in expanded


def test_local_graph_is_typed_and_evidence_aware():
    chunk = Chunk(
        doc_id="doc_test",
        chunk_id="chunk_test",
        workspace_uid="ws",
        text="Клапан DN50 PN16. Корпус выполнен из стали 12Х18Н10Т. Стандарт ГОСТ 33259.",
        page_start=1,
        page_end=1,
        section_path="/Технические характеристики",
        ordinal=0,
        metadata={"filename": "valve.txt"},
    )
    extraction = EntityRelationExtractor().extract_from_chunk(chunk)
    graph = _build_local_subgraph([(chunk, extraction)])
    node_types = {node["type"] for node in graph["nodes"]}
    edge_labels = {edge["label"] for edge in graph["edges"]}
    assert "TechnicalObject" in node_types
    assert "Parameter" in node_types
    assert "Material" in node_types
    assert "Standard" in node_types
    assert "OBJECT_HAS_PARAMETER" in edge_labels
    assert "OBJECT_COMPLIES_WITH_STANDARD" in edge_labels
    assert "FACT_SUPPORTED_BY_CHUNK" in edge_labels

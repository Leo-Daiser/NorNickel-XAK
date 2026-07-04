from __future__ import annotations

from app.ingestion.document_models import ParsedBlock, ParsedTable
from app.ingestion.parser_router import build_chunks_from_document, table_to_row_chunks


def test_parsed_table_becomes_table_row_chunks() -> None:
    table = ParsedTable(table_id="t1", doc_id="d1", headers=["material", "value"], rows=[["ВТ6", "1120"]])

    chunks = table_to_row_chunks(table, parser_name="pandas", source_name="experiments.csv")

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_kind"] == "table_row"
    assert chunks[0].metadata["table_id"] == "t1"
    assert chunks[0].metadata["row_id"] == "0"
    assert chunks[0].metadata["table_columns"] == "material | value"


def test_paragraph_block_becomes_paragraph_chunk() -> None:
    block = ParsedBlock(block_id="b1", doc_id="d1", block_type="paragraph", text="ВТ6 после отжига показал прочность 1120 MPa.", ordinal=0)

    chunks = build_chunks_from_document("d1", "article.txt", "file", None, "plain", [block], [], [], "")

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_kind"] == "paragraph"
    assert chunks[0].metadata["block_id"] == "b1"


def test_long_paragraph_becomes_text_windows() -> None:
    text = " ".join(f"token{i}" for i in range(1500))
    block = ParsedBlock(block_id="b-long", doc_id="d1", block_type="paragraph", text=text, ordinal=0)

    chunks = build_chunks_from_document("d1", "long.txt", "file", None, "plain", [block], [], [], "")

    assert len(chunks) > 1
    assert all(chunk.metadata["chunk_kind"] == "text_window" for chunk in chunks)


def test_chunk_ids_are_stable() -> None:
    block = ParsedBlock(block_id="b1", doc_id="d1", block_type="paragraph", text="Стабильный текст блока.", ordinal=0)

    first = build_chunks_from_document("d1", "a.txt", "file", None, "plain", [block], [], [], "")
    second = build_chunks_from_document("d1", "a.txt", "file", None, "plain", [block], [], [], "")

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]

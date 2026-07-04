from __future__ import annotations

from app.ingestion.document_models import DocumentIntelligenceResult, ParsedBlock, ParsedImageRef, ParsedTable
from app.models.schemas import Chunk


def test_parsed_block_serializes() -> None:
    block = ParsedBlock(block_id="b1", doc_id="d1", block_type="paragraph", text="Текст", ordinal=1)
    restored = ParsedBlock.model_validate(block.model_dump())
    assert restored.block_id == "b1"
    assert restored.block_type == "paragraph"


def test_parsed_table_serializes() -> None:
    table = ParsedTable(table_id="t1", doc_id="d1", headers=["material"], rows=[["ВТ6"]])
    restored = ParsedTable.model_validate(table.model_dump())
    assert restored.headers == ["material"]
    assert restored.rows == [["ВТ6"]]


def test_parsed_image_ref_serializes() -> None:
    image = ParsedImageRef(image_id="img1", doc_id="d1", alt_text="схема", source_path_or_url="scheme.png")
    restored = ParsedImageRef.model_validate(image.model_dump())
    assert restored.alt_text == "схема"


def test_document_intelligence_result_contains_chunks() -> None:
    chunk = Chunk(doc_id="d1", chunk_id="c1", text="Текст", page_start=1, page_end=1, section_path="/")
    result = DocumentIntelligenceResult(doc_id="d1", source_name="demo.txt", parser_name="plain", chunks=[chunk])
    assert result.chunks[0].chunk_id == "c1"

"""Block-level document intelligence models for ingestion."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..models.schemas import Chunk


BlockType = Literal[
    "title",
    "section_heading",
    "paragraph",
    "table",
    "table_row",
    "figure",
    "caption",
    "list_item",
    "footnote",
    "metadata",
    "unknown",
]


class ParsedBlock(BaseModel):
    block_id: str
    doc_id: str
    block_type: BlockType
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: str = "/"
    ordinal: int
    bbox: list[float] | None = None
    parent_block_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedTable(BaseModel):
    table_id: str
    doc_id: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: str = "/"
    caption: str | None = None
    headers: list[str]
    rows: list[list[str]]
    source_block_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedImageRef(BaseModel):
    image_id: str
    doc_id: str
    page: int | None = None
    section_path: str = "/"
    alt_text: str | None = None
    caption: str | None = None
    source_path_or_url: str | None = None
    bbox: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScannedPdfDetection(BaseModel):
    is_probably_scanned: bool
    extracted_text_chars: int
    page_count: int | None = None
    reason: str


class DocumentIntelligenceResult(BaseModel):
    doc_id: str
    source_name: str
    source_type: str = "file"
    parser_name: str
    parser_version: str | None = None
    text: str = ""
    blocks: list[ParsedBlock] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    images: list[ParsedImageRef] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


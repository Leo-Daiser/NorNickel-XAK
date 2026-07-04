"""SQLite catalog for durable local document/chunk metadata.

Neo4j remains the intended canonical graph when it is available. This
catalog is a pragmatic local durability layer for demo/fallback mode:
FastAPI can restart and rebuild the lexical index from stored chunks.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from ..models.schemas import Chunk, Document


class SQLiteCatalog:
    """Small durable catalog for documents and chunks."""

    def __init__(self, path: str | Path = "data/catalog.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    workspace_uid TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_uid TEXT NULL,
                    external_id TEXT NULL,
                    parser TEXT NOT NULL,
                    language TEXT NULL,
                    status TEXT NULL,
                    created_at TEXT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NULL DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_workspace_external
                ON documents (workspace_uid, external_id)
                WHERE external_id IS NOT NULL
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    workspace_uid TEXT NULL,
                    text TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    section_path TEXT NOT NULL,
                    ordinal INTEGER NULL,
                    char_start INTEGER NULL,
                    char_end INTEGER NULL,
                    token_count INTEGER NULL,
                    text_hash TEXT NULL,
                    preview TEXT NULL,
                    embedding_version TEXT NULL,
                    updated_at TEXT NULL DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_ordinal
                ON chunks (doc_id, ordinal)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_workspace
                ON chunks (workspace_uid)
                """
            )

    def upsert_document(self, document: Document, metadata: Optional[Dict[str, Any]] = None) -> None:
        existing_metadata = self.get_document_metadata(document.doc_id)
        merged_metadata = {**existing_metadata, **(metadata or {})}
        merged_metadata.setdefault("active", True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(
                    doc_id, workspace_uid, title, source_uid, external_id, parser,
                    language, status, created_at, updated_at, version, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), COALESCE(?, CURRENT_TIMESTAMP), ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    workspace_uid=excluded.workspace_uid,
                    title=excluded.title,
                    source_uid=excluded.source_uid,
                    external_id=excluded.external_id,
                    parser=excluded.parser,
                    language=excluded.language,
                    status=excluded.status,
                    updated_at=CURRENT_TIMESTAMP,
                    version=excluded.version,
                    metadata_json=excluded.metadata_json
                """,
                (
                    document.doc_id,
                    document.workspace_uid,
                    document.title,
                    document.source_uid,
                    document.external_id,
                    document.parser,
                    document.language,
                    document.status,
                    document.created_at,
                    document.updated_at,
                    document.version,
                    json.dumps(merged_metadata, ensure_ascii=False),
                ),
            )

    def replace_chunks(self, doc_id: str, chunks: Iterable[Chunk]) -> None:
        chunk_list = list(chunks)
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, doc_id, workspace_uid, text, page_start, page_end,
                    section_path, ordinal, char_start, char_end, token_count,
                    text_hash, preview, embedding_version, updated_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.workspace_uid,
                        chunk.text,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.section_path,
                        chunk.ordinal,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.token_count,
                        chunk.text_hash,
                        chunk.preview,
                        chunk.embedding_version,
                        chunk.updated_at,
                        json.dumps(chunk.metadata or {}, ensure_ascii=False),
                    )
                    for chunk in chunk_list
                ],
            )

    def list_documents(self) -> List[Document]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY updated_at DESC").fetchall()
        return [self._row_to_document(row) for row in rows]

    def list_document_records(self) -> List[Dict[str, Any]]:
        """Return documents with metadata and chunk counts for product UI/API."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT d.*, COUNT(c.chunk_id) AS chunks_count
                FROM documents d
                LEFT JOIN chunks c ON c.doc_id = d.doc_id
                GROUP BY d.doc_id
                ORDER BY d.updated_at DESC
                """
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            result.append(
                {
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "parser": row["parser"],
                    "language": row["language"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "version": row["version"],
                    "chunks": int(row["chunks_count"] or 0),
                    "active": bool(metadata.get("active", True)),
                    "source_type": metadata.get("source_type") or metadata.get("document_intelligence", {}).get("source_type"),
                    "source_name": metadata.get("source_name") or row["title"],
                    "source_url": metadata.get("source_url"),
                    "source_title": metadata.get("source_title") or row["title"],
                    "filename": metadata.get("filename") or row["title"],
                    "parser_diagnostics": metadata.get("parser_diagnostics") or {},
                    "document_intelligence": metadata.get("document_intelligence") or {},
                    "metadata": metadata,
                }
            )
        return result

    def set_document_active(self, doc_id: str, active: bool) -> bool:
        """Set active flag in document metadata. Returns False when document is missing."""

        metadata = self.get_document_metadata(doc_id)
        with self._connect() as conn:
            row = conn.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if not row:
                return False
            metadata["active"] = bool(active)
            conn.execute(
                """
                UPDATE documents
                SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE doc_id = ?
                """,
                (json.dumps(metadata, ensure_ascii=False), doc_id),
            )
        return True

    def set_documents_active(self, changes: Dict[str, bool]) -> Dict[str, int]:
        """Batch update active flags. Missing ids are counted, not raised."""

        updated = 0
        missing = 0
        with self._connect() as conn:
            for doc_id, active in changes.items():
                row = conn.execute("SELECT metadata_json FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
                if not row:
                    missing += 1
                    continue
                metadata = json.loads(row["metadata_json"] or "{}")
                metadata["active"] = bool(active)
                conn.execute(
                    """
                    UPDATE documents
                    SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE doc_id = ?
                    """,
                    (json.dumps(metadata, ensure_ascii=False), doc_id),
                )
                updated += 1
        return {"updated": updated, "missing": missing}

    def clear(self) -> Dict[str, int]:
        """Remove all local catalog documents and chunks."""

        before = self.counts()
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
        return before

    def is_document_active(self, doc_id: str) -> bool:
        """Return active flag, defaulting old catalog rows to active."""

        return bool(self.get_document_metadata(doc_id).get("active", True))

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._row_to_document(row) if row else None

    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT metadata_json FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if not row:
            return {}
        return json.loads(row["metadata_json"] or "{}")

    def list_chunks(self, doc_id: str | None = None, active_only: bool = True) -> List[Chunk]:
        with self._connect() as conn:
            if doc_id is None:
                if active_only:
                    rows = conn.execute(
                        """
                        SELECT c.*
                        FROM chunks c
                        JOIN documents d ON d.doc_id = c.doc_id
                        WHERE json_extract(d.metadata_json, '$.active') IS NULL
                           OR json_extract(d.metadata_json, '$.active') != 0
                        ORDER BY c.doc_id, c.ordinal
                        """
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT * FROM chunks ORDER BY doc_id, ordinal").fetchall()
            else:
                if active_only and not self.is_document_active(doc_id):
                    rows = []
                else:
                    rows = conn.execute(
                        "SELECT * FROM chunks WHERE doc_id = ? ORDER BY ordinal", (doc_id,)
                    ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def counts(self) -> Dict[str, int]:
        with self._connect() as conn:
            docs = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
            chunks = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
            active_docs = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM documents
                WHERE json_extract(metadata_json, '$.active') IS NULL
                   OR json_extract(metadata_json, '$.active') != 0
                """
            ).fetchone()["c"]
            active_chunks = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE json_extract(d.metadata_json, '$.active') IS NULL
                   OR json_extract(d.metadata_json, '$.active') != 0
                """
            ).fetchone()["c"]
        return {"documents": int(docs), "chunks": int(chunks), "active_documents": int(active_docs), "active_chunks": int(active_chunks)}

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            workspace_uid=row["workspace_uid"],
            title=row["title"],
            source_uid=row["source_uid"],
            external_id=row["external_id"],
            parser=row["parser"],
            language=row["language"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            version=row["version"],
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        metadata = json.loads(row["metadata_json"] or "{}")
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            workspace_uid=row["workspace_uid"],
            text=row["text"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            section_path=row["section_path"],
            ordinal=row["ordinal"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            token_count=row["token_count"],
            text_hash=row["text_hash"],
            preview=row["preview"],
            embedding_version=row["embedding_version"],
            updated_at=row["updated_at"],
            metadata=metadata,
        )

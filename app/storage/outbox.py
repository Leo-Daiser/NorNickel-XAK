"""SQLite transactional outbox for Neo4j -> Qdrant projection events."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List


class SQLiteOutbox:
    """Small durable outbox used by the projection worker.

    The outbox persists change events that need to be projected to
    Qdrant. For a hackathon setup SQLite is enough; later this can be
    swapped for Postgres without changing the API contract.
    """

    def __init__(self, path: str | Path = "data/outbox.sqlite3") -> None:
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
                CREATE TABLE IF NOT EXISTS sync_outbox (
                    event_id TEXT PRIMARY KEY,
                    dedupe_key TEXT UNIQUE,
                    aggregate_type TEXT NOT NULL,
                    aggregate_uid TEXT NOT NULL,
                    op TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sync_outbox_status_created
                ON sync_outbox (status, created_at)
                """
            )
            # Existing archives may have the older table without dedupe_key.
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(sync_outbox)").fetchall()
            }
            if "dedupe_key" not in columns:
                conn.execute("ALTER TABLE sync_outbox ADD COLUMN dedupe_key TEXT")
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_outbox_dedupe_key
                ON sync_outbox (dedupe_key)
                WHERE dedupe_key IS NOT NULL
                """
            )

    def enqueue(
        self,
        aggregate_type: str,
        aggregate_uid: str,
        op: str,
        version: int,
        payload: Dict[str, Any],
        dedupe_key: str | None = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        key = dedupe_key or f"{aggregate_type}:{aggregate_uid}:{op}:{version}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sync_outbox(event_id, dedupe_key, aggregate_type, aggregate_uid, op, version, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dedupe_key) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    status='pending',
                    last_error=NULL
                """,
                (event_id, key, aggregate_type, aggregate_uid, op, version, json.dumps(payload, ensure_ascii=False)),
            )
            row = conn.execute(
                "SELECT event_id FROM sync_outbox WHERE dedupe_key = ?", (key,)
            ).fetchone()
            if row:
                event_id = row["event_id"]
        return event_id

    def pending(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sync_outbox
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def mark_processed(self, event_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sync_outbox
                SET status = 'processed', processed_at = CURRENT_TIMESTAMP
                WHERE event_id = ?
                """,
                (event_id,),
            )

    def mark_failed(self, event_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sync_outbox
                SET attempts = attempts + 1, last_error = ?
                WHERE event_id = ?
                """,
                (error[:2000], event_id),
            )

"""
Graph database interface for knowledge graph operations.

This module encapsulates connections to Neo4j (or a fallback
in‑memory graph) and provides methods to upsert entities and
relations, as well as to retrieve neighbourhood subgraphs.  The
implementation follows the ideal design: separate storage of facts
from the chunks that support them, and use of Cypher queries for
efficient graph traversals.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..config import settings
from .neo4j_connection import create_neo4j_driver


class GraphDB:
    """Neo4j wrapper for inserting and querying the knowledge graph."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.database = database if database is not None else getattr(settings, "neo4j_database", "neo4j")
        self._driver = create_neo4j_driver(self.uri, self.user, password if password is not None else settings.neo4j_password)
        # Force connection check now. Without this the driver object is
        # created lazily and the API may think Neo4j is enabled while the
        # server is unreachable.
        self._driver.verify_connectivity()

    def close(self) -> None:
        self._driver.close()

    def session(self):
        """Open a Neo4j session using the configured database when provided."""
        if self.database:
            return self._driver.session(database=self.database)
        return self._driver.session()

    def run(self, query: str, **params):
        """Run one Cypher query and return materialized records."""
        with self.session() as session:
            return list(session.run(query, **params))

    @staticmethod
    def _node_to_dict(node: Any) -> Dict[str, Any]:
        """Convert a Neo4j Node object to a JSON-serialisable dict."""
        return {
            "element_id": getattr(node, "element_id", None),
            "labels": list(getattr(node, "labels", [])),
            "properties": dict(node),
        }

    @staticmethod
    def _relationship_to_dict(rel: Any) -> Dict[str, Any]:
        """Convert a Neo4j Relationship object to a JSON-serialisable dict."""
        start = getattr(rel, "start_node", None)
        end = getattr(rel, "end_node", None)
        return {
            "element_id": getattr(rel, "element_id", None),
            "type": getattr(rel, "type", None),
            "start_node": dict(start).get("uid") if start is not None else None,
            "end_node": dict(end).get("uid") if end is not None else None,
            "properties": dict(rel),
        }

    def upsert_entity(self, entity_id: str, label: str, properties: Dict[str, Any]) -> None:
        """Upsert a canonical Entity node.

        `entity_id` is stored as the stable `uid`. The previous prototype
        used property `id`; using `uid` keeps the graph schema consistent
        with constraints and relation queries.
        """
        query = (
            "MERGE (n:Entity {uid: $uid}) "
            "SET n.canonical_name = $label, n += $props"
        )
        with self.session() as session:
            session.run(query, uid=entity_id, label=label, props=properties)

    def upsert_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        qualifiers: Dict[str, Any] | None = None,
        confidence: float | None = None,
        evidence_chunk_ids: List[str] | None = None,
    ) -> None:
        """Upsert a relationship between two entities."""
        query = (
            "MERGE (s:Entity {uid: $sub_id}) "
            "MERGE (o:Entity {uid: $obj_id}) "
            "MERGE (s)-[r:RELATION {predicate: $pred}]->(o) "
            "SET r.qualifiers_json = $quals_json, r.confidence = $conf, "
            "    r.evidence_chunk_ids = $evidence"
        )
        with self.session() as session:
            session.run(
                query,
                sub_id=subject_id,
                obj_id=object_id,
                pred=predicate,
                quals_json=json.dumps(qualifiers or {}, ensure_ascii=False),
                conf=confidence,
                evidence=evidence_chunk_ids or [],
            )

    def stats(self) -> Dict[str, int]:
        """Return compact graph counts for diagnostics and demo readiness checks."""
        query = """
        MATCH (n)
        WITH labels(n) AS labels, count(n) AS count
        UNWIND labels AS label
        RETURN label, sum(count) AS count
        """
        rel_query = "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count"
        with self.session() as session:
            node_rows = list(session.run(query))
            rel_rows = list(session.run(rel_query))
            stats = {f"nodes_{row['label']}": int(row["count"]) for row in node_rows}
            for row in rel_rows:
                stats[f"relationships_{row['type']}"] = int(row["count"])
            return stats

    def clear_all(self) -> Dict[str, int]:
        """Delete all nodes and relationships from the configured Neo4j database."""

        before = self.stats()
        with self.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        return before

    def fetch_subgraph(self, entity_ids: List[str], hops: int = 1) -> List[Dict[str, Any]]:
        """Return nodes and relationships within a given number of hops around specified entities."""
        # This Cypher uses variable length paths limited by `hops` and returns
        # nodes and relationships needed for simple visualisation.  You may
        # customise the fields returned depending on your UI needs.
        query = (
            "MATCH p = (n:Entity)-[*1..$hops]-(m:Entity) "
            "WHERE n.uid IN $ids "
            "RETURN nodes(p) AS nodes, relationships(p) AS rels"
        )
        with self.session() as session:
            result = session.run(query, hops=hops, ids=entity_ids)
            subgraphs: List[Dict[str, Any]] = []
            for record in result:
                subgraphs.append(
                    {
                        "nodes": [self._node_to_dict(node) for node in record["nodes"]],
                        "relationships": [self._relationship_to_dict(rel) for rel in record["rels"]],
                    }
                )
            return subgraphs

    # --- Schema management ---

    def create_constraints(self) -> None:
        """Create constraints and indexes for the graph model.

        The method is intentionally tolerant: if one statement is not
        supported by the active Neo4j edition/version, the remaining
        schema statements are still attempted. This prevents a Community
        Edition setup from disabling the whole graph layer.
        """
        statements = [
            "CREATE CONSTRAINT workspace_uid IF NOT EXISTS FOR (w:Workspace) REQUIRE w.uid IS UNIQUE",
            "CREATE CONSTRAINT user_uid IF NOT EXISTS FOR (u:User) REQUIRE u.uid IS UNIQUE",
            "CREATE CONSTRAINT source_uid IF NOT EXISTS FOR (s:Source) REQUIRE s.uid IS UNIQUE",
            "CREATE CONSTRAINT document_uid IF NOT EXISTS FOR (d:Document) REQUIRE d.uid IS UNIQUE",
            "CREATE CONSTRAINT chunk_uid IF NOT EXISTS FOR (c:Chunk) REQUIRE c.uid IS UNIQUE",
            "CREATE CONSTRAINT entity_uid IF NOT EXISTS FOR (e:Entity) REQUIRE e.uid IS UNIQUE",
            "CREATE CONSTRAINT tag_uid IF NOT EXISTS FOR (t:Tag) REQUIRE t.uid IS UNIQUE",
            "CREATE INDEX chunk_doc_ordinal IF NOT EXISTS FOR (c:Chunk) ON (c.document_uid, c.ordinal)",
            "CREATE INDEX doc_workspace IF NOT EXISTS FOR (d:Document) ON (d.workspace_uid)",
            "CREATE INDEX doc_external IF NOT EXISTS FOR (d:Document) ON (d.workspace_uid, d.external_id)",
            "CREATE INDEX entity_norm_name IF NOT EXISTS FOR (e:Entity) ON (e.norm_name)",
            "CREATE FULLTEXT INDEX doc_titles IF NOT EXISTS FOR (d:Document) ON EACH [d.title]",
            "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (e:Entity) ON EACH [e.canonical_name, e.norm_name]",
        ]
        with self.session() as session:
            for stmt in statements:
                try:
                    session.run(stmt)
                except Exception:
                    continue

    # --- Node upserts ---

    def upsert_workspace(self, uid: str, slug: str, name: str, created_at: str | None = None) -> None:
        query = (
            "MERGE (w:Workspace {uid: $uid}) "
            "SET w.slug = $slug, w.name = $name, "
            "    w.created_at = coalesce(w.created_at, $created_at)"
        )
        with self.session() as session:
            session.run(query, uid=uid, slug=slug, name=name, created_at=created_at)

    def upsert_user(self, uid: str, email: str, display_name: str | None = None) -> None:
        query = (
            "MERGE (u:User {uid: $uid}) "
            "SET u.email = $email, u.display_name = $display_name"
        )
        with self.session() as session:
            session.run(query, uid=uid, email=email, display_name=display_name)

    def upsert_source(self, uid: str, type_: str, uri: str | None = None, checksum: str | None = None, imported_at: str | None = None) -> None:
        query = (
            "MERGE (s:Source {uid: $uid}) "
            "SET s.type = $type, s.uri = $uri, s.checksum = $checksum, s.imported_at = coalesce(s.imported_at, $imported_at)"
        )
        with self.session() as session:
            session.run(query, uid=uid, type=type_, uri=uri, checksum=checksum, imported_at=imported_at)

    def link_workspace_source(self, workspace_uid: str, source_uid: str) -> None:
        query = (
            "MATCH (w:Workspace {uid: $workspace_uid}), (s:Source {uid: $source_uid}) "
            "MERGE (w)-[:HAS_SOURCE]->(s)"
        )
        with self.session() as session:
            session.run(query, workspace_uid=workspace_uid, source_uid=source_uid)

    def link_document_source(self, document_uid: str, source_uid: str) -> None:
        query = (
            "MATCH (d:Document {uid: $document_uid}), (s:Source {uid: $source_uid}) "
            "MERGE (d)-[:DERIVED_FROM]->(s)"
        )
        with self.session() as session:
            session.run(query, document_uid=document_uid, source_uid=source_uid)

    def upsert_document(self, doc_id: str, workspace_uid: str, title: str, source_uid: str | None, external_id: str | None, parser: str, language: str | None, status: str | None, created_at: str | None, updated_at: str | None, version: int | None) -> None:
        query = (
            "MERGE (d:Document {uid: $doc_id}) "
            "SET d.workspace_uid = $workspace_uid, d.title = $title, d.source_uid = $source_uid, "
            "    d.external_id = $external_id, d.parser = $parser, d.language = $language, d.status = $status, "
            "    d.created_at = coalesce(d.created_at, $created_at), d.updated_at = $updated_at, d.version = $version"
        )
        rel_query = (
            "MATCH (w:Workspace {uid: $workspace_uid}), (d:Document {uid: $doc_id}) "
            "MERGE (w)-[:OWNS]->(d)"
        )
        with self.session() as session:
            session.run(query, doc_id=doc_id, workspace_uid=workspace_uid, title=title, source_uid=source_uid, external_id=external_id, parser=parser, language=language, status=status, created_at=created_at, updated_at=updated_at, version=version)
            session.run(rel_query, workspace_uid=workspace_uid, doc_id=doc_id)

    def upsert_chunk_node(self, chunk: Dict[str, Any]) -> None:
        """Upsert a chunk node and link it to its document.

        `chunk` should contain keys: uid, document_uid, workspace_uid, ordinal,
        text, page_start, page_end, section_path, char_start, char_end,
        token_count, text_hash, preview, embedding_version, updated_at.
        """
        query = (
            "MERGE (c:Chunk {uid: $uid}) "
            "SET c.document_uid = $document_uid, c.workspace_uid = $workspace_uid, c.ordinal = $ordinal, "
            "    c.page_start = $page_start, c.page_end = $page_end, c.section_path = $section_path, "
            "    c.char_start = $char_start, c.char_end = $char_end, c.token_count = $token_count, "
            "    c.text_hash = $text_hash, c.preview = $preview, c.embedding_version = $embedding_version, "
            "    c.updated_at = $updated_at"
        )
        rel_query = (
            "MATCH (d:Document {uid: $document_uid}), (c:Chunk {uid: $uid}) "
            "MERGE (d)-[:CONTAINS {ordinal: $ordinal}]->(c)"
        )
        with self.session() as session:
            session.run(query, **chunk)
            session.run(rel_query, document_uid=chunk["document_uid"], uid=chunk["uid"], ordinal=chunk.get("ordinal"))

    def tag_document(self, document_uid: str, tag_uid: str, tag_name: str) -> None:
        query_tag = "MERGE (t:Tag {uid: $tag_uid}) SET t.name = $tag_name"
        query_rel = "MATCH (d:Document {uid: $document_uid}) MATCH (t:Tag {uid: $tag_uid}) MERGE (d)-[:TAGGED_WITH]->(t)"
        with self.session() as session:
            session.run(query_tag, tag_uid=tag_uid, tag_name=tag_name)
            session.run(query_rel, document_uid=document_uid, tag_uid=tag_uid)

    def link_chunk_sequence(self, document_uid: str) -> None:
        """Create NEXT edges between consecutive chunks of a document."""
        query = (
            "MATCH (:Document {uid: $document_uid})-[:CONTAINS]->(c:Chunk) "
            "WITH c ORDER BY c.ordinal ASC "
            "WITH collect(c) AS chunks "
            "WHERE size(chunks) > 1 "
            "UNWIND range(0, size(chunks)-2) AS i "
            "WITH chunks[i] AS a, chunks[i+1] AS b, i "
            "MERGE (a)-[:NEXT {order: i}]->(b)"
        )
        with self.session() as session:
            session.run(query, document_uid=document_uid)

    def link_chunk_entity(
        self,
        chunk_uid: str,
        entity_uid: str,
        canonical_name: str,
        entity_type: str,
        norm_name: str | None = None,
        confidence: float | None = None,
        count: int | None = None,
    ) -> None:
        """Create or update an Entity node and MENTIONS edge from a chunk."""
        query = (
            "MERGE (e:Entity {uid: $entity_uid}) "
            "SET e.canonical_name = $canonical_name, e.type = $entity_type, e.norm_name = $norm_name "
            "WITH e "
            "MATCH (c:Chunk {uid: $chunk_uid}) "
            "MERGE (c)-[m:MENTIONS]->(e) "
            "SET m.confidence = $confidence, m.count = $count"
        )
        with self.session() as session:
            session.run(
                query,
                chunk_uid=chunk_uid,
                entity_uid=entity_uid,
                canonical_name=canonical_name,
                entity_type=entity_type,
                norm_name=norm_name,
                confidence=confidence,
                count=count,
            )

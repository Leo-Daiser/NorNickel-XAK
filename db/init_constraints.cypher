// Neo4j schema definition for the knowledge graph.
// Community-friendly constraints and indexes. Safe to run multiple times.

CREATE CONSTRAINT workspace_uid IF NOT EXISTS
FOR (w:Workspace) REQUIRE w.uid IS UNIQUE;

CREATE CONSTRAINT user_uid IF NOT EXISTS
FOR (u:User) REQUIRE u.uid IS UNIQUE;

CREATE CONSTRAINT source_uid IF NOT EXISTS
FOR (s:Source) REQUIRE s.uid IS UNIQUE;

CREATE CONSTRAINT document_uid IF NOT EXISTS
FOR (d:Document) REQUIRE d.uid IS UNIQUE;

CREATE CONSTRAINT chunk_uid IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.uid IS UNIQUE;

CREATE CONSTRAINT entity_uid IF NOT EXISTS
FOR (e:Entity) REQUIRE e.uid IS UNIQUE;

CREATE CONSTRAINT tag_uid IF NOT EXISTS
FOR (t:Tag) REQUIRE t.uid IS UNIQUE;

CREATE INDEX chunk_doc_ordinal IF NOT EXISTS
FOR (c:Chunk) ON (c.document_uid, c.ordinal);

CREATE INDEX doc_workspace IF NOT EXISTS
FOR (d:Document) ON (d.workspace_uid);

CREATE INDEX doc_external IF NOT EXISTS
FOR (d:Document) ON (d.workspace_uid, d.external_id);

CREATE INDEX entity_norm_name IF NOT EXISTS
FOR (e:Entity) ON (e.norm_name);

CREATE FULLTEXT INDEX doc_titles IF NOT EXISTS
FOR (d:Document) ON EACH [d.title];

CREATE FULLTEXT INDEX entity_names IF NOT EXISTS
FOR (e:Entity) ON EACH [e.canonical_name, e.norm_name];

# Knowledge Expansion Engine

This project expands the knowledge base deterministically. LLMs are not used as
fact extractors.

## Pipeline

Current expansion path:

```text
input documents
 -> parser_router / document intelligence fallback
 -> chunk catalog
 -> ExtractionPipeline(mode=deterministic, enable_llm=false)
 -> graph model adapter
 -> canonical fact normalization
 -> deduplication with evidence merge
 -> conflict and data-gap detection
 -> graph delta calculation
 -> fallback catalog/retrieval update
 -> optional Neo4j sync
 -> expansion report
```

The implementation entry point is `app/knowledge/expansion.py`.

## Document Identity And Versions

Uploaded files use:

- `content_hash`: SHA-256 of file content;
- `doc_id`: stable id from content hash;
- `source_name`: uploaded filename or URL-derived name;
- `document_version`: stable for the same filename/content hash, incremented
  when the same filename is uploaded with changed content;
- `active`: catalog metadata flag controlling participation in retrieval and
  analytical answers.

The minimal version model keeps old document versions in the catalog. It does
not implement temporal graph queries yet.

## Canonical Facts

Canonical facts are keyed by:

- canonical material;
- canonical regime;
- canonical property;
- normalized numeric value and unit when available;
- qualitative effect when available.

Source identity is omitted from the default canonical key so duplicate evidence
can be merged into one fact. Evidence is never discarded; repeated or new source
evidence is merged into the canonical fact evidence list.

## Delta Fields

The upload response and `/knowledge/expansion-report` expose:

- `new_nodes_count`;
- `new_edges_count`;
- `new_materials`;
- `new_regimes`;
- `new_properties`;
- `new_canonical_facts`;
- `duplicate_facts`;
- `corroborated_facts`;
- `conflict_groups_added`;
- `data_gaps_added`;
- `new_comparison_opportunities`;
- `new_research_questions`.

Research questions are generated from deterministic graph gaps/rules, not from
an LLM.

## Active/Inactive Semantics

Inactive documents are excluded from active catalog chunks. This affects:

- retrieval index rebuild;
- fallback graph repository;
- knowledge expansion report;
- strict Neo4j reads after graph sync.

The active flag is non-destructive. It does not delete documents, chunks or
facts. Neo4j sync marks document/chunk nodes active or inactive, and repository
queries only read facts supported by active evidence chunks.

## API

Read-mostly endpoints:

```text
GET /knowledge/summary
GET /knowledge/expansion-report
```

Explicit update endpoints:

```text
POST /knowledge/rebuild
POST /knowledge/sync-neo4j
```

`/ask` is unchanged.

## Current Source Support

The current ingestion layer treats sources as `Document` records with metadata.
It intentionally does not add source-specific ontology nodes.

Supported source paths:

- files uploaded through `/ingest/documents`;
- HTML files uploaded as normal files;
- URL web pages through `/ingest/url`;
- CSV/XLSX catalog-like files through the existing table parser and
  deterministic extraction;
- plain text and Markdown documents through fallback text parsing.

URL web pages are first-class sources through document metadata:

- `source_type = "url"`;
- `source_url` stores the raw final URL;
- `source_name` / `source_title` stores a readable HTML title when available;
- `content_hash` is calculated from URL plus fetched content;
- chunks keep `source_type`, `source_url`, `source_name`, `filename` and parser
  metadata.

Main UI surfaces the readable title or `domain · path` label. Raw URLs remain in
metadata/diagnostics for provenance.

Until a real corpus and detailed task specification are available,
source-specific adapters for employees, laboratories, equipment and topic tags
are intentionally not overfit.

## CLI

```bash
python scripts/knowledge_expansion_report.py
python scripts/knowledge_expansion_report.py --since-last-ingest
python scripts/knowledge_expansion_report.py --document-id <id>
python scripts/knowledge_expansion_report.py --json artifacts/knowledge_expansion_report.json
```

`--since-last-ingest` is currently a controlled limitation: the script reports
the current state and records that persisted since-last-run checkpoints are not
implemented yet.

## Evaluation

```bash
python evaluation/eval_knowledge_expansion.py
```

The eval runs in an economy-compatible fallback profile and checks:

- initial VT6/Ti-6Al-4V conflict;
- evidence on every accepted fact;
- adding 7075-T6 creates a strength comparison opportunity;
- `77 ksi` normalizes to approximately `530.9 MPa`;
- corrosion data gap detection;
- idempotent re-ingest;
- active/inactive exclusion and reactivation.

## Resource Contract

Knowledge expansion does not require:

- LLM extraction;
- embeddings;
- Qdrant;
- GPU.

Embeddings and guarded LLM polish can improve retrieval and phrasing, but the
source of truth remains deterministic extraction plus the graph/canonical fact
layer.

## Current Limitations

- No persisted since-last-run checkpoint table yet.
- No temporal graph query language for older document versions.
- Neo4j sync is idempotent and active-aware, but old inactive nodes are retained
  for provenance unless a future explicit cleanup command is added.

# Architecture

## Goal

Система реализует Technical Document Intelligence: превращает технические документы, таблицы и HTML/URL-ресурсы в структурированные факты и локальный knowledge graph.

## Pipeline

```text
Input files / URL
  -> parser_router
  -> Document + SourceChunk + metadata
  -> EntityRelationExtractor
  -> local graph builder
  -> RetrievalEngine
  -> /ask structured answer
  -> Streamlit UI
```

## Parser layer

Поддерживаемые форматы:

- PDF с текстовым слоем через `pypdf`;
- DOCX через `python-docx`;
- PPTX через `python-pptx`;
- HTML/URL через `BeautifulSoup`;
- CSV/XLSX через `pandas/openpyxl`;
- TXT/MD через plain UTF-8 fallback.

Каждый chunk хранит:

- `doc_id`;
- `filename`;
- `source_type`;
- `source_url`;
- `page_start/page_end`;
- `section_path`;
- `table_id`;
- `row_id`;
- `image_refs`;
- `parser_name`;
- `parser_error`;
- `char_start/char_end`;
- `text_hash`.

Табличные строки сохраняются как отдельные chunks:

```text
Column A: value | Column B: value | Column C: value
```

## Extraction layer

Rule-based extractor извлекает:

- `TechnicalObject`;
- `Part`;
- `ArticleNumber`;
- `Standard`;
- `Material`;
- `Parameter`;
- `Requirement`;
- `ImageArtifact`;
- `Experiment`;
- `ProcessRegime`;
- `Property`;
- `PropertyValue`;
- `DataGap`.

Основные отношения:

- `DOCUMENT_HAS_SECTION`;
- `SECTION_HAS_CHUNK`;
- `CHUNK_MENTIONS_ENTITY`;
- `OBJECT_HAS_PARAMETER`;
- `OBJECT_HAS_PART`;
- `PART_HAS_ARTICLE_NUMBER`;
- `OBJECT_MADE_OF_MATERIAL`;
- `OBJECT_COMPLIES_WITH_STANDARD`;
- `REQUIREMENT_APPLIES_TO_OBJECT`;
- `IMAGE_LINKED_TO_SECTION`;
- `MISSING_FOR`;
- `STUDIES`;
- `USES_REGIME`;
- `MEASURES`;
- `OF_PROPERTY`;
- `HAS_CHANGE`.

## Retrieval

Обязательный fallback:

```text
SimpleBM25
```

Optional режимы:

```text
RETRIEVAL_MODE=bm25|embedding|hybrid
ENABLE_LOCAL_EMBEDDINGS=false|true
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Если `sentence-transformers`, Qdrant или модель недоступны, система не падает и использует BM25.

Docker по умолчанию собирает минимальный fallback-образ из `requirements.txt`.
Этот режим не требует `torch`. Для локальных embeddings нужен
`requirements-embeddings.txt` или build arg `INSTALL_FULL=true`, который ставит
`requirements-full.txt`. Qdrant/Neo4j поднимаются только через compose profile
`full`.

## /ask contract

`/ask`:

1. определяет intent вопроса;
2. строит `QueryConstraints`;
3. для strict `material + regime + property` делает exact graph query;
4. для аналитических вопросов использует `AnalyticalQueryRouter`;
5. получает graph facts через `GraphRepository`;
6. ищет supporting evidence через retrieval;
7. строит compact `GraphContext`;
8. формирует grounded `answer`;
9. возвращает structured JSON:

```json
{
  "answer": "...",
  "facts": [],
  "technical_objects": [],
  "parts": [],
  "parameters": [],
  "standards": [],
  "materials": [],
  "requirements": [],
  "sources": [],
  "gaps": [],
  "subgraph": {"nodes": [], "edges": []}
}
```

## Analytical GraphRAG layer

Новый analytical слой расположен поверх strict QA и не заменяет его.

```text
question
  -> QueryPlanner
  -> QueryConstraints
  -> AnalyticalQueryRouter
  -> GraphRepository
  -> EvidenceSearch / EvidenceReranker
  -> GraphContextBuilder
  -> AnswerSynthesizer
  -> /ask response
```

Поддержанные analytical intents:

- `material_overview`;
- `regime_overview`;
- `property_overview`;
- `decision_history`;
- `gap_analysis`;
- `material_comparison`;
- `regime_comparison`;
- `similar_experiments`;
- `equipment_usage`;
- `lab_activity`;
- `team_activity`;
- `topic_search`;
- `graph_neighborhood`;
- `general_search`.

Evidence retrieval не создаёт facts. Он используется только для цитат и диагностики.
Если пользователь явно указал `material + regime + property`, strict path имеет приоритет и positive answer возможен только при exact chain в графе.

## Expert cockpit layer

Cockpit UI и API предназначены для ручной проверки системы жюри:

```text
Streamlit cockpit
  -> /ask
  -> /graph/stats
  -> /graph/entities
  -> /graph/entity/{entity_type}/{entity_id}
  -> /graph/neighborhood
  -> /graph/gaps
  -> /graph/decision-history
  -> /graph/similar-experiments
  -> /system/capabilities
  -> /demo/scenarios
```

Explorer API использует тот же `GraphRepository`, что и strict/analytics QA. Поэтому:

- при `KG_BACKEND=neo4j` cockpit читает materialized Neo4j graph;
- при `KG_BACKEND=auto` API выбирает Neo4j, если прямой `RETURN 1` check успешен, иначе использует fallback;
- при `KG_BACKEND=fallback` cockpit строится из SQLite catalog + accepted `ExtractionPipeline` facts;
- invalid labels проходят через whitelist и возвращают `400`, а не произвольный Cypher.

Neo4j availability is not treated as a permanent startup decision. `/health` and
`/system/capabilities` force a short retry path, so a service that initially
started before Neo4j can recover to `kg_backend_active=neo4j` after Neo4j becomes
reachable. Diagnostics expose URI/user and `neo4j_password_configured`, but never
the password value.

## Runtime presets and final hardening

UI/API expose three user-facing runtime presets instead of raw technical switches:

- `expert_max`: KG auto, parser auto, hybrid extraction, hybrid/template grounded answer.
- `strict_audit`: KG auto, deterministic extraction, template answer, no LLM polish.
- `offline_reliable`: fallback KG, fallback parser, deterministic extraction, template answer.

Preset selection is per request. It does not mutate process-wide environment variables. `/ask` accepts both legacy query params and JSON body; JSON body has priority and records this in diagnostics.

Fallback and Neo4j now share the same extraction truth contract:

```text
SQLite chunks
  -> ExtractionPipeline
  -> accepted ExtractionBundle
  -> ExperimentFact/DataGap projection
  -> GraphRepository / GraphWriter
```

Rejected extraction items do not become graph facts. The old `EntityRelationExtractor` remains only inside deterministic extraction, not as a separate fallback graph source of truth.

Ingestion hardening:

- URL ingestion allows only `http/https`, blocks localhost/private/link-local addresses by default, rechecks redirects and enforces response-size/time limits.
- Upload ingestion enforces file count, file size and extension allowlist.
- Release builder creates a clean zip and excludes runtime databases, audit logs, caches and secrets.

## Storage

Fallback storage:

- SQLite catalog: documents/chunks;
- SQLite outbox: optional Qdrant projection events;
- in-memory BM25 index.

Optional storage:

- Neo4j for canonical graph;
- Qdrant for dense vector projection.

## Degradation

| Missing component | Behaviour |
|---|---|
| GPU | CPU fallback работает. |
| Neo4j | `/ask` возвращает local JSON graph. |
| Qdrant | Retrieval остаётся на BM25. |
| Docling/MarkItDown | Используются ручные fallback parsers. |
| LLM API | Rule-based extractor продолжает работать. |
| Internet | Файловый ingestion и evaluation работают локально. |

## Production work left

- LLM structured extractor с валидацией JSON как optional layer.
- OCR/VLM для схем без текстового слоя.
- Улучшенная section hierarchy для сложных Word/PDF.
- Persistent local graph storage.
- Метрики качества на реальном корпусе организаторов.

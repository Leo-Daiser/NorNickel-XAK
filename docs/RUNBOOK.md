# RUNBOOK

Команды ниже рассчитаны на Windows PowerShell.

## 1. Перейти в проект

```powershell
cd C:\Users\WORK\Avito-Xakaton\hackathon_project
```

Если вы находитесь в `C:\Users\WORK\Avito-Xakaton`, выполните:

```powershell
cd .\hackathon_project
```

## 2. Установить зависимости

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Проверить:

```powershell
python --version
python -m pip list
```

## 3. Прогнать тесты и benchmark

```powershell
python -m pytest -q
python evaluation\eval_demo.py
python evaluation\eval_stress.py
python evaluation\eval_extraction.py
python evaluation\eval_parser.py
python evaluation\eval_analytics.py
python evaluation\eval_cockpit.py
python scripts\check_project.py
python scripts\check_release_package.py
python scripts\diagnose_stack.py --ingest-demo
```

Ожидаемо:

```text
pytest: все тесты passed
eval_demo.py: PASS по всем gold questions
eval_stress.py: PASS по дополнительным нестандартным вопросам
eval_extraction.py: PASS по extraction quality gates
eval_parser.py: PASS по parser/document intelligence gates
eval_analytics.py: PASS по analytical GraphRAG questions
eval_cockpit.py: PASS по expert cockpit readiness
check_project.py: SMOKE TEST PASSED
```

## 3.1. Запуск через Docker

Минимальный режим, который должен работать на любой машине с Docker:

```powershell
docker compose up -d --build
```

`--build` нужен только после изменения Dockerfile/requirements или если образа ещё нет. Для обычного перезапуска:

```powershell
docker compose up -d
docker compose restart api ui
```

Если зависимости уже скачаны, а нужно применять правки кода без пересборки образа:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart api ui
```

Проверить:

```powershell
curl http://localhost:8000/health
```

Открыть UI:

```text
http://localhost:8501
```

Остановить:

```powershell
docker compose down
```

Full demo-режим с Qdrant/Neo4j и установкой optional Python-зависимостей:

```powershell
$env:INSTALL_FULL="true"
$env:RETRIEVAL_MODE="hybrid"
$env:ENABLE_LOCAL_EMBEDDINGS="true"
docker compose -f docker-compose.yml -f docker-compose.optional.yml --profile full up -d --build
```

Если full-зависимости не нужны, не задавайте `INSTALL_FULL=true`: fallback-образ собирается быстрее и стабильнее.

## 4. Запустить API

```powershell
uvicorn app.api:app --reload --port 8000
```

Проверить:

```powershell
curl http://localhost:8000/health
```

`graph: disabled` допустим: это локальный fallback без Neo4j.

## 5. Запустить UI

Во втором PowerShell:

```powershell
cd C:\Users\WORK\Avito-Xakaton\hackathon_project
.\.venv\Scripts\Activate.ps1
streamlit run app/ui.py
```

Открыть:

```text
http://localhost:8501
```

## 6. Загрузить demo_data

В UI выберите файлы из:

```text
demo_data/
```

Рекомендуемый набор:

```text
valve_datasheet.html
pump_spec.docx
parts_catalog.csv
materials_table.xlsx
standards_requirements.txt
synthetic_vt6_heat_treatment.csv
readme_demo_scenario.md
```

После загрузки задайте вопросы из `evaluation/gold_questions.json`.

## 7. URL ingestion

Через Swagger:

```text
http://localhost:8000/docs
POST /ingest/url
```

Через PowerShell:

```powershell
curl -X POST "http://localhost:8000/ingest/url?url=https://example.com/page.html"
```

Если URL недоступен или не HTML, API вернёт понятную ошибку `400`.

## 8. Optional режимы

Qdrant/Neo4j:

```powershell
docker compose --profile full up -d --build
```

Embeddings:

```powershell
python -m pip install -r requirements-embeddings.txt
$env:RETRIEVAL_MODE="hybrid"
$env:ENABLE_LOCAL_EMBEDDINGS="true"
$env:EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

Если embeddings/Qdrant недоступны, retrieval автоматически остаётся на BM25.

Зависимости по режимам:

| Файл | Когда ставить | Тянет `torch` |
|---|---|---|
| `requirements.txt` | обычный запуск API/UI, fallback, BM25, парсинг PDF/DOCX/PPTX/HTML/CSV/XLSX/TXT | нет |
| `requirements-parsing.txt` | если нужен MarkItDown как дополнительный конвертер | нет, в норме |
| `requirements-embeddings.txt` | если нужны локальные dense embeddings через sentence-transformers | да, обычно |
| `requirements-ocr.txt` | если появятся сканы без текстового слоя | может тянуть тяжёлые OCR-зависимости |
| `requirements-research-heavy.txt` | экспериментальный ML/Docling/FlagEmbedding стек | да |
| `requirements-full.txt` | только для сильной demo-машины, когда осознанно нужны MarkItDown + embeddings | да |

## 9. Диагностика

Проверить количество документов и chunks:

```powershell
curl http://localhost:8000/health
```

Пересобрать BM25 index из SQLite:

```powershell
curl -X POST http://localhost:8000/admin/rebuild-index
```

Проверить retrieval:

```powershell
curl "http://localhost:8000/debug/retrieval?question=клапан%20DN50&top_k=5"
```

Проверить локально весь стек без внешних сервисов:

```powershell
python scripts\diagnose_stack.py --ingest-demo
```

Проверить реально настроенные optional embeddings:

```powershell
python scripts\diagnose_stack.py --live-embeddings --ingest-demo
```

Проверить реально настроенный optional LLM:

```powershell
python scripts\diagnose_stack.py --live-llm --ingest-demo
```

Команды live-режима запускайте только если модель embeddings/LLM уже установлена или доступна: иначе они корректно деградируют, но могут потратить время на попытку загрузки модели или сетевой запрос.

## 10. Structured extraction pipeline

Extraction запускается до materialization в Neo4j: документы и таблицы превращаются в валидированный `ExtractionBundle`, rejected-факты не пишутся в граф.

Минимальный deterministic режим без LLM:

```powershell
$env:EXTRACTION_MODE="deterministic"
$env:EXTRACTION_MIN_CONFIDENCE="0.55"
$env:EXTRACTION_ENABLE_LLM="false"
python evaluation\eval_extraction.py
python scripts\sync_graph_to_neo4j.py
```

Hybrid режим: deterministic extractor остаётся основой, LLM используется только если настроена. Если LLM недоступна, sync не падает:

```powershell
$env:EXTRACTION_MODE="hybrid"
$env:EXTRACTION_ENABLE_LLM="false"
python evaluation\eval_extraction.py
```

LLM-only режим требует настроенного provider и должен падать явно, если LLM недоступна:

```powershell
$env:EXTRACTION_MODE="llm"
$env:EXTRACTION_ENABLE_LLM="true"
python scripts\sync_graph_to_neo4j.py
```

Audit files:

```text
data/extraction_audit/accepted.jsonl
data/extraction_audit/rejected.jsonl
data/extraction_audit/diagnostics.jsonl
```

Проверить текущие настройки:

```powershell
curl http://localhost:8000/health
```

Ожидаемые поля:

```json
{
  "extraction": {
    "extraction_mode": "deterministic",
    "extraction_min_confidence": 0.55,
    "llm_extraction_available": false,
    "audit_enabled": true
  }
}
```

## 11. Document intelligence ingestion

Parser сохраняет структуру документа: блоки, таблицы, изображения и evidence-ready chunks.

Режимы:

```powershell
$env:PARSER_BACKEND="auto"
$env:ENABLE_OCR="false"
$env:OCR_BACKEND="none"
python evaluation\eval_parser.py
```

Семантика:

```text
auto       — пробует optional Docling/MarkItDown, затем fallback
fallback   — pypdf, pandas, python-docx, python-pptx, BeautifulSoup, plain text
docling    — требует установленный Docling, иначе явная ошибка
markitdown — требует MarkItDown, иначе явная ошибка
```

Для PDF без текстового слоя OCR не запускается по умолчанию. Вместо этого parser diagnostics покажет:

```json
{
  "scanned_pdf_detected": true,
  "ocr_enabled": false,
  "warnings": ["PDF appears scanned; OCR is disabled"]
}
```

Parser audit:

```text
data/parser_audit/parsed.jsonl
data/parser_audit/errors.jsonl
```

Проверить health:

```powershell
curl http://localhost:8000/health
```

Ожидаемые поля:

```json
{
  "parser_backend": "auto",
  "docling_available": false,
  "markitdown_available": false,
  "ocr_enabled": false,
  "parser_audit_enabled": true
}
```

## 12. Release hygiene

Перед сдачей архива проверьте, что не попали секреты и runtime-файлы:

```powershell
python scripts\check_release_package.py
```

В рабочей папке команда может упасть из-за `.env`, `data/*.sqlite3`, audit JSONL или `__pycache__`. Это нормально: такие файлы не должны входить в релиз. Проверка чистой папки:

```powershell
python scripts\check_release_package.py --path dist\hackathon_project_release
```

## 13. Hybrid GraphRAG analytical query engine

Аналитический слой `/ask` используется для свободных вопросов, где нет полного strict-набора `material + regime + property`.

Поддержанные режимы:

```text
strict exact matching — material + regime + property, без галлюцинаций
overview               — обзор по материалу, режиму или свойству
comparison             — сравнение материалов или режимов
history                — история решений по материалу
gaps                   — анализ пробелов в данных
search                 — похожие эксперименты и тематический поиск
neighborhood           — связанные сущности вокруг материала/режима/свойства
```

Text retrieval используется только как supporting evidence. Он не создаёт facts и не заменяет graph exact match.

PowerShell:

```powershell
$env:ANSWER_SYNTHESIS_MODE="template"
python evaluation\eval_analytics.py
```

Если Neo4j доступен:

```powershell
$env:KG_BACKEND="auto"
python scripts\sync_graph_to_neo4j.py
python evaluation\eval_analytics.py
```

Ожидаемые поля в `/ask`:

```json
{
  "analytical_intent": "material_overview",
  "answer_mode": "overview",
  "graph_context": {
    "facts_count": 12,
    "sources_count": 5,
    "evidence_count": 5,
    "subgraph_nodes": 18,
    "subgraph_edges": 24
  },
  "evidence": [],
  "diagnostics": {
    "evidence_backend": "bm25",
    "answer_synthesis_mode": "template"
  }
}
```

## 14. Product UI: one-screen GraphRAG demo

Запуск:

```powershell
uvicorn app.api:app --reload --port 8000
streamlit run app/ui.py
```

Streamlit показывает один основной экран. В sidebar остаются только компактные diagnostics, без навигации по внутренним разделам.
На странице доступны три режима качества:

- `Лучший ответ`;
- `Строгая проверка`;
- `Офлайн-режим`.

Основной поток:

```text
режим работы -> загрузка/выбор активных документов -> вопрос -> ответ -> интерактивный граф -> details
```

Документный блок позволяет:

- загрузить PDF/DOCX/PPTX/XLSX/CSV/HTML/TXT/MD;
- посмотреть chunks/parser diagnostics;
- включить или выключить документ из active corpus;
- обновить graph/retrieval по активным документам.

Рекомендуемый порядок демо для жюри:

```text
1. Проверить список документов и active flags.
2. Загрузить документы при необходимости.
3. Нажать Обновить граф по активным документам.
4. Лучший ответ: ВТ6 + отжиг + прочность.
5. Строгая проверка: ВТ6 + криообработка + вязкость.
6. Overview: Что уже делали по ВТ6?
7. Раскрыть facts/evidence/history/gaps/diagnostics.
8. Показать интерактивный graph zoom/pan/drag.
```

Проверка cockpit/API и качества ответа:

```powershell
python evaluation\eval_cockpit.py
python evaluation\eval_answer_quality.py
python evaluation\eval_ui_product.py
```

## Pre-demo checklist

Перед показом запускайте один контролируемый gate вместо ручной проверки
разрозненных экранов. Реальные API keys должны лежать только в локальном
`.env`; не копируйте их в `.env.example`, README, код, логи или чат.

Полная пересборка API с optional local embeddings:

```powershell
docker compose down
$env:EXTRA_REQUIREMENTS="requirements-embeddings.txt"
$env:RETRIEVAL_MODE="hybrid"
$env:ENABLE_LOCAL_EMBEDDINGS="true"
$env:EAGER_LOCAL_EMBEDDINGS="false"
$env:DIRECT_QDRANT_PROJECTION="false"
$env:EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
docker compose --profile full build --no-cache api
docker compose --profile full up -d
```

Проверки внутри контейнера:

```powershell
docker compose exec api python scripts/check_mistral_connection.py
docker compose exec api python scripts/check_embeddings_runtime.py
docker compose exec api python scripts/demo_gate.py
curl http://localhost:8000/health
```

Если нужно проверить только импорт `sentence-transformers`:

```powershell
docker compose exec api python -c "import sentence_transformers; print('sentence-transformers ok')"
```

Интерпретация `scripts/demo_gate.py`:

- `PASS` — проект можно показывать по основному demo path.
- `WARN` по embeddings допустим, если BM25 работает: это означает controlled
  degradation, а не падение retrieval.
- `FAIL` по Neo4j, LLM, preset mapping, `/ask` или raw leakage нужно исправить
  до показа.

Gate проверяет `/health`, активный Neo4j, готовность LLM, retrieval diagnostics,
runtime presets, несколько demo-вопросов, отсутствие `technical_answer`,
`doc_`, `chunk_`, `EXP-`, `SCI-` и raw graph labels в основном ответе, а также
security sanity-check для `.env` и release package.

### Knowledge expansion checks

Knowledge expansion работает детерминированно: новые документы проходят parsing,
chunking, extraction, canonical normalization, deduplication, conflict/data-gap
detection и delta report. LLM extraction не используется.

Локальный отчёт:

```powershell
python scripts/knowledge_expansion_report.py --json artifacts/knowledge_expansion_report.json
```

Controlled eval в economy-compatible режиме:

```powershell
python evaluation/eval_knowledge_expansion.py
```

Если Docker/Neo4j доступен:

```powershell
docker compose exec api python scripts/init_neo4j_schema.py
docker compose exec api python scripts/sync_graph_to_neo4j.py
docker compose exec api python scripts/smoke_neo4j_graph.py
docker compose exec api python scripts/knowledge_expansion_report.py
```

Инварианты:

- повторный ingest того же документа не увеличивает canonical facts;
- изменённое содержимое того же файла получает новую версию документа;
- inactive documents исключаются из active retrieval/graph answer path;
- accepted facts имеют evidence;
- conflicts и data gaps показываются в отчётах, а не скрываются;
- `/ask` contract не меняется.

### Extraction quality report

Локально, без доступа к контейнерной сети Neo4j, запускайте отчёт явно без
persisted graph scan:

```powershell
python scripts/extraction_quality_report.py --skip-neo4j
```

В Docker API-контейнере Neo4j-переменные уже передаются сервису, поэтому можно
проверить и persisted graph records:

```powershell
docker compose exec api python scripts/extraction_quality_report.py
```

Если локальный запуск пишет `Neo4j scan skipped: Neo4j connection settings are
not configured for this runtime.`, это не ошибка extraction layer. Это означает,
что отчёт построен по локальному catalog/corpus, но скрипт не получил
`NEO4J_URI`/`NEO4J_PASSWORD` для проверки записей в контейнерном Neo4j. Для
локального scan можно передать параметры явно:

```powershell
python scripts/extraction_quality_report.py `
  --neo4j-uri bolt://localhost:7687 `
  --neo4j-user neo4j `
  --neo4j-password <local-password> `
  --neo4j-database neo4j
```

Пароль не печатается в отчёте. В JSON смотрите поля
`neo4j_scan_status`, `neo4j_scan_warning` и
`legacy_neo4j_records_missing_normalized_fields`.

Полезные endpoints:

```powershell
curl http://localhost:8000/system/capabilities
curl http://localhost:8000/graph/stats
curl "http://localhost:8000/graph/entities?entity_type=Material&limit=20"
curl "http://localhost:8000/graph/neighborhood?entity_type=Material&entity_id=ВТ6"
curl http://localhost:8000/demo/scenarios
curl http://localhost:8000/documents
curl -X POST http://localhost:8000/system/test-llm
```

## Advanced demo mode: embeddings and grounded LLM

The project is designed to work in CPU-only fallback mode first. Dense retrieval and LLM synthesis are optional accelerators; failures automatically fall back to BM25 and rule-based answers.

### Hybrid retrieval without Qdrant

For Phase 1, use an in-memory sentence-transformers index instead of Qdrant.
This keeps Neo4j as the source of truth and uses embeddings only for candidate
retrieval.

```powershell
$env:RETRIEVAL_MODE="hybrid"
$env:ENABLE_LOCAL_EMBEDDINGS="true"
$env:EAGER_LOCAL_EMBEDDINGS="false"
$env:DIRECT_QDRANT_PROJECTION="false"
$env:EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
uvicorn app.api:app --reload
```

For Docker, install the optional dependency explicitly:

```powershell
$env:EXTRA_REQUIREMENTS="requirements-embeddings.txt"
docker compose down
docker compose --profile full up -d --build
```

`EAGER_LOCAL_EMBEDDINGS=false` keeps startup fast. The local vector index is
built lazily on the first dense query. If the dependency/model is missing, the
API continues with BM25 and reports a degradation reason.

Check status:

```powershell
curl "http://localhost:8000/health"
curl "http://localhost:8000/debug/retrieval?question=Какие%20параметры%20указаны%20для%20клапана%20DN50"
python evaluation/eval_semantic_retrieval.py
```

Expected healthy Phase 1 retrieval diagnostics:

```json
"retrieval": {
  "retrieval_mode": "hybrid",
  "bm25_ready": true,
  "embedding_dependency_available": true,
  "local_embeddings_enabled": true,
  "local_embeddings_ready": true,
  "hybrid_dense_enabled": true,
  "hybrid_degraded_reason": ""
}
```

If it degrades safely, expect `effective_retrieval_mode:
hybrid_degraded_to_bm25` and a concrete `hybrid_degraded_reason`, such as
`dependency missing`, `model load failed`, `indexing failed`, or
`disabled by config`.

Use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` as the
default CPU-friendly demo model. `BAAI/bge-m3` can be tested only as an
advanced heavier model when RAM/time are sufficient.

### Qdrant dense retrieval

For service-backed dense retrieval, start Qdrant and set:

```powershell
$env:RETRIEVAL_MODE="hybrid"
$env:DIRECT_QDRANT_PROJECTION="true"
$env:QDRANT_HOST="localhost"
$env:QDRANT_PORT="6333"
```

Uploaded chunks are projected to Qdrant best-effort. BM25 remains available as fallback.

### Neo4j strict KG backend

The strict graph QA layer can use Neo4j as the primary backend. Fallback SQLite/extractor remains available.

Backend modes:

```powershell
$env:KG_BACKEND="auto"      # use Neo4j if available, otherwise fallback
$env:KG_BACKEND="neo4j"     # require Neo4j; no silent fallback
$env:KG_BACKEND="fallback"  # force local fallback
```

In `auto` mode the API retries Neo4j from `/health` and `/system/capabilities`.
If a direct `RETURN 1` check succeeds, `kg_backend_active` must become `neo4j`;
startup-time connection refusals must not permanently pin the service to fallback.

Start Neo4j:

```powershell
docker compose --profile full up -d neo4j
```

Check direct connectivity:

```powershell
python scripts\check_neo4j_connection.py
```

Inside the Docker API container use the project path under `/code`:

```powershell
docker compose exec api python scripts/check_neo4j_connection.py
```

Apply schema:

```powershell
python scripts\init_neo4j_schema.py
```

Load documents through API/UI, then materialize the SQLite catalog into Neo4j:

```powershell
python scripts\sync_graph_to_neo4j.py
```

Run Neo4j graph smoke:

```powershell
python scripts\smoke_neo4j_graph.py
```

Check active backend:

```powershell
curl "http://localhost:8000/health"
```

Expected fields:

```json
{
  "kg_backend_configured": "auto",
  "kg_backend_active": "neo4j",
  "neo4j_available": true,
  "neo4j_error": "",
  "neo4j_password_configured": true,
  "kg_backend_decision": {
    "selected": "neo4j"
  }
}
```

Do not copy password values from local `.env`, `docker compose config`, terminal logs
or screenshots into docs/chats. Health diagnostics intentionally expose only
`neo4j_password_configured`.

### Optional grounded LLM synthesis

The LLM layer is not used as an uncontrolled “chat with PDF”. It receives only extracted facts, gaps and source snippets, then returns a grounded answer. Rule-based extraction remains source of truth.

Ollama example:

```powershell
$env:ENABLE_LLM="true"
$env:LLM_PROVIDER="ollama"
$env:LLM_BASE_URL="http://localhost:11434"
$env:LLM_MODEL="qwen2.5:7b-instruct"
uvicorn app.api:app --reload
```

OpenAI-compatible local server example:

```powershell
$env:ENABLE_LLM="true"
$env:LLM_PROVIDER="openai_compatible"
$env:LLM_BASE_URL="http://localhost:8001"
$env:LLM_MODEL="Qwen2.5-7B-Instruct"
```

If the LLM endpoint is unavailable, `/ask` returns `answer_mode: rule_based` and the demo still works.

### Free/API LLM mode before hackathon GPU is available

The API layer also supports OpenAI-compatible cloud providers for temporary demos.
The LLM is used only for query rewriting and grounded answer polishing: extracted
facts and sources remain the source of truth.

Mistral Studio / La Plateforme example:

```powershell
$env:LLM_ENABLED="true"
$env:LLM_PROVIDER="mistral"
$env:MISTRAL_API_KEY="..."
$env:MISTRAL_BASE_URL="https://api.mistral.ai/v1"
$env:MISTRAL_MODEL="mistral-small-latest"
$env:MISTRAL_TIMEOUT_SECONDS="60"
$env:MISTRAL_MAX_TOKENS="1200"
$env:MISTRAL_TEMPERATURE="0.2"
uvicorn app.api:app --reload --port 8000
```

Put the real Mistral key only into a local `.env` or local shell environment.
Do not write it into `.env.example`, README, committed docs, or source code.
If `mistral-small-latest` is unavailable for the account, choose an allowed
model in Mistral Studio / API Limits and set `MISTRAL_MODEL`.

OpenRouter example:

```powershell
$env:LLM_ENABLED="true"
$env:LLM_PROVIDER="openrouter"
$env:LLM_BASE_URL="https://openrouter.ai/api/v1"
$env:LLM_MODEL="<openrouter-model-slug>"
$env:LLM_API_KEY="sk-or-..."
uvicorn app.api:app --reload --port 8000
```

OpenRouter aliases are also supported:

```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
$env:OPENROUTER_MODEL="<openrouter-model-slug>"
$env:OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
uvicorn app.api:app --reload --port 8000
```

If `OPENROUTER_API_KEY` is set but no model is configured, `/health` reports
`OpenRouter API key found, but model is not configured` / `LLM_MODEL is missing`
instead of silently showing `provider=none`.

Provider auto-selection:

- `LLM_PROVIDER=mistral`: use Mistral; if the request fails and OpenRouter is
  configured, answer synthesis may fall back to OpenRouter, then to the
  deterministic template path.
- `LLM_PROVIDER=openrouter`: keep the OpenRouter path.
- `LLM_PROVIDER=auto`: use Mistral when `MISTRAL_API_KEY` exists, otherwise
  OpenRouter when `OPENROUTER_API_KEY` exists, otherwise offline/template.
- `LLM_PROVIDER=offline` or `LLM_ENABLED=false`: use offline/template.

Groq example:

```powershell
$env:ENABLE_LLM="true"
$env:LLM_PROVIDER="groq"
$env:LLM_MODEL="llama-3.3-70b-versatile"
$env:LLM_API_KEY="gsk_..."
uvicorn app.api:app --reload --port 8000
```

When LLM mode is enabled, `/ask` may return `answer_mode: llm_grounded` or
`llm_grounded_negative`. If the provider is unreachable or rate-limited, it
falls back to deterministic answers.

Check LLM configuration:

```powershell
curl http://localhost:8000/health
curl -X POST http://localhost:8000/system/test-llm
```

## Local knowledge graph contract

Without Neo4j, `/ask` returns a typed local graph:

- nodes: `Document`, `Section`, `SourceChunk`, `TechnicalObject`, `Part`, `ArticleNumber`, `Material`, `Standard`, `Parameter`, `Requirement`, `Experiment`, `PropertyValue`, `DataGap`;
- edges: `DOCUMENT_HAS_SECTION`, `SECTION_HAS_CHUNK`, `CHUNK_MENTIONS_ENTITY`, `OBJECT_HAS_PARAMETER`, `OBJECT_HAS_PART`, `PART_HAS_ARTICLE_NUMBER`, `OBJECT_MADE_OF_MATERIAL`, `OBJECT_COMPLIES_WITH_STANDARD`, `STUDIES`, `USES_REGIME`, `MEASURES`, `MISSING_FOR`, `FACT_SUPPORTED_BY_CHUNK`.

Every extracted fact has a source chunk through `source_chunk_id` and/or `FACT_SUPPORTED_BY_CHUNK`.

## Final hardening checks

### UI runtime presets

Streamlit показывает один пользовательский selector `Режим качества`:

- `Лучший ответ` — человекоориентированный grounded answer с caveats и выводом.
- `Строгая проверка` — audit-формат для демонстрации no-hallucination и exact matching.
- `Офлайн-режим` — полностью локальный fallback без внешних сервисов.

Запуск UI:

```powershell
streamlit run app/ui.py
```

Проверка presets:

```powershell
python evaluation\eval_runtime_presets.py
python evaluation\eval_answer_quality.py
```

### `/ask` JSON body

Старый формат остаётся:

```powershell
curl -X POST "http://localhost:8000/ask?question=Что%20уже%20делали%20по%20ВТ6&top_k=8"
```

Новый формат:

```powershell
curl -X POST "http://localhost:8000/ask" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Что уже делали по ВТ6?\",\"top_k\":8,\"preset_id\":\"expert_max\"}"
```

Если query params и JSON body переданы одновременно, JSON body имеет приоритет; это видно в `diagnostics.input_source` и `diagnostics.query_params_ignored`.

### Adversarial extraction

```powershell
python evaluation\eval_adversarial.py
```

Проверяются cases, где нельзя считать `E1`/`EXP-VT6-001` материалами, нельзя привязывать `8 %` к прочности, и gap-фраза `коррозионная стойкость не измерялась` не должна становиться positive measurement.

### Clean release archive

```powershell
python scripts\make_release_archive.py
python scripts\check_release_package.py --path dist\release_unpacked
```

Сдавать нужно `dist\hackathon_project_release.zip`, а не текущую рабочую папку. В архив не должны попадать `.env`, `data/`, `volumes/`, `*.sqlite3`, audit JSONL, `__pycache__`, `*.pyc`, logs.

### Final command sequence

```powershell
python -m pytest -q
python -m pytest -q -W error
python evaluation\eval_demo.py
python evaluation\eval_stress.py
python evaluation\eval_extraction.py
python evaluation\eval_parser.py
python evaluation\eval_analytics.py
python evaluation\eval_cockpit.py
python evaluation\eval_runtime_presets.py
python evaluation\eval_adversarial.py
python scripts\check_project.py
python scripts\make_release_archive.py
python scripts\check_release_package.py --path dist\release_unpacked
ruff check .
docker compose config
```

Optional Neo4j check:

```powershell
python scripts\init_neo4j_schema.py
python scripts\sync_graph_to_neo4j.py
python scripts\smoke_neo4j_graph.py
```

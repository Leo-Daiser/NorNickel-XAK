# CONTEXT_HANDOFF.md

## 1. Project overview

Проект: Python/FastAPI + Streamlit knowledge graph / GraphRAG система для научно-технических документов.

Цель: загружать документы, извлекать структурированные факты об экспериментах, материалах, режимах обработки, свойствах, оборудовании, лабораториях, выводах и пробелах в данных; материализовать knowledge graph в Neo4j; отвечать на строгие и аналитические вопросы без галлюцинаций.

Основной стек:

- Backend API: FastAPI.
- UI: Streamlit.
- Local storage: SQLite catalog/outbox.
- KG backend: Neo4j primary when available, validated fallback graph otherwise.
- Extraction: deterministic / hybrid / llm modes, typed `ExtractionBundle`.
- Ingestion: document intelligence parser with blocks/tables/images/chunks.
- Analytics: GraphRAG analytical router, graph context, evidence search/reranking, human answer synthesis.
- Optional services: Neo4j, Qdrant, OpenRouter-compatible LLM.
- Tests: pytest.
- Release: custom clean archive builder/checker.

Текущий проверенный статус:

- Core tests pass.
- Evaluation scripts pass.
- Docker API/UI rebuilt from current code.
- Docker `/health` verified to select Neo4j when Neo4j is reachable.
- Neo4j schema/sync/smoke verified inside API container.
- Release archive verified clean.

## 2. Current state

### Точно работает

- `/ask` supports JSON body and legacy query params.
- Strict QA exact material + regime + property behavior works.
- `no_exact_match` remains strict: partial matches are not treated as exact facts.
- Neo4j backend works as primary in `KG_BACKEND=auto` when Neo4j is reachable.
- Fallback graph works and is built through validated extraction path.
- Streamlit UI works with:
  - document upload;
  - document active checkbox management via editable table;
  - main question/answer flow;
  - human-readable answers;
  - compact answer graph;
  - details expanders for facts/evidence/history/gaps/diagnostics.
- OpenRouter config resolution works in health diagnostics when env is configured.
- Release archive excludes `.env`, `data/`, `volumes/`, SQLite DBs, pycache, logs and audit runtime artifacts.

### Предположительно работает

- Real OpenRouter completion endpoint should work if a valid model/key is configured, because health reports ready with current env. A real `/system/test-llm` network call was not part of the latest verification.
- Qdrant optional integration remains present, but current verification used BM25/local modes.

### Не проверено в последнем цикле

- Real user interaction through browser with Streamlit after final changes.
- Full LLM answer polish quality with an actual paid/free OpenRouter model response.
- Long-running production-style ingestion with large PDFs/OCR.

## 3. Key decisions

### Neo4j primary, fallback safe

Decision: `KG_BACKEND=auto` should use Neo4j if a direct `RETURN 1` check succeeds; otherwise fallback.

Why:

- Manual check from API container proved Neo4j was reachable.
- App health previously stayed on fallback because `graph_db=None` from startup failure was reused.
- Fix added retry/no permanent failure semantics and canonical connection helper.

Relevant files:

- `app/graph/neo4j_connection.py`
- `app/graph/graph_db.py`
- `app/api.py`
- `scripts/check_neo4j_connection.py`

### Main graph is answer evidence map, not raw DB dump

Decision: primary UI graph must be compact semantic answer graph, not force-directed raw subgraph.

Why:

- Raw graph showed internal labels like `Experiment`, `PropertyValue`, `SourceChunk`, relation labels, internal IDs.
- Users need semantic chain: material -> regime/property -> values/effects -> sources/warning.

Relevant files:

- `app/graph/answer_graph.py`
- `app/ui.py`
- `tests/test_answer_graph_builder.py`
- `tests/test_answer_graph_labels.py`
- `tests/test_answer_graph_rendering.py`
- `tests/test_ui_product_graph_contract.py`

### Human answer is API-level, not UI-only

Decision: answer rewriting happens in `app/answering/human_answer.py` via `enhance_answer_payload`, not in Streamlit.

Why:

- API and UI must be consistent.
- Runtime presets must produce different answer text at `/ask` level.
- Main answer must never leak `technical_answer`, internal IDs, raw effects.

Relevant files:

- `app/answering/human_answer.py`
- `app/api.py`
- `tests/test_human_answer_synthesis.py`
- `tests/test_comparison_answer_quality.py`

### Comparison answers normalize strength units

Decision: support `ksi -> MPa` conversion for strength comparison.

Why:

- Comparison question `Сравни ВТ6 и 7075-T6 по прочности` mixed MPa and ksi.
- User-facing answer must compare ranges in a common unit or explicitly caveat.

Relevant files:

- `app/domain/unit_normalization.py`
- `app/answering/human_answer.py`
- `app/graph/answer_graph.py`
- `tests/test_unit_normalization.py`
- `tests/test_comparison_answer_quality.py`

### Runtime presets stay user-facing

Presets:

- `expert_max` / `Лучший ответ`
- `strict_audit` / `Строгая проверка`
- `offline_reliable` / `Офлайн-режим`

Decision: preserve all three; they must produce visibly different main answers.

Relevant files:

- `app/runtime/presets.py`
- `evaluation/eval_runtime_presets.py`
- `tests/test_preset_answer_differences.py`

## 4. Known problems

### No currently known failing tests

Last verified:

- `python -m pytest -q`: passed.
- `python -m pytest -q -W error`: passed.

### Docker compose config can print local secrets

`docker compose config` reads local `.env` and may print sensitive values in terminal output. Do not copy those values into docs/chats. Release archive excludes `.env`.

### Host Neo4j script needs explicit password if env is missing

Observed:

```text
Neo4j password is missing. Set NEO4J_PASSWORD.
```

This is expected and now explicit. Set:

```powershell
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_PASSWORD="<neo4j-password>"
```

For Docker API container, env was present and check passed.

### First Docker rebuild timed out once

Initial `docker compose up -d --build api ui` timed out while downloading dependencies. Separate `docker compose build api ui` later completed, then `docker compose up -d api ui` recreated containers successfully.

### Real LLM call not recently verified

Health showed OpenRouter ready in container:

```json
{
  "enabled": true,
  "provider": "openrouter",
  "base_url": "https://openrouter.ai/api/v1",
  "model": "<configured-model>",
  "api_key_configured": true,
  "ready": true,
  "last_error": ""
}
```

But `/system/test-llm` with a real external model was not run in the final cycle.

## 5. Recent debugging context

### Problem debugged most recently

Symptoms:

- Manual Neo4j check inside API container worked:
  - URI: `bolt://neo4j:7687`
  - user: `neo4j`
  - password set
  - `RETURN 1` succeeded
- `/health` still showed:

```json
{
  "kg_backend_configured": "auto",
  "kg_backend_active": "fallback",
  "neo4j_available": false,
  "neo4j_error": "Connection refused..."
}
```

Root cause:

- API initialized `graph_db=None` when Neo4j was unavailable/refused during startup.
- For `KG_BACKEND=auto`, `_graph_db_for_repository()` returned stale `graph_db` without retry.
- Health and repository selection were therefore stuck in fallback after Neo4j became available.

Fix:

- Added `app/graph/neo4j_connection.py`.
- `GraphDB` uses canonical driver creation.
- `_graph_db_for_repository(..., force_retry=True)` used by `/health` and `/system/capabilities`.
- Short failure TTL prevents expensive repeated retries on every request while still allowing recovery.
- Health diagnostics now include:
  - `neo4j_uri`
  - `neo4j_user`
  - `neo4j_password_configured`
  - `kg_backend_decision`

### Verified command results from final cycle

Tests:

```powershell
python -m pytest -q
# 198 passed

python -m pytest -q -W error
# 198 passed
```

Evaluations:

```powershell
python evaluation/eval_demo.py
# 7 passed / 0 failed

python evaluation/eval_stress.py
# 5 passed / 0 failed

python evaluation/eval_extraction.py
# PASS

python evaluation/eval_parser.py
# PASS

python evaluation/eval_analytics.py
# PASS

python evaluation/eval_cockpit.py
# PASS

python evaluation/eval_runtime_presets.py
# PASS

python evaluation/eval_adversarial.py
# PASS

python evaluation/eval_answer_quality.py
# PASS

python evaluation/eval_ui_product.py
# PASS
```

Project/release/lint:

```powershell
python scripts/check_project.py
# SMOKE TEST PASSED

ruff check .
# All checks passed

python scripts/make_release_archive.py
# RELEASE ARCHIVE CREATED
# forbidden_files_count: 0

python scripts/check_release_package.py --path dist/release_unpacked
# RELEASE PACKAGE CHECK PASSED

docker compose config
# passed
```

Neo4j host check with explicit env:

```powershell
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_PASSWORD="<neo4j-password>"
python scripts/check_neo4j_connection.py
# available: true
# reason: Neo4j connection check succeeded with RETURN 1
```

Docker API container checks:

```powershell
docker compose build api ui
# built

docker compose up -d api ui
# api/ui recreated

docker compose exec -T api python hackathon_project/scripts/check_neo4j_connection.py
# available: true

docker compose exec -T api python hackathon_project/scripts/init_neo4j_schema.py
# statements: 17, applied: 17, errors: []

docker compose exec -T api python hackathon_project/scripts/sync_graph_to_neo4j.py
# accepted_experiments: 35
# accepted_measurements: 56
# accepted_gaps: 4
# relationships_written: 938

docker compose exec -T api python hackathon_project/scripts/smoke_neo4j_graph.py
# NEO4J GRAPH SMOKE TEST PASSED
```

Docker health check from inside API container:

```json
{
  "kg_backend_configured": "auto",
  "kg_backend_active": "neo4j",
  "neo4j_available": true,
  "neo4j_uri": "bolt://neo4j:7687",
  "neo4j_user": "neo4j",
  "neo4j_password_configured": true,
  "neo4j_error": "",
  "kg_backend_decision": {
    "configured": "auto",
    "selected": "neo4j",
    "reason": "Neo4j connection check succeeded with RETURN 1"
  }
}
```

No password value is included above.

## 6. Important files and directories

### API/UI

- `app/api.py`
  - FastAPI app.
  - `/ask`, `/health`, `/system/capabilities`, document endpoints, graph refresh, ingest endpoints.
  - Backend selection and runtime preset application.

- `app/ui.py`
  - Streamlit product UI.
  - Document upload/control.
  - Main question/answer view.
  - Compact answer graph rendering.

- `app/ui_helpers.py`
  - UI formatting helpers.
  - Document rows, evidence/fact rows, technical graph HTML helper.

### Configuration/runtime

- `app/config.py`
  - Environment-driven settings.
  - Neo4j, Qdrant, extraction, parser, LLM, upload/security config.

- `app/runtime/presets.py`
  - Three runtime presets.
  - Preset diagnostics and effective runtime mode.

### Graph

- `app/graph/neo4j_connection.py`
  - Canonical Neo4j driver/check helper.

- `app/graph/graph_db.py`
  - Neo4j wrapper.
  - Uses `create_neo4j_driver`.

- `app/graph/graph_repository.py`
  - `GraphRepository` protocol.
  - `CatalogGraphRepository` fallback.
  - `GraphRepositoryFactory`.

- `app/graph/neo4j_repository.py`
  - Strict graph repository over Neo4j.

- `app/graph/graph_writer.py`
  - Neo4j materialization writer/sync.

- `app/graph/schema.cypher`
  - Neo4j constraints/indexes.

- `app/graph/answer_graph.py`
  - Compact user-facing answer evidence map.

### Answering/analytics

- `app/answering/human_answer.py`
  - Human-readable answer synthesis.
  - Fact ranking.
  - Evidence population.
  - Preset-specific answer style.
  - Comparison answer quality.

- `app/answering/answer_builder.py`
  - Strict graph QA answer builder.

- `app/analytics/*`
  - Analytical intents, router, graph context, evidence search/rerank, answer synthesizer.

### Domain/extraction/ingestion

- `app/domain/query_constraints.py`
  - Query constraints model.

- `app/domain/unit_normalization.py`
  - `ksi -> MPa` conversion for comparison answers.

- `app/extraction/*`
  - Extraction pipeline, typed extraction models, deterministic/table/LLM extractors, validators, confidence, audit.

- `app/ingestion/*`
  - Document intelligence parser, structured chunks, scanned PDF detection, parser audit.

### Storage/security

- `app/storage/catalog.py`
  - SQLite catalog.
  - Document active flag.

- `app/security/url_safety.py`
  - SSRF protection and safe URL ingestion.

### Scripts

- `scripts/check_project.py`
  - Compile/smoke check.

- `scripts/check_release_package.py`
  - Verifies release dir/archive has no forbidden runtime files.

- `scripts/make_release_archive.py`
  - Builds clean `dist/hackathon_project_release.zip`.

- `scripts/check_neo4j_connection.py`
  - Direct Neo4j `RETURN 1` check through canonical helper.

- `scripts/init_neo4j_schema.py`
  - Applies schema.

- `scripts/sync_graph_to_neo4j.py`
  - Materializes accepted extraction facts to Neo4j.

- `scripts/smoke_neo4j_graph.py`
  - Neo4j exact/no-match smoke test.

### Data/release directories

- `demo_data/`
  - Demo documents/tables used by tests/evals.

- `data/`
  - Runtime local DB/audit data. Do not commit/release.

- `volumes/`
  - Docker runtime volumes. Do not commit/release.

- `dist/`
  - Generated release archive/unpacked release. Do not treat as source.

## 7. Runtime/config context

### Docker

Primary compose file:

- `docker-compose.yml`

Services:

- `api`
- `ui`
- `neo4j`
- `qdrant`

Useful Docker commands:

```powershell
docker compose ps
docker compose build api ui
docker compose up -d api ui
docker compose config
```

Run Neo4j checks inside API container:

```powershell
docker compose exec -T api python hackathon_project/scripts/check_neo4j_connection.py
docker compose exec -T api python hackathon_project/scripts/init_neo4j_schema.py
docker compose exec -T api python hackathon_project/scripts/sync_graph_to_neo4j.py
docker compose exec -T api python hackathon_project/scripts/smoke_neo4j_graph.py
```

### Neo4j

Important env vars:

```env
KG_BACKEND=auto
NEO4J_URI=bolt://neo4j:7687       # inside Docker network
NEO4J_URI=bolt://localhost:7687   # from host
NEO4J_USER=neo4j
NEO4J_PASSWORD=<neo4j-password>
NEO4J_DATABASE=neo4j
```

Do not expose real password.

Expected `/health` when Neo4j is available:

```json
{
  "kg_backend_configured": "auto",
  "kg_backend_active": "neo4j",
  "neo4j_available": true,
  "neo4j_error": "",
  "kg_backend_decision": {
    "selected": "neo4j"
  }
}
```

### Qdrant

Qdrant is optional. Current tests/evals are configured to avoid requiring dense retrieval:

```env
DIRECT_QDRANT_PROJECTION=false
ENABLE_LOCAL_EMBEDDINGS=false
RETRIEVAL_MODE=bm25
```

### OpenRouter / LLM

Supported env:

```env
LLM_ENABLED=true
LLM_PROVIDER=openrouter
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=<openrouter-api-key>
LLM_MODEL=<openrouter-model-slug>
```

Aliases supported:

```env
OPENROUTER_API_KEY=<openrouter-api-key>
OPENROUTER_MODEL=<openrouter-model-slug>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Do not commit real keys.

Health should not show `provider=none` if OpenRouter key/model are configured.

### UI

Main UI:

```powershell
streamlit run app/ui.py
```

Docker UI:

```text
http://localhost:8501
```

API:

```text
http://localhost:8000
```

Key endpoints:

- `GET /health`
- `GET /system/capabilities`
- `POST /system/test-llm`
- `POST /ask`
- `GET /documents`
- `PATCH /documents/{doc_id}/active`
- `POST /ingest/documents`
- `POST /graph/refresh`
- `GET /runtime/presets`
- `POST /runtime/validate-preset`
- `POST /runtime/run-preset-check`

`/ask` JSON body:

```json
{
  "question": "Сравни ВТ6 и 7075-T6 по прочности.",
  "top_k": 12,
  "preset_id": "expert_max"
}
```

## 8. Tests/evaluations

### Main test command

```powershell
python -m pytest -q
python -m pytest -q -W error
```

Latest verified result:

```text
198 passed
```

### Evaluation commands

```powershell
python evaluation/eval_demo.py
python evaluation/eval_stress.py
python evaluation/eval_extraction.py
python evaluation/eval_parser.py
python evaluation/eval_analytics.py
python evaluation/eval_cockpit.py
python evaluation/eval_runtime_presets.py
python evaluation/eval_adversarial.py
python evaluation/eval_answer_quality.py
python evaluation/eval_ui_product.py
```

Latest verified status: all PASS.

### Important regression tests

Neo4j:

- `tests/test_neo4j_connection_helper.py`
- `tests/test_neo4j_backend_activation.py`
- `tests/test_kg_backend_selection.py`
- `tests/test_neo4j_repository_queries.py`

Comparison/units:

- `tests/test_unit_normalization.py`
- `tests/test_comparison_answer_quality.py`
- `tests/test_answer_synthesizer.py`
- `tests/test_human_answer_synthesis.py`

Answer graph/UI:

- `tests/test_answer_graph_builder.py`
- `tests/test_answer_graph_labels.py`
- `tests/test_answer_graph_rendering.py`
- `tests/test_ui_product_graph_contract.py`
- `tests/test_graph_label_cleanup.py`
- `tests/test_interactive_graph_rendering.py`

Runtime presets:

- `tests/test_runtime_presets.py`
- `tests/test_runtime_preset_evaluation.py`
- `tests/test_preset_answer_differences.py`

Ingestion/document management:

- `tests/test_document_management_api.py`
- `tests/test_document_activation.py`
- `tests/test_ui_document_management.py`
- `tests/test_upload_and_refresh_flow.py`
- `tests/test_ingestion_security.py`

Release:

- `tests/test_make_release_archive.py`

### Release commands

```powershell
python scripts/make_release_archive.py
python scripts/check_release_package.py --path dist/release_unpacked
```

Expected:

```text
forbidden_files_count: 0
RELEASE PACKAGE CHECK PASSED
```

## 9. Do not break

These are hard invariants:

- Do not expose `.env`, API keys, Neo4j password, OpenRouter key, SQLite DBs, volumes, logs, audit JSONL in release.
- Do not remove fallback graph.
- Do not make Neo4j mandatory for normal tests.
- Do not make LLM mandatory.
- Do not make Qdrant mandatory.
- Do not make OCR mandatory.
- Do not reintroduce raw `technical_answer` as main `answer`.
- Do not show raw `increase`, `decrease`, `unknown` in main user answer.
- Do not show internal IDs in main answer:
  - `doc_`
  - `chunk_`
  - `EXP-`
  - `SCI-`
  - `Experiment doc_...`
- Do not make partial matches look like exact facts.
- Do not return positive strict answer without exact graph path.
- Do not replace compact answer graph with raw force-directed subgraph in main UI.
- Do not put nested Streamlit expanders inside document controls.
- Do not restore separate document toggle selectbox/button; active flag is edited via checkbox table.
- Do not break `/ask` JSON body.
- Do not break `/documents` active flag semantics.
- Do not let `KG_BACKEND=auto` stay fallback if direct Neo4j check succeeds.

## 10. Next actions

Priority 1:

1. Verify actual Streamlit UI manually in browser after final Docker rebuild.
2. Ask:
   - `Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?`
   - `Сравни ВТ6 и 7075-T6 по прочности.`
3. Confirm:
   - main answer is readable;
   - compact graph is visible;
   - technical graph is hidden in expander;
   - no raw IDs/effects in main answer.

Priority 2:

1. If valid OpenRouter key/model are available, run:

```powershell
curl -X POST http://localhost:8000/system/test-llm
```

2. Confirm success/latency/error behavior.

Priority 3:

1. Test upload of a new document through UI.
2. Click graph refresh.
3. Toggle active checkbox off/on.
4. Verify inactive document is excluded from answers.

Priority 4:

1. Review docs after all code changes:
   - `README.md`
   - `docs/RUNBOOK.md`
   - `docs/ARCHITECTURE.md`
2. Ensure docs match current UI/runtime behavior.

Priority 5:

1. Consider making Docker build faster by improving caching or pinning wheels if needed.
2. Do not change runtime behavior unless tests/evals are updated.

## 11. Useful prompts/commands

### Ask API with JSON body

```powershell
$body = @{
  question = "Сравни ВТ6 и 7075-T6 по прочности."
  top_k = 12
  preset_id = "expert_max"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/ask" -ContentType "application/json" -Body $body
```

### Check health

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected when Docker Neo4j is up:

```text
kg_backend_active = neo4j
neo4j_available = true
neo4j_error = ""
```

### Check capabilities

```powershell
Invoke-RestMethod http://localhost:8000/system/capabilities
```

### Neo4j host checks

```powershell
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_PASSWORD="<neo4j-password>"
python scripts/check_neo4j_connection.py
python scripts/init_neo4j_schema.py
python scripts/sync_graph_to_neo4j.py
python scripts/smoke_neo4j_graph.py
```

### Neo4j Docker checks

```powershell
docker compose exec -T api python hackathon_project/scripts/check_neo4j_connection.py
docker compose exec -T api python hackathon_project/scripts/init_neo4j_schema.py
docker compose exec -T api python hackathon_project/scripts/sync_graph_to_neo4j.py
docker compose exec -T api python hackathon_project/scripts/smoke_neo4j_graph.py
```

### Full verification set

```powershell
python -m pytest -q
python -m pytest -q -W error
python evaluation/eval_demo.py
python evaluation/eval_stress.py
python evaluation/eval_extraction.py
python evaluation/eval_parser.py
python evaluation/eval_analytics.py
python evaluation/eval_cockpit.py
python evaluation/eval_runtime_presets.py
python evaluation/eval_adversarial.py
python evaluation/eval_answer_quality.py
python evaluation/eval_ui_product.py
python scripts/check_project.py
python scripts/make_release_archive.py
python scripts/check_release_package.py --path dist/release_unpacked
ruff check .
docker compose config
```

### Suggested prompt for next agent

```text
You are working in an existing Python/FastAPI/Streamlit knowledge graph project.
Read CONTEXT_HANDOFF.md first. Preserve all "Do not break" invariants.
Before changes, run targeted tests around the area you touch.
Do not expose secrets. Do not rewrite architecture.
```

## 12. Open questions

1. Does `/system/test-llm` succeed with the currently configured real OpenRouter model/key?
   - Health reports ready, but final cycle did not run the real external request.

2. Is the current compact answer graph visually acceptable in browser at different screen sizes?
   - Tests verify HTML contract and labels, but browser visual inspection was not repeated after the final Neo4j/backend patch.

3. Should docs be updated to mention the latest Neo4j retry/health behavior?
   - Code is updated; README/RUNBOOK may need final wording review.

4. Should Docker build time be optimized?
   - Build succeeded, but initial dependency download was slow.

5. Should Qdrant be tested in a full semantic retrieval mode?
   - Current verified path relies on BM25/local validated graph and Neo4j.

6. Should large/scanned PDF ingestion be tested with OCR enabled?
   - OCR remains optional and disabled by default.


# Resource Efficiency Strategy

## Цель

Проект спроектирован small-model-first: извлечение фактов, нормализация, дедупликация, построение графа и базовые ответы работают без LLM. Нейросеть используется только как опциональный слой формулировки и проходит grounding guard. Это позволяет сохранять качество при низких ресурсах и отключать тяжелые компоненты без потери основной функциональности.

## Runtime Profiles

`RUNTIME_PROFILE` задает defaults. Явные env-переменные пользователя имеют приоритет.

| Profile | Retrieval | Embeddings | LLM | Назначение |
| --- | --- | --- | --- | --- |
| `economy_core` | BM25 | off | off/offline | Минимальные ресурсы: deterministic extraction, graph facts, template answer. |
| `economy_guarded_llm` | BM25 | off | guarded polish only | Дешевый режим с более читаемым ответом. LLM не извлекает факты. |
| `balanced_hybrid` | BM25 + local dense | MiniLM, lazy | off by default | Семантический retrieval при умеренных CPU/RAM. |
| `quality_full` | BM25 + local dense | MiniLM, lazy | guarded polish | Текущий полный demo path. Qdrant остается optional, не default. |

## Presentation Summary

| Mode | LLM | Embeddings | Image size | Demo regression | Use case |
| --- | --- | --- | --- | --- | --- |
| `economy_core` | no | no | ~0.20 GB measured locally (`economy-check`) | 7/7 | Minimal deployment and resource-efficiency proof. |
| `balanced_hybrid` | optional guarded | MiniLM 384d | current API ~2.888 GB measured locally | 7/7 | Better semantic retrieval with moderate CPU/RAM. |
| `quality_full` | guarded | hybrid | depends on optional dependencies | 7/7 | Maximum demo quality with guard-protected LLM polish. |

## Почему это resource-efficient

- Extraction deterministic: факты извлекаются regex/pattern/table logic, не через LLM.
- Graph reasoning deterministic: Neo4j/fallback graph хранит structured facts и provenance.
- Canonical fact layer снижает дубли и уменьшает шум в ответе.
- LLM не является source of truth: используется только для polish и блокируется grounding guard при неподтвержденных claims.
- Embeddings optional: `economy_core` не требует `sentence-transformers`, torch и GPU.
- BM25 baseline работает без dense model.
- Local dense модель по умолчанию: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, CPU-friendly, 384 dimensions.
- Local embeddings строятся lazy: `EAGER_LOCAL_EMBEDDINGS=false`.
- Qdrant projection не включен по умолчанию: `DIRECT_QDRANT_PROJECTION=false`.

## Как включить режимы

Рекомендуемый способ переключения локального `.env`:

```bash
python scripts/switch_runtime_profile.py economy --write --backup
python scripts/switch_runtime_profile.py balanced --write --backup
python scripts/switch_runtime_profile.py quality --write --backup
```

Без `--write` скрипт только показывает sanitized diff и не меняет `.env`.

Минимальный режим:

```bash
RUNTIME_PROFILE=economy_core
RETRIEVAL_MODE=bm25
ENABLE_LOCAL_EMBEDDINGS=false
ENABLE_LLM=false
LLM_PROVIDER=offline
ANSWER_SYNTHESIS_MODE=template
```

Дешевый режим с guarded LLM polish:

```bash
RUNTIME_PROFILE=economy_guarded_llm
RETRIEVAL_MODE=bm25
ENABLE_LOCAL_EMBEDDINGS=false
ENABLE_LLM=true
LLM_PROVIDER=mistral
MISTRAL_MODEL=mistral-small-latest
```

Сбалансированный hybrid retrieval:

```bash
RUNTIME_PROFILE=balanced_hybrid
EXTRA_REQUIREMENTS=requirements-embeddings.txt
RETRIEVAL_MODE=hybrid
ENABLE_LOCAL_EMBEDDINGS=true
EAGER_LOCAL_EMBEDDINGS=false
DIRECT_QDRANT_PROJECTION=false
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## Проверки

```bash
python scripts/resource_efficiency_report.py
python evaluation/eval_resource_ablation.py
python evaluation/eval_resource_ablation.py --target docker
python evaluation/eval_demo_regression.py
python scripts/demo_gate.py
```

Docker rebuild для economy image не должен ставить `sentence-transformers`:

```bash
docker compose down
docker compose --profile full build --no-cache api
docker compose --profile full up -d
```

Docker rebuild для hybrid image:

```bash
EXTRA_REQUIREMENTS=requirements-embeddings.txt docker compose --profile full build --no-cache api
```

На Windows PowerShell:

```powershell
$env:EXTRA_REQUIREMENTS="requirements-embeddings.txt"
docker compose --profile full build --no-cache api
```

## Как читать предупреждения

- `Hybrid degraded to BM25` не является fatal, если BM25 работает и ответы проходят eval.
- `LLM disabled` нормально для `economy_core`.
- Docker image `> RESOURCE_MAX_IMAGE_GB` дает warning по умолчанию и fail только при `RESOURCE_STRICT=true`.
- `LLM extraction enabled` для этого проекта считается плохим ресурсным режимом; source of truth должен оставаться deterministic extraction + graph.
- `Profile economy_core is overridden by explicit env settings` означает, что профиль выбран как economy, но конкретные env-переменные включили hybrid/embeddings/LLM. Для строгой демонстрации переключите `.env` через `scripts/switch_runtime_profile.py economy --write --backup`.
- `balanced_hybrid requested but dense retrieval is disabled/degraded` означает, что hybrid-профиль выбран, но dense слой не поднялся или недоступна dependency/model.

## Рекомендуемый режим для демонстрации resource efficiency

Для организаторов, которые оценивают экономию ресурсов, сначала показывать `economy_core`: система отвечает по графу без LLM и embeddings. Затем запускать `eval_resource_ablation.py`, чтобы показать, какой прирост дают guarded LLM polish и `balanced_hybrid`, и что базовый режим не разваливается без тяжелых моделей.

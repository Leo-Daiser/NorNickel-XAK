# Default-Xak — R&D Knowledge Graph

Проект для хакатона: система для загрузки научно-технических документов, извлечения проверяемых фактов, построения knowledge graph и ответа на исследовательские вопросы.

Система рассчитана на задачи R&D-карты знаний в горно-металлургической и материаловедческой тематике: материалы, процессы, режимы, свойства, числовые параметры, оборудование, источники, лаборатории, эксперты, выводы, противоречия и пробелы в данных.

## Что умеет проект

- Загружает PDF, DOCX, PPTX, XLSX, CSV, HTML, TXT, MD.
- Извлекает сущности и факты из текстов и таблиц.
- Нормализует материалы, процессы, свойства, единицы измерения и числовые значения.
- Показывает источники и evidence для фактов.
- Находит противоречия и пробелы в данных.
- Отвечает на вопросы через API `/ask`.
- Показывает ответ, источники, диагностику и граф в Streamlit UI.
- Работает в экономичном режиме без LLM и без embeddings.
- Может использовать Neo4j как графовую БД.

Базовый режим для запуска у новичка — `economy_core`: без внешних API-ключей, без LLM, без dense embeddings. Этого достаточно, чтобы поднять проект, загрузить документы и проверить UI.

---

## Быстрый запуск через Docker на Windows

### 1. Установить зависимости

Нужно установить:

- Git;
- Docker Desktop.

После установки Docker Desktop должен быть запущен.

Проверь в PowerShell:

```powershell
docker --version
docker compose version
```

Если команды не работают, перезапусти Docker Desktop или компьютер.

---

### 2. Скачать проект

```powershell
git clone https://github.com/Leo-Daiser/Default-Xak.git
cd Default-Xak
```

---

### 3. Создать локальный `.env`

В репозиторий нельзя коммитить реальные ключи и локальные базы. Поэтому `.env` создается у каждого локально.

Для первого запуска скопируй экономичный профиль:

```powershell
Copy-Item .env.economy.example .env
```

Этот режим:

- не требует API-ключей;
- не использует LLM;
- не использует embeddings;
- работает через BM25 + deterministic extraction + graph/fallback logic.

Проверь, что файл появился:

```powershell
dir .env
```

---

### 4. Собрать и запустить контейнеры

```powershell
docker compose --profile full up -d --build
```

Первый запуск может занять несколько минут.

Проверить контейнеры:

```powershell
docker compose ps
```

Ожидаемо должны быть контейнеры:

```text
api
ui
neo4j
qdrant
```

Главные адреса:

```text
Streamlit UI:  http://localhost:8501
API docs:      http://localhost:8000/docs
API health:    http://localhost:8000/health
Neo4j Browser: http://localhost:7474
```

Проверка API из PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Открыть UI:

```powershell
Start-Process http://localhost:8501
```

---

## Как пользоваться UI

1. Открой `http://localhost:8501`.
2. В блоке загрузки документов выбери файлы.
3. Для быстрого теста можно взять файлы из папки проекта `evaluation/test_corpus`.
4. Нажми загрузку документов.
5. После загрузки нажми обновление графа/индексов в UI.
6. Задай вопрос в поле вопроса.
7. Посмотри ответ, источники, факты, диагностику и граф.

Примеры вопросов:

```text
Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?
Сравни ВТ6 и 7075-T6 по прочности.
Какие есть противоречия или неоднородные данные по прочности?
Какие пробелы в данных найдены?
Найди evidence по прочности 7075-T6 после aging.
```

Для полного ТЗ также проверяются вопросы вида:

```text
Какие методы обессоливания воды подходят при сульфатах и хлоридах 200–300 мг/л?
Какие решения циркуляции католита при электроэкстракции никеля описаны в мировой практике?
Покажи эксперименты по распределению Au, Ag и МПГ между штейном и шлаком за последние 5 лет.
Какие способы закачки шахтных вод применялись в России и за рубежом?
```

---

## Важные команды Docker

Остановить проект:

```powershell
docker compose down
```

Перезапустить без пересборки:

```powershell
docker compose --profile full up -d
```

Пересобрать полностью:

```powershell
docker compose --profile full up -d --build
```

Посмотреть логи API:

```powershell
docker compose logs api --tail=100
```

Посмотреть логи UI:

```powershell
docker compose logs ui --tail=100
```

Посмотреть логи Neo4j:

```powershell
docker compose logs neo4j --tail=100
```

Полностью сбросить контейнеры и локальные volumes:

```powershell
docker compose down -v
docker compose --profile full up -d --build
```

---

## Проверки после запуска

Проверить API:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Проверить UI:

```powershell
Invoke-WebRequest http://localhost:8501/_stcore/health
```

Ожидаемый ответ UI health:

```text
ok
```

Проверить ключевые eval внутри контейнера API:

```powershell
docker compose exec api python evaluation/eval_ui_product.py
docker compose exec api python evaluation/eval_grounding_guard.py
docker compose exec api python evaluation/eval_answer_quality.py
docker compose exec api python evaluation/eval_tz_query_readiness.py
```

Запустить быстрые UI/API тесты:

```powershell
docker compose exec api python -m pytest -q tests/test_streamlit_preset_mapping.py tests/test_streamlit_no_nested_expanders.py tests/test_upload_and_refresh_flow.py tests/test_document_management_api.py tests/test_api_ask.py
```

---

## Если что-то не работает

### Docker пишет, что `.env` не найден

Создай `.env`:

```powershell
Copy-Item .env.economy.example .env
```

Потом снова запусти:

```powershell
docker compose --profile full up -d --build
```

### UI не открывается

```powershell
docker compose ps
docker compose logs ui --tail=100
```

Проверь, что API доступен из UI-контейнера:

```powershell
docker compose exec ui python -c "import urllib.request; print(urllib.request.urlopen('http://api:8000/health').read().decode()[:500])"
```

### API не открывается

```powershell
docker compose ps
docker compose logs api --tail=100
```

Проверь health:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

### Neo4j не подключается

Для первого запуска это не критично: проект умеет работать через fallback.

Проверить подключение из API-контейнера:

```powershell
docker compose exec api python scripts/check_neo4j_connection.py
```

Если Neo4j нужен явно, проверь `.env`:

```env
NEO4J_DOCKER_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

### Загрузка файлов идет долго

Это нормально для больших PDF, DOCX и XLSX. UI загружает файлы по одному, чтобы один тяжелый или битый файл не ронял весь batch.

Если файл является сканом без текстового слоя, ожидаемый статус — `ocr_required`. Это не crash, а честная диагностика: нужен OCR-профиль.

---

## Режимы работы

### economy_core

Режим по умолчанию для первого запуска.

```text
без LLM
без embeddings
BM25 retrieval
deterministic extraction
template answer
```

Используется файл:

```text
.env.economy.example
```

### balanced_hybrid

Локальный hybrid retrieval с MiniLM embeddings.

```powershell
Copy-Item .env.balanced.example .env
docker compose --profile full up -d --build
```

Если `sentence-transformers` или модель недоступны, система не падает: retrieval деградирует в BM25, а причина отображается в `/health`.

### quality_full

Режим с hybrid retrieval и LLM polish. Требует API-ключ Mistral или другого совместимого provider.

```powershell
Copy-Item .env.quality.example .env
notepad .env
```

В `.env` нужно вручную добавить ключ. Реальные ключи нельзя коммитить.

---

## Как подключить Mistral опционально

Для первого запуска это не нужно.

Если нужен LLM polish:

```env
RUNTIME_PROFILE=economy_guarded_llm
ENABLE_LLM=true
LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
MISTRAL_BASE_URL=https://api.mistral.ai/v1
MISTRAL_MODEL=mistral-small-latest
```

Перезапуск:

```powershell
docker compose down
docker compose --profile full up -d --build
```

Проверка:

```powershell
docker compose exec api python scripts/check_mistral_connection.py
Invoke-RestMethod http://localhost:8000/health
```

Важно: LLM не является источником истины. В базовой архитектуре факты извлекаются и проверяются отдельно, а LLM используется только как опциональный слой формулировки ответа.

---

## Структура проекта

```text
app/          основной код API, UI, retrieval, extraction, graph, answering
docs/         архитектура, runbook, отчеты по готовности
evaluation/   eval-скрипты и тестовый корпус
tests/        pytest-тесты
scripts/      служебные скрипты
requirements*.txt  зависимости для разных режимов
```

---

## Что нельзя коммитить

Не добавлять в репозиторий локальные runtime-файлы:

```text
.env
data/
data_storage/
volumes/
dist/
artifacts/
__pycache__/
*.pyc
*.sqlite3
*.log
*.zip
```

Перед коммитом проверяй:

```powershell
git status --short
git diff --cached --name-only
```

Если случайно добавил тяжелую папку:

```powershell
git restore --staged data_storage/
git restore --staged data/
git restore --staged artifacts/
```

---

## Краткая архитектура

```text
Документы
  → parsing / chunking
  → deterministic extraction
  → canonical facts + evidence
  → conflict / gap detection
  → local catalog + optional Neo4j graph
  → retrieval / graph query
  → guarded answer + sources + graph UI
```

Главный принцип: LLM не пишет финальные факты напрямую в граф. Факты должны иметь источник, evidence и проходить проверку.

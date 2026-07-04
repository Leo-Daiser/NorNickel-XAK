# NorNickel-XAK — R&D Knowledge Graph

Система для хакатона: загрузка научно-технических документов, извлечение проверяемых фактов, построение knowledge graph и ответы на исследовательские вопросы с источниками/evidence.

Проект рассчитан на R&D-задачи в горно-металлургической и материаловедческой тематике: материалы, процессы, режимы, свойства, числовые параметры, оборудование, источники, выводы, противоречия и пробелы в данных.

## Возможности

- Streamlit UI для загрузки документов и вопросов.
- FastAPI backend с endpoint `/ask`.
- Поддержка PDF, DOCX, PPTX, XLSX, CSV, HTML, TXT, MD.
- Parsing/chunking документов и локальный каталог.
- Детерминированное извлечение фактов.
- Нормализация материалов, процессов, свойств, единиц и числовых значений.
- Evidence/sources для найденных фактов.
- Поиск противоречий и пробелов.
- Опциональный Neo4j graph backend.
- Опциональный Qdrant/vector backend.
- Базовый режим без внешних LLM API-ключей.

Базовый режим для проверки — `economy_core`: BM25 + deterministic extraction + fallback/graph logic, без внешнего LLM и без локальной embedding-модели.

---

## Быстрый запуск локально через Docker

### 1. Требования

Нужно установить:

- Git;
- Docker Desktop.

Проверка:

```powershell
docker --version
docker compose version
```

### 2. Скачать проект

```powershell
git clone https://github.com/Leo-Daiser/NorNickel-XAK.git
cd NorNickel-XAK
```

### 3. Создать `.env`

```powershell
Copy-Item .env.example .env
```

Для первого запуска реальные API-ключи не нужны. Базовый запуск работает локально.

### 4. Запустить сервисы

```powershell
docker compose --profile full up -d --build
```

Первый запуск может занять несколько минут.

Проверить контейнеры:

```powershell
docker compose ps
```

Ожидаемые сервисы:

```text
api
ui
neo4j
qdrant
```

Адреса:

```text
Streamlit UI:  http://localhost:8501
API docs:      http://localhost:8000/docs
API health:    http://localhost:8000/health
Neo4j Browser: http://localhost:7474
Qdrant API:    http://localhost:6333
```

Neo4j Browser:

```text
login:    neo4j
password: hackathon_password
```

### 5. Проверить запуск

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-WebRequest http://localhost:8501/_stcore/health
Start-Process http://localhost:8501
```

---

## Данные

Исходный датасет хакатона **не включен** в репозиторий.

Для локального запуска с полным корпусом нужно распаковать выданный датасет в папку:

```text
data_storage/
```

Ожидаемая структура:

```text
data_storage/
  Доклады/
  Журналы/
  Материалы конференций/
  Обзоры/
  Статьи/
```

Папка `data_storage/` намеренно исключена из Git, потому что содержит исходные данные хакатона.

Для быстрого теста без приватного датасета можно использовать тестовые документы из:

```text
evaluation/test_corpus/
```

---

## Как загрузить документы

### Вариант A — через UI

1. Открыть `http://localhost:8501`.
2. Загрузить документы через блок загрузки файлов.
3. После загрузки обновить граф/индексы в UI.
4. Задать вопрос.
5. Проверить ответ, источники, evidence, диагностику и граф.

### Вариант B — загрузить часть реального корпуса скриптом

После запуска Docker можно загрузить выборку из `data_storage/` через API:

```powershell
python -m pip install requests
python scripts/stage_real_corpus_demo.py --input data_storage --count 25 --reset --sync-neo4j
```

Dry-run без загрузки:

```powershell
python scripts/stage_real_corpus_demo.py --input data_storage --count 25 --dry-run
```

Скрипт загружает файлы по одному, обновляет граф после загрузки и сохраняет отчет в `artifacts/`.

---

## Примеры вопросов

Для тестового корпуса:

```text
Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?
Сравни ВТ6 и 7075-T6 по прочности.
Какие есть противоречия или неоднородные данные по прочности?
Какие пробелы в данных найдены?
Найди evidence по прочности 7075-T6 после aging.
```

Для полного корпуса хакатона:

```text
Какие материалы, процессы и свойства встречаются в загруженных источниках?
Какие методы обессоливания воды подходят при сульфатах и хлоридах 200–300 мг/л?
Какие решения циркуляции католита при электроэкстракции никеля описаны в источниках?
Покажи эксперименты по распределению Au, Ag и МПГ между штейном и шлаком.
Какие способы закачки шахтных вод применялись в России и за рубежом?
Какие пробелы в данных найдены в активном корпусе?
Есть ли противоречия или неоднородные данные по численным параметрам?
```

---

## Полезные команды

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

Логи:

```powershell
docker compose logs api --tail=100
docker compose logs ui --tail=100
docker compose logs neo4j --tail=100
```

Сбросить контейнеры и локальные runtime-данные:

```powershell
docker compose down -v
Remove-Item -Recurse -Force data, volumes, artifacts -ErrorAction SilentlyContinue
docker compose --profile full up -d --build
```

---

## Проверки внутри контейнера

Быстрые API/UI smoke-тесты:

```powershell
docker compose exec api python -m pytest -q tests/test_api_ask.py tests/test_document_management_api.py tests/test_upload_and_refresh_flow.py tests/test_streamlit_preset_mapping.py tests/test_streamlit_no_nested_expanders.py
```

Ключевые eval-проверки:

```powershell
docker compose exec api python evaluation/eval_ui_product.py
docker compose exec api python evaluation/eval_grounding_guard.py
docker compose exec api python evaluation/eval_answer_quality.py
docker compose exec api python evaluation/eval_tz_query_readiness.py
```

---

## Troubleshooting

### `.env` не найден

```powershell
Copy-Item .env.example .env
docker compose --profile full up -d --build
```

### UI не открывается

```powershell
docker compose ps
docker compose logs ui --tail=100
docker compose exec ui python -c "import urllib.request; print(urllib.request.urlopen('http://api:8000/health').read().decode()[:500])"
```

### API не открывается

```powershell
docker compose ps
docker compose logs api --tail=100
Invoke-RestMethod http://localhost:8000/health
```

### Neo4j не подключается

Для первого запуска это не критично: проект умеет работать через fallback.

Проверка:

```powershell
docker compose exec api python scripts/check_neo4j_connection.py
```

Параметры локального Neo4j:

```text
NEO4J_DOCKER_URI=bolt://neo4j:7687
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=hackathon_password
NEO4J_DATABASE=neo4j
```

### Загрузка файлов идет долго

Это нормально для больших PDF, DOCX и XLSX. Если файл является сканом без текстового слоя, ожидаемый статус — `ocr_required`; это не crash, а диагностика необходимости OCR-профиля.

---

## Режимы работы

### economy_core

Режим для первого запуска и локальной проверки:

```text
без внешнего LLM
без локальных dense embeddings
BM25 retrieval
deterministic extraction
template/fallback answer
```

### balanced_hybrid

Гибридный retrieval с локальными embeddings. Требует больше ресурсов и локальную модель в `models/` или доступную загрузку модели.

В `.env`:

```text
RUNTIME_PROFILE=balanced_hybrid
```

После изменения:

```powershell
docker compose --profile full up -d --build
```

Если embeddings недоступны, система должна деградировать в BM25/fallback, а причина будет видна в `/health`.

### economy_guarded_llm / quality_full

Опциональные режимы с LLM polish. Для локального запуска организаторам они не обязательны. Реальные ключи нельзя коммитить.

Важно: LLM не является источником истины. Факты извлекаются и проверяются отдельно, а LLM используется только как опциональный слой формулировки ответа.

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

Runtime-папки, которые создаются локально и не хранятся в Git:

```text
data/
artifacts/
volumes/
models/
data_storage/
```

---

## Что нельзя коммитить

```text
.env
data_storage/
data/
volumes/
artifacts/
models/
__pycache__/
*.pyc
*.sqlite3
*.log
*.zip
```

Проверка перед коммитом:

```powershell
git status --short
git diff --cached --name-only
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

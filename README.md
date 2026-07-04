# NorNickel-XAK — R&D Knowledge Graph

Система для хакатона: загрузка научно-технических документов, извлечение проверяемых фактов, построение knowledge graph и ответы на исследовательские вопросы с источниками/evidence.

Проект рассчитан на R&D-задачи в горно-металлургической и материаловедческой тематике: материалы, процессы, режимы, свойства, числовые параметры, оборудование, источники, выводы, противоречия и пробелы в данных.

## Что умеет проект

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

Рекомендуемый режим для локальной проверки — `economy_core`: BM25 + deterministic extraction + fallback/graph logic, без внешнего LLM и без локальной embedding-модели.

---

## Быстрый запуск для проверяющих

### 1. Требования

Нужно установить:

- Git;
- Docker Desktop;
- Docker Compose v2;
- 4 GB+ свободной RAM для полного запуска с Neo4j и Qdrant.

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

### 3. Создать `.env` и локальные runtime-папки

Это обязательный шаг. Без этих папок Docker Compose на части окружений может не поднять сервисы из-за bind-mount путей.

PowerShell:

```powershell
Copy-Item .env.example .env -Force
New-Item -ItemType Directory -Force data, artifacts, models, data_storage, volumes, volumes\neo4j, volumes\qdrant | Out-Null
```

Linux/macOS/Git Bash:

```bash
cp .env.example .env
mkdir -p data artifacts models data_storage volumes/neo4j volumes/qdrant
```

Для первого запуска реальные API-ключи не нужны. По умолчанию используется локальный режим `economy_core` без внешнего LLM.

### 4. Запустить сервисы

Полный локальный запуск с UI, API, Neo4j и Qdrant:

```powershell
docker compose --env-file .env --profile full up -d --build
```

Первый build может занять несколько минут.

Проверить все контейнеры, включая остановленные:

```powershell
docker compose --profile full ps -a
```

Ожидаемые сервисы:

```text
api
ui
neo4j
qdrant
```

Если какой-то контейнер в статусе `Exited` или `Restarting`, сразу смотрите логи:

```powershell
docker compose --profile full logs --tail=200 api ui neo4j qdrant
```

Адреса после успешного запуска:

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

PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-WebRequest http://localhost:8501/_stcore/health
Start-Process http://localhost:8501
```

Linux/macOS/Git Bash:

```bash
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

Если UI открылся на `http://localhost:8501`, проект готов к ручному тестированию.

---

## Данные

Исходный датасет хакатона **не включен** в репозиторий.

Для локального запуска с полным корпусом нужно распаковать выданный организаторами датасет в папку:

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

Папка `data_storage/` намеренно исключена из Git, потому что содержит исходные данные хакатона. В Docker Compose она монтируется внутрь API-контейнера как read-only путь `/code/hackathon_project/data_storage`.

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

Сначала распакуйте датасет в `data_storage/`, затем запустите:

```powershell
docker compose --profile full exec api python scripts/stage_real_corpus_demo.py --input /code/hackathon_project/data_storage --count 25 --reset --sync-neo4j
```

Dry-run без загрузки:

```powershell
docker compose --profile full exec api python scripts/stage_real_corpus_demo.py --input /code/hackathon_project/data_storage --count 25 --dry-run
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
docker compose --profile full down
```

Перезапустить без пересборки:

```powershell
docker compose --env-file .env --profile full up -d
```

Пересобрать полностью:

```powershell
docker compose --env-file .env --profile full up -d --build
```

Логи:

```powershell
docker compose --profile full logs api --tail=100
docker compose --profile full logs ui --tail=100
docker compose --profile full logs neo4j --tail=100
docker compose --profile full logs qdrant --tail=100
```

---

## Troubleshooting

### Контейнеры не запустились после `docker compose --profile full up -d --build`

1. Проверьте, что `.env` и runtime-папки созданы:

```powershell
Test-Path .env
Test-Path data
Test-Path models
Test-Path data_storage
Test-Path volumes\neo4j
Test-Path volumes\qdrant
```

2. Если чего-то нет, создайте заново:

```powershell
Copy-Item .env.example .env -Force
New-Item -ItemType Directory -Force data, artifacts, models, data_storage, volumes, volumes\neo4j, volumes\qdrant | Out-Null
```

3. Запустите с явным `.env`:

```powershell
docker compose --env-file .env --profile full up -d --build
```

4. Посмотрите статусы и логи:

```powershell
docker compose --profile full ps -a
docker compose --profile full logs --tail=200 api ui neo4j qdrant
```

### UI не открывается

```powershell
docker compose --profile full ps -a
docker compose --profile full logs ui --tail=100
docker compose --profile full exec ui python -c "import urllib.request; print(urllib.request.urlopen('http://api:8000/health').read().decode()[:500])"
```

### API не открывается

```powershell
docker compose --profile full ps -a
docker compose --profile full logs api --tail=100
Invoke-RestMethod http://localhost:8000/health
```

### Neo4j не подключается

Для первого запуска это не критично: проект умеет работать через fallback.

Проверка:

```powershell
docker compose --profile full exec api python scripts/check_neo4j_connection.py
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
docker compose --env-file .env --profile full up -d --build
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

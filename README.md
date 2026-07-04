# SciKnot — Scientific GraphRAG для технических документов

Проект для хакатона: система загружает научно-технические документы, извлекает проверяемые факты, строит граф знаний и отвечает на исследовательские вопросы с источниками, цитатами и аудитом происхождения ответа.

Основной интерфейс — Streamlit UI. Backend — FastAPI. Граф — Neo4j. Поиск — BM25 / hybrid retrieval. LLM-провайдер можно подключить через Yandex Cloud AI.

---

## 1. Что нужно установить

Перед запуском установите:

- Git;
- Docker Desktop;
- Docker Compose v2.

Проверьте установку:

```powershell
docker --version
docker compose version
```

Для нормального запуска желательно иметь не меньше 4–8 GB свободной RAM.

---

## 2. Скачать проект

```powershell
git clone https://github.com/Leo-Daiser/NorNickel-XAK.git
cd NorNickel-XAK
```

---

## 3. Создать `.env` и runtime-папки

Сначала создайте `.env` из примера:

```powershell
Copy-Item .env.example .env -Force
```

Создайте локальные папки, которые не хранятся в Git:

```powershell
New-Item -ItemType Directory -Force data, artifacts, models, data_storage, volumes, volumes\neo4j, volumes\qdrant | Out-Null
```

Эти папки нужны для runtime-данных, кэшей, загружаемых документов и volume-данных сервисов.

---

## 4. Как подключить Yandex Cloud AI

Откройте файл `.env`:

```powershell
notepad .env
```

Добавьте или замените в нём блок LLM-настроек:

```env
LLM_PROVIDER=yandex
YANDEX_API_KEY=ВАШ_SECRET_KEY_ОТ_YANDEX_CLOUD_AI
YC_API_KEY=ВАШ_SECRET_KEY_ОТ_YANDEX_CLOUD_AI
YANDEX_FOLDER_ID=ВАШ_FOLDER_ID
YANDEX_BASE_URL=https://ai.api.cloud.yandex.net/v1
YANDEX_MODEL_URI=gpt://ВАШ_FOLDER_ID/yandexgpt-5.1
```

Пример, как должна выглядеть строка `YANDEX_MODEL_URI`:

```env
YANDEX_MODEL_URI=gpt://b1xxxxxxxxxxxxxxxxx/yandexgpt-5.1
```

Где взять значения:

- `YANDEX_API_KEY` / `YC_API_KEY` — secret key API-ключа сервисного аккаунта Yandex Cloud AI;
- `YANDEX_FOLDER_ID` — ID каталога Yandex Cloud;
- `YANDEX_MODEL_URI` — URI модели, где вместо `ВАШ_FOLDER_ID` указан ваш реальный folder id.

Важно:

- не коммитьте `.env` в Git;
- не отправляйте API-ключ в чат или публичный репозиторий;
- если ключ случайно попал в публичное место, удалите его в Yandex Cloud и создайте новый.

Проверить, что `.env` игнорируется Git:

```powershell
git check-ignore -v .env
```

Если команда ничего не вывела, добавьте `.env` в `.gitignore`:

```powershell
Add-Content .gitignore "`n.env"
```

---

## 5. Запуск проекта

Запустите полный стек:

```powershell
docker compose --env-file .env --profile full up -d --build
```

Первый build может идти **10–15 минут**. Это нормально: Docker собирает образы и устанавливает зависимости.

Проверить статус контейнеров:

```powershell
docker compose --profile full ps -a
```

После успешного запуска откройте:

```text
Streamlit UI:  http://localhost:8501
API docs:      http://localhost:8000/docs
API health:    http://localhost:8000/health
Neo4j Browser: http://localhost:7474
```

Neo4j Browser:

```text
login:    neo4j
password: hackathon_password
```

---

## 6. Проверка запуска

Проверьте API:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Проверьте UI:

```powershell
Invoke-WebRequest http://localhost:8501/_stcore/health
Start-Process http://localhost:8501
```

В UI вверху должно быть видно примерно:

```text
API: ok
Neo4j: подключен
LLM: готов
```

Если `LLM: готов` не появился, проверьте значения `LLM_PROVIDER`, `YANDEX_API_KEY`, `YANDEX_FOLDER_ID` и `YANDEX_MODEL_URI` в `.env`.

---

## 7. Как загрузить документы

Откройте UI:

```text
http://localhost:8501
```

Дальше:

1. Перейдите к блоку загрузки документов.
2. Загрузите PDF, DOCX, TXT, MD, HTML, XLSX, CSV или PPTX.
3. Дождитесь обработки.
4. При необходимости нажмите обновление графа / active corpus в UI.
5. Задайте исследовательский вопрос.

Если у вас есть полный корпус организаторов, распакуйте его в папку:

```text
data_storage/
```

Папка `data_storage/` не хранится в Git, потому что может содержать исходные данные хакатона.

---

## 8. Быстрые вопросы для проверки

Для демонстрации в режиме **Лучший ответ** можно использовать:

```text
Какие технологические решения электроэкстракции никеля описаны?
```

```text
Что известно об электроэкстракции никеля из сульфатных или хлоридных растворов?
```

Дополнительные вопросы:

```text
Какие способы охлаждения применяются для глубоких рудников?
Какие основные источники тепла возникают в глубоких подземных рудниках?
Какие пробелы в данных найдены в активном корпусе?
Есть ли противоречия или неоднородные данные по численным параметрам?
```

---

## 9. Режимы работы в UI

### Лучший ответ

Основной режим для демонстрации. Система сначала пытается найти подтверждённые факты в графе. Если точных structured facts недостаточно, она может показать навигационный ответ по найденным источникам и evidence-фрагментам.

### Строгая проверка

Аудиторский режим. Он не должен подмешивать частично релевантные chunks как основной verified-ответ. Если точных `AcceptedFact` нет, система честно показывает, что exact-подтверждение не найдено.

### Офлайн-режим

Режим без внешнего LLM. Подходит для проверки базового retrieval, графа, загрузки документов и deterministic fallback-ответов.

---

## 10. PDF-отчёт

В UI можно выгрузить ответ в PDF. Отчёт содержит:

- вопрос;
- краткий вывод;
- найденные факты или частично релевантные результаты;
- ограничения анализа;
- использованные источники и цитаты;
- служебную информацию по графу.

PDF формируется с поддержкой русской кириллицы.

---

## 11. Остановить проект

```powershell
docker compose --profile full down
```

Перезапустить без пересборки:

```powershell
docker compose --env-file .env --profile full up -d
```

Полностью пересобрать:

```powershell
docker compose --env-file .env --profile full up -d --build
```

Посмотреть логи:

```powershell
docker compose --profile full logs api --tail=100
docker compose --profile full logs ui --tail=100
docker compose --profile full logs neo4j --tail=100
```

---

## 12. Если что-то не запустилось

Проверьте контейнеры:

```powershell
docker compose --profile full ps -a
```

Посмотрите логи:

```powershell
docker compose --profile full logs --tail=200 api ui neo4j qdrant
```

Проверьте, что созданы runtime-папки:

```powershell
Test-Path .env
Test-Path data
Test-Path artifacts
Test-Path models
Test-Path data_storage
Test-Path volumes\neo4j
Test-Path volumes\qdrant
```

Если чего-то нет, создайте заново:

```powershell
Copy-Item .env.example .env -Force
New-Item -ItemType Directory -Force data, artifacts, models, data_storage, volumes, volumes\neo4j, volumes\qdrant | Out-Null
```

Потом снова запустите:

```powershell
docker compose --env-file .env --profile full up -d --build
```

---

## 13. Главные идеи проекта

Проект сделан как проверяемая GraphRAG-система для R&D-документов.

Ключевая идея — не просто генерировать текстовый ответ, а связывать ответ с документами, источниками, evidence и графом знаний.

Что делает система:

- принимает технические документы разных форматов;
- разбивает документы на chunks;
- извлекает структурированные факты;
- связывает факты с источниками и цитатами;
- строит knowledge graph в Neo4j;
- ищет релевантные документы и факты по исследовательскому вопросу;
- отделяет подтверждённые `AcceptedFact` от навигационных evidence-фрагментов;
- показывает аудит происхождения ответа;
- формирует PDF-отчёт для эксперта.

Главное отличие от обычного чат-бота: система показывает, откуда взят ответ, какие источники использовались и является ли вывод строго подтверждённым фактом или только навигационным результатом по найденным фрагментам.

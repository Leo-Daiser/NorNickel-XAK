# Real Corpus Staging

This project keeps uploaded documents in the persistent API catalog and Neo4j
projection. Deleting `demo_data` from the workspace does not remove already
ingested documents. Use the staging flow below to replace the active corpus
with a clean sample from `data_storage`.

## Clean stage command

```powershell
docker compose build api ui
docker compose up -d api ui neo4j

python scripts/stage_real_corpus_demo.py `
  --input data_storage `
  --count 25 `
  --max-file-mb 25 `
  --reset `
  --sync-neo4j `
  --timeout 300 `
  --report artifacts/real_corpus_stage_report.json `
  --questions artifacts/real_corpus_ui_questions.md
```

`--reset` calls `POST /admin/reset-corpus?confirm=RESET_ACTIVE_CORPUS` and
clears the local catalog, runtime retrieval index and Neo4j projection before
uploading the selected files. Do not use it if you want to keep the current
corpus.

## UI questions for the staged corpus

1. Какие материалы, процессы и свойства встречаются в загруженных источниках?
2. Какие методы обессоливания воды подходят для обогатительной фабрики при сульфатах и хлоридах 200-300 мг/л?
3. Какие решения по циркуляции католита при электроэкстракции никеля описаны в источниках?
4. Какие системы очистки газов и печи взвешенной плавки применяются для удаления SO2?
5. Покажите эксперименты и публикации по распределению Au, Ag и МПГ между штейном и шлаком.
6. Какие способы закачки шахтных вод в глубокие горизонты упоминаются в российских и зарубежных источниках?
7. Какие пробелы в данных найдены в активном корпусе?
8. Есть ли противоречия или неоднородные данные по численным параметрам?
9. Какие источники подтверждают найденные выводы по гидрометаллургии?
10. Какие лаборатории, команды или авторы упоминаются в загруженных документах?

## Active document workflow

In Streamlit, document checkboxes are edited in a form:

1. Change the `Активен` checkboxes.
2. Press `Применить изменения`.
3. Press `Обновить граф по активным документам`.

This avoids running a heavy API/Neo4j sync for every checkbox click.

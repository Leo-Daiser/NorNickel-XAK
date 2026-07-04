# File Corpus Readiness

## 1. Current Format Support

The production ingestion path is `ParserRouter -> chunks -> deterministic extraction -> canonical facts`.

Currently supported extensions are:

- `.pdf`: fallback parser uses `pypdf` text extraction. If optional Docling is installed and `PARSER_BACKEND=auto`, Docling may be used first.
- `.docx`: fallback parser uses `python-docx` paragraphs and tables.
- `.pptx`: fallback parser uses `python-pptx` text frames and tables.
- `.xlsx`: fallback parser uses `pandas.read_excel` and turns rows into table-row chunks.
- `.csv`: fallback parser uses `pandas.read_csv` with delimiter inference.
- `.html` / `.htm`: fallback parser uses BeautifulSoup, removes `script`, `style`, `nav`, `footer`, and extracts text, tables and images.
- `.md` / `.txt`: fallback parser uses line/heading/markdown-table parsing.

Optional parser backends:

- `docling`: useful for richer PDFs, if installed.
- `markitdown`: useful for Office/HTML conversion, if installed.
- OCR is not executed by default. `ENABLE_OCR=false` in resource-efficient profiles.

## 2. Where Data Can Be Lost

- Image-only/scanned PDFs can have zero extracted text with `pypdf`. This must be reported as `ocr_required`, not as a scientific no-data result.
- PDFs with a poor text layer can produce low-density text, broken units and broken material names.
- DOCX/PPTX tables can lose structure when cells contain nested formatting or merged cells.
- CSV delimiter inference can fail on severely malformed legacy files.
- HTML parsing removes navigation/footer noise, but malformed tables can still lose headers.
- Markdown/TXT OCR exports can contain broken line wraps and mixed Cyrillic/Latin unit symbols.

## 3. Text-Layer Quality Detection

The readiness profiler measures:

- `text_chars`;
- estimated pages;
- characters per page;
- `text_density`: `empty`, `very_low`, `low`, `medium`, `high`;
- dirty OCR signals such as split `M Pa`, mixed `МРа`, `НV`, `ВТ 6`, `7075 Т6`, hyphenated line breaks and many short OCR lines.

Low density is a parser/text-layer warning. It is not the same as extraction failure.

## 4. OCR-Required Detection

For PDFs, the parser checks page count and extracted text length.

If a PDF has pages but text length is below `SCANNED_PDF_MIN_TEXT_CHARS`, the document is marked with:

- `parse_status = ocr_required`;
- warning `ocr_required`;
- no claim that scientific facts are absent.

OCR is intentionally not enabled in `economy_core`. The system reports the need for OCR rather than silently hallucinating missing facts.

## 5. Table Handling

Tables are converted into row-level chunks with:

- table id;
- row id;
- table columns;
- source file metadata;
- stable chunk ids.

The table extractor reads common columns:

- material/alloy;
- regime/process;
- property/metric;
- value/result;
- unit;
- effect/conclusion;
- gap/data_gap.

Table-heavy documents are flagged so reviewers can inspect parser quality.

## 6. Parser Failure vs Extraction Miss

The profiler separates:

- `parser_failure`: parser crashed or file cannot be read;
- `ocr_required`: parser worked but text layer is absent/too sparse;
- `zero_facts`: parser produced chunks but deterministic extraction found no accepted facts/gaps;
- `extraction_miss`: facts may be absent because current deterministic patterns do not cover the wording;
- `retrieval_miss`: facts exist but a query does not retrieve them;
- `answer_synthesis_error`: facts are present but answer formatting/routing is wrong.

This separation is required for product readiness. A scanned PDF is not an extraction miss.

## 7. Product Metrics

Per file:

- parser backend;
- parse status;
- text density;
- page/table/image/chunk counts;
- raw and canonical facts;
- facts without evidence;
- conflicts;
- data gaps;
- warnings.

Corpus-level:

- files by extension;
- parse status counts;
- parser failures;
- OCR-required documents;
- zero-fact documents;
- dirty OCR documents;
- table-heavy documents;
- total chunks;
- total raw/canonical facts;
- facts without evidence;
- conflict groups;
- data gaps.

## 8. Resource Efficiency

The readiness profiler is small-model-first:

- no LLM calls;
- no LLM extraction;
- no embeddings;
- no Qdrant;
- no Neo4j dependency;
- deterministic parser and extraction only.

The report can run in `economy_core` and is suitable for low-resource deployment checks.

## 9. Economy Core Requirements

In `economy_core`:

- `ENABLE_LLM=false`;
- `LLM_PROVIDER=offline`;
- `ENABLE_LOCAL_EMBEDDINGS=false`;
- `RETRIEVAL_MODE=bm25`;
- facts are extracted deterministically;
- every accepted fact must have evidence;
- scanned/image-only documents must be marked as OCR-required instead of being treated as no-data evidence.

## 10. Commands

Build the product readiness report:

```powershell
python scripts/corpus_readiness_report.py --input data_storage --profile-mode inventory --output artifacts/corpus_readiness_report.json --markdown artifacts/corpus_readiness_report.md
```

Build the unified `data_storage` readiness dashboard:

```powershell
python scripts/data_storage_readiness_dashboard.py --input data_storage --output artifacts/data_storage_readiness_dashboard.json --markdown artifacts/data_storage_readiness_dashboard.md
```

This dashboard combines inventory, direct-ingest planning, archive staging, legacy Office conversion planning and OCR/large-PDF queue status. It does not execute OCR, conversion, archive extraction, embeddings or LLM calls.

Profile a bounded sample with real parsing:

```powershell
python scripts/corpus_readiness_report.py --input data_storage --profile-mode auto --max-parse-mb 5 --sample-per-group 20 --output artifacts/data_storage_sample_readiness.json --markdown artifacts/data_storage_sample_readiness.md
```

Parse a single selected file fully:

```powershell
python scripts/corpus_readiness_report.py --input "data_storage\Статьи\some_file.pdf" --profile-mode full --output artifacts/single_file_readiness.json --markdown artifacts/single_file_readiness.md
```

Build a safe batch ingestion plan without calling the API:

```powershell
python scripts/batch_ingest_corpus.py --input data_storage --dry-run --max-file-mb 25 --report artifacts/batch_ingest_plan.json
```

Run a small live API smoke on ready files only:

```powershell
python evaluation/eval_batch_ingest_smoke.py --input data_storage --sample-size 5 --max-file-mb 10 --timeout 180
```

If API is not running, this eval reports a controlled WARN. If it reports `runtime_profile_economy_core_overridden_by_env`, the API is still usable, but the current `.env` overrides the nominal profile with LLM/embeddings settings.

Build the conversion/OCR backlog for files that cannot be directly ingested:

```powershell
python scripts/conversion_backlog_report.py --input data_storage --max-file-mb 25 --output artifacts/conversion_backlog_report.json --markdown artifacts/conversion_backlog_report.md
```

Inspect archives without extracting them:

```powershell
python scripts/archive_staging_report.py --input data_storage --output artifacts/archive_staging_report.json --markdown artifacts/archive_staging_report.md
```

Extract safe supported files from ZIP archives into ignored staging storage:

```powershell
python scripts/archive_staging_report.py --input data_storage --extract-zip --staging-dir artifacts/archive_staging --output artifacts/archive_staging_report.json --markdown artifacts/archive_staging_report.md
```

RAR and multipart archives are not extracted by the Python script. They require a controlled external extractor and must keep the original archive/member provenance in the generated manifest.

Plan legacy Office conversion without converting files:

```powershell
python scripts/legacy_office_conversion_report.py --input data_storage --output artifacts/legacy_office_conversion_report.json --markdown artifacts/legacy_office_conversion_report.md
```

Run conversion only when LibreOffice/soffice is installed and the staging policy is acceptable:

```powershell
python scripts/legacy_office_conversion_report.py --input data_storage --convert --staging-dir artifacts/legacy_office_staging --output artifacts/legacy_office_conversion_report.json --markdown artifacts/legacy_office_conversion_report.md
```

If `soffice_available=false`, this is not a parser failure. It means the conversion tool is not configured in the current runtime. The legacy files remain in the backlog until converted into `.docx`, `.xlsx` or `.pptx`.

Plan OCR and large-PDF processing without running heavy tools:

```powershell
python scripts/ocr_large_pdf_report.py --input data_storage --max-file-mb 25 --output artifacts/ocr_large_pdf_report.json --markdown artifacts/ocr_large_pdf_report.md
```

This report checks whether `ocrmypdf`, `tesseract` and `pdftotext` are available. Missing tools are reported as controlled blockers, not as parser failures. OCR/text outputs must be staged as derived artifacts with original-source provenance.

Ingest a bounded sample through the API, one file per request:

```powershell
python scripts/batch_ingest_corpus.py --input data_storage --sample-per-group 10 --max-file-mb 25 --timeout 180 --report artifacts/batch_ingest_report.json
```

Resume after interruption:

```powershell
python scripts/batch_ingest_corpus.py --input data_storage --sample-per-group 10 --max-file-mb 25 --timeout 300 --state artifacts/batch_ingest_state.json --report artifacts/batch_ingest_report.json
```

Run the full ready-file batch with resume state:

```powershell
python scripts/batch_ingest_corpus.py --input data_storage --max-file-mb 25 --timeout 300 --state artifacts/batch_ingest_state.json --report artifacts/batch_ingest_report.json
```

By default, one bad/slow file is reported as `WARN` and does not stop the corpus run. Use strict mode only for CI gates:

```powershell
python scripts/batch_ingest_corpus.py --input data_storage --max-file-mb 25 --timeout 300 --fail-on-file-error
```

Read-timeout retry is opt-in because a timed-out request may still finish server-side and retrying can create duplicate ingestion attempts. Canonical dedup protects facts, but retries should be explicit:

```powershell
python scripts/batch_ingest_corpus.py --input data_storage --max-file-mb 25 --timeout 300 --timeout-retries 1 --retry-timeout-multiplier 2
```

Do not upload the whole `data_storage` folder through Streamlit as one request. Large corpora must go through the batch command so each file has an individual status, timeout, resume state and parser diagnostic.

Run file/corpus readiness eval:

```powershell
python evaluation/eval_file_corpus_readiness.py
```

Run final-TZ query constraint readiness eval:

```powershell
python evaluation/eval_tz_query_readiness.py
```

Run final-TZ live answer readiness eval against the current API/corpus:

```powershell
python evaluation/eval_tz_answer_readiness.py --preset-id offline_reliable
```

Run end-to-end dirty corpus eval:

```powershell
python evaluation/eval_dirty_demo_corpus.py
```

Interpretation:

- `PASS`: no blocking parser/extraction/provenance failures.
- `WARN`: controlled limitation, for example unsupported archive format or OCR-required scanned PDF.
- `FAIL`: parser crash, facts without evidence, raw leak, hallucinated numeric no-data answer, or economy profile violation.

For the provided `data_storage` corpus, the first pass should be inventory-first:

- many files are large PDFs and conference proceedings;
- archives such as `.zip`, `.rar`, `.001`, `.002` require a controlled extraction step before parsing;
- legacy `.doc`, `.xls`, `.docm` files require conversion or explicit parser adapters;
- image-like sources are marked as OCR-required, not as scientific no-data evidence.
- direct batch ingest should process ready files only; blocked files should go through the conversion/OCR backlog first.

This keeps `economy_core` real: no LLM, no embeddings, no OCR by default, and no hidden parser failures.

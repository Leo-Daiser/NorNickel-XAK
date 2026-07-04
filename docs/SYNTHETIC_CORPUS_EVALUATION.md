# Synthetic Corpus Evaluation

## Scope

This phase evaluates truthfulness and robustness on a synthetic/dirty corpus without expanding the production ontology blindly.

The corpus is intentionally mixed:

- clean factual documents;
- CSV/HTML/Markdown semi-structured sources;
- OCR-like noisy text;
- conflicting measurements;
- duplicate and near-duplicate evidence;
- versioned reports;
- web-page HTML fixtures;
- negative/adversarial sources;
- unsupported coverage probes for labs, teams and equipment.

Unsupported coverage probes are not a requirement to extract new entity or relation types. They verify that the system does not crash and does not invent facts outside the current material/regime/property/gap pipeline.

## Technical Subgraph Decision

The noisy technical subgraph canvas was removed from the main UX. It duplicated the readable answer map while exposing raw IDs and internal relation names. The UI now keeps:

- the compact enriched answer graph;
- the expanded answer map;
- the collapsed `Аудит графа` section with structured node/edge tables;
- raw provenance only in diagnostics/details.

This preserves auditability without showing a debug graph to users.

## Evaluation Method

`evaluation/eval_corpus_truthfulness.py` performs component-wise evaluation rather than full-text answer matching.

It checks:

- material detection;
- regime detection;
- property detection;
- numeric value correctness;
- normalized value correctness;
- conflict detection;
- data-gap detection;
- provenance presence;
- raw technical leak rate;
- unsupported numeric claim rate;
- no-data hallucination rate;
- web ingestion answer quality;
- expansion delta correctness;
- idempotency;
- active/inactive filtering.

Reports are written to:

- `artifacts/eval_corpus_truthfulness.json`;
- `artifacts/eval_corpus_truthfulness.md`;
- `artifacts/synthetic_corpus_analysis.md`.

## Current Result Snapshot

Latest local runs on the synthetic corpus:

| Profile | Summary | Raw leaks | Unsupported numeric claims | Main warnings |
|---|---:|---:|---:|---|
| economy_core | WARN | 0.0 | 0.0 | over-broad gap answer; weak generic conflict query |
| balanced_hybrid | WARN | 0.0 | 0.0 | same as economy_core in host isolated mode |

The synthetic corpus does not currently show a quality gain from balanced_hybrid in the isolated host eval. That is not surprising: the query bank is mostly graph/extraction-driven and the corpus is small. Hybrid retrieval remains useful for multilingual evidence lookup in live Docker, but the corpus eval does not prove a large uplift yet.

## What Works Reliably

- Clean material/regime/property measurements are extracted with evidence.
- `ksi -> MPa` normalization is preserved in answers and diagnostics.
- Duplicate and near-duplicate facts do not inflate canonical fact counts.
- Conflict groups are detected for repeated material/regime/property values.
- Negative material queries do not produce unsupported unit-bearing numbers.
- Web HTML fixtures ingest as first-class sources and keep readable source names.
- Economy mode answers the core graph questions without LLM extraction, embeddings or external services.

## What Degrades On Dirty Data

- OCR-like split numbers and corrupted units can reduce extraction recall.
- Noisy documents with unrelated numeric boilerplate can make retrieval broader than desired.
- Gap-oriented questions can still receive grounded but over-broad numeric summaries from the generic offline answer path.

## What Degrades On Web Pages

- Navigation-heavy HTML can introduce irrelevant candidate text.
- HTML tables are useful when parser output preserves row text, but malformed tables can lose structure.
- URL ingestion is mocked in eval; real web variability still needs corpus-specific validation.

## Truthfulness Risks

- The main remaining risk is not raw hallucination, but over-broad grounded answers: the answer may include true facts that do not answer the exact gap/conflict query.
- Generic conflict questions can miss explicit conflict wording even when conflict groups exist in diagnostics.
- Coverage probes for labs/equipment are intentionally not extracted into production graph entities yet.

## Priority Fixes Before Real Corpus

1. Improve answer scoping for gap and conflict queries before adding new ontology.
2. Add only corpus-confirmed OCR/unit typo normalization rules.
3. Keep source-specific adapters behind real corpus evidence.
4. Preserve economy_core as the baseline quality/resource proof.

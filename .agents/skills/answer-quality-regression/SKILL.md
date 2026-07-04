---
name: answer-quality-regression
description: Use when changing answer synthesis, material comparison, unit conversion, evidence formatting, runtime presets, or technical_answer handling.
---

# Answer Quality Regression

## Purpose

Keep user-facing scientific answers readable, evidence-grounded, and free of raw graph internals.

## Read first

- `app/answering/human_answer.py`
- `app/domain/unit_normalization.py`
- `app/analytics/`
- `app/runtime/presets.py`
- `evaluation/eval_answer_quality.py`
- `evaluation/eval_runtime_presets.py`
- `tests/test_comparison_answer_quality.py`
- `tests/test_unit_normalization.py`

## Invariants

- Main answer must not expose `technical_answer`.
- Main answer must not contain internal IDs: `doc_`, `chunk_`, `EXP-`, `SCI-`.
- Main answer must not leak raw graph effect labels: `increase`, `decrease`, `unknown`.
- Strength comparisons must normalize `ksi` to `MPa` when possible.
- If regimes, states, or units differ, include a comparability caveat.
- `expert_max`, `strict_audit`, and `offline_reliable` must produce meaningfully different answers.

## Required checks

Run:

```bash
python -m pytest -q tests/test_comparison_answer_quality.py tests/test_unit_normalization.py
python evaluation/eval_answer_quality.py
python evaluation/eval_runtime_presets.py
```

## Canonical demo query

```text
Сравни ВТ6 и 7075-T6 по прочности.
```

Expected behavior:

- human-readable comparison;
- MPa normalization or explicit caveat;
- no raw technical answer;
- no internal graph IDs;
- conclusion with confidence/caveat.

---
name: streamlit-demo-regression
description: Use when changing Streamlit UI, document management, graph rendering, demo scenarios, upload flow, or visual product behavior.
---

# Streamlit Demo Regression

## Purpose

Protect the demo UI from becoming a developer cockpit or unreadable graph dump.

## Read first

- `app/ui.py`
- `app/graph/answer_graph.py`
- `evaluation/eval_ui_product.py`
- `tests/test_ui_product_graph_contract.py`
- `tests/test_answer_graph_builder.py`
- `tests/test_streamlit_no_nested_expanders.py`

## Invariants

- UI should work as one coherent demo page.
- Document management must use editable `Активен` checkbox.
- Do not reintroduce lower selectbox/button for active/inactive document toggling.
- Do not use nested `st.expander`.
- Main graph must be compact answer graph, not raw technical subgraph.
- Main graph should avoid labels like `doc_`, `chunk_`, `EXP-`, `SCI-`, `Experiment`, `SourceChunk`, `MEASURES`, `OF_PROPERTY`.
- Raw technical graph may exist only in a closed technical expander.

## Required checks

Run:

```bash
python -m pytest -q tests/test_ui_product_graph_contract.py tests/test_answer_graph_builder.py tests/test_streamlit_no_nested_expanders.py
python evaluation/eval_ui_product.py
```

## Optional Playwright MCP check

If Playwright MCP is available and UI is running:

- open `http://localhost:8501`;
- verify upload area exists;
- verify document table has `Активен` checkbox;
- ask demo question;
- verify answer graph is readable and compact;
- verify raw technical graph is not the first visible graph.

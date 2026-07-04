# Project Instructions for Codex

This is a scientific knowledge graph / GraphRAG hackathon project.

Before making changes, prefer the relevant skill:

- `neo4j-backend-triage` for Neo4j/fallback/backend activation.
- `answer-quality-regression` for answer synthesis, unit conversion, evidence, presets.
- `streamlit-demo-regression` for UI and graph rendering.
- `release-security-gate` before packaging or sharing archives.

Do not expose secrets. Do not commit `.env`.

Core verification commands:

```bash
python -m pytest -q
python evaluation/eval_answer_quality.py
python evaluation/eval_runtime_presets.py
python evaluation/eval_ui_product.py
python scripts/check_project.py
```

If full pytest fails because optional UI dependencies are missing, run targeted tests and report the missing dependency explicitly.

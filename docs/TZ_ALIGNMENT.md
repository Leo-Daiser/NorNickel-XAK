# TZ Alignment Notes

## Current Position

The project remains small-model-first:

- deterministic parsing and extraction;
- canonical facts with evidence;
- normalized numeric constraints;
- Neo4j/fallback graph projection;
- BM25 with optional hybrid embeddings;
- optional guarded LLM polish only.

LLM extraction is intentionally not used as a source of truth.

## Covered Query Constraints

The deterministic query planner currently extracts:

- materials/substances: nickel, copper, sulfates, chlorides, Ca, Mg, Na, Au, Ag, PGM, matte, slag, mine water, catholyte, electrolyte;
- process regimes: desalination, hydrometallurgy, pyrometallurgy, electrowinning, catholyte circulation, heap leaching, gas cleaning, SO2 removal, mine-water injection, flash smelting furnace / PVP;
- properties/parameters: concentration, dry residue/TDS, flow velocity, temperature, throughput, recovery, metal yield, distribution, corrosion resistance, economic indicators;
- equipment constraints: electrowinning cells, diaphragm cells, flash smelting furnaces/PVP, gas cleaning systems, pumps, pilot plants;
- geography: Russia/domestic practice, foreign practice, worldwide practice, China, USA, Canada;
- time filters: explicit year ranges and relative filters such as “last 5 years”;
- numeric ranges/operators with units such as mg/L, mg/dm3, m/s, m3/h, t/day, %, ppm, MPa, ksi, HV, RUB/t, USD/t.

Coverage is verified by:

```powershell
python evaluation/eval_tz_query_readiness.py
```

Live API answer readiness is checked separately:

```powershell
python evaluation/eval_tz_answer_readiness.py --preset-id offline_reliable
```

This second gate distinguishes product-critical failures from corpus coverage
gaps. Missing evidence is a WARN; raw leaks, unsupported numeric claims and
grounding violations are FAIL.

## What Is Not Claimed Yet

The system does not yet claim production-grade extraction for every entity type listed in the final TZ:

- experts/authors/laboratories from unstructured text are not fully extracted as verified graph facts;
- access control and audit roles are not implemented as a production security subsystem;
- OCR is detected as required but not executed by default;
- RAR/multipart archives and legacy Office files require controlled preprocessing;
- semantic embeddings are optional and may degrade to BM25 when dependency is missing;
- no SHACL/OWL validation layer is currently enforced.

These are deliberate limitations, not hidden behavior.

## Next Practical Priorities

1. Run resumable batch ingestion for ready `data_storage` files.
2. Add controlled preprocessing for ZIP, legacy Office and OCR/large PDFs.
3. Build corpus-specific glossary from provided dictionaries when available.
4. Add verified extraction/eval for experts, laboratories and equipment only after source fields or reliable patterns are present.
5. Add FAIR-style export/reporting once graph facts from the real corpus are populated.

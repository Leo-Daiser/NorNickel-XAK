# Metallurgy R&D TZ Readiness

## Current phase

The system is being aligned with the full hackathon specification for a
resource-efficient scientific knowledge graph in mining and metallurgy.

This phase deliberately does not add LLM extraction. The source of truth remains:

```text
documents / URL
-> deterministic parsing and extraction
-> canonical facts
-> normalization, deduplication, conflicts, gaps
-> fallback graph + Neo4j projection
-> deterministic answer / optional guarded LLM polish
```

## Implemented in this phase

- Domain aliases expanded from alloy-only examples toward mining/metallurgy R&D:
  - hydrometallurgy, pyrometallurgy, desalination, electrowinning;
  - catholyte circulation, heap leaching, gas cleaning, SO2 removal;
  - nickel, copper, sulfates, chlorides, Ca, Mg, Na, Au, Ag, PGM;
  - matte, slag, mine water, catholyte, electrolyte;
  - dry residue, concentration, flow velocity, recovery, distribution, economic indicators.
- Deterministic numeric constraint parser added:
  - ranges: `200-300 mg/L`, `200–300 мг/л`;
  - operators: `<=`, `≤`, `>=`, `≥`, `не более`, `не менее`;
  - units: `mg/L`, `мг/л`, `мг/дм3`, `мг/дм³`, `m/s`, `м/с`, `t/day`, `т/сут`, `%`, `ppm`.
- Query planner now captures:
  - numeric constraints;
  - geography filters such as Russia, foreign practice, world practice;
  - relative time filters such as "last 5 years";
  - broad TZ questions as overview queries instead of forcing exact graph match.
- Deterministic extraction now covers the first process-parameter facts needed
  by the specification:
  - dissolved ion concentrations such as sulfates, chlorides, Ca, Mg, Na in `mg/L`;
  - dry residue / TDS constraints in `mg/L`;
  - catholyte/solution flow velocity in `m/s`;
  - throughput/capacity, recovery, metal yield and basic economic indicators
    when a value and unit are explicitly present near the property.
- Added `evaluation/eval_tz_query_readiness.py` for the four key sample questions from the specification.

## What this proves

The project can now parse the structure of the specification's most important
query patterns without using an LLM:

- water desalination with multi-component concentration constraints;
- catholyte circulation in nickel electrowinning with world-practice filter;
- Au/Ag/PGM distribution between matte and slag with a relative time filter;
- mine-water injection with Russia vs foreign-practice filtering.

## Still incomplete

The parser now understands these questions, and the extraction layer can capture
the first process-parameter measurements. The project still needs corpus-backed
evidence to answer the full technology-review scenarios well. The current
production facts are stronger for alloy/material-property examples than for
full mining-metallurgy technology review scenarios.

Priority next fixes:

1. Expand deterministic extraction coverage for process-parameter facts:
   distribution coefficients, detailed CAPEX/OPEX, climatic conditions and
   process-specific operating windows.
2. Add source metadata extraction:
   publication year, geography, source type, internal/external flag, reliability level.
3. Add technology-comparison answer template:
   method, material/media, conditions, metric, geography, source count, confidence, limitations.
4. Add expert/laboratory extraction from report metadata and author/executor sections.
5. Add a controlled metallurgy TZ fixture corpus with evidence for the four sample queries.

## Resource position

The resource-efficient strategy remains intact:

- `economy_core` can run without LLM and embeddings;
- LLM remains optional polish only;
- numeric and domain query parsing are deterministic;
- embeddings are optional retrieval expansion, not source of truth.

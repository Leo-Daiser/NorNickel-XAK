# Extraction Pipeline

This document describes the extraction code that is actually used in the
current project. It is an implementation inventory, not a product roadmap.

## 1. Pipeline Overview

The active extraction path is deterministic and evidence-bound:

```text
Chunk
 -> ExtractionPipeline.extract_from_chunk
 -> TableExtractor or DeterministicExtractor
 -> validate_items
 -> ExtractionBundle
 -> bundle_to_experiment_facts / bundle_to_data_gaps
 -> CatalogGraphRepository fallback graph or GraphWriter Neo4j sync
```

Main modules:

- `app/extraction/pipeline.py`: orchestrates deterministic/table/optional LLM extraction, validation and audit writing.
- `app/extraction/table_extractor.py`: extracts structured rows from serialized table chunks.
- `app/extraction/deterministic.py`: adapts the legacy rule extractor and adds direct materials-science text patterns.
- `app/extraction/extraction.py`: legacy regex entity/relation extractor.
- `app/extraction/resolver.py`: canonicalizes raw material/regime/property/unit strings.
- `app/extraction/validators.py`: rejects invalid/unsafe extracted items.
- `app/extraction/to_graph_models.py`: converts accepted bundles into graph model objects.
- `app/graph/graph_repository.py`: builds fallback graph facts from catalog chunks.
- `app/graph/graph_writer.py`: materializes accepted facts into Neo4j.

`StructuredLLMExtractor` exists in `app/extraction/llm_structured.py`, but LLM extraction is not part of the safe default demo path. In `hybrid` mode it is used only when `EXTRACTION_ENABLE_LLM=true` and the extractor is available; otherwise the pipeline records `llm_extractor_unavailable`.

## 2. Input/Output Of Each Stage

Input object:

- `app.models.schemas.Chunk`
- Important fields: `chunk_id`, `doc_id`, `text`, `page_start`, `page_end`, `section_path`, `metadata`.

Typed extraction output:

- `ExtractionBundle`
- `entities: list[ExtractedEntity]`
- `experiments: list[ExtractedExperiment]`
- `data_gaps: list[ExtractedDataGap]`
- `rejected_items: list[RejectedExtraction]`
- `diagnostics: dict`

Graph conversion output:

- `ExperimentFact`
- `Measurement`
- `DataGap`
- `Evidence`
- `Measurement` includes both original and normalized numeric fields:
  `value_original`, `unit_original`, `value_normalized`, `unit_normalized`,
  `normalization_family`.

Fallback graph:

- `CatalogGraphRepository` re-extracts active catalog chunks and caches `ExperimentFact` and `DataGap` objects in memory.

Neo4j graph:

- `GraphWriter.sync_catalog` runs `ExtractionPipeline` over active catalog chunks and writes accepted facts into Neo4j with idempotent `MERGE` queries.

## 3. Entity Extraction

Typed entity types in `ExtractedEntity`:

- `Material`
- `ProcessRegime`
- `Property`
- `Equipment`
- `Laboratory`
- `ResearchTeam`
- `Employee`
- `TopicTag`

Legacy extractor also detects broader graph/debug entities:

- `TechnicalObject`
- `Part`
- `ArticleNumber`
- `Standard`
- `Parameter`
- `Requirement`
- `ImageArtifact`
- `Experiment`
- `PropertyValue`
- `PropertyChange`
- `Conclusion`
- `DataGap`

Typed extraction keeps only the entity types allowed by `ExtractedEntity`.
Legacy technical entities can still appear in raw subgraph/debug paths, but not
as accepted typed `ExtractedEntity` objects.

Entity pattern sources:

- `MATERIAL_PATTERNS` in `app/extraction/extraction.py`
- `_MATERIAL_CANDIDATES` in `app/extraction/deterministic.py`
- table column aliases in `app/extraction/table_extractor.py`

Material examples currently supported:

- `ВТ6`, `VT6`, `Ti-6Al-4V`
- `7075`, `7075-T6`
- `12Х18Н10Т`
- `09Г2С`

## 4. Measurement Extraction

Measurement object:

- `ExtractedMeasurement`
- fields: `property_raw`, `property_canonical`, `value`, `unit`, `effect`,
  optional baseline/delta fields, `confidence`, `evidence`.

Primary measurement routes:

- Table rows: `TableExtractor.extract_from_chunk`
  - Reads columns like `property`, `value`, `unit`, `effect`.
  - Builds one `ExtractedMeasurement` when a property is present.
- Legacy text regex: `EntityRelationExtractor.extract_from_chunk`
  - Uses `MEASUREMENT_RE`.
  - Infers property from nearby terms and unit context.
  - Creates legacy `MEASURES`, `OF_PROPERTY`, `HAS_CHANGE` relations.
- Direct text patterns: `_extract_measurements` in `deterministic.py`
  - Russian strength: `предел прочности`, `прочность`, `прочности`
  - English strength: `tensile strength`, `ultimate tensile strength`
  - Units: `MPa`, `МПа`, `GPa`, `ksi`
  - Ductility: `удлинение ... %`
  - Hardness: `HV`, `HRC`
  - Qualitative corrosion effect: corrosion resistance increase/decrease
    without creating a numeric value.

Important safety behavior:

- A numeric value without a recognized property is not accepted as a typed
  measurement.
- `validators.validate_measurement` rejects numeric measurements when the
  numeric value exists but the property term is not near the value inside the
  evidence quote.
- `_apply_binding_guard` in `deterministic.py` lowers measurement confidence
  when material/regime and property/value are in different sentences without a
  clear linking phrase.
- Corrosion gap phrases like `коррозионная стойкость не измерялась` are rejected
  as positive measurements.
- Qualitative measurements can be accepted only when they have a known property
  and a non-`unknown` effect.

## 5. Unit Normalization

Extraction-time unit canonicalization:

- `app/extraction/resolver.py::resolve_unit`
- aliases include `МПа -> MPa`, `GPa`, `ksi`, `HV`, `HRC`, `%`, `C`, `h`, `min`.

Validation-time normalization:

- `app/extraction/validators.py::validate_measurement`
- normalizes units through `resolve_unit`
- accepts only `VALID_UNITS`

Graph/API-time numeric normalization:

- `app/domain/fact_normalization.py::measurement_normalization_fields`
- Stores explicit original and normalized fields for graph/API facts.
- Supported conversion: strength `ksi -> MPa`.
- `МПа`/`мПа` canonicalizes to `MPa`; `MPa` stays `MPa`.

Answer-time strength conversion:

- `app/domain/unit_normalization.py::normalize_strength_to_mpa`
- converts `ksi -> MPa` using `6.894757`
- returns a conversion note such as `77 ksi ≈ 531 MPa`

Not implemented yet:

- General unit conversion for all properties.
- Dimension-aware unit families in Neo4j.
- Temperature conversion from Fahrenheit/Kelvin.

## 6. Synonym Normalization

Canonical aliases:

- `app/domain/aliases.py`
- `MATERIAL_ALIASES`
- `REGIME_ALIASES`
- `PROPERTY_ALIASES`

Canonical functions:

- `canonical_material`
- `canonical_regime`
- `canonical_property`

Resolver functions:

- `resolve_material`
- `resolve_regime`
- `resolve_property`

Current examples:

- `Ti-6Al-4V`, `VT6` -> `ВТ6`
- `7075` -> `7075-T6`
- `annealing`, `annealed` -> `отжиг`
- `aging`, `aged` -> `старение`
- `heat treatment` -> `термообработка`
- `tensile strength`, `ultimate tensile strength` -> `прочность`
- `corrosion resistance` -> `коррозионная стойкость`

## 7. Relation Construction

Legacy relation types in `EntityRelationExtractor`:

- `STUDIES`
- `USES_REGIME`
- `MEASURES`
- `OF_PROPERTY`
- `HAS_CHANGE`
- `USES_EQUIPMENT`
- `PERFORMED_BY`
- `SUPPORTED_BY`
- `MISSING_FOR`
- `HAS_MEASUREMENT`
- `OBJECT_HAS_PARAMETER`
- `OBJECT_HAS_PART`
- `OBJECT_MADE_OF_MATERIAL`
- `OBJECT_COMPLIES_WITH_STANDARD`
- `PART_HAS_ARTICLE_NUMBER`
- `REQUIREMENT_APPLIES_TO_OBJECT`
- `IMAGE_LINKED_TO_SECTION`
- `CHUNK_MENTIONS_ENTITY`

Typed graph model relations are implicit in `ExperimentFact` and materialized by
`GraphWriter` into Neo4j:

- `Experiment -[:SUPPORTED_BY]-> DocumentChunk`
- `Experiment -[:USES_MATERIAL]-> Material`
- `Experiment -[:HAS_REGIME]-> ProcessRegime`
- `Experiment -[:MEASURED]-> Measurement`
- `Measurement -[:OF_PROPERTY]-> Property`
- `Measurement -[:SUPPORTED_BY]-> DocumentChunk`
- `Experiment -[:USED_EQUIPMENT]-> Equipment`
- `Experiment -[:PERFORMED_BY]-> ResearchTeam`
- `Experiment -[:PERFORMED_AT]-> Laboratory`
- `Experiment -[:LED_TO]-> Conclusion`
- `DataGap -[:GAP_FOR_ENTITY]-> Material`
- `DataGap -[:GAP_FOR_REGIME]-> ProcessRegime`
- `DataGap -[:GAP_FOR_PROPERTY]-> Property`
- `DataGap -[:SUPPORTED_BY]-> DocumentChunk`

## 8. Confidence Scoring

Main scoring function:

- `app/extraction/confidence.py::experiment_confidence`

Signals:

- material present: `+0.25`
- regime present: `+0.20`
- measurement property present: `+0.20`
- numeric value with unit: `+0.15`
- equipment/lab/team present: `+0.10`
- conclusion or known effect: `+0.10`
- missing evidence: `-0.20`
- ambiguous extraction: `-0.20`

Default minimum:

- `EXTRACTION_MIN_CONFIDENCE=0.55`

Item-level confidences are set in extractors:

- table evidence: usually `0.95`
- table material: `0.9`
- table regime/measurement: `0.86`
- direct strength measurement: `0.88-0.90`
- direct corrosion qualitative measurement: `0.72`
- legacy relations: usually `0.65-0.86`
- sentence/window binding guard can lower measurement confidence when material,
  regime and measurement are too far apart inside a chunk.

Not implemented yet:

- Learned confidence calibration.
- Per-document trust scoring.
- Cross-source agreement scoring.

## 9. Evidence Binding

Every accepted entity, experiment, measurement or gap must have evidence:

- `EvidenceSpan`
- `ExtractionSource`

Source fields:

- `document_id`
- `chunk_id`
- `source_name`
- `page`
- `section_path`
- `block_type`
- `row_index`
- `column_name`

Binding flow:

- `_source_from_chunk` creates source metadata from `Chunk`.
- Deterministic/table extractors attach `EvidenceSpan`.
- Validators reject entities, measurements, experiments and gaps without evidence.
- `to_graph_models._to_evidence` converts `EvidenceSpan` to domain `Evidence`.
- `GraphWriter` writes `DocumentChunk` nodes and `SUPPORTED_BY` edges.
- Duplicate canonical facts are collapsed by `app/domain/fact_normalization.py`
  while preserving all evidence spans in the merged fact.

Canonical fact key:

- material canonical name
- regime canonical name
- property canonical name
- normalized numeric value when available
- normalized unit when available
- qualitative effect when it is known
- optional source identity for use cases where source-level identity is needed

Raw identifiers stay in diagnostics/raw graph tables. User-facing answer text and
main answer graph should not expose `doc_`, `chunk_`, `EXP-`, `SCI-`, or raw
technical labels.

## 10. Known Limitations

- Extraction is deterministic and pattern-based by default.
- LLM extraction exists but is not enabled in the safe demo path.
- General English scientific prose coverage is limited.
- Direct patterns cover a small set of material/property/regime combinations.
- Unit normalization is currently limited to strength `ksi -> MPa` plus common
  MPa spellings.
- Multi-sentence relation scope is heuristic; it lowers confidence or rejects
  obvious weak bindings, but it is not a full NLP parser.
- Conflict resolution is report-only, not used to suppress facts automatically.
- Rejected candidates are available only from extraction runs/audit, not as a
  first-class persisted catalog table.
- Table extraction depends on serialized row text with recognizable column names.

## 11. How To Improve Next

Recommended next steps:

1. Add a persisted extraction audit index/table for accepted and rejected
   candidates.
2. Expand sentence/window-level scoping with more controlled corpus examples.
3. Add more controlled golden extraction cases before expanding patterns.
4. Add conflict policy: source priority and disagreement flags.
5. Add broader unit-family metadata beyond strength.
6. Add optional LLM extraction only behind an explicit audit gate, with schema
   validation and deterministic rejection rules.
7. Add provenance-oriented UI for source snippets, without raw graph ids in the
   main user path.

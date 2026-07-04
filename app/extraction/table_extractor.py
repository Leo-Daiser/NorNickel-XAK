"""Column-aware extractor for CSV/XLSX/catalog table row chunks."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from ..models.schemas import Chunk
from .confidence import experiment_confidence
from .deterministic import _source_from_chunk
from .models import (
    CandidateFact,
    EvidenceSpan,
    ExtractedDataGap,
    ExtractedEntity,
    ExtractedExperiment,
    ExtractedMeasurement,
    ExtractedRegime,
    ExtractionBundle,
    RejectedExtraction,
)
from .resolver import clean_raw, resolve_equipment, resolve_material, resolve_property, resolve_regime, resolve_unit


MATERIAL_COLUMNS = ["material", "материал", "сплав", "alloy", "grade", "марка", "вещество", "руда", "концентрат", "штейн", "шлак", "раствор", "электролит", "католит", "commodity", "media", "medium"]
DESCRIPTOR_COLUMNS = ["column_1", "feed", "сырье", "исходный материал", "объект", "object", "продукт", "product", "sample", "образец"]
REGIME_COLUMNS = ["regime", "режим", "process", "обработка", "термообработка", "process_regime", "процесс", "технология", "метод", "method", "operation", "treatment"]
PROPERTY_COLUMNS = ["property", "свойство", "показатель", "metric", "parameter", "параметр", "условие", "condition", "indicator"]
VALUE_COLUMNS = ["value", "значение", "result", "результат", "capacity", "мощность", "производительность", "effect", "эффект"]
UNIT_COLUMNS = ["unit", "ед.", "единица", "units", "ед. изм.", "единицы"]
EQUIPMENT_COLUMNS = ["equipment", "оборудование", "установка", "stand", "device", "cell", "furnace", "reactor"]
LAB_COLUMNS = ["laboratory", "лаборатория", "lab"]
TEAM_COLUMNS = ["team", "команда", "group", "группа"]
EMPLOYEE_COLUMNS = ["employee", "сотрудник", "researcher", "исполнитель", "author", "expert", "эксперт", "автор"]
CONCLUSION_COLUMNS = ["conclusion", "вывод", "effect", "эффект"]
TOPIC_TAG_COLUMNS = ["tag", "topic", "тематика", "тег"]
EXPERIMENT_COLUMNS = ["experiment_id", "experiment", "id эксперимента", "опыт", "exp_id"]
GAP_COLUMNS = ["data_gap", "gap", "пробел", "нет данных"]
SOURCE_COLUMNS = ["source", "источник", "publication", "публикация", "reference"]
YEAR_COLUMNS = ["year", "год", "date", "дата", "period", "период"]
GEOGRAPHY_COLUMNS = ["country", "страна", "region", "регион", "geography", "география"]
FACILITY_COLUMNS = ["facility", "plant", "mine", "завод", "рудник", "предприятие", "объект"]
CLAIM_COLUMNS = ["claim", "утверждение", "вывод", "conclusion", "recommendation", "рекомендация", "note", "примечание"]
TOPIC_COLUMNS = ["topic", "тематика", "theme", "направление", "область", "area"]
TECHNOLOGY_COLUMNS = ["technology", "технология", "solution", "решение", "method", "метод", "approach", "способ", "system", "система"]
TARGET_PROBLEM_COLUMNS = ["target_problem", "problem", "задача", "проблема", "application", "назначение", "purpose", "для"]


class TableExtractor:
    """Extract experiments from serialized table row chunks."""

    extractor_version = "table_v1"

    def extract_from_chunk(self, chunk: Chunk) -> ExtractionBundle:
        row = parse_serialized_row(chunk.text or "")
        source_name = str(chunk.metadata.get("source_name") or chunk.metadata.get("filename") or chunk.doc_id)
        bundle = ExtractionBundle(document_id=chunk.doc_id, source_name=source_name, extractor_version=self.extractor_version)
        quote = chunk.text.strip()
        if not row:
            bundle.rejected_items.append(RejectedExtraction(item_type="table_row", reason="empty_row", raw_payload=chunk.text or ""))
            return bundle
        source = _source_from_chunk(chunk)
        evidence = EvidenceSpan(source=source, quote=quote, confidence=0.95)
        material_values = _multi_values(_pick(row, MATERIAL_COLUMNS))
        bundle.candidate_facts.extend(_candidate_facts_from_row(row, chunk, source_name, evidence))
        if not material_values and not any(clean_raw(value) for value in row.values()):
            bundle.rejected_items.append(RejectedExtraction(item_type="table_row", reason="empty_row", raw_payload=row, evidence=[evidence]))
            return bundle
        if not material_values:
            bundle.rejected_items.append(RejectedExtraction(item_type="table_row", reason="missing_material", raw_payload=row, evidence=[evidence]))
            return bundle

        regime_raw = _pick(row, REGIME_COLUMNS)
        property_raw = _pick(row, PROPERTY_COLUMNS)
        value_raw = _pick(row, VALUE_COLUMNS)
        unit = resolve_unit(_pick(row, UNIT_COLUMNS))
        effect_raw = _pick(row, CONCLUSION_COLUMNS)
        gap_raw = _pick(row, GAP_COLUMNS)

        materials = _dedupe_entities([
            ExtractedEntity(entity_type="Material", raw_name=value, canonical_name=resolve_material(value), confidence=0.9, evidence=[evidence])
            for value in material_values
            if clean_raw(value)
        ])
        regimes: list[ExtractedRegime] = []
        if regime_raw:
            regimes.append(ExtractedRegime(raw_name=regime_raw, canonical_name=resolve_regime(regime_raw), confidence=0.86, evidence=[evidence]))

        measurements: list[ExtractedMeasurement] = []
        if property_raw:
            measurements.append(
                ExtractedMeasurement(
                    property_raw=property_raw,
                    property_canonical=_property_for_unit(property_raw, unit),
                    value=_float_or_none(value_raw),
                    unit=unit,
                    effect=_effect_from_text(effect_raw or value_raw),
                    confidence=0.86 if value_raw or effect_raw else 0.5,
                    evidence=[evidence],
                )
            )

        equipment = _dedupe_entities([_entity("Equipment", item, evidence) for item in _multi_values(_pick(row, EQUIPMENT_COLUMNS))])
        laboratories = _dedupe_entities([_entity("Laboratory", item, evidence) for item in _multi_values(_pick(row, LAB_COLUMNS))])
        teams = _dedupe_entities([_entity("ResearchTeam", item, evidence) for item in _multi_values(_pick(row, TEAM_COLUMNS))])
        employees = _dedupe_entities([_entity("Employee", item, evidence) for item in _multi_values(_pick(row, EMPLOYEE_COLUMNS))])
        topic_tags = _dedupe_entities([_entity("TopicTag", item, evidence) for item in _multi_values(_pick(row, TOPIC_TAG_COLUMNS))])
        conclusions = [text for text in [effect_raw] if text]

        if measurements:
            experiment_id = clean_raw(_pick(row, EXPERIMENT_COLUMNS)) or _stable_experiment_id(source_name, source.row_index, materials, regimes, measurements)
            experiment = ExtractedExperiment(
                experiment_id=experiment_id,
                materials=materials,
                regimes=regimes,
                measurements=measurements,
                equipment=equipment,
                laboratories=laboratories,
                teams=teams,
                employees=employees,
                conclusions=conclusions,
                topic_tags=topic_tags,
                evidence=[evidence],
                confidence=0.0,
            )
            confidence = experiment_confidence(experiment)
            experiment = experiment.model_copy(update={"confidence": confidence}) if hasattr(experiment, "model_copy") else experiment.copy(update={"confidence": confidence})
            bundle.experiments.append(experiment)
            bundle.entities.extend([*materials, *equipment, *laboratories, *teams, *employees, *topic_tags])
        else:
            bundle.entities.extend(materials)
            bundle.rejected_items.append(RejectedExtraction(item_type="table_row", reason="missing_regime_or_measurement", raw_payload=row, evidence=[evidence]))

        if gap_raw:
            material = materials[0].canonical_name if materials else None
            regime = regimes[0].canonical_name if regimes else None
            prop = resolve_property(property_raw) if property_raw else None
            raw = "|".join([material or "", regime or "", prop or "", gap_raw])
            bundle.data_gaps.append(
                ExtractedDataGap(
                    gap_id=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
                    material=material,
                    regime=regime,
                    property=prop or _property_from_gap(gap_raw),
                    reason=gap_raw,
                    confidence=0.85,
                    evidence=[evidence],
                )
            )

        bundle.diagnostics = {"row_index": source.row_index, "columns": list(row)}
        return bundle


def parse_serialized_row(text: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for line in (text or "").splitlines():
        if line.startswith("Table:") or line.startswith("Table columns:"):
            continue
        for part in line.split(" | "):
            if ":" not in part:
                continue
            indexed = re.match(r"^(?P<key>[^:]+:\s*\d+)\s*:\s*(?P<value>.+)$", part)
            if indexed:
                key, value = indexed.group("key"), indexed.group("value")
            else:
                key, value = part.split(":", 1)
            clean_key = _norm_col(key)
            clean_value = value.strip()
            if clean_key and clean_value:
                row[clean_key] = clean_value
    return row


def _pick(row: dict[str, str], aliases: Iterable[str]) -> str:
    alias_set = {_norm_col(alias) for alias in aliases}
    for key, value in row.items():
        if key in alias_set:
            return value
    for key, value in row.items():
        if any(alias in key for alias in alias_set):
            return value
    return ""


def _norm_col(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().replace("ё", "е"))


def _multi_values(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[;\n]+", value)
    return [clean_raw(part) for part in parts if clean_raw(part)]


def _float_or_none(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[-+]?\d+(?:[\.,]\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def _effect_from_text(value: str | None) -> str:
    norm = str(value or "").lower()
    if any(term in norm for term in ["increase", "increased", "повыс", "увелич"]):
        return "increase"
    if any(term in norm for term in ["decrease", "decreased", "сниз", "уменьш"]):
        return "decrease"
    if any(term in norm for term in ["unchanged", "без измен", "no change"]):
        return "no_change"
    return "unknown"


def _entity(entity_type: str, raw: str, evidence: EvidenceSpan) -> ExtractedEntity:
    canonical = resolve_equipment(raw) if entity_type == "Equipment" else clean_raw(raw)
    return ExtractedEntity(entity_type=entity_type, raw_name=raw, canonical_name=canonical, confidence=0.75, evidence=[evidence])


def _dedupe_entities(items: list[ExtractedEntity]) -> list[ExtractedEntity]:
    seen = set()
    result: list[ExtractedEntity] = []
    for item in items:
        key = (item.entity_type, item.canonical_name)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _stable_experiment_id(source_name: str, row_index: int | None, materials, regimes, measurements) -> str:
    raw = "|".join(
        [
            source_name,
            str(row_index or 0),
            ",".join(item.canonical_name for item in materials),
            ",".join(item.canonical_name for item in regimes),
            ",".join(item.property_canonical for item in measurements),
        ]
    )
    return "table_exp_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _property_from_gap(text: str) -> str | None:
    prop = resolve_property(text)
    return prop if prop != clean_raw(text) else None


def _candidate_facts_from_row(row: dict[str, str], chunk: Chunk, source_name: str, evidence: EvidenceSpan) -> list[CandidateFact]:
    candidates: list[CandidateFact] = []
    property_raw = _pick(row, PROPERTY_COLUMNS)
    value_raw = _pick(row, VALUE_COLUMNS)
    table_context = _table_context_text(row, chunk, evidence)
    unit = _row_unit(row, property_raw, value_raw, table_context)
    regime_raw = _pick(row, REGIME_COLUMNS)
    material_raw = _material_or_descriptor(row)
    if any([property_raw, value_raw, regime_raw, material_raw]):
        property_canonical = _property_for_unit(property_raw, unit)
        fact_type = _fact_type_for_row(row, property_canonical, unit)
        subject = _subject_for_row(row, material_raw, regime_raw)
        value = _float_or_none(value_raw)
        candidate_id = _candidate_id(
            fact_type,
            chunk.doc_id,
            chunk.chunk_id,
            subject,
            property_canonical,
            value,
            unit,
            evidence.source.row_index,
        )
        candidates.append(
            CandidateFact(
                candidate_id=candidate_id,
                fact_type=fact_type,
                extractor_name=f"{TableExtractor.extractor_version}:structured_table_adapter",
                document_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                source_name=source_name,
                subject=subject,
                predicate=_predicate_for_fact_type(fact_type),
                object={
                    "property": property_canonical if property_raw else "",
                    "property_raw": property_raw,
                    "process": resolve_regime(regime_raw) if regime_raw else "",
                    "year": clean_raw(_pick(row, YEAR_COLUMNS)),
                    "source_note": clean_raw(_pick(row, SOURCE_COLUMNS)),
                },
                value=value,
                unit=unit,
                evidence_quote=evidence.quote,
                raw_span=" | ".join(f"{key}: {value}" for key, value in row.items()),
                context_window=table_context,
                confidence=_candidate_confidence(fact_type, property_raw, value_raw, regime_raw, material_raw, row),
                document_type="unknown",
            )
        )
    candidates.extend(_assay_candidate_facts_from_row(row, chunk, source_name, evidence, material_raw, regime_raw))
    if not value_raw:
        candidates.extend(_wide_capacity_candidate_facts_from_row(row, chunk, source_name, evidence, material_raw, regime_raw, table_context))
    expertise = _expertise_candidate_from_row(row, chunk, source_name, evidence, material_raw, regime_raw)
    if expertise is not None:
        candidates.append(expertise)
    publication = _publication_claim_candidate_from_row(row, chunk, source_name, evidence, material_raw, regime_raw)
    if publication is not None:
        candidates.append(publication)
    technology = _technology_solution_candidate_from_row(row, chunk, source_name, evidence, material_raw, regime_raw)
    if technology is not None:
        candidates.append(technology)
    return _dedupe_candidate_facts(candidates)


def _fact_type_for_row(row: dict[str, str], property_canonical: str, unit: str | None) -> str:
    text = " ".join([*row.keys(), *row.values()]).lower().replace("ё", "е")
    if any(marker in text for marker in ["capex", "opex", "стоимост", "затрат", "руб/т", "usd/t", "eur/t", "€/t", "cost"]):
        return "EconomicIndicatorFact"
    if any(marker in text for marker in ["capacity", "production capacity", "производительность", "throughput", "мощность", "facility", "plant", "mine"]):
        return "FacilityCapacityFact"
    if property_canonical in {"концентрация", "сухой остаток", "скорость потока", "расход", "извлечение", "выход металла", "распределение", "температура", "pH", "давление"}:
        return "ProcessParameterFact"
    if unit in {"USD/t", "EUR/t", "RUB/t", "USD/m3", "RUB/m3", "mln RUB", "mln RUB/year"}:
        return "EconomicIndicatorFact"
    return "ExperimentResultFact"


def _subject_for_row(row: dict[str, str], material_raw: str, regime_raw: str) -> dict[str, str]:
    return {
        "material": resolve_material(material_raw) if material_raw else "",
        "material_raw": clean_raw(material_raw),
        "process": resolve_regime(regime_raw) if regime_raw else "",
        "process_raw": clean_raw(regime_raw),
        "equipment": resolve_equipment(_pick(row, EQUIPMENT_COLUMNS)),
        "facility": clean_raw(_pick(row, FACILITY_COLUMNS)),
        "geography": clean_raw(_pick(row, GEOGRAPHY_COLUMNS)),
        "expert_or_lab": clean_raw(_pick(row, [*EMPLOYEE_COLUMNS, *LAB_COLUMNS, *TEAM_COLUMNS])),
    }


def _material_or_descriptor(row: dict[str, str]) -> str:
    return clean_raw(_pick(row, MATERIAL_COLUMNS) or _pick(row, DESCRIPTOR_COLUMNS))


def _assay_candidate_facts_from_row(
    row: dict[str, str],
    chunk: Chunk,
    source_name: str,
    evidence: EvidenceSpan,
    material_raw: str,
    regime_raw: str,
) -> list[CandidateFact]:
    subject = _subject_for_row(row, material_raw, regime_raw)
    result: list[CandidateFact] = []
    for column, raw_value in row.items():
        assay = _parse_assay_column(column, raw_value)
        if assay is None:
            continue
        analyte, unit, value, value_min, value_max = assay
        candidate_id = _candidate_id(
            "ProcessParameterFact",
            chunk.doc_id,
            chunk.chunk_id,
            subject,
            "содержание",
            analyte,
            raw_value,
            unit,
            evidence.source.row_index,
        )
        result.append(
            CandidateFact(
                candidate_id=candidate_id,
                fact_type="ProcessParameterFact",
                extractor_name=f"{TableExtractor.extractor_version}:structured_table_adapter",
                document_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                source_name=source_name,
                subject=subject,
                predicate="REPORTS_PROCESS_PARAMETER",
                object={
                    "property": "содержание",
                    "property_raw": column,
                    "parameter": "содержание",
                    "analyte": analyte,
                    "process": resolve_regime(regime_raw) if regime_raw else "",
                    "value_min": value_min,
                    "value_max": value_max,
                    "raw_value": clean_raw(raw_value),
                    "year": clean_raw(_pick(row, YEAR_COLUMNS)),
                    "source_note": clean_raw(_pick(row, SOURCE_COLUMNS)),
                },
                value=value,
                unit=unit,
                evidence_quote=evidence.quote,
                raw_span=f"{column}: {raw_value}",
                context_window=evidence.quote,
                confidence=0.84 if (subject.get("material_raw") or subject.get("process")) else 0.72,
                document_type="unknown",
            )
        )
    return result


def _wide_capacity_candidate_facts_from_row(
    row: dict[str, str],
    chunk: Chunk,
    source_name: str,
    evidence: EvidenceSpan,
    material_raw: str,
    regime_raw: str,
    table_context: str,
) -> list[CandidateFact]:
    text = " ".join([*row.keys(), *row.values(), source_name, table_context]).lower().replace("ё", "е")
    if not any(marker in text for marker in ["capacity", "production capacity", "throughput", "производительность", "мощность"]):
        return []
    descriptor = _capacity_descriptor(row)
    numeric_cells = _numeric_cells(row)
    if not descriptor or not numeric_cells:
        return []
    unit = _capacity_unit_from_table_context(text)
    subject = _subject_for_row(row, material_raw or _commodity_from_text(text), regime_raw)
    if not subject.get("material") and not subject.get("facility") and not subject.get("geography"):
        subject["material"] = resolve_material(_commodity_from_text(text))
        subject["material_raw"] = _commodity_from_text(text)
    result: list[CandidateFact] = []
    for key, raw_value, value in numeric_cells[:6]:
        obj = {
            "property": "производительность",
            "property_raw": descriptor,
            "parameter": "производительность",
            "metric": "capacity",
            "raw_value": clean_raw(raw_value),
            "year": _year_from_text(key) or clean_raw(_pick(row, YEAR_COLUMNS)),
            "source_note": clean_raw(_pick(row, SOURCE_COLUMNS)),
        }
        result.append(
            CandidateFact(
                candidate_id=_candidate_id("FacilityCapacityFact", chunk.doc_id, chunk.chunk_id, subject, descriptor, key, raw_value, unit, evidence.source.row_index),
                fact_type="FacilityCapacityFact",
                extractor_name=f"{TableExtractor.extractor_version}:structured_table_adapter",
                document_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                source_name=source_name,
                subject=subject,
                predicate="REPORTS_CAPACITY",
                object=obj,
                value=value,
                unit=unit,
                evidence_quote=evidence.quote,
                raw_span=f"{descriptor} | {key}: {raw_value}",
                context_window=table_context,
                confidence=0.8 if unit else 0.72,
                document_type="unknown",
            )
        )
    return result


def _expertise_candidate_from_row(
    row: dict[str, str],
    chunk: Chunk,
    source_name: str,
    evidence: EvidenceSpan,
    material_raw: str,
    regime_raw: str,
) -> CandidateFact | None:
    expert = clean_raw(_pick(row, EMPLOYEE_COLUMNS))
    lab = clean_raw(_pick(row, LAB_COLUMNS))
    team = clean_raw(_pick(row, TEAM_COLUMNS))
    topic = clean_raw(_pick(row, TOPIC_COLUMNS) or _pick(row, TOPIC_TAG_COLUMNS) or _pick(row, PROPERTY_COLUMNS))
    if not (expert or lab or team):
        return None
    if not (topic or regime_raw or material_raw):
        return None
    subject = _subject_for_row(row, material_raw, regime_raw)
    subject.update({"expert": expert, "laboratory": lab, "team": team})
    return CandidateFact(
        candidate_id=_candidate_id("ExpertiseFact", chunk.doc_id, chunk.chunk_id, subject, topic, evidence.source.row_index),
        fact_type="ExpertiseFact",
        extractor_name=f"{TableExtractor.extractor_version}:structured_table_adapter",
        document_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        source_name=source_name,
        subject=subject,
        predicate="HAS_EXPERTISE",
        object={
            "topic": topic,
            "process": resolve_regime(regime_raw) if regime_raw else "",
            "material": resolve_material(material_raw) if material_raw else "",
            "source_note": clean_raw(_pick(row, SOURCE_COLUMNS)),
        },
        evidence_quote=evidence.quote,
        raw_span=" | ".join(f"{key}: {value}" for key, value in row.items()),
        context_window=evidence.quote,
        confidence=0.82,
        document_type="unknown",
    )


def _publication_claim_candidate_from_row(
    row: dict[str, str],
    chunk: Chunk,
    source_name: str,
    evidence: EvidenceSpan,
    material_raw: str,
    regime_raw: str,
) -> CandidateFact | None:
    source_note = clean_raw(_pick(row, SOURCE_COLUMNS))
    claim = clean_raw(_pick(row, CLAIM_COLUMNS))
    topic = clean_raw(_pick(row, TOPIC_COLUMNS) or _pick(row, PROPERTY_COLUMNS) or _pick(row, TOPIC_TAG_COLUMNS))
    if not (claim or source_note):
        return None
    if not (topic or regime_raw or material_raw):
        return None
    return CandidateFact(
        candidate_id=_candidate_id("PublicationClaimFact", chunk.doc_id, chunk.chunk_id, source_note, claim, topic, evidence.source.row_index),
        fact_type="PublicationClaimFact",
        extractor_name=f"{TableExtractor.extractor_version}:structured_table_adapter",
        document_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        source_name=source_name,
        subject=_subject_for_row(row, material_raw, regime_raw),
        predicate="PUBLICATION_CLAIM",
        object={
            "topic": topic,
            "claim": claim or source_note,
            "source_note": source_note,
            "year": clean_raw(_pick(row, YEAR_COLUMNS)),
            "geography": clean_raw(_pick(row, GEOGRAPHY_COLUMNS)),
        },
        evidence_quote=evidence.quote,
        raw_span=" | ".join(f"{key}: {value}" for key, value in row.items()),
        context_window=evidence.quote,
        confidence=0.8 if claim else 0.74,
        document_type="unknown",
    )


def _technology_solution_candidate_from_row(
    row: dict[str, str],
    chunk: Chunk,
    source_name: str,
    evidence: EvidenceSpan,
    material_raw: str,
    regime_raw: str,
) -> CandidateFact | None:
    technology = clean_raw(_pick(row, TECHNOLOGY_COLUMNS) or regime_raw)
    target_problem = clean_raw(_pick(row, TARGET_PROBLEM_COLUMNS) or _pick(row, TOPIC_COLUMNS) or _pick(row, PROPERTY_COLUMNS))
    claim = clean_raw(_pick(row, CLAIM_COLUMNS) or _pick(row, CONCLUSION_COLUMNS) or _pick(row, SOURCE_COLUMNS))
    if not (claim or target_problem):
        return None
    context = " ".join([*row.keys(), *row.values(), evidence.quote]).lower().replace("ё", "е")
    has_marker = any(
        marker in context
        for marker in [
            "метод",
            "способ",
            "технолог",
            "схема",
            "система",
            "решение",
            "примен",
            "использ",
            "method",
            "technology",
            "approach",
            "system",
            "solution",
            "used for",
            "applied for",
            "designed for",
        ]
    )
    if not has_marker:
        return None
    if not (technology or target_problem or material_raw or regime_raw or _pick(row, EQUIPMENT_COLUMNS)):
        return None
    claim_text = claim or evidence.quote
    subject = _subject_for_row(row, material_raw, regime_raw)
    return CandidateFact(
        candidate_id=_candidate_id("TechnologySolutionFact", chunk.doc_id, chunk.chunk_id, technology, target_problem, claim_text, evidence.source.row_index),
        fact_type="TechnologySolutionFact",
        extractor_name=f"{TableExtractor.extractor_version}:structured_table_adapter",
        document_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        source_name=source_name,
        subject=subject,
        predicate="DESCRIBES_TECHNOLOGY_SOLUTION",
        object={
            "technology": technology or subject.get("process") or target_problem,
            "solution_name": technology or subject.get("process") or target_problem,
            "target_problem": target_problem,
            "process_context": subject.get("process") or "",
            "material": resolve_material(material_raw) if material_raw else "",
            "equipment": subject.get("equipment") or "",
            "applicable_conditions": clean_raw(_pick(row, ["condition", "условие", "conditions", "условия"])),
            "claim": claim_text,
            "source_note": clean_raw(_pick(row, SOURCE_COLUMNS)),
            "year": clean_raw(_pick(row, YEAR_COLUMNS)),
            "geography": clean_raw(_pick(row, GEOGRAPHY_COLUMNS)),
        },
        evidence_quote=evidence.quote,
        raw_span=" | ".join(f"{key}: {value}" for key, value in row.items()),
        context_window=evidence.quote,
        confidence=0.82 if (technology and (target_problem or material_raw or regime_raw)) else 0.76,
        document_type="unknown",
    )


def _parse_assay_column(column: str, raw_value: str) -> tuple[str, str, float, float | None, float | None] | None:
    header = clean_raw(column)
    value = clean_raw(raw_value)
    if not value:
        return None
    match = re.fullmatch(
        r"(?P<name>[A-Za-zА-Яа-яЁё0-9+\-]+)\s*,?\s*(?P<unit>%|г/т|g/t|ppm|mg/kg)?",
        header,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    analyte = match.group("name").strip()
    unit = resolve_unit(match.group("unit") or "")
    if unit not in {"%", "г/т", "g/t", "ppm", "mg/kg"}:
        return None
    if not _looks_like_assay_analyte(analyte):
        return None
    value_min, value_max = _numeric_range(value)
    if value_min is None:
        return None
    return analyte, unit, value_min, value_min, value_max


def _row_unit(row: dict[str, str], property_raw: str, value_raw: str, context: str = "") -> str | None:
    explicit = resolve_unit(_pick(row, UNIT_COLUMNS))
    if explicit:
        return explicit
    combined = " ".join([property_raw or "", value_raw or "", context or "", *row.keys()])
    return _capacity_unit_from_table_context(combined) or resolve_unit(_unit_token_from_text(combined))


def _table_context_text(row: dict[str, str], chunk: Chunk, evidence: EvidenceSpan) -> str:
    metadata = chunk.metadata or {}
    metadata_keys = [
        "table_caption",
        "caption",
        "table_title",
        "title",
        "section_title",
        "previous_paragraph",
        "next_paragraph",
        "before_table",
        "after_table",
        "table_context",
        "source_name",
        "filename",
    ]
    metadata_parts = [str(metadata.get(key) or "") for key in metadata_keys]
    return " ".join(
        part
        for part in [
            evidence.quote,
            chunk.section_path or "",
            " ".join(row.keys()),
            *metadata_parts,
        ]
        if part
    )


def _property_for_unit(property_raw: str, unit: str | None) -> str:
    canonical = resolve_property(property_raw)
    if canonical == "расход" and unit == "m/s":
        return "скорость потока"
    if canonical == "скорость потока" and unit in {"m3/h", "t/day"}:
        return "расход"
    return canonical


def _unit_token_from_text(text: str) -> str:
    match = re.search(
        r"(мг\s*/\s*(?:л|дм[³3])|mg\s*/\s*(?:l|dm3)|г\s*/\s*л|g\s*/\s*l|м\s*/\s*с|m\s*/\s*s|"
        r"м3\s*/\s*ч|m3\s*/\s*h|т\s*/\s*сут|t\s*/\s*(?:day|d)|т\s*/\s*год|t\s*/\s*(?:y|year)|"
        r"кт\s*/\s*год|тыс\.?\s*т\s*/\s*год|ktpa|kt\s*/\s*(?:y|year)|млн\.?\s*т\s*/\s*год|mtpa|mt\s*/\s*(?:y|year)|"
        r"руб\.?\s*/\s*т|RUB\s*/\s*t|USD\s*/\s*t|EUR\s*/\s*t|€\s*/\s*t|\$\s*/\s*t|%|ppm|г/т|g/t)",
        text or "",
        flags=re.IGNORECASE,
    )
    return match.group(0) if match else ""


def _capacity_unit_from_text(text: str) -> str | None:
    token = _unit_token_from_text(text)
    unit = resolve_unit(token)
    return unit if unit in {"t/day", "t/y", "kt/y", "Mt/y", "m3/h", "%"} else None


def _capacity_unit_from_table_context(text: str) -> str | None:
    unit = _capacity_unit_from_text(text)
    if not unit:
        return None
    norm = str(text or "").lower().replace("ё", "е")
    capacity_markers = ["capacity", "production capacity", "throughput", "производительность", "мощность"]
    if any(marker in norm for marker in capacity_markers):
        return unit
    return None


def _capacity_descriptor(row: dict[str, str]) -> str:
    for value in row.values():
        clean = clean_raw(value)
        norm = clean.lower().replace("ё", "е")
        if any(marker in norm for marker in ["capacity", "production capacity", "throughput", "производительность", "мощность"]):
            return clean
    for key in row:
        norm = key.lower().replace("ё", "е")
        if any(marker in norm for marker in ["capacity", "production capacity", "throughput", "производительность", "мощность"]):
            return key
    return ""


def _numeric_cells(row: dict[str, str]) -> list[tuple[str, str, float]]:
    cells: list[tuple[str, str, float]] = []
    for key, raw in row.items():
        value = _float_or_none(raw)
        if value is None:
            continue
        cells.append((key, raw, value))
    return cells


def _commodity_from_text(text: str) -> str:
    norm = str(text or "").lower().replace("ё", "е")
    for raw in ["copper", "медь", "nickel", "никель", "gold", "золото", "silver", "серебро"]:
        if raw in norm:
            return raw
    return ""


def _year_from_text(text: str) -> str:
    match = re.search(r"\b(?:19|20)\d{2}\b", str(text or ""))
    return match.group(0) if match else ""


def _looks_like_assay_analyte(value: str) -> bool:
    norm = value.lower().replace("ё", "е")
    if norm in {"ni", "cu", "fe", "s", "au", "ag", "мпг", "pgm", "pt", "pd", "rh", "co", "zn", "pb"}:
        return True
    return bool(re.fullmatch(r"[A-Z][a-z]?\d*(?:O\d*)?", value))


def _numeric_range(value: str) -> tuple[float | None, float | None]:
    numbers = re.findall(r"\d+(?:[\.,]\d+)?", value)
    if not numbers:
        return None, None
    parsed = [float(item.replace(",", ".")) for item in numbers[:2]]
    if len(parsed) == 1:
        return parsed[0], None
    return min(parsed[0], parsed[1]), max(parsed[0], parsed[1])


def _predicate_for_fact_type(fact_type: str) -> str:
    return {
        "FacilityCapacityFact": "REPORTS_CAPACITY",
        "EconomicIndicatorFact": "REPORTS_ECONOMIC_INDICATOR",
        "ProcessParameterFact": "REPORTS_PROCESS_PARAMETER",
    }.get(fact_type, "REPORTS_TABLE_RESULT")


def _candidate_confidence(
    fact_type: str,
    property_raw: str,
    value_raw: str,
    regime_raw: str,
    material_raw: str,
    row: dict[str, str],
) -> float:
    if fact_type == "FacilityCapacityFact" and value_raw and (
        material_raw or _pick(row, FACILITY_COLUMNS) or _pick(row, GEOGRAPHY_COLUMNS)
    ):
        return 0.82
    if fact_type == "EconomicIndicatorFact" and value_raw and _pick(row, UNIT_COLUMNS):
        return 0.82
    if fact_type == "ProcessParameterFact" and property_raw and value_raw and (regime_raw or material_raw):
        return 0.82
    if property_raw and (value_raw or regime_raw or material_raw):
        return 0.82
    return 0.58


def _candidate_id(*parts) -> str:
    raw = "|".join(str(part) for part in parts)
    return "table_cand_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _dedupe_candidate_facts(items: list[CandidateFact]) -> list[CandidateFact]:
    seen = set()
    result: list[CandidateFact] = []
    for item in items:
        if item.candidate_id in seen:
            continue
        seen.add(item.candidate_id)
        result.append(item)
    return result

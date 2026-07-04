"""Deterministic extraction adapter producing typed ExtractionBundle objects."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ..models.schemas import Chunk
from ..domain.fact_normalization import canonical_fact_key
from .confidence import experiment_confidence
from .extraction import EntityRelationExtractor
from .models import (
    CandidateFact,
    EvidenceSpan,
    ExtractedDataGap,
    ExtractedEntity,
    ExtractedExperiment,
    ExtractedMeasurement,
    ExtractedRegime,
    ExtractionBundle,
    ExtractionSource,
)
from .resolver import resolve_equipment, resolve_material, resolve_property, resolve_regime, resolve_unit


class DeterministicExtractor:
    """Wrap the existing rule-based extractor in the typed extraction contract."""

    extractor_version = "deterministic_v2"

    def __init__(self, legacy_extractor: EntityRelationExtractor | None = None) -> None:
        self.legacy_extractor = legacy_extractor or EntityRelationExtractor()

    def extract_from_chunk(self, chunk: Chunk) -> ExtractionBundle:
        chunks = self._split_experiment_segments(chunk)
        bundle = ExtractionBundle(
            document_id=chunk.doc_id,
            source_name=str(chunk.metadata.get("source_name") or chunk.metadata.get("filename") or chunk.doc_id),
            extractor_version=self.extractor_version,
            diagnostics={"segments": len(chunks), "ambiguous_multiple_experiment_ids": len(chunks) > 1},
        )
        for segment in chunks:
            partial = self._extract_single_chunk(segment)
            bundle.entities.extend(partial.entities)
            bundle.experiments.extend(partial.experiments)
            bundle.data_gaps.extend(partial.data_gaps)
            bundle.rejected_items.extend(partial.rejected_items)
            bundle.candidate_facts.extend(partial.candidate_facts)
        bundle.entities = _dedupe_entities(bundle.entities)
        bundle.experiments = _dedupe_experiments(bundle.experiments)
        bundle.data_gaps = _dedupe_gaps(bundle.data_gaps)
        return bundle

    def _extract_single_chunk(self, chunk: Chunk) -> ExtractionBundle:
        extraction = self.legacy_extractor.extract_from_chunk(chunk)
        source = _source_from_chunk(chunk)
        default_evidence = EvidenceSpan(source=source, quote=(chunk.text or "")[:700], confidence=0.8)
        prop_by_value: dict[str, str] = {}
        change_by_value: dict[str, str] = {}
        for rel in extraction.relations:
            if rel.predicate == "OF_PROPERTY":
                prop_by_value[rel.subject] = resolve_property(rel.object)
            elif rel.predicate == "HAS_CHANGE":
                change_by_value[rel.subject] = _normalize_effect(str((rel.qualifiers or {}).get("direction") or rel.object))

        entities: list[ExtractedEntity] = []
        for entity in extraction.entities:
            converted = _entity_from_legacy(entity, default_evidence)
            if converted is not None:
                entities.append(converted)

        experiment_map: dict[str, dict[str, Any]] = {}
        data_gaps: list[ExtractedDataGap] = []
        for rel in extraction.relations:
            evidence = _evidence_from_relation(rel, default_evidence)
            if rel.predicate == "MISSING_FOR":
                data_gaps.append(_gap_from_text(rel.subject, rel.object, evidence))
                continue
            if rel.predicate not in {"STUDIES", "USES_REGIME", "MEASURES", "USES_EQUIPMENT", "PERFORMED_BY"}:
                continue
            exp = experiment_map.setdefault(
                rel.subject,
                {"materials": [], "regimes": [], "measurements": [], "equipment": [], "laboratories": [], "evidence": [], "conclusions": []},
            )
            exp["evidence"].append(evidence)
            if rel.predicate == "STUDIES":
                if _is_experiment_identifier(rel.object):
                    continue
                exp["materials"].append(_entity("Material", rel.object, resolve_material(rel.object), evidence, 0.82))
            elif rel.predicate == "USES_REGIME":
                exp["regimes"].append(ExtractedRegime(raw_name=rel.object, canonical_name=resolve_regime(rel.object), confidence=0.78, evidence=[evidence]))
            elif rel.predicate == "USES_EQUIPMENT":
                exp["equipment"].append(_entity("Equipment", rel.object, rel.object, evidence, 0.72))
            elif rel.predicate == "PERFORMED_BY":
                if _is_team_mention(rel.object, evidence.quote):
                    exp.setdefault("teams", []).append(_entity("ResearchTeam", rel.object, rel.object, evidence, 0.72))
                else:
                    exp["laboratories"].append(_entity("Laboratory", rel.object, rel.object, evidence, 0.72))
            elif rel.predicate == "MEASURES":
                qualifiers = rel.qualifiers or {}
                property_name = prop_by_value.get(rel.object) or resolve_property(rel.object)
                if not property_name:
                    continue
                exp["measurements"].append(
                    ExtractedMeasurement(
                        property_raw=rel.object,
                        property_canonical=property_name,
                        value=_float_or_none(qualifiers.get("value")),
                        unit=qualifiers.get("unit"),
                        effect=_normalize_effect(change_by_value.get(rel.object) or qualifiers.get("direction")),
                        confidence=rel.confidence,
                        evidence=[evidence],
                    )
                )

        conclusion_entities = [entity.canonical_name for entity in extraction.entities if entity.entity_type == "Conclusion"]
        experiments: list[ExtractedExperiment] = []
        for experiment_id, data in experiment_map.items():
            experiment = ExtractedExperiment(
                experiment_id=experiment_id,
                materials=_dedupe_entities(data["materials"]),
                regimes=_dedupe_regimes(data["regimes"]),
                measurements=_dedupe_measurements(data["measurements"]),
                equipment=_dedupe_entities(data["equipment"]),
                laboratories=_dedupe_entities(data["laboratories"]),
                teams=_dedupe_entities(data.get("teams", [])),
                conclusions=list(dict.fromkeys(conclusion_entities)),
                evidence=_dedupe_evidence(data["evidence"] or [default_evidence]),
                confidence=0.0,
            )
            confidence = experiment_confidence(experiment, ambiguous=False)
            experiment = experiment.model_copy(update={"confidence": confidence}) if hasattr(experiment, "model_copy") else experiment.copy(update={"confidence": confidence})
            experiments.append(experiment)

        for entity in extraction.entities:
            if entity.entity_type == "DataGap":
                data_gaps.append(_gap_from_text(entity.canonical_name, None, default_evidence))

        pattern_bundle = _extract_direct_text_patterns(chunk, default_evidence)
        entities.extend(pattern_bundle.entities)
        experiments.extend(pattern_bundle.experiments)
        data_gaps.extend(pattern_bundle.data_gaps)
        candidate_facts = list(pattern_bundle.candidate_facts)

        return ExtractionBundle(
            document_id=chunk.doc_id,
            source_name=str(chunk.metadata.get("source_name") or chunk.metadata.get("filename") or chunk.doc_id),
            extractor_version=self.extractor_version,
            entities=_dedupe_entities(entities),
            experiments=_dedupe_experiments(experiments),
            data_gaps=_dedupe_gaps(data_gaps),
            candidate_facts=candidate_facts,
            diagnostics={"chunk_id": chunk.chunk_id},
        )

    @staticmethod
    def _split_experiment_segments(chunk: Chunk) -> list[Chunk]:
        text = chunk.text or ""
        marker_pattern = (
            r"(?:эксперимент|experiment)\s+[A-ZА-Я0-9_.-]+\s*:"
            r"|(?:experiment_id|experiment\s+id|id\s+эксперимента)\s*[:=]"
        )
        markers = re.findall(marker_pattern, text, flags=re.IGNORECASE)
        if len(markers) <= 1:
            return [chunk]
        segments = [
            part.strip(" .;\n\t")
            for part in re.split(rf"(?={marker_pattern})", text, flags=re.IGNORECASE)
            if part.strip(" .;\n\t")
        ]
        if len(segments) <= 1:
            segments = [line.strip() for line in text.splitlines() if line.strip()]
        result: list[Chunk] = []
        for idx, segment in enumerate(segments):
            if not segment:
                continue
            update = {
                "chunk_id": f"{chunk.chunk_id}:seg{idx}",
                "text": segment,
                "ordinal": (chunk.ordinal or 0) * 1000 + idx,
                "metadata": {**(chunk.metadata or {}), "parent_chunk_id": chunk.chunk_id, "segment_id": idx},
            }
            result.append(chunk.model_copy(update=update) if hasattr(chunk, "model_copy") else chunk.copy(update=update))
        return result or [chunk]


_EXPERIMENT_ID_RE = re.compile(r"\b(?:E\d+|EXP-[A-ZА-Я0-9_.-]+)\b", re.IGNORECASE)
_PERSON_RE = re.compile(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.")
_LAB_RE = re.compile(r"\b(?:в\s+)?(лаборатори[ия]\s+[^;,.]+)", re.IGNORECASE)
_TEAM_RE = re.compile(
    r"\b(?:команд[аы]|групп[аы]|research\s+team|team)\s*[:\-]?\s*"
    r"(?P<name>[A-Za-zА-Яа-яЁё0-9_.\- ]{2,70}?)"
    r"(?=\s+(?:выполн\w*|провод\w*|использ\w*|работал\w*|работала\w*|in\s+laboratory|used|using|worked)\b|;|\.|\n|$)",
    re.IGNORECASE,
)
_TOPIC_RE = re.compile(
    r"\b(?:тем(?:а|атика)|topic|tag|тег)\s*[:\-]\s*"
    r"(?P<name>[^;.\n|]{2,90})",
    re.IGNORECASE,
)
_EQUIPMENT_LABEL_RE = re.compile(
    r"\b(?:оборудование|установка|прибор|equipment|device|stand)\s*[:\-]\s*"
    r"(?P<name>[^;.\n|]{2,90})",
    re.IGNORECASE,
)
_EQUIPMENT_CANDIDATES = [
    "ванна электроэкстракции",
    "ванны электроэкстракции",
    "diaphragm cell",
    "диафрагменная ячейка",
    "диафрагменные ячейки",
    "печь взвешенной плавки",
    "ПВП",
    "flash smelting furnace",
    "fluidized bed furnace",
    "система очистки газов",
    "системы очистки газов",
    "газоочистная установка",
    "reactor",
    "furnace",
    "microscope",
    "твердомер",
    "микроскоп",
]
_MATERIAL_CANDIDATES = [
    "ВТ6",
    "VT6",
    "Ti-6Al-4V",
    "7075-T6",
    "7075",
    "12Х18Н10Т",
    "09Г2С",
    "никель",
    "никеля",
    "никелевым",
    "медь",
    "медный",
    "медным",
    "медного",
    "сульфаты",
    "сульфат",
    "SO2",
    "SO₂",
    "диоксид серы",
    "sulfur dioxide",
    "хлориды",
    "хлорид",
    "Ca",
    "Mg",
    "Na",
    "Au",
    "Ag",
    "МПГ",
    "штейн",
    "шлак",
    "шахтные воды",
    "шахтных вод",
    "католит",
    "электролит",
]
_REGIME_CANDIDATES = [
    "отжиг",
    "старение",
    "закалка",
    "криообработка",
    "термообработка",
    "annealing",
    "annealed",
    "aging",
    "aged",
    "quenching",
    "heat treatment",
    "обессоливание",
    "обессоливания",
    "desalination",
    "водоочистка",
    "электроэкстракция",
    "электроэкстракции",
    "электролиз",
    "электролиза",
    "electrowinning",
    "electroextraction",
    "electrolysis",
    "циркуляция католита",
    "catholyte circulation",
    "кучное выщелачивание",
    "heap leaching",
    "выщелачивание",
    "leaching",
    "закачка шахтных вод",
    "закачки шахтных вод",
    "deep well injection",
    "пирометаллургия",
    "пирометаллургическом",
    "pyrometallurgy",
    "ПВП",
    "печь взвешенной плавки",
    "flash smelting furnace",
    "fluidized bed furnace",
    "газоочистка",
    "gas cleaning",
    "удаление SO2",
    "удаление диоксида серы",
    "SO2 removal",
    "sulfur dioxide removal",
]


def _extract_direct_text_patterns(chunk: Chunk, evidence: EvidenceSpan) -> ExtractionBundle:
    """Extract common Russian materials-science phrasings missed by legacy rules."""
    text = chunk.text or ""
    sanitized = _EXPERIMENT_ID_RE.sub(" ", text)
    materials = _extract_materials(sanitized, evidence)
    regimes = _extract_regimes(sanitized, evidence)
    measurements = _extract_measurements(text, evidence)
    laboratories = [_entity("Laboratory", match.group(1), match.group(1).strip(), evidence, 0.74) for match in _LAB_RE.finditer(text)]
    employees = [_entity("Employee", match.group(0), match.group(0).strip(), evidence, 0.72) for match in _PERSON_RE.finditer(text)]
    equipment = _extract_equipment(text, evidence)
    teams = _extract_teams(text, evidence)
    topic_tags = _extract_topic_tags(text, evidence)
    gaps = _extract_gap_patterns(text, evidence)
    source_candidates = _extract_source_adapter_candidates(text, evidence)
    entities: list[ExtractedEntity] = [
        *materials,
        *[_entity("ProcessRegime", regime.raw_name, regime.canonical_name, evidence, regime.confidence) for regime in regimes],
        *[_entity("Property", measurement.property_raw, measurement.property_canonical, evidence, measurement.confidence) for measurement in measurements],
        *equipment,
        *laboratories,
        *teams,
        *employees,
        *topic_tags,
    ]
    if not materials or not (regimes or measurements):
        return ExtractionBundle(
            document_id=evidence.source.document_id,
            source_name=evidence.source.source_name,
            extractor_version=DeterministicPatternVersion.VALUE,
            entities=_dedupe_entities(entities),
            data_gaps=gaps,
            candidate_facts=source_candidates,
            diagnostics={"direct_patterns": True, "experiment_created": False},
        )

    experiment_id = _extract_experiment_id(text, chunk)
    measurements = _apply_binding_guard(text, materials, regimes, measurements)
    experiment = ExtractedExperiment(
        experiment_id=experiment_id,
        materials=_dedupe_entities(materials),
        regimes=_dedupe_regimes(regimes),
        measurements=_dedupe_measurements(measurements),
        equipment=_dedupe_entities(equipment),
        laboratories=_dedupe_entities(laboratories),
        teams=_dedupe_entities(teams),
        employees=_dedupe_entities(employees),
        topic_tags=_dedupe_entities(topic_tags),
        evidence=[evidence],
        confidence=0.0,
    )
    confidence = experiment_confidence(experiment)
    experiment = experiment.model_copy(update={"confidence": confidence}) if hasattr(experiment, "model_copy") else experiment.copy(update={"confidence": confidence})
    return ExtractionBundle(
        document_id=evidence.source.document_id,
        source_name=evidence.source.source_name,
        extractor_version=DeterministicPatternVersion.VALUE,
        entities=_dedupe_entities(entities),
        experiments=[experiment],
        data_gaps=gaps,
        candidate_facts=source_candidates,
        diagnostics={"direct_patterns": True, "experiment_created": True},
    )


class DeterministicPatternVersion:
    VALUE = "deterministic_text_patterns_v1"


def _extract_materials(text: str, evidence: EvidenceSpan) -> list[ExtractedEntity]:
    result: list[ExtractedEntity] = []
    lowered = text.lower()
    for raw in _MATERIAL_CANDIDATES:
        if not _material_candidate_in_text(raw, lowered):
            continue
        canonical = resolve_material(raw)
        if canonical and not _is_experiment_identifier(raw):
            result.append(_entity("Material", raw, canonical, evidence, 0.84))
    return _dedupe_entities(result)


def _material_candidate_in_text(raw: str, lowered_text: str) -> bool:
    candidate = raw.lower()
    if candidate in {"ca", "mg", "na", "au", "ag"}:
        return bool(re.search(rf"(?<![A-Za-zА-Яа-я0-9]){re.escape(candidate)}(?![A-Za-zА-Яа-я0-9])", lowered_text, flags=re.IGNORECASE))
    return candidate in lowered_text


def _extract_equipment(text: str, evidence: EvidenceSpan) -> list[ExtractedEntity]:
    result: list[ExtractedEntity] = []
    lowered = text.lower()
    for raw in _EQUIPMENT_CANDIDATES:
        if raw.lower() in lowered:
            result.append(_entity("Equipment", raw, _canonical_equipment(raw), evidence, 0.76))
    for match in _EQUIPMENT_LABEL_RE.finditer(text):
        name = _clean_named_entity(match.group("name"))
        if name:
            result.append(_entity("Equipment", name, _canonical_equipment(name), evidence, 0.72))
    return _dedupe_entities(result)


def _extract_teams(text: str, evidence: EvidenceSpan) -> list[ExtractedEntity]:
    result: list[ExtractedEntity] = []
    for match in _TEAM_RE.finditer(text):
        name = _clean_named_entity(match.group("name"))
        if name:
            result.append(_entity("ResearchTeam", name, name, evidence, 0.72))
    return _dedupe_entities(result)


def _extract_topic_tags(text: str, evidence: EvidenceSpan) -> list[ExtractedEntity]:
    result: list[ExtractedEntity] = []
    for match in _TOPIC_RE.finditer(text):
        for item in re.split(r"\s*(?:,|/|\||;|\bи\b|\band\b)\s*", match.group("name"), flags=re.IGNORECASE):
            name = _clean_named_entity(item)
            if name:
                result.append(_entity("TopicTag", name, name, evidence, 0.68))
    return _dedupe_entities(result)


def _canonical_equipment(raw: str) -> str:
    value = _clean_named_entity(raw)
    normalized = value.lower().replace("ё", "е")
    aliases = {
        "пвп": "печь взвешенной плавки",
        "fluidized bed furnace": "печь взвешенной плавки",
        "flash smelting furnace": "печь взвешенной плавки",
        "ванны электроэкстракции": "ванна электроэкстракции",
        "диафрагменные ячейки": "диафрагменная ячейка",
        "системы очистки газов": "система очистки газов",
    }
    return aliases.get(normalized, value)


def _clean_named_entity(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" .;|\"'«»"))


def _extract_regimes(text: str, evidence: EvidenceSpan) -> list[ExtractedRegime]:
    result: list[ExtractedRegime] = []
    lowered = text.lower().replace("ё", "е")
    temp_match = re.search(r"(\d+(?:[,.]\d+)?)\s*(?:°\s*)?[CСс]", text)
    duration_match = re.search(r"(\d+(?:[,.]\d+)?)\s*(?:ч|h|час)", text, flags=re.IGNORECASE)
    for raw in _REGIME_CANDIDATES:
        if raw.lower().replace("ё", "е") not in lowered:
            continue
        result.append(
            ExtractedRegime(
                raw_name=raw,
                canonical_name=resolve_regime(raw),
                temperature=_float_or_none(temp_match.group(1)) if temp_match else None,
                temperature_unit="C" if temp_match else None,
                duration=_float_or_none(duration_match.group(1)) if duration_match else None,
                duration_unit="h" if duration_match else None,
                confidence=0.82,
                evidence=[evidence],
            )
        )
    return _dedupe_regimes(result)


def _extract_measurements(text: str, evidence: EvidenceSpan) -> list[ExtractedMeasurement]:
    measurements: list[ExtractedMeasurement] = []
    strength_pattern = re.compile(
        r"(?:предел\s+прочности\s*(?:σв|sigma_b)?|прочност[ьи])"
        r"[^.;,\n]{0,80}?(увелич\w+\s+до\s+|составил\w*\s+|=|:)?"
        r"(\d+(?:[,.]\d+)?)\s*(МПа|MPa|мПа|ГПа|GPa|ksi)",
        re.IGNORECASE,
    )
    for match in strength_pattern.finditer(text):
        window = match.group(0).lower()
        measurements.append(
            ExtractedMeasurement(
                property_raw="предел прочности",
                property_canonical=resolve_property("прочность"),
                value=_float_or_none(match.group(2)),
                unit=resolve_unit(match.group(3)),
                effect="increase" if "увелич" in window else "unknown",
                confidence=0.88,
                evidence=[evidence],
            )
        )

    english_strength_pattern = re.compile(
        r"(?:ultimate\s+tensile\s+strength|tensile\s+strength|strength)"
        r"[^.;,\n]{0,80}?(?:of\s+|=|:|reached\s+|was\s+|is\s+)?"
        r"(\d+(?:[,.]\d+)?)\s*(ksi|MPa|GPa)",
        re.IGNORECASE,
    )
    for match in english_strength_pattern.finditer(text):
        measurements.append(
            ExtractedMeasurement(
                property_raw="tensile strength",
                property_canonical=resolve_property("tensile strength"),
                value=_float_or_none(match.group(1)),
                unit=resolve_unit(match.group(2)),
                effect=_normalize_effect(_qualitative_effect_near(text, match.start(), match.end())),
                confidence=0.9,
                evidence=[evidence],
            )
        )

    corrosion_effect_pattern = re.compile(
        r"(коррозионн\w+\s+стойк\w+|corrosion\s+resistance)[^.;\n]{0,80}?"
        r"(повыс\w+|увелич\w+|increase\w*|сниз\w+|уменьш\w+|decrease\w*)",
        re.IGNORECASE,
    )
    for match in corrosion_effect_pattern.finditer(text):
        measurements.append(
            ExtractedMeasurement(
                property_raw=match.group(1),
                property_canonical=resolve_property(match.group(1)),
                value=None,
                unit=None,
                effect=_normalize_effect(match.group(2)),
                confidence=0.72,
                evidence=[evidence],
            )
        )

    ductility_pattern = re.compile(r"(?:относительное\s+)?удлинение[^.;,\n]{0,40}?(\d+(?:[,.]\d+)?)\s*%", re.IGNORECASE)
    for match in ductility_pattern.finditer(text):
        measurements.append(
            ExtractedMeasurement(
                property_raw="удлинение",
                property_canonical=resolve_property("пластичность"),
                value=_float_or_none(match.group(1)),
                unit="%",
                confidence=0.86,
                evidence=[evidence],
            )
        )

    hardness_patterns = [
        re.compile(r"(?:тв[её]рдость[^.;,\n]{0,25}?)?(HV|HRC)\s*(\d+(?:[,.]\d+)?)", re.IGNORECASE),
        re.compile(r"тв[её]рдость[^.;,\n]{0,30}?(\d+(?:[,.]\d+)?)\s*(HV|HRC)", re.IGNORECASE),
    ]
    for pattern in hardness_patterns:
        for match in pattern.finditer(text):
            unit, value = (match.group(1), match.group(2)) if match.group(1).upper() in {"HV", "HRC"} else (match.group(2), match.group(1))
            measurements.append(
                ExtractedMeasurement(
                    property_raw="твёрдость",
                    property_canonical=resolve_property("твёрдость"),
                    value=_float_or_none(value),
                    unit=resolve_unit(unit),
                    confidence=0.86,
                    evidence=[evidence],
                )
            )
    measurements.extend(_extract_process_parameter_measurements(text, evidence))
    return _dedupe_measurements(measurements)


_NUMERIC_UNIT = (
    r"(?P<op><=|>=|≤|≥|<|>|=|до|от|не\s+более|не\s+менее)?\s*"
    r"(?P<value>\d+(?:[,.]\d+)?)"
    r"(?:\s*[-–]\s*\d+(?:[,.]\d+)?)?\s*"
    r"(?P<unit>мг/л|мг/дм3|мг/дм³|mg/l|mg/dm3|mg/dm³|г/л|g/l|м/с|m/s|м3/ч|м³/ч|m3/h|m³/h|т/сут|т/сутки|t/day|t/d|%|ppm)"
)
_ECONOMIC_UNIT = (
    r"(?P<op><=|>=|≤|≥|<|>|=|до|от|не\s+более|не\s+менее)?\s*"
    r"(?P<value>\d+(?:[,.]\d+)?)\s*"
    r"(?P<unit>USD/t|\$/t|usd/t|RUB/t|rub/t|руб/т|руб\.?/т|руб/м3|руб/м³|RUB/m3|rub/m3|USD/m3|usd/m3|млн\s*руб(?:/год)?|млн\s*RUB(?:/year)?)"
)

_PROCESS_PARAMETER_PATTERNS: list[tuple[re.Pattern[str], str, str, float]] = [
    (
        re.compile(rf"(?:концентрац\w+\s+)?(?P<raw>сульфат\w*|sulfates?|sulphates?|so4)[^.;\n]{{0,60}}?{_NUMERIC_UNIT}", re.IGNORECASE),
        "концентрация",
        "сульфаты",
        0.83,
    ),
    (
        re.compile(rf"(?:концентрац\w+\s+)?(?P<raw>хлорид\w*|chlorides?|cl)[^.;\n]{{0,60}}?{_NUMERIC_UNIT}", re.IGNORECASE),
        "концентрация",
        "хлориды",
        0.83,
    ),
    (
        re.compile(rf"(?:концентрац\w+\s+)?(?P<raw>Ca|кальци\w+|calcium)[^.;\n]{{0,50}}?{_NUMERIC_UNIT}", re.IGNORECASE),
        "концентрация",
        "Ca",
        0.8,
    ),
    (
        re.compile(rf"(?:концентрац\w+\s+)?(?P<raw>Mg|магни\w+|magnesium)[^.;\n]{{0,50}}?{_NUMERIC_UNIT}", re.IGNORECASE),
        "концентрация",
        "Mg",
        0.8,
    ),
    (
        re.compile(rf"(?:концентрац\w+\s+)?(?P<raw>Na|натри\w+|sodium)[^.;\n]{{0,50}}?{_NUMERIC_UNIT}", re.IGNORECASE),
        "концентрация",
        "Na",
        0.8,
    ),
    (
        re.compile(rf"(?P<raw>сухой\s+остаток|total\s+dissolved\s+solids|tds)[^.;\n]{{0,70}}?{_NUMERIC_UNIT}", re.IGNORECASE),
        "сухой остаток",
        "сухой остаток",
        0.84,
    ),
    (
        re.compile(rf"(?P<raw>скорост\w+\s+(?:поток\w+|циркуляц\w+)|flow\s+(?:velocity|rate)|circulation\s+rate)[^.;\n]{{0,70}}?{_NUMERIC_UNIT}", re.IGNORECASE),
        "скорость потока",
        "скорость потока",
        0.82,
    ),
    (
        re.compile(rf"(?P<raw>производительност\w*|capacity|throughput)[^.;\n]{{0,70}}?{_NUMERIC_UNIT}", re.IGNORECASE),
        "производительность",
        "производительность",
        0.8,
    ),
    (
        re.compile(rf"(?P<raw>извлечени\w+|recovery)[^.;\n]{{0,70}}?{_NUMERIC_UNIT}", re.IGNORECASE),
        "извлечение",
        "извлечение",
        0.82,
    ),
    (
        re.compile(rf"(?P<raw>выход\s+металл\w+|metal\s+yield|yield)[^.;\n]{{0,70}}?{_NUMERIC_UNIT}", re.IGNORECASE),
        "выход металла",
        "выход металла",
        0.8,
    ),
    (
        re.compile(
            rf"(?P<raw>распределени\w+|коэффициент\s+распределени\w+|distribution(?:\s+coefficient)?)"
            rf"[^.;\n]{{0,90}}?{_NUMERIC_UNIT}",
            re.IGNORECASE,
        ),
        "распределение",
        "распределение",
        0.81,
    ),
    (
        re.compile(
            rf"(?P<raw>CAPEX|OPEX|капитальн\w+\s+затрат\w+|эксплуатационн\w+\s+затрат\w+|"
            rf"технико-экономическ\w+\s+показател\w+|экономическ\w+\s+показател\w+|стоимост\w+)"
            rf"[^.;,\n]{{0,45}}?{_ECONOMIC_UNIT}",
            re.IGNORECASE,
        ),
        "экономический показатель",
        "экономический показатель",
        0.76,
    ),
]


def _extract_process_parameter_measurements(text: str, evidence: EvidenceSpan) -> list[ExtractedMeasurement]:
    measurements: list[ExtractedMeasurement] = []
    for pattern, property_name, raw_fallback, confidence in _PROCESS_PARAMETER_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.groupdict().get("raw") or raw_fallback
            unit = resolve_unit(match.group("unit"))
            value = _float_or_none(match.group("value"))
            if value is None or not unit:
                continue
            measurements.append(
                ExtractedMeasurement(
                    property_raw=raw,
                    property_canonical=resolve_property(property_name),
                    value=value,
                    unit=unit,
                    effect="unknown",
                    confidence=confidence,
                    evidence=[evidence],
                )
            )
    return measurements


def _extract_gap_patterns(text: str, evidence: EvidenceSpan) -> list[ExtractedDataGap]:
    gaps: list[ExtractedDataGap] = []
    if re.search(r"коррозионн\w+\s+стойк\w+[^.;\n]{0,40}не\s+измер", text, flags=re.IGNORECASE):
        gaps.append(_gap_from_text("коррозионная стойкость не измерялась", "коррозионная стойкость", evidence))
    if re.search(r"(?:необходим\w+|нужн\w+|требу\w+)[^.;\n]{0,80}дополнительн\w+\s+данн\w+[^.;\n]{0,60}коррозионн\w+\s+стойк", text, flags=re.IGNORECASE):
        gaps.append(_gap_from_text("нужны дополнительные данные по коррозионной стойкости", "коррозионная стойкость", evidence))
    if re.search(
        r"(corrosion\s+resistance|corrosion\s+data)[^.;\n]{0,90}?"
        r"(no\s+(?:numerical|numeric)?\s*corrosion\s+data|no\s+(?:numerical|numeric)\s+data|not\s+reported|were\s+not\s+reported)",
        text,
        flags=re.IGNORECASE,
    ):
        gaps.append(_gap_from_text("no numerical corrosion data were reported", "corrosion resistance", evidence))
    return gaps


_CLAIM_MARKERS = [
    "вывод",
    "заключение",
    "рекомендуется",
    "показано",
    "установлено",
    "reported",
    "concluded",
    "recommended",
    "suggests",
    "indicates",
    "demonstrated",
]
_EXPERTISE_MARKERS = [
    "эксперт",
    "автор",
    "researcher",
    "laboratory",
    "лаборатория",
    "lab",
    "institute",
    "department",
    "team",
    "команда",
    "competence",
    "expertise",
    "компетенц",
]
_ECONOMIC_MARKERS = [
    "capex",
    "opex",
    "cost",
    "operating cost",
    "capital cost",
    "стоимост",
    "затрат",
    "эконом",
    "руб/т",
    "usd/t",
    "$/t",
    "eur/t",
    "€/t",
]
_TECHNOLOGY_SOLUTION_MARKERS = [
    "метод",
    "способ",
    "технология",
    "схема",
    "система",
    "решение",
    "применяется",
    "используется",
    "позволяет",
    "предназначен",
    "рекомендуется",
    "method",
    "technology",
    "approach",
    "system",
    "solution",
    "used for",
    "applied for",
    "designed for",
    "recommended",
]
_CIRCULATION_SOLUTION_MARKERS = [
    "циркуляция",
    "рециркуляция",
    "подача католита",
    "возврат католита",
    "поток католита",
    "движение католита",
    "расход католита",
    "скорость потока",
    "схема циркуляции",
    "циркуляционный контур",
    "католит",
    "электролит",
    "catholyte circulation",
    "electrolyte circulation",
    "recirculation",
    "catholyte flow",
    "electrolyte flow",
    "flow velocity",
    "flow rate",
    "circulation loop",
    "circulation scheme",
]
_PROPERTY_CONTEXT_CANDIDATES = [
    "концентрация",
    "сухой остаток",
    "tds",
    "скорость потока",
    "flow velocity",
    "расход",
    "flow rate",
    "извлечение",
    "recovery",
    "распределение",
    "distribution",
    "содержание",
    "content",
    "grade",
    "производительность",
    "capacity",
    "throughput",
    "capex",
    "opex",
    "стоимость",
    "cost",
]
_TEXT_ECONOMIC_RE = re.compile(_ECONOMIC_UNIT, re.IGNORECASE)
_LAB_NAME_TEXT_RE = re.compile(
    r"\b(?:лаборатори[ия]|laboratory|lab|institute|department|институт|отдел)\s+"
    r"(?P<name>[A-Za-zА-Яа-яЁё0-9_.\- ]{2,90}?)(?=;|,|\.|\n|\s+(?:по|for|of|with|работ\w*|исслед\w*|study|topic)\b|$)",
    re.IGNORECASE,
)
_AUTHOR_TEXT_RE = re.compile(
    r"\b(?:author|автор|researcher|эксперт)\s*[:\-]?\s*"
    r"(?P<name>[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё'.\-]+(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё'.\-]+){0,3})",
    re.IGNORECASE,
)


def _extract_source_adapter_candidates(text: str, evidence: EvidenceSpan) -> list[CandidateFact]:
    """Extract typed source candidates from prose without using LLMs.

    These candidates are not graph facts yet.  They are accepted only after the
    direct candidate validator confirms evidence, schema and topic context.
    """

    candidates: list[CandidateFact] = []
    for sentence in _iter_candidate_sentences(text):
        norm = sentence.lower().replace("ё", "е")
        context = _domain_context(sentence)
        if any(marker in norm for marker in _CIRCULATION_SOLUTION_MARKERS):
            candidate = _circulation_solution_candidate_from_text(sentence, evidence, context)
            if candidate is not None:
                candidates.append(candidate)
        if any(marker in norm for marker in _TECHNOLOGY_SOLUTION_MARKERS):
            candidate = _technology_solution_candidate_from_text(sentence, evidence, context)
            if candidate is not None:
                candidates.append(candidate)
        if any(marker in norm for marker in _CLAIM_MARKERS):
            candidate = _publication_claim_candidate_from_text(sentence, evidence, context)
            if candidate is not None:
                candidates.append(candidate)
        if any(marker in norm for marker in _EXPERTISE_MARKERS) or _PERSON_RE.search(sentence) or _LAB_NAME_TEXT_RE.search(sentence):
            candidate = _expertise_candidate_from_text(sentence, evidence, context)
            if candidate is not None:
                candidates.append(candidate)
        if any(marker in norm for marker in _ECONOMIC_MARKERS):
            candidates.extend(_economic_candidates_from_text(sentence, evidence, context))
    return _dedupe_candidate_facts(candidates)


def _circulation_solution_candidate_from_text(sentence: str, evidence: EvidenceSpan, context: dict[str, list[str]]) -> CandidateFact | None:
    norm = sentence.lower().replace("ё", "е")
    has_media = any(marker in norm for marker in ["католит", "catholyte", "электролит", "electrolyte"])
    has_flow = any(marker in norm for marker in ["циркуляц", "рециркуляц", "поток", "расход", "скорость", "flow", "circulation", "recirculation"])
    has_context = any(marker in norm for marker in ["электроэкстракц", "electrowinning", "electroextraction", "никел", "nickel"])
    if not (has_media and has_flow and has_context):
        return None
    claim = _clean_claim_span(sentence)
    media = "католит" if any(marker in norm for marker in ["католит", "catholyte"]) else "электролит"
    subprocess = "циркуляция католита" if media == "католит" else "циркуляция электролита"
    process = "электроэкстракция" if any(marker in norm for marker in ["электроэкстракц", "electrowinning", "electroextraction"]) else context["processes"][0] if context["processes"] else ""
    subject = _context_subject(context)
    subject.update(
        {
            "material": subject.get("material") or ("никель" if any(marker in norm for marker in ["никел", "nickel"]) else ""),
            "process": process,
            "subprocess": subprocess,
            "media": media,
        }
    )
    solution = _circulation_solution_label(sentence, subprocess)
    return CandidateFact(
        candidate_id=_source_candidate_id("TechnologySolutionFact", evidence, sentence, subprocess, media, solution),
        fact_type="TechnologySolutionFact",
        extractor_name=f"{DeterministicPatternVersion.VALUE}:extractive_circulation_solution_adapter",
        document_id=evidence.source.document_id,
        chunk_id=evidence.source.chunk_id,
        source_name=evidence.source.source_name,
        subject=subject,
        predicate="DESCRIBES_TECHNOLOGY_SOLUTION",
        object={
            "technology": solution,
            "solution_name": solution,
            "target_problem": subprocess,
            "process": process,
            "process_context": process,
            "subprocess": subprocess,
            "material": subject.get("material") or "",
            "media": media,
            "equipment": context["equipment"][0] if context["equipment"] else "",
            "applicable_conditions": _technology_conditions(sentence),
            "claim": claim,
            "source_note": claim,
        },
        evidence_quote=claim,
        raw_span=claim,
        context_window=sentence,
        confidence=0.84,
        document_type="unknown",
    )


def _technology_solution_candidate_from_text(sentence: str, evidence: EvidenceSpan, context: dict[str, list[str]]) -> CandidateFact | None:
    domain = _first_context_value(context, "processes", "materials", "equipment", "properties")
    if not domain:
        return None
    claim = _clean_claim_span(sentence)
    solution = _technology_solution_name(sentence, context)
    subject = _context_subject(context)
    subject["media"] = context["materials"][0] if context["materials"] else ""
    return CandidateFact(
        candidate_id=_source_candidate_id("TechnologySolutionFact", evidence, sentence, domain, solution),
        fact_type="TechnologySolutionFact",
        extractor_name=f"{DeterministicPatternVersion.VALUE}:extractive_technology_solution_adapter",
        document_id=evidence.source.document_id,
        chunk_id=evidence.source.chunk_id,
        source_name=evidence.source.source_name,
        subject=subject,
        predicate="DESCRIBES_TECHNOLOGY_SOLUTION",
        object={
            "technology": solution or domain,
            "solution_name": solution or domain,
            "target_problem": _technology_target_problem(sentence, context),
            "process_context": context["processes"][0] if context["processes"] else "",
            "subprocess": context["processes"][0] if context["processes"] and "циркуляц" in context["processes"][0].lower().replace("ё", "е") else "",
            "material": context["materials"][0] if context["materials"] else "",
            "media": context["materials"][0] if context["materials"] and context["materials"][0] in {"католит", "электролит"} else "",
            "equipment": context["equipment"][0] if context["equipment"] else "",
            "applicable_conditions": _technology_conditions(sentence),
            "claim": claim,
            "source_note": claim,
        },
        evidence_quote=claim,
        raw_span=claim,
        context_window=sentence,
        confidence=0.81 if (context["processes"] and (context["materials"] or context["equipment"] or context["properties"])) else 0.77,
        document_type="unknown",
    )


def _circulation_solution_label(sentence: str, fallback: str) -> str:
    norm = sentence.lower().replace("ё", "е")
    if "схема" in norm or "scheme" in norm:
        return "схема циркуляции"
    if "контур" in norm or "loop" in norm:
        return "циркуляционный контур"
    if "расход" in norm or "flow rate" in norm:
        return "расход католита"
    if "скорость" in norm or "flow velocity" in norm:
        return "скорость потока католита"
    return fallback


def _publication_claim_candidate_from_text(sentence: str, evidence: EvidenceSpan, context: dict[str, list[str]]) -> CandidateFact | None:
    topic = _first_context_value(context, "properties", "processes", "materials", "equipment")
    if not topic:
        return None
    subject = _context_subject(context)
    return CandidateFact(
        candidate_id=_source_candidate_id("PublicationClaimFact", evidence, sentence, topic),
        fact_type="PublicationClaimFact",
        extractor_name=f"{DeterministicPatternVersion.VALUE}:extractive_claim_adapter",
        document_id=evidence.source.document_id,
        chunk_id=evidence.source.chunk_id,
        source_name=evidence.source.source_name,
        subject=subject,
        predicate="PUBLICATION_CLAIM",
        object={
            "topic": topic,
            "claim": _clean_claim_span(sentence),
            "property": context["properties"][0] if context["properties"] else "",
            "process": context["processes"][0] if context["processes"] else "",
            "material": context["materials"][0] if context["materials"] else "",
            "equipment": context["equipment"][0] if context["equipment"] else "",
        },
        evidence_quote=_clean_claim_span(sentence),
        raw_span=_clean_claim_span(sentence),
        context_window=sentence,
        confidence=0.79,
        document_type="unknown",
    )


def _expertise_candidate_from_text(sentence: str, evidence: EvidenceSpan, context: dict[str, list[str]]) -> CandidateFact | None:
    topic = _first_context_value(context, "processes", "materials", "properties", "equipment")
    if not topic:
        return None
    expert = ""
    lab = ""
    team = ""
    person = _PERSON_RE.search(sentence) or _AUTHOR_TEXT_RE.search(sentence)
    if person:
        expert = _clean_named_entity(person.group(0) if not person.groupdict().get("name") else person.group("name"))
    lab_match = _LAB_NAME_TEXT_RE.search(sentence) or _LAB_RE.search(sentence)
    if lab_match:
        lab = _clean_named_entity(lab_match.group(0) if "name" in lab_match.groupdict() else lab_match.group(1))
    team_match = _TEAM_RE.search(sentence)
    if team_match:
        team = _clean_named_entity(team_match.group("name"))
    if not (expert or lab or team):
        return None
    subject = _context_subject(context)
    subject.update({"expert": expert, "laboratory": lab, "team": team})
    return CandidateFact(
        candidate_id=_source_candidate_id("ExpertiseFact", evidence, sentence, topic, expert or lab or team),
        fact_type="ExpertiseFact",
        extractor_name=f"{DeterministicPatternVersion.VALUE}:extractive_expertise_adapter",
        document_id=evidence.source.document_id,
        chunk_id=evidence.source.chunk_id,
        source_name=evidence.source.source_name,
        subject=subject,
        predicate="HAS_EXPERTISE",
        object={
            "topic": topic,
            "process": context["processes"][0] if context["processes"] else "",
            "material": context["materials"][0] if context["materials"] else "",
            "equipment": context["equipment"][0] if context["equipment"] else "",
        },
        evidence_quote=_clean_claim_span(sentence),
        raw_span=_clean_claim_span(sentence),
        context_window=sentence,
        confidence=0.78,
        document_type="unknown",
    )


def _economic_candidates_from_text(sentence: str, evidence: EvidenceSpan, context: dict[str, list[str]]) -> list[CandidateFact]:
    candidates: list[CandidateFact] = []
    subject = _context_subject(context)
    for match in _TEXT_ECONOMIC_RE.finditer(sentence):
        unit = resolve_unit(match.group("unit"))
        value = _float_or_none(match.group("value"))
        candidates.append(
            CandidateFact(
                candidate_id=_source_candidate_id("EconomicIndicatorFact", evidence, sentence, match.group(0)),
                fact_type="EconomicIndicatorFact",
                extractor_name=f"{DeterministicPatternVersion.VALUE}:extractive_economic_adapter",
                document_id=evidence.source.document_id,
                chunk_id=evidence.source.chunk_id,
                source_name=evidence.source.source_name,
                subject=subject,
                predicate="REPORTS_ECONOMIC_INDICATOR",
                object={
                    "property": "экономический показатель",
                    "indicator": _economic_indicator(sentence),
                    "technology": context["processes"][0] if context["processes"] else context["equipment"][0] if context["equipment"] else "",
                    "process": context["processes"][0] if context["processes"] else "",
                    "material": context["materials"][0] if context["materials"] else "",
                    "raw_value": match.group(0),
                },
                value=value,
                unit=unit,
                evidence_quote=_clean_claim_span(sentence),
                raw_span=match.group(0),
                context_window=sentence,
                confidence=0.8,
                document_type="unknown",
            )
        )
    return candidates


def _technology_solution_name(sentence: str, context: dict[str, list[str]]) -> str:
    text = _clean_claim_span(sentence)
    match = re.search(
        r"(?:метод|способ|технологи[яи]|схема|система|решение|method|technology|approach|system|solution)\s+"
        r"(?P<name>[A-Za-zА-Яа-яЁё0-9₂/+\- ]{3,100}?)(?=,|;|\.|\s+(?:для|по|при|used|applied|designed|recommended|позволяет|применяется|используется)\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _clean_claim_span(match.group("name"))[:120]
    return _first_context_value(context, "processes", "equipment", "properties", "materials")


def _technology_target_problem(sentence: str, context: dict[str, list[str]]) -> str:
    text = _clean_claim_span(sentence)
    match = re.search(
        r"(?:для|по|направлен[аоы]?\s+на|предназначен[аоы]?\s+для|used\s+for|applied\s+for|designed\s+for)\s+"
        r"(?P<problem>[A-Za-zА-Яа-яЁё0-9₂/+\- ]{3,120}?)(?=,|;|\.|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _clean_claim_span(match.group("problem"))[:160]
    return _first_context_value(context, "properties", "processes", "materials", "equipment")


def _technology_conditions(sentence: str) -> str:
    text = _clean_claim_span(sentence)
    match = re.search(
        r"(?:при|в условиях|under|at)\s+(?P<conditions>[A-Za-zА-Яа-яЁё0-9₂/+\-–., ]{3,140}?)(?=;|\.|$)",
        text,
        flags=re.IGNORECASE,
    )
    return _clean_claim_span(match.group("conditions"))[:180] if match else ""


def _domain_context(sentence: str) -> dict[str, list[str]]:
    lowered = sentence.lower().replace("ё", "е")
    materials = [
        resolve_material(raw)
        for raw in _MATERIAL_CANDIDATES
        if _material_candidate_in_text(raw, lowered)
    ]
    processes = [
        resolve_regime(raw)
        for raw in _REGIME_CANDIDATES
        if raw.lower().replace("ё", "е") in lowered
    ]
    equipment = [
        resolve_equipment(raw) or _canonical_equipment(raw)
        for raw in _EQUIPMENT_CANDIDATES
        if raw.lower().replace("ё", "е") in lowered
    ]
    properties = [
        resolve_property(raw)
        for raw in _PROPERTY_CONTEXT_CANDIDATES
        if raw.lower().replace("ё", "е") in lowered
    ]
    return {
        "materials": _dedupe_texts(materials),
        "processes": _dedupe_texts(processes),
        "equipment": _dedupe_texts(equipment),
        "properties": _dedupe_texts(properties),
    }


def _context_subject(context: dict[str, list[str]]) -> dict[str, str]:
    return {
        "material": context["materials"][0] if context["materials"] else "",
        "process": context["processes"][0] if context["processes"] else "",
        "equipment": context["equipment"][0] if context["equipment"] else "",
    }


def _first_context_value(context: dict[str, list[str]], *keys: str) -> str:
    for key in keys:
        values = context.get(key) or []
        if values:
            return values[0]
    return ""


def _iter_candidate_sentences(text: str) -> list[str]:
    rows = re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
    result = []
    for row in rows:
        sentence = re.sub(r"\s+", " ", row).strip(" .;\t")
        if 20 <= len(sentence) <= 900:
            result.append(sentence)
    if not result and 20 <= len(str(text or "").strip()) <= 900:
        result.append(re.sub(r"\s+", " ", str(text).strip()))
    return result[:20]


def _clean_claim_span(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip(" .;\n\t")[:700]


def _economic_indicator(sentence: str) -> str:
    norm = sentence.lower().replace("ё", "е")
    if "capex" in norm or "капитальн" in norm:
        return "CAPEX"
    if "opex" in norm or "эксплуатацион" in norm or "operating cost" in norm:
        return "OPEX"
    if "тариф" in norm:
        return "tariff"
    return "cost"


def _source_candidate_id(fact_type: str, evidence: EvidenceSpan, *parts: object) -> str:
    raw = "|".join(
        [
            fact_type,
            str(evidence.source.document_id or ""),
            str(evidence.source.chunk_id or ""),
            *(str(part) for part in parts),
        ]
    )
    return "text_cand_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _dedupe_texts(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if str(item or "").strip()))


def _dedupe_candidate_facts(items: list[CandidateFact]) -> list[CandidateFact]:
    seen: set[str] = set()
    result: list[CandidateFact] = []
    for item in items:
        if item.candidate_id in seen:
            continue
        seen.add(item.candidate_id)
        result.append(item)
    return result


def _extract_experiment_id(text: str, chunk: Chunk) -> str:
    match = _EXPERIMENT_ID_RE.search(text)
    if match:
        return match.group(0)
    raw = f"{chunk.chunk_id}|{text[:120]}"
    return f"EXP-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _is_experiment_identifier(value: str | None) -> bool:
    return bool(_EXPERIMENT_ID_RE.fullmatch(str(value or "").strip()))


def _source_from_chunk(chunk: Chunk, column_name: str | None = None) -> ExtractionSource:
    row_id = chunk.metadata.get("row_id")
    try:
        row_index = int(row_id) if row_id is not None else None
    except ValueError:
        row_index = None
    return ExtractionSource(
        document_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        source_name=str(chunk.metadata.get("source_name") or chunk.metadata.get("filename") or chunk.doc_id),
        page=chunk.page_start,
        section_path=chunk.section_path,
        block_type=str(chunk.metadata.get("chunk_kind") or "text"),
        row_index=row_index,
        column_name=column_name,
    )


def _entity(entity_type: str, raw: str, canonical: str, evidence: EvidenceSpan, confidence: float) -> ExtractedEntity:
    return ExtractedEntity(entity_type=entity_type, raw_name=raw, canonical_name=canonical, confidence=confidence, evidence=[evidence])


def _entity_from_legacy(entity, evidence: EvidenceSpan) -> ExtractedEntity | None:
    if entity.entity_type == "Material":
        if _is_experiment_identifier(entity.canonical_name):
            return None
        return _entity("Material", entity.canonical_name, resolve_material(entity.canonical_name), evidence, 0.8)
    if entity.entity_type in {"ProcessRegime", "ProcessCondition"}:
        return _entity("ProcessRegime", entity.canonical_name, resolve_regime(entity.canonical_name), evidence, 0.72)
    if entity.entity_type == "Property":
        return _entity("Property", entity.canonical_name, resolve_property(entity.canonical_name), evidence, 0.72)
    if entity.entity_type in {"Equipment", "Laboratory", "ResearchTeam", "Employee", "TopicTag"}:
        if entity.entity_type == "Laboratory" and _is_team_mention(entity.canonical_name, evidence.quote):
            return _entity("ResearchTeam", entity.canonical_name, entity.canonical_name, evidence, 0.7)
        return _entity(entity.entity_type, entity.canonical_name, entity.canonical_name, evidence, 0.7)
    return None


def _is_team_mention(name: str | None, text: str | None) -> bool:
    clean_name = re.escape(str(name or "").strip())
    if not clean_name:
        return False
    return bool(
        re.search(
            rf"\b(?:команд[аы]|групп[аы]|research\s+team|team)\s*[:\-]?\s*{clean_name}\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def _evidence_from_relation(rel, default: EvidenceSpan) -> EvidenceSpan:
    if not rel.evidence:
        return default
    src = rel.evidence[0]
    quote = src.quote or default.quote
    source = ExtractionSource(
        document_id=src.doc_id,
        chunk_id=src.chunk_id,
        source_name=default.source.source_name,
        page=src.page_start,
        section_path=default.source.section_path,
        block_type=default.source.block_type,
        row_index=default.source.row_index,
    )
    return EvidenceSpan(source=source, quote=quote, confidence=rel.confidence or default.confidence)


def _gap_from_text(text: str, missing_for: str | None, evidence: EvidenceSpan) -> ExtractedDataGap:
    joined = f"{text} {missing_for or ''} {evidence.quote or ''}"
    material = _first_canonical(joined, resolve_material, ["ВТ6", "7075-T6", "12Х18Н10Т", "09Г2С"])
    regime = _first_canonical(joined, resolve_regime, ["отжиг", "старение", "закалка", "криообработка", "термообработка", "heat treatment"])
    property_name = None
    if re.search(r"коррозионн\w+\s+стойк|corrosion\s+resistance|corrosion\s+data", str(joined or ""), flags=re.IGNORECASE):
        property_name = "коррозионная стойкость"
    property_name = property_name or _first_canonical(missing_for or "", resolve_property, ["коррозионная стойкость", "прочность", "твёрдость", "пластичность", "вязкость"])
    property_name = property_name or _first_canonical(joined, resolve_property, ["коррозионная стойкость", "прочность", "твёрдость", "пластичность", "вязкость"])
    reason = re.sub(r"\s+", " ", str(text or "").strip(" .;|"))
    raw = "|".join([material or "", regime or "", property_name or "", reason])
    return ExtractedDataGap(
        gap_id=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
        material=material,
        regime=regime,
        property=property_name,
        reason=reason,
        confidence=0.75,
        evidence=[evidence],
    )


def _first_canonical(text: str, resolver, candidates: list[str]) -> str | None:
    text_norm = str(text or "").lower().replace("ё", "е")
    for candidate in candidates:
        canonical = resolver(candidate)
        canonical_norm = str(canonical or "").lower().replace("ё", "е")
        if canonical and canonical_norm in text_norm:
            return canonical
        if resolver(text) == canonical:
            return canonical
    return None


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _normalize_effect(value: str | None) -> str:
    norm = str(value or "").lower()
    if norm in {"increase", "increased"} or "повыс" in norm or "увелич" in norm:
        return "increase"
    if norm in {"decrease", "decreased"} or "сниз" in norm or "уменьш" in norm:
        return "decrease"
    if norm in {"unchanged", "no_change"} or "без измен" in norm:
        return "no_change"
    return "unknown"


def _qualitative_effect_near(text: str, start: int, end: int) -> str | None:
    window = text[max(0, start - 80): min(len(text), end + 80)]
    match = re.search(r"повыс\w+|увелич\w+|increase\w*|сниз\w+|уменьш\w+|decrease\w*|без\s+измен\w+|unchanged|no\s+change", window, re.IGNORECASE)
    return match.group(0) if match else None


def _dedupe_entities(items: list[ExtractedEntity]) -> list[ExtractedEntity]:
    seen = set()
    result = []
    for item in items:
        key = (item.entity_type, item.canonical_name)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_regimes(items: list[ExtractedRegime]) -> list[ExtractedRegime]:
    seen = set()
    result = []
    for item in items:
        key = item.canonical_name
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_measurements(items: list[ExtractedMeasurement]) -> list[ExtractedMeasurement]:
    by_key: dict[str, ExtractedMeasurement] = {}
    for item in items:
        key = canonical_fact_key(property_name=item.property_canonical, value=item.value, unit=item.unit, effect=item.effect)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = item
            continue
        evidence = _dedupe_evidence([*existing.evidence, *item.evidence])
        confidence = max(existing.confidence, item.confidence)
        by_key[key] = existing.model_copy(update={"evidence": evidence, "confidence": confidence}) if hasattr(existing, "model_copy") else existing.copy(update={"evidence": evidence, "confidence": confidence})
    return list(by_key.values())


def _apply_binding_guard(
    text: str,
    materials: list[ExtractedEntity],
    regimes: list[ExtractedRegime],
    measurements: list[ExtractedMeasurement],
) -> list[ExtractedMeasurement]:
    if not measurements or not materials:
        return measurements
    sentences = _sentence_spans(text)
    if not sentences:
        return measurements
    material_sentences = _entity_sentence_indexes(sentences, [item.raw_name for item in materials] + [item.canonical_name for item in materials])
    regime_sentences = _entity_sentence_indexes(sentences, [item.raw_name for item in regimes] + [item.canonical_name for item in regimes])
    guarded: list[ExtractedMeasurement] = []
    for measurement in measurements:
        measurement_sentences = _measurement_sentence_indexes(sentences, measurement)
        if not measurement_sentences:
            guarded.append(_copy_measurement_confidence(measurement, measurement.confidence - 0.25))
            continue
        penalty = 0.0
        if material_sentences and material_sentences.isdisjoint(measurement_sentences):
            penalty += 0.25
        if regime_sentences and regime_sentences.isdisjoint(measurement_sentences) and not _has_cross_sentence_link(text):
            penalty += 0.15
        if _has_weak_binding_marker(text) and (material_sentences.isdisjoint(measurement_sentences) or regime_sentences.isdisjoint(measurement_sentences)):
            penalty += 0.20
        guarded.append(_copy_measurement_confidence(measurement, measurement.confidence - penalty))
    return guarded


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"[.!?;\n]+", text):
        end = match.start()
        sentence = text[start:end].strip()
        if sentence:
            result.append((start, end, sentence.lower().replace("ё", "е")))
        start = match.end()
    tail = text[start:].strip()
    if tail:
        result.append((start, len(text), tail.lower().replace("ё", "е")))
    return result


def _entity_sentence_indexes(sentences: list[tuple[int, int, str]], values: list[str]) -> set[int]:
    normalized = [str(value or "").lower().replace("ё", "е") for value in values if value]
    return {idx for idx, (_, _, sentence) in enumerate(sentences) if any(value and value in sentence for value in normalized)}


def _measurement_sentence_indexes(sentences: list[tuple[int, int, str]], measurement: ExtractedMeasurement) -> set[int]:
    terms = [measurement.property_raw, measurement.property_canonical]
    if measurement.property_canonical == "прочность":
        terms.extend(["прочност", "tensile strength", "ultimate tensile strength", "strength"])
    if measurement.property_canonical == "коррозионная стойкость":
        terms.extend(["коррозион", "corrosion resistance"])
    value = "" if measurement.value is None else f"{measurement.value:g}".lower()
    result: set[int] = set()
    for idx, (_, _, sentence) in enumerate(sentences):
        has_property = any(str(term or "").lower().replace("ё", "е") in sentence for term in terms if term)
        has_value = not value or value in sentence
        if has_property and has_value:
            result.add(idx)
    return result


def _has_cross_sentence_link(text: str) -> bool:
    lowered = text.lower().replace("ё", "е")
    return any(marker in lowered for marker in ["после этого", "после обработки", "after this", "after treatment", "resulting in"])


def _has_weak_binding_marker(text: str) -> bool:
    lowered = text.lower().replace("ё", "е")
    return any(marker in lowered for marker in ["без связи", "другой таблиц", "unrelated", "without relation"])


def _copy_measurement_confidence(measurement: ExtractedMeasurement, confidence: float) -> ExtractedMeasurement:
    value = max(0.0, min(1.0, confidence))
    return measurement.model_copy(update={"confidence": value}) if hasattr(measurement, "model_copy") else measurement.copy(update={"confidence": value})


def _dedupe_evidence(items: list[EvidenceSpan]) -> list[EvidenceSpan]:
    seen = set()
    result = []
    for item in items:
        key = (item.source.document_id, item.source.chunk_id, item.quote)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_experiments(items: list[ExtractedExperiment]) -> list[ExtractedExperiment]:
    by_id: dict[str, ExtractedExperiment] = {}
    for item in items:
        existing = by_id.get(item.experiment_id)
        if existing is None:
            by_id[item.experiment_id] = item
            continue
        updates = {
            "materials": _dedupe_entities([*existing.materials, *item.materials]),
            "regimes": _dedupe_regimes([*existing.regimes, *item.regimes]),
            "measurements": _dedupe_measurements([*existing.measurements, *item.measurements]),
            "equipment": _dedupe_entities([*existing.equipment, *item.equipment]),
            "laboratories": _dedupe_entities([*existing.laboratories, *item.laboratories]),
            "teams": _dedupe_entities([*existing.teams, *item.teams]),
            "employees": _dedupe_entities([*existing.employees, *item.employees]),
            "conclusions": list(dict.fromkeys([*existing.conclusions, *item.conclusions])),
            "evidence": _dedupe_evidence([*existing.evidence, *item.evidence]),
            "confidence": max(existing.confidence, item.confidence),
        }
        by_id[item.experiment_id] = existing.model_copy(update=updates) if hasattr(existing, "model_copy") else existing.copy(update=updates)
    return list(by_id.values())


def _dedupe_gaps(items: list[ExtractedDataGap]) -> list[ExtractedDataGap]:
    seen = set()
    result = []
    for item in items:
        key = (item.material, item.regime, item.property, item.reason)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result

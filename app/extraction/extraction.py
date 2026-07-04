"""Structured and deterministic entity/relation extraction.

The full system is designed for LLM structured output. For a reliable
hackathon baseline this module also includes deterministic extraction of
materials, process conditions and properties. That guarantees that the
graph is not empty even without an LLM or external API.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from typing import Dict, List

from ..models.schemas import Chunk, ExtractionResult, MaterialEntity, RelationAssertion, SourceRef


MATERIAL_PATTERNS = [
    re.compile(r"\b(Alloy|Steel|Aluminum|Titanium|Copper|Nickel)\s+[A-Za-z0-9\-]+\b", re.IGNORECASE),
    re.compile(r"\b(?:сплав|сталь|стали|алюминиевый\s+сплав|алюминий|алюминия|титан|медь|никель)\s+[A-Za-zА-Яа-я0-9\-]+\b", re.IGNORECASE),
    re.compile(r"\b12[ХX]18[НH]10[ТT]\b", re.IGNORECASE),
    re.compile(r"\b09Г2С\b", re.IGNORECASE),
    re.compile(r"\bAISI\s*(?:304|321)\b", re.IGNORECASE),
    re.compile(r"\bTi-?6Al-?4V\b", re.IGNORECASE),
    re.compile(r"\b7075-T6\b", re.IGNORECASE),
    re.compile(r"\b7075\b", re.IGNORECASE),
    # Generic alloy codes containing both letters and digits (e.g., VT6, 7075, Ti6Al4V).
    # Matches sequences starting with one or two letters (Latin or Cyrillic), followed by digits, and optionally letters/digits/hyphens.
    re.compile(r"\b(?=[A-Za-zА-Яа-я]*\d)(?=[\d]*[A-Za-zА-Яа-я])[A-Za-zА-Яа-я]{1,3}[A-Za-zА-Яа-я0-9\-/]{1,6}\b"),
]

MATERIAL_CANONICAL_REPLACEMENTS = {
    "вт6": "ВТ6",
    "vt6": "ВТ6",
    "сплав вт6": "ВТ6",
    "сплав 7075": "7075",
    "сплава 7075": "7075",
    "alloy vt6": "ВТ6",
    "сталь 12х18н10т": "12Х18Н10Т",
    "steel 12х18н10т": "12Х18Н10Т",
    "12х18н10т": "12Х18Н10Т",
    "алюминиевый сплав 7075": "7075",
    "алюминия 7075": "7075",
    "алюминий 7075": "7075",
    "aluminum 7075": "7075",
    "alloy 7075": "7075",
    "7075": "7075-T6",
    "7075-t6": "7075-T6",
    "09г2с": "09Г2С",
    "aisi 304": "AISI 304",
    "aisi304": "AISI 304",
    "aisi 321": "AISI 321",
    "aisi321": "AISI 321",
    "ti-6al-4v": "Ti-6Al-4V",
    "ti6al4v": "Ti-6Al-4V",
}

PROPERTY_TERMS = {
    "strength": "Strength",
    "hardness": "Hardness",
    "corrosion": "Corrosion resistance",
    "ductility": "Ductility",
    "прочность": "Прочность",
    "прочности": "Прочность",
    "прочке": "Прочность",
    "прочк": "Прочность",
    "твердость": "Твердость",
    "твёрдость": "Твердость",
    "корроз": "Коррозионная стойкость",
    "пластичность": "Пластичность",
    # Russian stems and variations
    "прочн": "Прочность",
    "тверд": "Твердость",
    "твёрд": "Твердость",
    "коррозия": "Коррозионная стойкость",
    "коррозионная": "Коррозионная стойкость",
    "пластичн": "Пластичность",
    "plasticity": "Ductility",
    "toughness": "Strength",
}

PROCESS_TERMS = {
    "anneal": "Annealing",
    "annealed": "Annealing",
    "quench": "Quenching",
    "quenched": "Quenching",
    "aging": "Aging",
    "aged": "Aging",
    "отжиг": "Отжиг",
    "закал": "Закалка",
    "старен": "Старение",
    # morphological variations in Russian
    "отожжен": "Отжиг",
    "отжига": "Отжиг",
    "отжиге": "Отжиг",
    "закалки": "Закалка",
    "закален": "Закалка",
    "старения": "Старение",
    "старением": "Старение",
    # English variations
    "annealing": "Annealing",
    "quenching": "Quenching",
}

TECHNICAL_OBJECT_RE = re.compile(
    r"\b(?P<type>клапан|насос|корпус|узел|деталь|valve|pump|assembly|body|shaft|impeller)"
    r"(?:\s+(?P<name>[A-Za-zА-Яа-я0-9\-]+))?"
    r"(?:\s+(?P<size>DN\s*\d+|Ду\s*\d+))?",
    re.IGNORECASE,
)

PART_RE = re.compile(
    r"\b(?P<part>корпус|рабочее колесо|колесо|уплотнение|вал|крышка|body|impeller|seal|shaft|cover)\b",
    re.IGNORECASE,
)

ARTICLE_RE = re.compile(
    r"\b(?:ART-[A-Z0-9]+(?:-[A-Z0-9]+)+|[A-Z]{2,}-\d{3,}[A-Z0-9\-]*|A\d{3}-B\d{2}|SEAL-\d+X\d+|\d{2}\.\d{3}\.\d{3}|VALVE-DN\d+-PN\d+)\b",
    re.IGNORECASE,
)

STANDARD_RE = re.compile(
    r"\b(?:ГОСТ\s*\d+(?:-\d+)?|ISO\s*\d+(?:-\d+)?|ASTM\s*[A-Z]\d+|EN\s*\d+(?:-\d+)?|ТУ\s*\d+(?:-\d+)?)\b",
    re.IGNORECASE,
)

PARAMETER_RES = [
    re.compile(r"\b(?P<name>DN|Ду)\s*(?P<value>\d+)\b", re.IGNORECASE),
    re.compile(r"\b(?P<name>PN)\s*(?P<value>\d+)\b", re.IGNORECASE),
    re.compile(r"\b(?P<name>P)\s*=\s*(?P<value>\d+(?:[\.,]\d+)?)\s*(?P<unit>MPa|МПа|bar|бар)\b", re.IGNORECASE),
    re.compile(r"\b(?P<value>\d+(?:[\.,]\d+)?)\s*(?P<unit>MPa|МПа|bar|бар)\b", re.IGNORECASE),
    re.compile(r"\b(?P<name>T)\s*=\s*(?P<value>[+\-]?\d+(?:[\.,]\d+)?)\s*°?\s*(?P<unit>C|С)\b", re.IGNORECASE),
    re.compile(r"(?P<value>от\s*[+\-]?\d+\s*до\s*[+\-]?\d+)\s*°?\s*(?P<unit>C|С)", re.IGNORECASE),
    re.compile(r"[Ø⌀]\s*(?P<value>\d+(?:[\.,]\d+)?)\s*(?P<unit>мм|mm)\b", re.IGNORECASE),
    re.compile(r"\b(?P<value>\d+(?:[\.,]\d+)?\s*[x×]\s*\d+(?:[\.,]\d+)?\s*[x×]\s*\d+(?:[\.,]\d+)?)\s*(?P<unit>мм|mm)\b", re.IGNORECASE),
    re.compile(r"\b(?P<name>производительность|расход|capacity|flow)\s*[:=]?\s*(?P<value>\d+(?:[\.,]\d+)?)\s*(?P<unit>м3/ч|m3/h|л/с|l/s)\b", re.IGNORECASE),
]

REQUIREMENT_RE = re.compile(
    r"(?:требование|requirement|должен|должна|необходимо|required)\s*[:\-]?\s*(?P<text>[^.\n|;]+)",
    re.IGNORECASE,
)

IMAGE_RE = re.compile(
    r"Image:\s*url:\s*(?P<url>[^|]+)\|\s*alt:\s*(?P<alt>[^|]*)\|\s*caption:\s*(?P<caption>[^|]*)\|\s*section_path:\s*(?P<section>.+)",
    re.IGNORECASE,
)

IMAGE_PATH_RE = re.compile(
    r"(?P<url>(?:images|img|assets)[A-Za-z0-9_./\\-]*\.(?:png|jpg|jpeg|svg|webp))",
    re.IGNORECASE,
)

EQUIPMENT_RE = re.compile(
    r"(?:equipment|оборудование|установка|станок|печь|камера|машина|прибор)\s*[:\-]?\s*"
    r"(?P<name>[A-Za-zА-Яа-я0-9\-\s\"«»]+?)(?=\s*\||\.|;|\n|$)",
    re.IGNORECASE,
)

LAB_RE = re.compile(
    r"(?:laboratory|lab|research team|team|лаборатория|команда|группа)\s*[:\-]?\s*"
    r"(?P<name>[A-Za-zА-Яа-я0-9\-\s\"«»]+?)(?=\s*\||\.|;|\n|$)",
    re.IGNORECASE,
)

EXPERIMENT_RE = re.compile(
    r"(?:experiment|эксперимент|опыт|exp[_\-\s]?)\s*[:#№]?\s*(?P<name>[A-Za-zА-Яа-я0-9_\-]+)",
    re.IGNORECASE,
)

EXPERIMENT_ID_RE = re.compile(
    r"(?:experiment_id|experiment\s+id|id\s+эксперимента|experiment)\s*[:=]\s*(?P<name>[A-Za-zА-Яа-я0-9_\-]+)",
    re.IGNORECASE,
)

CONCLUSION_RE = re.compile(
    r"(?:conclusion|вывод|итог|result|результат)\s*[:\-]\s*(?P<text>[^.\n|;]+(?:[.][^.\n|;]+)?)",
    re.IGNORECASE,
)

DATA_GAP_RE = re.compile(
    r"(?:(?:data[_\s]+gap|(?<!_)\bgap\b|пробел)[ \t]*[:\-][ \t]*(?P<text1>[^.\n|;]+)"
    r"|(?P<text2>(?:нет данных|не измерялась|не измеряли|отсутствуют данные)[^.\n|;]*))",
    re.IGNORECASE,
)

MEASUREMENT_RE = re.compile(
    r"(?P<value>\d+(?:[\.,]\d+)?)\s*(?P<unit>°?C|°?С|K|MPa|GPa|HV|HRC|%|ppm|h|hr|hrs|hour|hours|мин|мин\.\s*|ч|ч\.?)",
    re.IGNORECASE,
)

# Direction keywords for crude effect extraction. These terms are checked
# near numerical measurements to infer whether a property increased,
# decreased or remained unchanged.  This simple heuristic is used for
# hackathon scenarios and is not meant to replace proper event
# extraction.
INCREASE_TERMS = [
    "increase", "increased", "rise", "rose", "higher", "повыс", "увелич", "возрос", "вырос", "больше"
]
DECREASE_TERMS = [
    "decrease", "decreased", "lower", "reduced", "reduce", "less", "сниз", "уменьш", "меньше"
]
UNCHANGED_TERMS = [
    "unchanged", "no change", "без изменения", "не изм", "не изменилось"
]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def clean_laboratory_name(text: str) -> str:
    value = text.strip(" «»\"")
    value = re.split(r"\b(?:выполнила|выполнил|проводит|провела|занималась|занимался|conclusion|вывод)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.split(r"\s*\|\s*", value, maxsplit=1)[0]
    value = value.strip(" .:;«»\"")
    if normalise(value) == "термообработки":
        return "Лаборатория термообработки"
    if normalise(value) == "легких сплавов":
        return "Лаборатория легких сплавов"
    return value


def canonical_material_name(text: str) -> str:
    raw = re.sub(r"\s+", " ", text.strip())
    key = raw.lower().replace("ё", "е")
    key = key.replace("x", "х") if "12x18" in key else key
    if key.startswith("vt6"):
        return "ВТ6"
    return MATERIAL_CANONICAL_REPLACEMENTS.get(key, raw)


def canonical_object_name(match: re.Match[str]) -> str:
    obj_type = match.group("type").strip()
    name = (match.group("name") or "").strip()
    size = (match.group("size") or "").strip()
    if size:
        size = size.upper().replace(" ", "").replace("ДУ", "DN")
    if name and re.fullmatch(r"(DN|Ду)\s*\d+", name, re.IGNORECASE):
        size = name.upper().replace(" ", "").replace("ДУ", "DN")
        name = ""
    return " ".join(part for part in [obj_type, name, size] if part)


def normalise_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    mapping = {"мпа": "MPa", "бар": "bar", "с": "C", "мм": "mm"}
    return mapping.get(unit.lower().replace("°", ""), unit)


def is_false_material(candidate: str) -> bool:
    upper = candidate.strip().upper()
    if re.fullmatch(r"(DN|PN|ДУ)\s*\d+[-A-ZА-Яа-я]*", upper):
        return True
    if "/" in upper or upper in {"M3", "М3", "М3/Ч"}:
        return True
    if upper in {"BODY", "IMP", "SEAL", "PUMP", "VALVE", "NPK", "T6"}:
        return True
    if upper.startswith(("ART-", "VALVE-DN", "SEAL-")):
        return True
    return False


def infer_property_from_context(unit: str | None, context: str, properties: List[str]) -> str | None:
    lower = context.lower()
    pressure_markers = ["pressure", "давление", "напор", "p=", "p =", "pn", "bar", "бар"]
    if unit in {"MPa", "GPa", "bar"} and any(marker in lower for marker in pressure_markers):
        return None
    if unit == "%" and ("пластич" in lower or "ductility" in lower or "elongation" in lower):
        return "Пластичность"
    if unit == "%" and ("корроз" in lower or "corrosion" in lower):
        return "Коррозионная стойкость"
    for term, canonical in PROPERTY_TERMS.items():
        if term in lower:
            return canonical
    if unit in {"HV", "HRC"}:
        return "Твердость"
    return None


def infer_direction(context: str) -> str | None:
    lower = context.lower()
    for term in INCREASE_TERMS:
        if term in lower:
            return "increase"
    for term in DECREASE_TERMS:
        if term in lower:
            return "decrease"
    for term in UNCHANGED_TERMS:
        if term in lower:
            return "unchanged"
    return None


def infer_direction_near(text: str, start: int, end: int) -> str | None:
    """Choose the closest effect keyword around a measurement."""
    window_start = max(0, start - 80)
    window_end = min(len(text), end + 80)
    context = text[window_start:window_end].lower()
    anchor = start - window_start
    candidates: List[tuple[int, str]] = []
    for direction, terms in (
        ("increase", INCREASE_TERMS),
        ("decrease", DECREASE_TERMS),
        ("unchanged", UNCHANGED_TERMS),
    ):
        for term in terms:
            pos = context.find(term)
            while pos >= 0:
                candidates.append((abs(pos - anchor), direction))
                pos = context.find(term, pos + 1)
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    return None


def process_regime_name(process: str, text: str) -> str:
    """Build a compact process regime label from nearby temperature/time values."""
    temps = []
    times = []
    for match in re.finditer(r"(?:temperature_c|температура)\s*:\s*(\d+(?:[\.,]\d+)?)", text, re.IGNORECASE):
        temps.append(f"{match.group(1).replace(',', '.')} C")
    for match in re.finditer(r"(?:time_h|время)\s*:\s*(\d+(?:[\.,]\d+)?)", text, re.IGNORECASE):
        times.append(f"{match.group(1).replace(',', '.')} h")
    for match in MEASUREMENT_RE.finditer(text):
        value = match.group("value").replace(",", ".")
        unit_raw = match.group("unit").lower().replace("°", "")
        if unit_raw in {"c", "с", "k"}:
            try:
                if float(value) < 30:
                    continue
            except ValueError:
                pass
            temps.append(f"{value} C" if unit_raw in {"c", "с"} else f"{value} K")
        elif unit_raw in {"h", "hr", "hrs", "hour", "hours", "ч", "ч."}:
            times.append(f"{value} h")
    details = []
    if temps:
        details.append(temps[0])
    if times:
        details.append(times[0])
    return f"{process} ({', '.join(details)})" if details else process


PROCESS_FIELD_RE = re.compile(
    r"(?:process_regime|process|режим(?:\s+обработки)?)\s*:\s*(?P<value>[^|\n.;]+)",
    re.IGNORECASE,
)


def explicit_processes(text: str) -> List[str]:
    processes: List[str] = []
    for match in PROCESS_FIELD_RE.finditer(text):
        value = match.group("value").lower()
        for term, canonical in PROCESS_TERMS.items():
            if term in value and canonical not in processes:
                processes.append(canonical)
    return processes


def table_field(text: str, field_name: str) -> str:
    match = re.search(rf"(?:^|\|)\s*{re.escape(field_name)}\s*:\s*(?P<value>[^|]+)", text, re.IGNORECASE)
    return match.group("value").strip() if match else ""


def stable_entity_uid(workspace_uid: str | None, entity_type: str, name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{workspace_uid or 'default'}:{entity_type}:{normalise(name)}"))


class EntityRelationExtractor:
    """Hybrid extractor with deterministic fallback.

    The constructor keeps a `model_name` parameter so the class can later
    be extended with Outlines/vLLM/OpenAI structured outputs without
    changing the API layer.
    """

    def __init__(self, model_name: str = "deterministic") -> None:
        self.model_name = model_name

    def extract_from_chunk(self, chunk: Chunk) -> ExtractionResult:
        text = chunk.text or ""
        entities: List[MaterialEntity] = []
        relations: List[RelationAssertion] = []
        seen = set()
        entity_by_key: Dict[tuple, MaterialEntity] = {}

        def add_entity(name: str, entity_type: str) -> MaterialEntity:
            key = (entity_type, normalise(name))
            if key in seen:
                # Return a compatible object for relation building.
                return entity_by_key[key]
            seen.add(key)
            entity = MaterialEntity(
                canonical_name=name.strip(),
                entity_type=entity_type,
                aliases=[],
                norm_name=normalise(name),
            )
            entities.append(entity)
            entity_by_key[key] = entity
            return entity

        materials: List[MaterialEntity] = []
        for pattern in MATERIAL_PATTERNS:
            for match in pattern.finditer(text):
                raw_material = match.group(0)
                if is_false_material(raw_material):
                    continue
                materials.append(add_entity(canonical_material_name(raw_material), "Material"))

        lower = text.lower()
        properties: List[MaterialEntity] = []
        for term, canonical in PROPERTY_TERMS.items():
            if term in lower:
                properties.append(add_entity(canonical, "Property"))

        process_regimes: List[MaterialEntity] = []
        explicit_process_values = explicit_processes(text)
        if explicit_process_values:
            process_candidates = explicit_process_values
        else:
            process_candidates = []
            for term, canonical in PROCESS_TERMS.items():
                if term in lower and canonical not in process_candidates:
                    process_candidates.append(canonical)
        for canonical in process_candidates:
            add_entity(canonical, "ProcessCondition")
            process_regimes.append(add_entity(process_regime_name(canonical, text), "ProcessRegime"))

        equipment_entities: List[MaterialEntity] = []
        for match in EQUIPMENT_RE.finditer(text):
            equipment_entities.append(add_entity(match.group("name").strip(" «»\""), "Equipment"))

        lab_entities: List[MaterialEntity] = []
        for match in LAB_RE.finditer(text):
            lab_name = clean_laboratory_name(match.group("name"))
            if lab_name and normalise(lab_name) not in {"conclusion", "вывод"}:
                lab_entities.append(add_entity(lab_name, "Laboratory"))

        technical_objects: List[MaterialEntity] = []
        for match in TECHNICAL_OBJECT_RE.finditer(text):
            name = canonical_object_name(match)
            if name:
                technical_objects.append(add_entity(name, "TechnicalObject"))

        parts: List[MaterialEntity] = []
        for match in PART_RE.finditer(text):
            parts.append(add_entity(match.group("part").strip(), "Part"))

        articles: List[MaterialEntity] = []
        for match in ARTICLE_RE.finditer(text):
            articles.append(add_entity(match.group(0).upper(), "ArticleNumber"))

        standards: List[MaterialEntity] = []
        for match in STANDARD_RE.finditer(text):
            standards.append(add_entity(re.sub(r"\s+", " ", match.group(0).strip().upper()), "Standard"))

        parameters: List[MaterialEntity] = []
        for regex in PARAMETER_RES:
            for match in regex.finditer(text):
                name = match.groupdict().get("name") or "Parameter"
                value = (match.groupdict().get("value") or match.group(0)).replace(",", ".")
                unit = normalise_unit(match.groupdict().get("unit"))
                if name.lower() == "ду":
                    name = "DN"
                label = f"{name.upper() if len(name) <= 3 else name}: {value}{(' ' + unit) if unit else ''}"
                parameters.append(add_entity(label, "Parameter"))

        requirements: List[MaterialEntity] = []
        for match in REQUIREMENT_RE.finditer(text):
            req_text = match.group("text").strip()
            if req_text:
                requirements.append(add_entity(req_text, "Requirement"))

        image_entities: List[MaterialEntity] = []
        for match in IMAGE_RE.finditer(text):
            url = match.group("url").strip()
            alt = match.group("alt").strip()
            caption = match.group("caption").strip()
            label = caption or alt or url
            image_entities.append(add_entity(label, "ImageArtifact"))
        for match in IMAGE_PATH_RE.finditer(text):
            url = match.group("url").strip()
            nearby_start = max(0, match.start() - 80)
            nearby = text[nearby_start:match.start()].lower()
            caption = "схема" if "схем" in nearby or "монтаж" in nearby else "image"
            image_entities.append(add_entity(f"{caption}: {url}", "ImageArtifact"))

        experiment_id_match = EXPERIMENT_ID_RE.search(text)
        experiment_match = EXPERIMENT_RE.search(text)
        if experiment_id_match:
            experiment_name = experiment_id_match.group("name").strip()
        elif experiment_match and normalise(experiment_match.group("name")) not in {"s", "id", "_id"}:
            experiment_name = experiment_match.group("name").strip()
        elif materials or process_regimes or properties:
            experiment_name = f"Experiment {chunk.doc_id}:{chunk.ordinal if chunk.ordinal is not None else 0}"
        else:
            experiment_name = ""
        experiment = add_entity(experiment_name, "Experiment") if experiment_name else None

        evidence = SourceRef(
            doc_id=chunk.doc_id,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_id=chunk.chunk_id,
            quote=text[:500],
        )

        context_objects = technical_objects[:3]
        if not context_objects and parts:
            context_objects = parts[:1]

        for obj in technical_objects:
            for param in parameters[:12]:
                relations.append(RelationAssertion(subject=obj.canonical_name, predicate="OBJECT_HAS_PARAMETER", object=param.canonical_name, confidence=0.78, evidence=[evidence]))
            for part in parts[:8]:
                relations.append(RelationAssertion(subject=obj.canonical_name, predicate="OBJECT_HAS_PART", object=part.canonical_name, confidence=0.72, evidence=[evidence]))
            for material in materials[:8]:
                relations.append(RelationAssertion(subject=obj.canonical_name, predicate="OBJECT_MADE_OF_MATERIAL", object=material.canonical_name, confidence=0.72, evidence=[evidence]))
            for standard in standards[:8]:
                relations.append(RelationAssertion(subject=obj.canonical_name, predicate="OBJECT_COMPLIES_WITH_STANDARD", object=standard.canonical_name, confidence=0.75, evidence=[evidence]))

        for part in parts[:8]:
            for article in articles[:8]:
                relations.append(RelationAssertion(subject=part.canonical_name, predicate="PART_HAS_ARTICLE_NUMBER", object=article.canonical_name, confidence=0.7, evidence=[evidence]))

        for requirement in requirements:
            targets = context_objects or technical_objects or materials or standards
            for target in targets[:3]:
                relations.append(RelationAssertion(subject=requirement.canonical_name, predicate="REQUIREMENT_APPLIES_TO_OBJECT", object=target.canonical_name, confidence=0.68, evidence=[evidence]))

        for image in image_entities:
            relations.append(RelationAssertion(subject=image.canonical_name, predicate="IMAGE_LINKED_TO_SECTION", object=chunk.section_path or "/", confidence=0.8, evidence=[evidence]))

        if technical_objects or parts or articles or standards or parameters or requirements or image_entities:
            for entity in technical_objects + parts + articles + standards + parameters + requirements + image_entities:
                relations.append(RelationAssertion(subject=chunk.chunk_id, predicate="CHUNK_MENTIONS_ENTITY", object=entity.canonical_name, confidence=0.8, evidence=[evidence]))

        if experiment:
            for material in materials[:5]:
                relations.append(RelationAssertion(subject=experiment.canonical_name, predicate="STUDIES", object=material.canonical_name, confidence=0.75, evidence=[evidence]))
            for regime in process_regimes[:5]:
                relations.append(RelationAssertion(subject=experiment.canonical_name, predicate="USES_REGIME", object=regime.canonical_name, confidence=0.72, evidence=[evidence]))
            for equipment in equipment_entities[:5]:
                relations.append(RelationAssertion(subject=experiment.canonical_name, predicate="USES_EQUIPMENT", object=equipment.canonical_name, confidence=0.7, evidence=[evidence]))
            for lab in lab_entities[:5]:
                relations.append(RelationAssertion(subject=experiment.canonical_name, predicate="PERFORMED_BY", object=lab.canonical_name, confidence=0.7, evidence=[evidence]))

        # MarkItDown often converts engineering datasheet tables to markdown
        # rows where units live in the header, not near each numeric cell. The
        # generic measurement regex cannot infer that "920" belongs to MPa in
        # "Tensile Strength | MPa". Handle this common material-property table
        # shape explicitly.
        if "tensile strength" in lower and "yield strength" in lower and ("mpa" in lower or "ksi" in lower):
            strength_unit = "MPa" if "mpa" in lower else "ksi"
            cells = [
                cell.strip()
                for cell in text.split("|")
                if cell.strip() and cell.strip() != "---"
            ]
            numeric_re = re.compile(r"^\d+(?:[\.,]\d+)?$")
            for idx, cell in enumerate(cells):
                cell_lower = cell.lower()
                if "ti-6al-4v" not in cell_lower and "vt6" not in cell_lower and "7075" not in cell_lower:
                    continue
                tail = cells[idx + 1: idx + 8]
                numbers = [value.replace(",", ".") for value in tail if numeric_re.match(value)]
                if strength_unit == "MPa" and len(numbers) < 4:
                    continue
                if strength_unit == "ksi" and len(numbers) < 2:
                    continue
                material = add_entity(canonical_material_name(cell), "Material")
                if material not in materials:
                    materials.append(material)
                if strength_unit == "MPa":
                    table_values = [
                        ("Tensile strength", numbers[1], "MPa"),
                        ("Yield strength", numbers[3], "MPa"),
                    ]
                    if len(numbers) >= 5:
                        table_values.append(("Elongation", numbers[4], "%"))
                else:
                    table_values = [
                        ("Tensile strength", numbers[0], "ksi"),
                        ("Yield strength", numbers[1], "ksi"),
                    ]
                    if len(numbers) >= 3:
                        table_values.append(("Elongation", numbers[2], "%"))
                table_src = SourceRef(
                    doc_id=chunk.doc_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_id=chunk.chunk_id,
                    quote=cell[:160],
                )
                for property_name, value, unit in table_values:
                    property_entity = add_entity(property_name, "Property")
                    property_value = add_entity(f"{property_name} = {value} {unit}", "PropertyValue")
                    relations.append(
                        RelationAssertion(
                            subject=property_value.canonical_name,
                            predicate="OF_PROPERTY",
                            object=property_entity.canonical_name,
                            qualifiers={"unit": unit, "value": value},
                            confidence=0.82,
                            evidence=[table_src],
                        )
                    )
                    relations.append(
                        RelationAssertion(
                            subject=material.canonical_name,
                            predicate="HAS_MEASUREMENT",
                            object=property_value.canonical_name,
                            qualifiers={"unit": unit, "value": value},
                            confidence=0.78,
                            evidence=[table_src],
                        )
                    )
                    if experiment:
                        relations.append(
                            RelationAssertion(
                                subject=experiment.canonical_name,
                                predicate="MEASURES",
                                object=property_value.canonical_name,
                                qualifiers={"unit": unit, "value": value, "direction": ""},
                                confidence=0.75,
                                evidence=[table_src],
                            )
                        )

        table_property = table_field(text, "property")
        table_value = table_field(text, "value").replace(",", ".")
        table_unit = normalise_unit(table_field(text, "unit"))
        table_effect = table_field(text, "effect").lower()
        if table_property and table_value and table_unit:
            property_name = table_property
            for term, canonical in PROPERTY_TERMS.items():
                if term in table_property.lower():
                    property_name = canonical
                    break
            direction = ""
            if table_effect.startswith("increas") or table_effect.startswith("повыс") or table_effect.startswith("увелич"):
                direction = "increase"
            elif table_effect.startswith("decreas") or table_effect.startswith("сниз") or table_effect.startswith("уменьш"):
                direction = "decrease"
            elif table_effect.startswith("unchanged") or "без измен" in table_effect:
                direction = "unchanged"
            property_entity = add_entity(property_name, "Property")
            property_value = add_entity(f"{property_name} = {table_value} {table_unit}", "PropertyValue")
            qualifiers = {"unit": table_unit, "value": table_value}
            if direction:
                qualifiers["direction"] = direction
                change_entity = add_entity(direction, "PropertyChange")
                relations.append(
                    RelationAssertion(
                        subject=property_value.canonical_name,
                        predicate="HAS_CHANGE",
                        object=change_entity.canonical_name,
                        qualifiers={"direction": direction},
                        confidence=0.82,
                        evidence=[evidence],
                    )
                )
            relations.append(
                RelationAssertion(
                    subject=property_value.canonical_name,
                    predicate="OF_PROPERTY",
                    object=property_entity.canonical_name,
                    qualifiers={"unit": table_unit, "value": table_value},
                    confidence=0.86,
                    evidence=[evidence],
                )
            )
            if experiment:
                relations.append(
                    RelationAssertion(
                        subject=experiment.canonical_name,
                        predicate="MEASURES",
                        object=property_value.canonical_name,
                        qualifiers=qualifiers,
                        confidence=0.84,
                        evidence=[evidence],
                    )
                )
            for material in materials[:5]:
                relations.append(
                    RelationAssertion(
                        subject=material.canonical_name,
                        predicate="HAS_MEASUREMENT",
                        object=property_value.canonical_name,
                        qualifiers=qualifiers,
                        confidence=0.82,
                        evidence=[evidence],
                    )
                )

        for match in MEASUREMENT_RE.finditer(text):
            # normalise decimal separators
            value = match.group("value").replace(",", ".")
            raw_unit = match.group("unit")
            # canonicalise units: harmonise synonyms (°C vs C, hours vs h, minutes)
            unit_map = {
                "°c": "C",
                "°с": "C",
                "c": "C",
                "с": "C",
                "k": "K",
                "mpa": "MPa",
                "gpa": "GPa",
                "hv": "HV",
                "hrc": "HRC",
                "%": "%",
                "h": "h",
                "hr": "h",
                "hrs": "h",
                "hour": "h",
                "hours": "h",
                "мин": "h",
                "мин.": "h",
                "ч": "h",
                "ч.": "h",
                "ppm": "ppm",
            }
            unit_lower = raw_unit.lower().replace("°", "")
            unit = unit_map.get(unit_lower, raw_unit)
            if unit in {"C", "K", "h"}:
                continue
            pressure_window = text[max(0, match.start() - 80): min(len(text), match.end() + 80)].lower()
            if unit in {"MPa", "GPa", "bar"} and any(marker in pressure_window for marker in ["pressure", "давление", "напор", "p=", "p =", "pn"]):
                continue
            measurement_label = f"{value} {unit}"
            # Determine effect direction heuristically by looking around the measurement
            # Define a window of characters before and after the measurement
            window_start = max(0, match.start() - 50)
            window_end = min(len(text), match.end() + 50)
            context_text = text[window_start:window_end]
            direction = infer_direction_near(text, match.start(), match.end())
            property_name = infer_property_from_context(unit, context_text, [p.canonical_name for p in properties])
            property_entity = add_entity(property_name, "Property") if property_name else None
            property_value_name = f"{property_name or 'Value'} = {measurement_label}"
            measurement = add_entity(property_value_name, "PropertyValue")
            if direction:
                change_entity = add_entity(direction, "PropertyChange")
                relations.append(
                    RelationAssertion(
                        subject=measurement.canonical_name,
                        predicate="HAS_CHANGE",
                        object=change_entity.canonical_name,
                        qualifiers={"direction": direction},
                        confidence=0.68,
                        evidence=[evidence],
                    )
                )
            if property_entity:
                relations.append(
                    RelationAssertion(
                        subject=measurement.canonical_name,
                        predicate="OF_PROPERTY",
                        object=property_entity.canonical_name,
                        qualifiers={"unit": unit, "value": value},
                        confidence=0.72,
                        evidence=[evidence],
                    )
                )
            if experiment:
                relations.append(
                    RelationAssertion(
                        subject=experiment.canonical_name,
                        predicate="MEASURES",
                        object=measurement.canonical_name,
                        qualifiers={"unit": unit, "value": value, "direction": direction or ""},
                        confidence=0.68,
                        evidence=[evidence],
                    )
                )
            if materials:
                src = SourceRef(
                    doc_id=chunk.doc_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_id=chunk.chunk_id,
                    quote=text[max(0, match.start() - 80): min(len(text), match.end() + 80)],
                )
                for material in materials[:3]:
                    qualifiers = {"unit": unit, "value": value}
                    if direction:
                        qualifiers["direction"] = direction
                    relations.append(
                        RelationAssertion(
                            subject=material.canonical_name,
                            predicate="HAS_MEASUREMENT",
                            object=measurement.canonical_name,
                            qualifiers=qualifiers,
                            confidence=0.65,
                            evidence=[src],
                        )
                    )

        for match in CONCLUSION_RE.finditer(text):
            conclusion = add_entity(match.group("text").strip(), "Conclusion")
            relations.append(
                RelationAssertion(
                    subject=conclusion.canonical_name,
                    predicate="SUPPORTED_BY",
                    object=chunk.chunk_id,
                    confidence=0.7,
                    evidence=[evidence],
                )
            )

        for match in DATA_GAP_RE.finditer(text):
            gap_text = (match.group("text1") or match.group("text2") or "").strip()
            if not gap_text:
                continue
            gap = add_entity(gap_text, "DataGap")
            target = None
            if technical_objects:
                target = technical_objects[0].canonical_name
            if materials:
                target = materials[0].canonical_name
            if properties:
                target = properties[0].canonical_name
            if process_regimes:
                target = process_regimes[0].canonical_name
            if target:
                relations.append(
                    RelationAssertion(
                        subject=gap.canonical_name,
                        predicate="MISSING_FOR",
                        object=target,
                        confidence=0.7,
                        evidence=[evidence],
                    )
                )

        return ExtractionResult(entities=entities, relations=relations, unresolved_terms=[])

    def batch_extract(self, chunks: List[Chunk]) -> List[ExtractionResult]:
        return [self.extract_from_chunk(chunk) for chunk in chunks]

    @staticmethod
    def entity_counts(result: ExtractionResult) -> Counter:
        return Counter((entity.entity_type, entity.norm_name or normalise(entity.canonical_name)) for entity in result.entities)

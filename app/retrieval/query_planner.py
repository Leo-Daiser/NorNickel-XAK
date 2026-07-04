"""Deterministic query planner for ontology-driven graph QA."""

from __future__ import annotations

import re

from ..domain.aliases import EQUIPMENT_ALIASES, MATERIAL_ALIASES, PROPERTY_ALIASES, REGIME_ALIASES, TOPIC_TAG_ALIASES
from ..domain.normalization import canonical_material, canonical_property, canonical_regime, normalize_text
from ..domain.numeric_constraints import extract_numeric_constraints
from ..domain.query_constraints import QueryConstraints, QueryIntent


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


class QueryPlanner:
    """Parse a raw user question into canonical graph constraints."""

    MATERIAL_PATTERNS = [
        re.compile(r"\b12[ХX]18[НH]10[ТT]\b", re.IGNORECASE),
        re.compile(r"\b09Г2С\b", re.IGNORECASE),
        re.compile(r"\b7075(?:-T6)?\b", re.IGNORECASE),
        re.compile(r"\bVT6\b", re.IGNORECASE),
        re.compile(r"\bВТ6\b", re.IGNORECASE),
        re.compile(r"\bTi-?6Al-?4V\b", re.IGNORECASE),
        re.compile(r"\b(?:сплав(?:у|а|ом|е)?|сталь|стали|alloy|steel)\s+(?P<name>[A-Za-zА-Яа-я0-9\-]+)\b", re.IGNORECASE),
    ]

    OBJECT_MARKERS = ["насос", "pump", "клапан", "valve", "dn50", "npk-200"]
    EQUIPMENT_MARKERS = ["оборудован", "установк", "печь", "прибор", "твердомер", "equipment"]
    TEAM_MARKERS = ["лаборатор", "команд", "группа", "laboratory", "team"]
    GAP_MARKERS = ["пробел", "нет данных", "не хватает", "не исслед", "gap"]
    HISTORY_MARKERS = ["история решений", "цепочка решений", "что пробовали", "историю решений"]
    BROAD_REVIEW_MARKERS = ["какие", "покажите все", "описаны", "миров", "практик", "подходят", "способы", "решения"]
    GEOGRAPHY_ALIASES = {
        "россия": "Россия",
        "россии": "Россия",
        "российск": "Россия",
        "отечествен": "Россия",
        "зарубеж": "зарубежная практика",
        "за рубеж": "зарубежная практика",
        "за рубежом": "зарубежная практика",
        "мировая практика": "мировая практика",
        "мировой практике": "мировая практика",
        "worldwide": "мировая практика",
        "global": "мировая практика",
        "abroad": "зарубежная практика",
        "китай": "Китай",
        "china": "Китай",
        "сша": "США",
        "usa": "США",
        "canada": "Канада",
        "канада": "Канада",
    }

    def parse(self, question: str) -> QueryConstraints:
        """Return canonical constraints for the provided question."""
        raw_question = question or ""
        q = normalize_text(raw_question)
        materials = self._materials(raw_question)
        regimes = self._aliases_in_text(q, REGIME_ALIASES, canonical_regime)
        properties = self._aliases_in_text(q, PROPERTY_ALIASES, canonical_property)
        equipment = self._aliases_in_text(q, EQUIPMENT_ALIASES, lambda value: self._canonical_from_alias(value, EQUIPMENT_ALIASES))
        topic_tags = self._aliases_in_text(q, TOPIC_TAG_ALIASES, lambda value: self._canonical_from_alias(value, TOPIC_TAG_ALIASES))
        numeric_constraints = extract_numeric_constraints(raw_question)
        geographies = self._geographies(q)
        time_filters = self._time_filters(q)
        typed_plan = self._typed_fact_plan(q, materials, regimes, properties, equipment, geographies, numeric_constraints, time_filters)

        intent = QueryIntent.UNKNOWN
        if any(marker in q for marker in self.HISTORY_MARKERS):
            intent = QueryIntent.DECISION_HISTORY
        elif any(marker in q for marker in self.GAP_MARKERS):
            intent = QueryIntent.GAP_ANALYSIS
        elif materials and regimes and properties and self._is_exact_fact_query(q, materials, regimes, properties, numeric_constraints, geographies, time_filters):
            intent = QueryIntent.MATERIAL_REGIME_PROPERTY_EFFECT
        elif equipment or any(marker in q for marker in self.EQUIPMENT_MARKERS):
            intent = QueryIntent.EQUIPMENT_USAGE
        elif any(marker in q for marker in self.TEAM_MARKERS):
            intent = QueryIntent.TEAM_ACTIVITY
        elif any(marker in q for marker in self.OBJECT_MARKERS):
            intent = QueryIntent.ENTITY_OVERVIEW
        elif materials or regimes or properties or equipment or topic_tags or numeric_constraints or geographies or time_filters:
            intent = QueryIntent.ENTITY_OVERVIEW

        return QueryConstraints(
            intent=intent,
            raw_question=raw_question,
            materials=materials,
            regimes=regimes,
            properties=properties,
            equipment=equipment,
            topic_tags=topic_tags,
            numeric_constraints=numeric_constraints,
            geographies=geographies,
            time_filters=time_filters,
            target_fact_types=typed_plan["target_fact_types"],
            answer_mode=typed_plan["answer_mode"],
            require_exact_match=bool(
                intent == QueryIntent.MATERIAL_REGIME_PROPERTY_EFFECT
                and self._is_exact_fact_query(q, materials, regimes, properties, numeric_constraints, geographies, time_filters)
            ),
        )

    def _materials(self, question: str) -> list[str]:
        values: list[str] = []
        for pattern in self.MATERIAL_PATTERNS:
            for match in pattern.finditer(question or ""):
                name = match.groupdict().get("name") or match.group(0)
                values.append(canonical_material(name))
        q = normalize_text(question)
        for alias in MATERIAL_ALIASES:
            if normalize_text(alias) in q:
                values.append(canonical_material(alias))
        return _unique(values)

    @staticmethod
    def _aliases_in_text(text: str, aliases: dict[str, str], canonicalizer) -> list[str]:
        values: list[str] = []
        matched_aliases: list[str] = []
        compact = re.sub(r"[\s_\-]+", "", text)
        for alias in sorted(aliases, key=lambda item: len(normalize_text(item)), reverse=True):
            alias_norm = normalize_text(alias)
            alias_compact = re.sub(r"[\s_\-]+", "", alias_norm)
            if alias_norm in text or alias_compact in compact:
                if any(alias_norm in matched or alias_compact in re.sub(r"[\s_\-]+", "", matched) for matched in matched_aliases):
                    continue
                matched_aliases.append(alias_norm)
                values.append(canonicalizer(alias))
        return _unique(values)

    @staticmethod
    def _canonical_from_alias(value: str, aliases: dict[str, str]) -> str:
        normalized = normalize_text(value)
        return aliases.get(normalized, value)

    def _geographies(self, normalized_question: str) -> list[str]:
        values: list[str] = []
        for marker, label in self.GEOGRAPHY_ALIASES.items():
            if marker in normalized_question:
                values.append(label)
        return _unique(values)

    def _is_exact_fact_query(
        self,
        normalized_question: str,
        materials: list[str],
        regimes: list[str],
        properties: list[str],
        numeric_constraints: list[dict],
        geographies: list[str],
        time_filters: list[dict],
    ) -> bool:
        if numeric_constraints or geographies or time_filters:
            return False
        if len(materials) != 1 or len(regimes) != 1 or len(properties) != 1:
            return False
        if any(marker in normalized_question for marker in self.BROAD_REVIEW_MARKERS):
            return False
        return True

    @staticmethod
    def _time_filters(normalized_question: str) -> list[dict[str, int | str]]:
        filters: list[dict[str, int | str]] = []
        for match in re.finditer(r"(?:за\s+)?последн\w+\s+(?P<years>\d{1,2})\s+лет", normalized_question):
            filters.append({"type": "relative_years", "years": int(match.group("years"))})
        years = [int(item) for item in re.findall(r"\b(?:19|20)\d{2}\b", normalized_question)]
        if years:
            filters.append({"type": "year_range", "start_year": min(years), "end_year": max(years)})
        return filters

    def _typed_fact_plan(
        self,
        normalized_question: str,
        materials: list[str],
        regimes: list[str],
        properties: list[str],
        equipment: list[str],
        geographies: list[str],
        numeric_constraints: list[dict],
        time_filters: list[dict],
    ) -> dict[str, list[str] | str | None]:
        target: list[str] = []
        mode: str | None = None
        q = normalized_question
        asks_for_solution = any(
            term in q
            for term in [
                "какие методы",
                "какие решения",
                "какие системы",
                "какие способы",
                "какие технологии",
                "что применяется",
                "описан",
                "подходят",
                "method",
                "solution",
                "technology",
                "system",
                "approach",
            ]
        )

        if asks_for_solution:
            target.extend(["TechnologySolutionFact", "PublicationClaimFact"])
            mode = "technology_solution_search"

        if any(term in q for term in ["лаборатор", "команд", "автор", "эксперт", "исполнитель", "researcher", "author"]):
            target.extend(["ExpertiseFact", "PublicationClaimFact"])
            mode = "expert_search"

        if any(term in q for term in ["католит", "электроэкстракц", "скорость потока", "циркуляц", "расход"]):
            target.extend(["TechnologySolutionFact", "ProcessParameterFact", "PublicationClaimFact"])
            mode = mode or ("process_parameter_search" if any(term in q for term in ["скорость", "расход", "параметр"]) else "technology_solution_search")

        if any(term in q for term in ["распредел", "извлечен", "извлечение", "содержание"]) and (
            any(item in {"Au", "Ag", "МПГ", "штейн", "шлак"} for item in materials)
            or any(term in q for term in ["au", "ag", "мпг", "штейн", "шлак"])
        ):
            target.extend(["ExperimentResultFact", "ProcessParameterFact", "PublicationClaimFact"])
            mode = "experiment_catalog_search"

        if any(term in q for term in ["закачка шахтных вод", "шахтн", "технико-эконом", "экономическ", "капекс", "opex", "стоимост"]):
            target.extend(["TechnologySolutionFact", "EconomicIndicatorFact", "PublicationClaimFact", "ProcessParameterFact"])
            mode = "domestic_vs_foreign_practice" if geographies or any(term in q for term in ["росси", "зарубеж", "миров"]) else "technology_comparison"

        if any(term in q for term in ["обессол", "водоочист", "сухой остаток", "концентрац"]) or numeric_constraints:
            target.extend(["TechnologySolutionFact", "ProcessParameterFact", "PublicationClaimFact"])
            mode = mode or ("process_parameter_search" if any(term in q for term in ["сухой остаток", "концентрац", "мг/л", "мг/дм"]) or numeric_constraints else "technology_solution_search")

        if equipment or any(term in q for term in ["оборудован", "установк", "печь", "ванн", "ячейк", "reactor", "furnace"]):
            target.extend(["TechnologySolutionFact", "ProcessParameterFact", "PublicationClaimFact", "FacilityCapacityFact"])
            mode = mode or "technology_comparison"

        if time_filters and not target:
            target.extend(["ExperimentResultFact", "PublicationClaimFact"])
            mode = "literature_review"

        if properties and not target:
            target.extend(["ExperimentResultFact", "ProcessParameterFact"])
            mode = "generic_typed_fact_summary"

        return {"target_fact_types": _unique(target), "answer_mode": mode}

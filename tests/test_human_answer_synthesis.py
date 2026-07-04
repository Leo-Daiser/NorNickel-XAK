from __future__ import annotations

from app.answering.human_answer import enhance_answer_payload


def _exact_payload() -> dict:
    return {
        "answer": "legacy technical answer",
        "status": "ok",
        "answer_mode": "graph_exact",
        "analytical_intent": "strict_material_regime_property",
        "constraints": {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]},
        "facts": [
            {
                "experiment_id": "SCI-VT6-AN-900",
                "material": "Titanium Alpha-Beta, ВТ6",
                "regime": "отжиг",
                "property": "прочность",
                "value": 1120.0,
                "unit": "MPa",
                "effect": "increase",
                "evidence": [
                    {
                        "document_id": "doc_demo",
                        "chunk_id": "chunk_demo",
                        "source_name": "demo.csv",
                        "quote": "ВТ6 отжиг прочность 1120 MPa",
                    }
                ],
            },
            {
                "experiment_id": "EXP-1cc92a75d794",
                "material": "ВТ6",
                "regime": "отжиг",
                "property": "прочность",
                "value": 980.0,
                "unit": "MPa",
                "effect": "unknown",
                "evidence": [
                    {
                        "document_id": "doc_demo2",
                        "chunk_id": "chunk_demo2",
                        "source_name": "demo.txt",
                        "quote": "ВТ6 отжиг прочность 980 MPa",
                    }
                ],
            },
        ],
        "sources": [],
        "subgraph": {"nodes": [{"id": "Material:ВТ6"}], "edges": []},
        "graph_context": {},
        "diagnostics": {},
        "retrieval": {},
    }


def test_strict_positive_human_answer_hides_internal_ids() -> None:
    payload = enhance_answer_payload(_exact_payload(), "expert_max")
    answer = payload["answer"]
    assert "ВТ6" in answer
    assert "отжиг" in answer
    assert "прочность" in answer
    assert "1120" in answer
    assert "Ограничения" in answer
    for forbidden in ["doc_", "chunk_", "EXP-", "SCI-", "effect:", "unknown"]:
        assert forbidden not in answer


def test_strict_negative_human_answer_is_clear() -> None:
    payload = enhance_answer_payload(
        {
            "answer": "Ближайшие данные: закалка при 1050 °C для другого материала.",
            "status": "no_exact_match",
            "constraints": {"materials": ["ВТ6"], "regimes": ["криообработка"], "properties": ["вязкость"]},
            "facts": [],
            "sources": [],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "diagnostics": {},
            "retrieval": {},
        },
        "expert_max",
    )
    answer = payload["answer"].lower()
    assert "точных данных" in answer
    assert "вт6" in answer
    assert "криообработка" in answer
    assert "вязкость" in answer
    assert "нельзя считать ответом" in answer


def test_comparison_answer_warns_about_comparability() -> None:
    payload = _exact_payload()
    payload["answer_mode"] = "comparison"
    payload["analytical_intent"] = "material_comparison"
    payload["constraints"] = {"materials": ["ВТ6", "7075-T6"], "regimes": [], "properties": ["прочность"]}
    answer = enhance_answer_payload(payload, "expert_max")["answer"].lower()
    assert "сравнение ограничено" in answer
    assert "не прямое экспериментальное сравнение" in answer


def test_overview_without_facts_does_not_claim_experiments_found() -> None:
    payload = enhance_answer_payload(
        {
            "answer": "legacy",
            "status": "partial",
            "answer_mode": "overview",
            "analytical_intent": "material_overview",
            "constraints": {"materials": ["X999"], "raw_question": "Что известно о сплаве X999 при лазерной обработке?"},
            "facts": [],
            "sources": [{"source_name": "nearest.txt", "quote": "unrelated evidence"}],
            "evidence": [{"source_name": "nearest.txt", "quote": "unrelated evidence"}],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "diagnostics": {"llm_answer_polished": True},
            "retrieval": {},
        },
        "expert_max",
    )
    answer = payload["answer"].lower()

    assert "структурированных фактов" in answer
    assert "нет подтверждённых" in answer or "не найдено" in answer
    assert "найдены связанные эксперименты" not in answer
    assert "1050" not in answer


def test_source_grounded_answer_uses_navigation_title_not_experiment_template() -> None:
    payload = enhance_answer_payload(
        {
            "answer": "Структурированных AcceptedFact недостаточно, поэтому ответ дан как навигационный по источникам.",
            "status": "partial",
            "answer_mode": "source_grounded_answer",
            "answer_is_verified": False,
            "source_grounded_answer_used": True,
            "facts": [],
            "sources": [
                {
                    "source_name": "deep_mine_cooling.pdf",
                    "page_start": 6,
                    "quote": "Источники тепла включают самосжатие воздуха, геотермальный поток и тепло оборудования.",
                }
            ],
            "evidence": [],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "diagnostics": {},
            "retrieval": {},
        },
        "expert_max",
    )
    answer = payload["answer"]

    assert "Ответ по найденным источникам" in answer
    assert "Подтверждённые экспериментальные данные" not in answer
    assert payload["human_answer"]["title"] == "Ответ по найденным источникам"
    assert payload["answer_is_verified"] is False


def test_technology_review_answer_shows_conditions_sources_and_limits() -> None:
    payload = enhance_answer_payload(
        {
            "answer": "legacy",
            "status": "ok",
            "answer_mode": "overview",
            "analytical_intent": "entity_overview",
            "constraints": {
                "raw_question": "Какие методы обессоливания воды подходят для обогатительной фабрики?",
                "materials": ["шахтные воды", "сульфаты", "хлориды"],
                "regimes": ["обессоливание"],
                "properties": ["концентрация", "сухой остаток"],
                "numeric_constraints": [
                    {"parameter": "сульфаты", "min": 200, "max": 300, "unit": "mg/L"},
                    {"parameter": "сухой остаток", "operator": "<=", "value": 1000, "unit": "mg/L"},
                ],
                "geographies": ["Россия", "зарубежная практика"],
                "time_filters": [],
            },
            "facts": [
                {
                    "material": "сульфаты",
                    "regime": "обессоливание",
                    "property": "концентрация",
                    "value": 250,
                    "unit": "mg/L",
                    "effect": "unknown",
                    "evidence": [
                        {
                            "source_name": "internal_report.csv",
                            "document_id": "doc_demo",
                            "chunk_id": "chunk_demo",
                            "quote": "сульфаты 250 мг/л после обессоливания",
                            "source_metadata": {
                                "practice_scope": "domestic",
                                "publication_year": 2024,
                                "reliability_level": "medium",
                            },
                        }
                    ],
                }
            ],
            "sources": [],
            "evidence": [],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "diagnostics": {},
            "retrieval": {"source_metadata_filter": {"applied": True, "matched_chunks": 1}},
        },
        "expert_max",
    )
    answer = payload["answer"].lower()

    assert "технологический обзор" in answer
    assert "условия из вопроса" in answer
    assert "сульфаты: 200-300 mg/l" in answer
    assert "сухой остаток: ≤1000 mg/l" in answer
    assert "обессоливание" in answer
    assert "отечественная практика" in answer
    assert "достоверность: средняя" in answer
    assert "llm не использовался как источник фактов" in answer
    for forbidden in ["doc_demo", "chunk_demo", "EXP-", "SCI-", "unknown"]:
        assert forbidden.lower() not in answer


def test_broad_tz_query_with_uncovered_constraints_returns_coverage_gap() -> None:
    payload = enhance_answer_payload(
        {
            "answer": "legacy object summary",
            "status": "ok",
            "answer_mode": "overview",
            "analytical_intent": "equipment_usage",
            "constraints": {
                "raw_question": "Какие схемы подачи электролита в ванны электроэкстракции никеля описаны в мировой практике?",
                "materials": ["никель", "электролит"],
                "regimes": ["электроэкстракция"],
                "properties": [],
                "equipment": ["ванна электроэкстракции", "диафрагменная ячейка"],
                "geographies": ["мировая практика"],
                "topic_tags": [],
                "numeric_constraints": [],
                "time_filters": [],
            },
            "facts": [
                {
                    "predicate": "PART_HAS_ARTICLE_NUMBER",
                    "object": "насос",
                    "material": "",
                    "regime": "",
                    "property": "",
                    "value": None,
                    "unit": "",
                    "evidence": [{"source_name": "pump_catalog.pdf", "quote": "насос и артикул"}],
                }
            ],
            "technical_objects": [{"name": "насос"}],
            "sources": [{"source_name": "pump_catalog.pdf", "quote": "насос и артикул"}],
            "evidence": [{"source_name": "pump_catalog.pdf", "quote": "насос и артикул"}],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "diagnostics": {},
            "retrieval": {},
        },
        "expert_max",
    )

    answer = payload["answer"].lower()

    assert "технологический обзор" in answer
    assert "покрытие неполное" in answer
    assert "нет данных" in answer or "нет подтверждённого" in answer
    assert "никель" in answer
    assert "электроэкстракция" in answer
    assert "ванна электроэкстракции" in answer
    assert "диафрагменная ячейка" in answer
    assert "мировая практика" in answer
    assert "сводка по объекту насос" not in answer
    for forbidden in ["doc_", "chunk_", "EXP-", "SCI-"]:
        assert forbidden.lower() not in answer


def test_exact_vt6_answer_does_not_trigger_broad_coverage_gap() -> None:
    payload = enhance_answer_payload(_exact_payload(), "expert_max")
    answer = payload["answer"].lower()

    assert "подтверждённые экспериментальные данные" in answer
    assert "покрытие неполное" not in answer


def test_material_inventory_with_relation_predicates_is_not_object_summary() -> None:
    payload = enhance_answer_payload(
        {
            "answer": "legacy",
            "status": "ok",
            "answer_mode": "rule_based",
            "intent": "material_inventory",
            "constraints": {"raw_question": "Какие материалы, процессы и свойства встречаются?"},
            "facts": [
                {
                    "subject": "chunk1",
                    "predicate": "CHUNK_MENTIONS_ENTITY",
                    "object": "Parameter: 500 MPa",
                    "source_chunk_id": "chunk1",
                    "doc_id": "doc1",
                },
                {
                    "subject": "ки",
                    "predicate": "STUDIES",
                    "object": "сплав сендаст",
                    "source_chunk_id": "chunk1",
                    "doc_id": "doc1",
                },
            ],
            "sources": [{"source_name": "doc_1a3fbdf07eab31e97b57ee22_real.pdf", "quote": "сплав сендаст 500 MPa"}],
            "evidence": [{"source_name": "doc_1a3fbdf07eab31e97b57ee22_real.pdf", "quote": "сплав сендаст 500 MPa"}],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "diagnostics": {},
            "retrieval": {},
        },
        "expert_max",
    )

    answer = payload["answer"].lower()
    assert "сводка по объекту" not in answer
    assert "техническому объекту" not in answer
    assert "обзор активного корпуса" in answer
    assert "сплав сендаст" in answer
    assert "500 MPa".lower() in answer
    assert "материал не указан" not in answer
    assert "none" not in answer
    assert "doc_" not in answer
    assert "real.pdf" in answer


def test_lab_activity_human_answer_lists_laboratories_cleanly() -> None:
    payload = enhance_answer_payload(
        {
            "answer": "legacy",
            "status": "ok",
            "answer_mode": "overview",
            "analytical_intent": "lab_activity",
            "constraints": {"raw_question": "Какие лаборатории или команды выполняли эксперименты?"},
            "facts": [],
            "laboratories": [
                {"canonical_name": "Лаборатория легких сплавов"},
                {"canonical_name": "Лаборатория термообработки"},
            ],
            "teams": [{"name": "Команда гидрометаллургии"}],
            "employees": [{"name": "Иванов И.И."}],
            "sources": [],
            "evidence": [],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "diagnostics": {},
            "retrieval": {},
        },
        "expert_max",
    )
    answer = payload["answer"]

    assert "Лаборатории, команды и исполнители" in answer
    assert "Лаборатория легких сплавов" in answer
    assert "Лаборатория термообработки" in answer
    assert "Команда гидрометаллургии" in answer
    assert "Иванов И.И." in answer
    for forbidden in ["doc_", "chunk_", "EXP-", "SCI-"]:
        assert forbidden not in answer

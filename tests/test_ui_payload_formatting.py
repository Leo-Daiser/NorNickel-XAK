from __future__ import annotations

from app.ui_helpers import (
    answer_evidence_summary_rows,
    answer_source_metadata_rows,
    conflict_explanation_rows,
    evidence_to_rows,
    evidence_to_user_rows,
    facts_to_rows,
    facts_to_user_rows,
    friendly_source_name,
    graph_context_stats,
    no_exact_match_warning,
    subgraph_to_tables,
)


def test_facts_convert_to_rows() -> None:
    rows = facts_to_rows({"facts": [{"experiment_id": "EXP-1", "material": "ВТ6"}]})
    assert rows == [{"experiment_id": "EXP-1", "material": "ВТ6"}]


def test_evidence_convert_to_table_rows() -> None:
    rows = evidence_to_rows(
        {
            "evidence": [
                {
                    "source_name": "demo.txt",
                    "document_id": "doc1",
                    "chunk_id": "chunk1",
                    "score": 0.9,
                    "retrieval_backend": "bm25",
                    "quote": "ВТ6 отжиг",
                }
            ]
        }
    )
    assert rows[0]["source_name"] == "demo.txt"
    assert rows[0]["doc_id"] == "doc1"


def test_user_fact_and_evidence_rows_hide_raw_ids() -> None:
    payload = {
        "facts": [
            {
                "experiment_id": "EXP-1",
                "material": "ВТ6",
                "property": "прочность",
                "value": 980,
                "unit": "MPa",
                "evidence": [
                    {
                        "source_name": "demo.txt",
                        "document_id": "doc_abc",
                        "chunk_id": "chunk_abc",
                        "quote": "После отжига ВТ6 предел прочности составил 980 MPa.",
                    }
                ],
            }
        ],
        "evidence": [
            {
                "source_name": "demo.txt",
                "document_id": "doc_abc",
                "chunk_id": "chunk_abc",
                "quote": "После отжига ВТ6 предел прочности составил 980 MPa.",
            }
        ],
    }

    fact_rows = facts_to_user_rows(payload)
    evidence_rows = evidence_to_user_rows(payload)
    rendered = f"{fact_rows} {evidence_rows}"

    assert "Experiment ID" not in fact_rows[0]
    assert "Chunk ID" not in fact_rows[0]
    assert "Фрагмент" not in evidence_rows[0]
    assert "doc_abc" not in rendered
    assert "chunk_abc" not in rendered
    assert "Основание" in fact_rows[0]


def test_answer_evidence_summary_uses_normalized_and_original_units_without_raw_ids() -> None:
    payload = {
        "facts": [
            {
                "experiment_id": "EXP-7075",
                "material": "7075-T6",
                "regime": "старение",
                "property": "прочность",
                "value": 77,
                "unit": "ksi",
                "value_original": 77,
                "unit_original": "ksi",
                "value_normalized": 530.9,
                "unit_normalized": "MPa",
                "evidence": [
                    {
                        "source_name": "article_7075.txt",
                        "document_id": "doc_7075",
                        "chunk_id": "chunk_7075",
                        "quote": "The 7075-T6 aluminum alloy showed tensile strength of 77 ksi after aging treatment.",
                    }
                ],
            }
        ]
    }

    rows = answer_evidence_summary_rows(payload)
    rendered = str(rows)

    assert rows[0]["Факт"] == "7075-T6 · старение · прочность: 530.9 MPa"
    assert rows[0]["Исходное значение"] == "77 ksi"
    assert rows[0]["Источник"] == "Статья по 7075-T6"
    assert "77 ksi" in rows[0]["Фрагмент"]
    for forbidden in ["doc_7075", "chunk_7075", "EXP-7075", "SourceChunk", "PropertyValue"]:
        assert forbidden not in rendered


def test_answer_source_metadata_rows_group_sources_without_raw_ids() -> None:
    payload = {
        "evidence": [
            {
                "source_name": "doc_abc_internal_report.csv",
                "document_id": "doc_abc",
                "chunk_id": "chunk_abc",
                "quote": "российская практика электроэкстракции никеля",
                "source_metadata": {
                    "publication_year": 2024,
                    "geographies": ["Россия"],
                    "practice_scope": "domestic",
                    "source_type_detected": "internal_report",
                    "reliability_level": "medium",
                },
            },
            {
                "source_name": "world_review.html",
                "source_url": "https://example.org/reports/nickel-electrowinning.html?utm_source=demo",
                "source_type": "url",
                "title": "Nickel electrowinning world practice",
                "quote": "world practice describes catholyte circulation",
                "source_metadata": {
                    "publication_year": 2023,
                    "geographies": ["мировая практика"],
                    "practice_scope": "foreign_or_global",
                    "source_type_detected": "publication",
                    "reliability_level": "high",
                },
            },
        ]
    }

    rows = answer_source_metadata_rows(payload)
    rendered = str(rows)

    assert any(row["Практика"] == "Отечественная практика" for row in rows)
    assert any(row["Практика"] == "Зарубежная/мировая практика" for row in rows)
    assert any(row["Год"] == "2024" for row in rows)
    assert any(row["Тип источника"] == "Публикация/обзор" for row in rows)
    assert any(row["Достоверность"] == "высокая" for row in rows)
    for forbidden in ["doc_abc", "chunk_abc", "utm_source", "EXP-", "SCI-"]:
        assert forbidden not in rendered


def test_conflict_explanation_is_human_readable_without_raw_ids() -> None:
    payload = {
        "diagnostics": {
            "fact_conflicts": [
                {
                    "material": "ВТ6",
                    "regime": "отжиг",
                    "property": "прочность",
                    "values": [
                        {"value": 980, "unit": "MPa", "value_original": 980, "unit_original": "MPa"},
                        {"value": 1120, "unit": "MPa", "value_original": 1120, "unit_original": "MPa"},
                    ],
                    "sources_count": 2,
                    "possible_reason": "sources report different numeric values for the same material/regime/property; check source conditions",
                }
            ]
        }
    }

    rows = conflict_explanation_rows(payload)
    rendered = str(rows)

    assert "Для ВТ6 после отжига найдены разные значения прочности: 980 MPa и 1120 MPa" in rows[0]["Описание"]
    assert "различаются параметры режима" in rows[0]["Описание"]
    for forbidden in ["doc_", "chunk_", "EXP-", "SCI-", "increase", "decrease", "unknown"]:
        assert forbidden not in rendered


def test_subgraph_convert_to_node_edge_tables() -> None:
    nodes, edges = subgraph_to_tables(
        {
            "nodes": [{"id": "Material:ВТ6"}],
            "edges": [{"source": "Experiment:E1", "target": "Material:ВТ6"}],
        }
    )
    assert nodes[0]["id"] == "Material:ВТ6"
    assert edges[0]["target"] == "Material:ВТ6"


def test_missing_optional_fields_do_not_crash() -> None:
    assert facts_to_rows({}) == []
    assert evidence_to_rows({}) == []
    assert subgraph_to_tables(None) == ([], [])
    assert graph_context_stats({})["facts_count"] == 0


def test_no_exact_match_warning_is_generated() -> None:
    warning = no_exact_match_warning({"status": "no_exact_match"})
    assert warning
    assert "Точного факта" in warning


def test_answer_evidence_summary_keeps_readable_source_after_doc_prefix_strip() -> None:
    payload = {
        "facts": [
            {
                "material": "ВТ6",
                "regime": "отжиг",
                "property": "прочность",
                "value": 980,
                "unit": "MPa",
                "evidence": [
                    {
                        "source_name": "doc_29765440445b821ce5d3075b_synthetic_vt6_heat_treatment.csv",
                        "quote": (
                            "Table columns: experiment_id | material | process_regime | property | value | unit | effect\n"
                            "experiment_id: SCI-VT6 | material: ВТ6 | process_regime: отжиг | "
                            "property: прочность | value: 980 | unit: MPa | effect: decreased"
                        ),
                    }
                ],
            }
        ]
    }

    rows = answer_evidence_summary_rows(payload)
    rendered = str(rows)

    assert rows[0]["Источник"] == "Данные по термообработке ВТ6"
    assert "материал: ВТ6" in rows[0]["Фрагмент"]
    assert "Table columns" not in rows[0]["Фрагмент"]
    assert "doc_" not in rendered
    assert "SCI-" not in rendered
    assert "эффект: снижена" in rows[0]["Фрагмент"]
    assert "increased" not in rendered
    assert "decreased" not in rendered


def test_friendly_source_name_hides_demo_technical_tokens() -> None:
    assert friendly_source_name("doc_29765440445b821ce5d3075b_synthetic_vt6_heat_treatment.csv") == "Данные по термообработке ВТ6"
    assert (
        friendly_source_name(
            "doc_29765440445b821ce5d3075b_synthetic_vt6_heat_treatment.csv",
            material="7075-T6",
        )
        == "Данные по термообработке 7075-T6"
    )
    assert friendly_source_name("article_vt6.txt") == "Статья по ВТ6"
    assert friendly_source_name("ti6al4v.html") == "Материал по Ti-6Al-4V"
    assert friendly_source_name(
        "https://example.org/reports/vt6-annealing.html?utm_source=demo",
        source_type="url",
    ) == "example.org · vt6 annealing"
    assert friendly_source_name(
        "VT6 annealing study",
        source_type="url",
        source_url="https://example.org/reports/vt6-annealing.html?utm_source=demo",
    ) == "VT6 annealing study"
    assert friendly_source_name("source_technical_name.txt") == "Источник из корпуса"
    rendered = " ".join(
        [
            friendly_source_name("doc_29765440445b821ce5d3075b_synthetic_vt6_heat_treatment.csv"),
            friendly_source_name("source_technical_name.txt"),
        ]
    )
    for forbidden in ["synthetic", "doc_", "29765440445b821ce5d3075b", "chunk_"]:
        assert forbidden not in rendered


def test_raw_source_provenance_stays_available_in_raw_rows() -> None:
    raw_source = "doc_29765440445b821ce5d3075b_synthetic_vt6_heat_treatment.csv"
    payload = {
        "evidence": [
            {
                "source_name": raw_source,
                "document_id": "doc_abc",
                "chunk_id": "chunk_abc",
                "quote": "quote",
            }
        ]
    }

    raw_rows = evidence_to_rows(payload)
    user_rows = evidence_to_user_rows(payload)

    assert raw_rows[0]["source_name"] == raw_source
    assert raw_rows[0]["doc_id"] == "doc_abc"
    assert raw_rows[0]["chunk_id"] == "chunk_abc"
    assert user_rows[0]["Источник"] == "Данные по термообработке ВТ6"


def test_url_source_keeps_raw_url_only_in_raw_rows() -> None:
    payload = {
        "evidence": [
            {
                "source_name": "VT6 annealing study",
                "source_type": "url",
                "source_url": "https://example.org/reports/vt6-annealing.html?utm_source=demo",
                "quote": "После отжига сплава ВТ6 предел прочности составил 980 MPa.",
            }
        ]
    }

    raw_rows = evidence_to_rows(payload)
    user_rows = evidence_to_user_rows(payload)

    assert raw_rows[0]["source_url"].endswith("utm_source=demo")
    assert user_rows[0]["Источник"] == "VT6 annealing study"
    assert "utm_" not in user_rows[0]["Источник"]

"""Formatting helpers for the Streamlit product UI."""

from __future__ import annotations

import math
import os
import re
import html
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


INTERNAL_ID_RE = re.compile(
    r"\b(?:doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|EXP-[A-Za-z0-9_-]+|SCI-[A-Za-z0-9_-]+|"
    r"g\d{4,}|[A-Za-z]+_from_chunk|material_from_chunk)\b"
)
ANSWER_STATE_NO_DATA = "no_data"
ANSWER_STATE_PARTIAL = "partial"
ANSWER_STATE_FULL = "full"
NO_DATA_TITLE = "❌ Нет точных данных по запросу"
NO_DATA_HINTS = [
    "Попробуйте уточнить материал / процесс / свойство / диапазон",
    "Проверьте активность документов",
]
PARTIAL_DATA_TITLE = "⚠ Найдены частично релевантные результаты"


def format_answer_markdown(payload: dict[str, Any]) -> str:
    """Return the main answer without exposing raw JSON or missing values."""

    answer = str(payload.get("answer") or "").strip()
    return _clean_public_text(translate_system_message(answer)) or "Ответ не сформирован."


def answer_text_sections(payload: dict[str, Any]) -> dict[str, list[str] | str]:
    """Return a structured answer outline for the Russian UI."""

    human = payload.get("human_answer") if isinstance(payload.get("human_answer"), dict) else {}
    title = _clean_public_text(human.get("title")) if human else ""
    summary = _clean_public_text(human.get("summary")) if human else ""
    if title and summary:
        conclusion = f"**{title}**\n\n{summary}"
    else:
        conclusion = _strip_markdown_sections(format_answer_markdown(payload))
    findings = [_clean_public_text(item) for item in (human.get("key_findings") or []) if _clean_public_text(item)] if human else []
    caveats = [_clean_public_text(item) for item in (human.get("caveats") or []) if _clean_public_text(item)] if human else []
    recommendation = _clean_public_text(human.get("recommendation")) if human else ""
    if recommendation:
        caveats.append(f"Вывод: {recommendation}")
    confidence = _clean_public_text(human.get("confidence_label")) if human else ""
    if confidence:
        conclusion = f"{conclusion}\n\nУверенность: {confidence}"
    return {
        "summary": conclusion or "Ответ не сформирован.",
        "findings": findings,
        "caveats": caveats,
    }


def build_answer_report_model(
    payload: dict[str, Any],
    *,
    question: str | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build one normalized user-facing report model for UI, Markdown and PDF."""

    generated = generated_at or datetime.now(timezone.utc)
    sections = answer_text_sections(payload)
    findings = _nonempty_strings(sections.get("findings") or [])
    if not findings:
        findings = _findings_from_summary_rows(payload)
    limitations = _dedupe_strings(
        [
            *_nonempty_strings(sections.get("caveats") or []),
            *(row.get("Описание") for row in caveat_rows(payload) if row.get("Описание")),
        ]
    )
    conflicts = _dedupe_strings(row.get("Описание") for row in conflict_explanation_rows(payload) if row.get("Описание"))
    facts = _confirmed_fact_report_rows(payload)
    sources = _source_report_rows(payload)
    stats = graph_context_stats(payload)
    return {
        "question": _clean_public_text(question or _payload_question(payload)) or "Вопрос не сохранен.",
        "generated_at": generated,
        "generated_at_label": generated.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "answer_title": "Ответ GraphRAG",
        "short_summary": _clean_public_text(str(sections.get("summary") or "")) or "Ответ не сформирован.",
        "confidence": _report_confidence(payload),
        "statistics": {
            "facts_count": stats["facts_count"],
            "sources_count": stats["sources_count"],
            "citations_count": stats["evidence_count"],
            "graph_nodes_count": stats["subgraph_nodes"],
        },
        "findings": findings,
        "confirmed_facts": facts,
        "conflicts": conflicts,
        "limitations": limitations,
        "sources": sources,
        "evidence_quotes": [row.get("Цитата") for row in sources if row.get("Цитата")],
    }


def answer_display_state(payload: dict[str, Any]) -> str:
    """Classify the /ask payload for Streamlit presentation."""

    facts = payload.get("primary_facts") or payload.get("facts") or []
    evidence = payload.get("evidence") or payload.get("sources") or []
    has_facts = any(isinstance(item, dict) for item in facts)
    has_evidence = any(isinstance(item, dict) for item in evidence)
    has_graph = answer_has_display_graph_data(payload)
    has_partial = bool(partial_matches_to_rows(payload))
    if has_partial and not (has_facts and has_graph):
        return ANSWER_STATE_PARTIAL
    if not (has_facts or has_evidence or has_graph):
        return ANSWER_STATE_NO_DATA
    return ANSWER_STATE_FULL


def answer_has_display_graph_data(payload: dict[str, Any]) -> bool:
    """Return true when the payload has enough data to render the answer graph."""

    facts = payload.get("primary_facts") or payload.get("facts") or []
    if any(isinstance(item, dict) for item in facts):
        return True
    nodes, edges = subgraph_to_tables(payload.get("subgraph"))
    if nodes or edges:
        return True
    stats = graph_context_stats(payload)
    return bool(stats["subgraph_nodes"] or stats["subgraph_edges"])


def partial_matches_to_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten partial matches into display-safe table rows."""

    partial = payload.get("partial_matches") or {}
    if not isinstance(partial, dict):
        return []
    result: list[dict[str, Any]] = []
    for group, rows in partial.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            item = row if isinstance(row, dict) else {"value": row}
            public = {str(key): translate_system_message(_clean_public_text(value)) for key, value in item.items() if value not in (None, "", [])}
            if not public:
                continue
            result.append({"Тип совпадения": str(group), **public})
    return result


def caveat_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return visible caveats and data gaps without empty placeholders."""

    result: list[dict[str, Any]] = []
    warning = no_exact_match_warning(payload)
    if warning:
        result.append({"Тип": "Ограничение", "Описание": warning})
    for row in payload.get("data_gaps") or payload.get("gaps") or []:
        if isinstance(row, dict):
            description = row.get("reason") or row.get("description") or row.get("property") or row.get("material")
        else:
            description = row
        text = translate_system_message(_clean_public_text(description))
        if text:
            result.append({"Тип": "Пробел данных", "Описание": text})
    for row in conflict_explanation_rows(payload):
        if row.get("Описание"):
            result.append({"Тип": "Неоднородность", "Описание": row["Описание"]})
    return result


def build_answer_markdown_export(payload: dict[str, Any], *, question: str | None = None, generated_at: datetime | None = None) -> str:
    """Build a readable Markdown report from the normalized report model."""

    report = build_answer_report_model(payload, question=question, generated_at=generated_at)
    return build_report_markdown(report)


def build_report_markdown(report: dict[str, Any]) -> str:
    """Render a normalized report model as public Markdown."""

    parts = [
        "# Ответ GraphRAG",
        "",
        f"Дата генерации: {_markdown_text(report.get('generated_at_label'))}",
        "",
        "## Вопрос",
        _markdown_text(report.get("question")),
        "",
        "## Краткий вывод",
        _markdown_text(report.get("short_summary")),
        "",
        "## Что найдено",
    ]
    findings = _nonempty_strings(report.get("findings") or [])
    parts.extend(f"- {_markdown_text(item)}" for item in findings) if findings else parts.append("- Существенные находки не выделены.")
    facts = _public_table_rows(report.get("confirmed_facts") or [])
    parts.extend(["", "## Подтвержденные факты"])
    parts.append(_markdown_table(facts) if facts else "Подтвержденные факты по этому запросу не найдены.")
    conflicts = _nonempty_strings(report.get("conflicts") or [])
    parts.extend(["", "## Найденные противоречия"])
    if conflicts:
        parts.extend(f"- {_markdown_text(item)}" for item in conflicts)
    else:
        parts.append("Явных противоречий по этому запросу не найдено.")
    limitations = _nonempty_strings(report.get("limitations") or [])
    parts.extend(["", "## Ограничения анализа"])
    if limitations:
        parts.extend(f"- {_markdown_text(item)}" for item in limitations)
    else:
        parts.append("Ограничения не указаны.")
    sources = _public_table_rows(report.get("sources") or [])
    parts.extend(["", "## Использованные источники"])
    parts.append(_markdown_table(sources) if sources else "Источники по этому запросу не найдены.")
    stats = report.get("statistics") if isinstance(report.get("statistics"), dict) else {}
    parts.extend(
        [
            "",
            "## Служебная информация",
            f"- Фактов: {_safe_int(stats.get('facts_count'))}",
            f"- Источников: {_safe_int(stats.get('sources_count'))}",
            f"- Цитат: {_safe_int(stats.get('citations_count'))}",
            f"- Узлов графа: {_safe_int(stats.get('graph_nodes_count'))}",
        ]
    )
    return "\n".join(parts).strip() + "\n"


def build_answer_pdf_export(payload: dict[str, Any], *, question: str | None = None, generated_at: datetime | None = None) -> bytes:
    """Build a Unicode visual PDF report from the normalized report model."""

    report = build_answer_report_model(payload, question=question, generated_at=generated_at)
    markdown = build_report_markdown(report)
    lines = _wrap_pdf_lines(_markdown_to_plain_text(markdown), width=88)
    return _pillow_text_pdf(lines, title="Ответ GraphRAG")


def format_status_badge(payload: dict[str, Any]) -> str:
    """Return a compact status label."""

    status = str(payload.get("status") or "unknown")
    mapping = {
        "ok": "Найдены подтверждённые данные",
        "partial": "Найден частичный контекст",
        "no_exact_match": "Точного факта нет",
        "error": "Ошибка",
    }
    return mapping.get(status, status)


def facts_to_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return display-safe raw fact rows."""

    rows = payload.get("facts") or []
    return [row if isinstance(row, dict) else {"value": row} for row in rows]


def facts_to_user_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return fact rows with user-oriented column names."""

    rows = payload.get("primary_facts") or payload.get("facts") or []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = {
                "Материал": _clean_public_text(row.get("material")),
                "Режим": _clean_public_text(row.get("regime")),
                "Свойство": _clean_public_text(row.get("property")),
                "Значение": row.get("value") if row.get("value") is not None else row.get("raw_value"),
                "Ед.": _clean_public_text(row.get("unit")),
                "Нормализовано": _normalized_value_label(row),
                "Эффект": _effect_label(row.get("effect")),
                "Оборудование": _join(row.get("equipment")),
                "Лаборатория": _join(row.get("laboratory") or row.get("laboratories")),
                "Основание": _fact_basis(row),
            }
        if any(item.get(key) for key in ("Материал", "Режим", "Свойство", "Значение", "Основание")):
            result.append(item)
    return result


def evidence_to_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return display-safe evidence/source rows."""

    evidence = payload.get("evidence") or []
    sources = payload.get("sources") or []
    rows = evidence if evidence else sources
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            result.append({"quote": str(row)})
            continue
        result.append(
            {
                "source_name": row.get("source_name") or row.get("title") or row.get("filename"),
                "source_url": row.get("source_url"),
                "source_type": row.get("source_type"),
                "title": row.get("title"),
                "filename": row.get("filename"),
                "doc_id": row.get("document_id") or row.get("doc_id"),
                "chunk_id": row.get("chunk_id"),
                "page": row.get("page") or row.get("page_start"),
                "score": row.get("score"),
                "retrieval_backend": row.get("retrieval_backend"),
                "evidence_type": row.get("evidence_type"),
                "quote": row.get("quote") or "Цитата не сохранена; источник доступен в деталях.",
            }
        )
    return result


def evidence_to_user_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return evidence rows with readable column names."""

    result: list[dict[str, Any]] = []
    for row in evidence_to_rows(payload):
        quote = _clean_public_text(row.get("quote"))
        source = friendly_source_name(
            row.get("source_name"),
            source_url=row.get("source_url"),
            source_type=row.get("source_type"),
            title=row.get("title"),
        )
        item = {
            "Источник": source,
            "Страница": row.get("page"),
            "Тип": _evidence_type_label(row.get("evidence_type") or row.get("retrieval_backend")),
            "Цитата": quote,
        }
        if source or quote:
            result.append(item)
    return result


def answer_evidence_summary_rows(payload: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    """Return compact user-facing fact/evidence rows for the main answer."""

    facts = payload.get("primary_facts") or payload.get("facts") or []
    result: list[dict[str, Any]] = []
    seen = set()
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        row = _answer_evidence_summary_row(payload, fact)
        if not row:
            continue
        identity = (
            row.get("Материал"),
            row.get("Режим"),
            row.get("Свойство"),
            row.get("Значение"),
            row.get("Источник"),
            row.get("Фрагмент"),
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(row)
        if len(result) >= limit:
            break
    return result


def answer_source_metadata_rows(payload: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    """Return compact user-facing source grouping by practice/year/type/reliability."""

    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in _source_metadata_input_rows(payload):
        metadata = _source_metadata_from_row(row)
        if not metadata:
            continue
        source = _public_source_name(row)
        if source == "Источник из корпуса" and not any(metadata.values()):
            continue
        practice = _practice_label(metadata.get("practice_scope"), metadata.get("geographies"))
        year = _year_label(metadata.get("publication_year"))
        source_type = _source_type_label(metadata.get("source_type_detected") or row.get("source_type"))
        reliability = _reliability_label(metadata.get("reliability_level"))
        key = (practice, year, source_type, reliability)
        item = grouped.setdefault(
            key,
            {
                "Практика": practice,
                "Год": year,
                "Тип источника": source_type,
                "Достоверность": reliability,
                "Источников": 0,
                "Примеры": [],
            },
        )
        examples = item["Примеры"]
        if source not in examples and len(examples) < 3:
            examples.append(source)
        item["Источников"] += 1
    rows = list(grouped.values())
    rows.sort(key=lambda item: (_source_group_rank(item), -int(item.get("Источников") or 0)))
    for item in rows:
        item["Примеры"] = ", ".join(item["Примеры"]) if item["Примеры"] else "источники корпуса"
    return rows[:limit]


def conflict_explanation_rows(payload: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    """Return compact conflict explanations for the main UI without raw ids."""

    diagnostics = payload.get("diagnostics") or {}
    conflicts = diagnostics.get("fact_conflicts") or payload.get("fact_conflicts") or []
    result: list[dict[str, Any]] = []
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            continue
        material = _clean_public_text(conflict.get("material")) or "материала"
        regime = _clean_public_text(conflict.get("regime"))
        prop = _property_for_sentence(_clean_public_text(conflict.get("property")) or "свойства")
        values = _conflict_values_label(conflict)
        if not values:
            continue
        regime_text = f" после {_regime_for_sentence(regime)}" if regime else ""
        reason = _conflict_reason_label(conflict.get("possible_reason"))
        description = (
            f"Для {material}{regime_text} найдены разные значения {prop}: {values}. "
            f"Возможная причина: {reason}."
        )
        result.append(
            {
                "Описание": _clean_public_text(description),
                "Источников": conflict.get("sources_count"),
            }
        )
        if len(result) >= limit:
            break
    return result


def subgraph_to_tables(subgraph: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return node and edge tables from a UI-compatible subgraph."""

    if not isinstance(subgraph, dict):
        return [], []
    nodes = subgraph.get("nodes") or []
    edges = subgraph.get("edges") or []
    return (
        [node for node in nodes if isinstance(node, dict)],
        [edge for edge in edges if isinstance(edge, dict)],
    )


def graph_to_display(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a UI-compatible graph payload."""

    subgraph = payload.get("subgraph")
    return subgraph if isinstance(subgraph, dict) else {"nodes": [], "edges": []}


def no_exact_match_warning(payload: dict[str, Any]) -> str | None:
    """Return user-facing trust warning for no-exact-match answers."""

    if payload.get("status") != "no_exact_match":
        return None
    return (
        "Точного факта в графе не найдено. Ниже показаны только частичные "
        "совпадения, evidence и предполагаемый пробел в данных; это не "
        "положительный ответ на исходное сочетание ограничений."
    )


def graph_context_stats(payload: dict[str, Any]) -> dict[str, int]:
    """Return stable graph context counters with defaults."""

    context = payload.get("graph_context") or {}
    return {
        "facts_count": int(context.get("facts_count") or len(payload.get("facts") or [])),
        "sources_count": int(context.get("sources_count") or len(payload.get("sources") or [])),
        "evidence_count": int(context.get("evidence_count") or len(payload.get("evidence") or [])),
        "subgraph_nodes": int(context.get("subgraph_nodes") or len((payload.get("subgraph") or {}).get("nodes") or [])),
        "subgraph_edges": int(context.get("subgraph_edges") or len((payload.get("subgraph") or {}).get("edges") or [])),
    }


def build_compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Return top-row product metrics."""

    stats = graph_context_stats(payload)
    return {
        "Факты": stats["facts_count"],
        "Источники": stats["sources_count"],
        "Цитаты": stats["evidence_count"],
        "Узлы графа": stats["subgraph_nodes"],
    }


def report_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact metrics from the normalized report model."""

    stats = report.get("statistics") if isinstance(report.get("statistics"), dict) else {}
    return {
        "Факты": _safe_int(stats.get("facts_count")),
        "Источники": _safe_int(stats.get("sources_count")),
        "Цитаты": _safe_int(stats.get("citations_count")),
        "Узлы графа": _safe_int(stats.get("graph_nodes_count")),
    }


def translate_system_message(value: Any) -> str:
    """Translate known backend/system messages for the Russian UI."""

    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip().lower().rstrip(".")
    exact = {
        "no numerical corrosion data were reported": "Не найдено числовых данных по коррозионной стойкости.",
        "no exact match found": "Точное совпадение не найдено.",
        "no exact graph path found": "Точный путь в графе не найден.",
        "not specified": "Не указано.",
        "source not specified": "Источник не указан.",
        "llm provider is offline; template fallback is active": "LLM отключен; используется шаблонный ответ.",
    }
    if normalized in exact:
        return exact[normalized]
    replacements = [
        (r"no numerical ([a-z\s-]+) data (?:were|was) reported", r"Не найдено числовых данных: \1."),
        (r"no ([a-z\s-]+) data (?:were|was) reported", r"Не найдено данных: \1."),
        (r"missing ([a-z\s-]+)", r"Не указано: \1."),
    ]
    for pattern, replacement in replacements:
        if re.fullmatch(pattern, normalized):
            translated = re.sub(pattern, replacement, normalized)
            return _translate_domain_terms(translated)
    return text


def diagnostics_to_safe_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Return diagnostics useful for users without dumping every payload field."""

    diagnostics = payload.get("diagnostics") or {}
    retrieval = payload.get("retrieval") or {}
    return {
        "preset_id": diagnostics.get("preset_id"),
        "preset_title": diagnostics.get("preset_title"),
        "kg_backend_active": retrieval.get("kg_backend_active") or diagnostics.get("kg_backend_active"),
        "answer_mode": payload.get("answer_mode"),
        "analytical_intent": payload.get("analytical_intent"),
        "fact_conflicts_count": len(diagnostics.get("fact_conflicts") or []),
        "warnings": diagnostics.get("warnings") or [],
    }


def documents_to_rows(payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format /documents payload for document management UI."""

    items = payload.get("items") if isinstance(payload, dict) else payload
    result: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        diagnostics = item.get("parser_diagnostics") or {}
        intelligence = item.get("document_intelligence") or {}
        result.append(
            {
                "Документ": (item.get("source_name") or item.get("source_title") or item.get("title")) if (item.get("source_type") or intelligence.get("source_type")) == "url" else item.get("filename") or item.get("title"),
                "Тип": item.get("source_type") or intelligence.get("source_type") or "file",
                "Chunks": item.get("chunks"),
                "Активен": bool(item.get("active", True)),
                "Дата загрузки": item.get("updated_at") or item.get("created_at"),
                "Parser": item.get("parser"),
                "Blocks": intelligence.get("blocks_count") or diagnostics.get("blocks_count"),
                "Tables": intelligence.get("tables_count") or diagnostics.get("tables_count"),
                "doc_id": item.get("doc_id"),
            }
        )
    return result


def format_upload_summary(
    result: dict[str, Any],
    *,
    graph_updated: bool = False,
    documents_payload: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a user-facing upload summary without raw ingestion JSON."""

    items = _ingested_items(result)
    failed_statuses = {"upload_error", "parse_failed", "skipped_empty", "error", "failed"}
    uploaded = 0
    chunks = 0
    for item in items:
        status = str(item.get("status") or "").lower()
        if status not in failed_statuses:
            uploaded += 1
        chunks += _safe_int(item.get("chunks"))
    active_documents = _active_documents_count(documents_payload)
    if active_documents is None:
        active_documents = _safe_int(result.get("active_documents") or result.get("active_documents_count"))
    return {
        "Документы загружены": uploaded,
        "Фрагменты добавлены": chunks,
        "Граф/индекс обновлен": "да" if graph_updated else "нет",
        "Активных документов теперь": active_documents,
    }


def _confirmed_fact_report_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary_by_fact = answer_evidence_summary_rows(payload, limit=20)
    source_by_key = {
        (
            row.get("Материал"),
            row.get("Режим"),
            row.get("Свойство"),
            row.get("Значение"),
        ): row.get("Источник")
        for row in summary_by_fact
    }
    rows: list[dict[str, Any]] = []
    for row in facts_to_user_rows(payload):
        key = (row.get("Материал"), row.get("Режим"), row.get("Свойство"), row.get("Значение") or row.get("Нормализовано"))
        source = source_by_key.get(key) or _source_from_basis(row.get("Основание"))
        item = {
            "Материал": row.get("Материал") or "не указано",
            "Режим": row.get("Режим") or "не указано",
            "Свойство": row.get("Свойство") or "не указано",
            "Значение": row.get("Значение") if row.get("Значение") not in (None, "") else "не указано",
            "Нормализовано": row.get("Нормализовано") or "",
            "Источник": source or "источник корпуса",
        }
        rows.append(_clean_report_row(item))
    return _dedupe_rows(rows)


def _source_report_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in evidence_to_user_rows(payload):
        item = {
            "Источник": row.get("Источник") or "Источник из корпуса",
            "Страница": row.get("Страница") or "",
            "Тип": row.get("Тип") or "Источник",
            "Цитата": row.get("Цитата") or "",
        }
        rows.append(_clean_report_row(item))
    return _dedupe_rows(rows)


def _findings_from_summary_rows(payload: dict[str, Any]) -> list[str]:
    rows = answer_evidence_summary_rows(payload, limit=5)
    findings = []
    for row in rows:
        fact = _clean_public_text(row.get("Факт"))
        if fact:
            findings.append(fact)
    return findings


def _report_confidence(payload: dict[str, Any]) -> str:
    human = payload.get("human_answer") if isinstance(payload.get("human_answer"), dict) else {}
    value = _clean_public_text(human.get("confidence_label")) if human else ""
    return value or ""


def _source_from_basis(value: Any) -> str:
    text = _clean_public_text(value)
    if not text:
        return ""
    return text.split(":", 1)[0].strip()


def _public_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_clean_report_row(row) for row in rows if isinstance(row, dict)]


def _clean_report_row(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        clean_key = _clean_public_text(key)
        if not clean_key or clean_key in _FORBIDDEN_EXPORT_FIELDS:
            continue
        if isinstance(value, int | float):
            result[clean_key] = value
            continue
        if isinstance(value, dict):
            continue
        if isinstance(value, list):
            clean_value = ", ".join(_clean_public_text(item) for item in value if _clean_public_text(item))
            result[clean_key] = clean_value
            continue
        clean_value = _clean_public_text(value)
        result[clean_key] = clean_value
    return result


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for row in rows:
        identity = tuple(sorted((str(key), str(value)) for key, value in row.items()))
        if identity in seen:
            continue
        seen.add(identity)
        result.append(row)
    return result


def _nonempty_strings(values: Any) -> list[str]:
    if not isinstance(values, list | tuple | set):
        values = [values]
    return [text for value in values if (text := _clean_public_text(value))]


def _dedupe_strings(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_public_text(value)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


_FORBIDDEN_EXPORT_FIELDS = {
    "Partial matches",
    "same_material",
    "same_material_and_regime",
    "same_material_and_property",
    "same_regime_and_property",
    "experiment_id",
    "document_id",
    "doc_id",
    "chunk_id",
    "raw JSON",
    "normalization_family",
    "raw_value",
    "baseline_value",
    "delta_abs",
    "delta_rel_percent",
    "source_name",
}


def upload_expansion_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return readable knowledge growth rows from ingestion payload."""

    rows: list[dict[str, Any]] = []
    for item in _ingested_items(result):
        expansion = item.get("knowledge_expansion") or {}
        if not expansion:
            continue
        rows.append(
            {
                "Документ": item.get("source_name") or item.get("filename") or item.get("url"),
                "Новые факты": expansion.get("new_canonical_facts_count", 0),
                "Объединено дублей": expansion.get("duplicate_facts_count", 0),
                "Подтверждены факты": expansion.get("corroborated_facts_count", 0),
                "Конфликты": expansion.get("conflict_groups_added_count", 0),
                "Новые связи": expansion.get("new_comparison_opportunities_count", 0),
                "Пробелы": expansion.get("data_gaps_added_count", 0),
            }
        )
    return rows


def document_metadata_summary(document: dict[str, Any]) -> dict[str, Any]:
    """Return a short document card before raw technical metadata."""

    diagnostics = document.get("parser_diagnostics") or {}
    intelligence = document.get("document_intelligence") or {}
    title = (
        document.get("source_name")
        or document.get("source_title")
        or document.get("filename")
        or document.get("title")
        or document.get("doc_id")
        or "Документ"
    )
    return {
        "Название": title,
        "Статус": document.get("status") or "загружен",
        "Фрагменты": document.get("chunks") or 0,
        "Дата загрузки": document.get("updated_at") or document.get("created_at") or "не указана",
        "Активен": bool(document.get("active", True)),
        "Парсер": document.get("parser"),
        "Блоки": intelligence.get("blocks_count") or diagnostics.get("blocks_count"),
        "Таблицы": intelligence.get("tables_count") or diagnostics.get("tables_count"),
    }


def active_document_changes(original_rows: Any, edited_rows: Any) -> list[tuple[str, bool]]:
    """Return changed (doc_id, active) pairs from document data editor rows."""

    original = _rows_from_any(original_rows)
    edited = _rows_from_any(edited_rows)
    original_active = {str(row.get("doc_id")): bool(row.get("Активен")) for row in original if row.get("doc_id")}
    changes: list[tuple[str, bool]] = []
    for row in edited:
        doc_id = row.get("doc_id")
        if not doc_id:
            continue
        new_active = bool(row.get("Активен"))
        if original_active.get(str(doc_id)) != new_active:
            changes.append((str(doc_id), new_active))
    return changes


def _rows_from_any(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        return list(rows.to_dict("records"))
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _ingested_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    items = result.get("ingested") if isinstance(result, dict) else []
    if isinstance(items, dict):
        items = [items]
    return [item for item in items or [] if isinstance(item, dict)]


def _active_documents_count(payload: dict[str, Any] | list[dict[str, Any]] | None) -> int | None:
    if payload is None:
        return None
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return None
    return sum(1 for item in items if isinstance(item, dict) and bool(item.get("active", True)))


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def clean_graph_label(node: dict[str, Any]) -> str:
    """Return a short human-readable graph label without internal ids."""

    raw = str(node.get("label") or node.get("name") or node.get("canonical_name") or "").strip()
    node_type = str(node.get("type") or node.get("label_type") or "Entity")
    if not raw or INTERNAL_ID_RE.search(raw) or len(raw) > 80:
        raw = _fallback_node_label(node_type, node)
    raw = raw.replace("PropertyValue", "").replace("SourceChunk", "Источник").replace("Experiment", "Эксперимент").strip(" :\n")
    raw = re.sub(r"\s+", " ", raw)
    raw = raw.replace("effect: increase", "эффект: рост")
    raw = raw.replace("effect: decrease", "эффект: снижение")
    raw = raw.replace("effect: unknown", "эффект не указан")
    return _shorten(raw or node_type, 42)


def graph_to_interactive_html(subgraph: dict[str, Any] | None, *, max_nodes: int = 20, max_edges: int = 30) -> str:
    """Build self-contained interactive SVG graph HTML with zoom/pan and fixed nodes."""

    nodes, edges = subgraph_to_tables(subgraph)
    nodes = nodes[:max_nodes]
    allowed = {str(node.get("id")) for node in nodes}
    edges = [edge for edge in edges if str(edge.get("source")) in allowed and str(edge.get("target")) in allowed][:max_edges]
    if not nodes:
        return "<div style='padding:16px;color:#64748b'>Для ответа нет связанного графа.</div>"

    width, height = 860, 520
    center_x, center_y = width / 2, height / 2
    radius_x, radius_y = 300, 180
    positioned: dict[str, dict[str, Any]] = {}
    for idx, node in enumerate(nodes):
        angle = 2 * math.pi * idx / max(len(nodes), 1)
        node_id = str(node.get("id"))
        positioned[node_id] = {
            **node,
            "x": center_x + radius_x * math.cos(angle),
            "y": center_y + radius_y * math.sin(angle),
            "display_label": clean_graph_label(node),
        }
    edge_lines = []
    for edge in edges:
        source = positioned.get(str(edge.get("source")))
        target = positioned.get(str(edge.get("target")))
        if not source or not target:
            continue
        label = _shorten(str(edge.get("label") or edge.get("type") or ""), 24)
        edge_lines.append(
            f"<line class='edge' x1='{source['x']:.1f}' y1='{source['y']:.1f}' x2='{target['x']:.1f}' y2='{target['y']:.1f}' />"
            f"<text class='edge-label' x='{(source['x'] + target['x']) / 2:.1f}' y='{(source['y'] + target['y']) / 2:.1f}'>{html.escape(label)}</text>"
        )
    node_items = []
    for idx, (node_id, node) in enumerate(positioned.items()):
        color = _node_color(str(node.get("type") or "Entity"))
        label = html.escape(str(node["display_label"]))
        title = html.escape(f"{node.get('type', 'Entity')}: {node['display_label']}")
        node_items.append(
            f"<g class='node' data-node-id='node-{idx}' transform='translate({node['x']:.1f},{node['y']:.1f})'>"
            f"<title>{title}</title><circle r='26' fill='{color}'></circle>"
            f"<text text-anchor='middle' y='42'>{label}</text></g>"
        )
    return f"""
<div class="kg-graph-wrap">
  <div class="kg-graph-help">Колесо — масштаб · фон — перемещение карты · узлы зафиксированы</div>
  <svg id="kgGraphSvg" viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">
    <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#64748b"></path></marker></defs>
    <g id="kgViewport">{"".join(edge_lines)}{"".join(node_items)}</g>
  </svg>
</div>
<style>
.kg-graph-wrap {{ border:1px solid #d8dee9; border-radius:8px; background:#ffffff; position:relative; }}
.kg-graph-help {{ position:absolute; right:10px; top:8px; font:12px Arial; color:#64748b; z-index:2; background:rgba(255,255,255,.86); padding:2px 6px; border-radius:4px; }}
#kgGraphSvg {{ cursor:grab; touch-action:none; }}
.edge {{ stroke:#64748b; stroke-width:1.2; marker-end:url(#arrow); opacity:.72; }}
.edge-label {{ fill:#475569; font:10px Arial; paint-order:stroke; stroke:#fff; stroke-width:3px; stroke-linejoin:round; }}
.node circle {{ stroke:#334155; stroke-width:1.2; filter: drop-shadow(0 2px 3px rgba(15,23,42,.18)); }}
.node text {{ fill:#0f172a; font:12px Arial; pointer-events:none; }}
.node {{ cursor:default; }}
</style>
<script>
(function() {{
 const svg = document.getElementById('kgGraphSvg');
 const viewport = document.getElementById('kgViewport');
 let state = {{x:0, y:0, scale:1}};
 let drag = null;
 function apply() {{ viewport.setAttribute('transform', `translate(${{state.x}},${{state.y}}) scale(${{state.scale}})`); }}
 function isNodeHit(ev) {{
   return Array.from(document.querySelectorAll('.node')).some(function(node) {{
     const box = node.getBoundingClientRect();
     return ev.clientX >= box.left && ev.clientX <= box.right && ev.clientY >= box.top && ev.clientY <= box.bottom;
   }});
 }}
 svg.addEventListener('wheel', function(ev) {{
   ev.preventDefault();
   const delta = ev.deltaY < 0 ? 1.12 : 0.89;
   state.scale = Math.max(0.25, Math.min(4, state.scale * delta));
   apply();
 }}, {{passive:false}});
 svg.addEventListener('pointerdown', function(ev) {{
   const node = ev.target.closest && ev.target.closest('.node');
   drag = (node || isNodeHit(ev)) ? null : {{kind: 'pan', startX: ev.clientX, startY: ev.clientY, x: state.x, y: state.y}};
   svg.setPointerCapture(ev.pointerId);
 }});
 svg.addEventListener('pointermove', function(ev) {{
   if (!drag) return;
   const dx = ev.clientX - drag.startX, dy = ev.clientY - drag.startY;
   state.x = drag.x + dx; state.y = drag.y + dy; apply();
 }});
 svg.addEventListener('pointerup', function(ev) {{ drag = null; try {{ svg.releasePointerCapture(ev.pointerId); }} catch(e) {{}} }});
 apply();
}})();
</script>
"""


def _fallback_node_label(node_type: str, node: dict[str, Any]) -> str:
    props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
    if node_type == "Material":
        return str(props.get("canonical_name") or "Материал")
    if node_type in {"ProcessRegime", "Regime"}:
        return str(props.get("canonical_name") or "Режим")
    if node_type in {"Property", "Measurement"}:
        value = props.get("value")
        unit = props.get("unit") or ""
        prop = props.get("property") or props.get("canonical_name") or "Свойство"
        return f"{prop} {value or ''} {unit}".strip()
    if node_type == "Experiment":
        return "Эксперимент"
    if node_type in {"Document", "DocumentChunk", "SourceChunk"}:
        return "Источник"
    if node_type == "DataGap":
        return "Пробел в данных"
    return node_type


def _node_color(node_type: str) -> str:
    return {
        "Material": "#bfdbfe",
        "ProcessRegime": "#bbf7d0",
        "Property": "#fde68a",
        "Measurement": "#fed7aa",
        "Experiment": "#ddd6fe",
        "Equipment": "#fecdd3",
        "Laboratory": "#ccfbf1",
        "ResearchTeam": "#ccfbf1",
        "DataGap": "#fecaca",
        "Document": "#e2e8f0",
        "DocumentChunk": "#e2e8f0",
        "SourceChunk": "#e2e8f0",
    }.get(node_type, "#f1f5f9")


def _shorten(value: str, limit: int) -> str:
    value = INTERNAL_ID_RE.sub("", value).strip()
    return value[: limit - 1] + "…" if len(value) > limit else value


def _join(value: Any) -> str | None:
    if isinstance(value, list):
        joined = ", ".join(_clean_public_text(item) for item in value if _clean_public_text(item))
        return joined or None
    cleaned = _clean_public_text(value)
    return cleaned or None


def _answer_evidence_summary_row(payload: dict[str, Any], fact: dict[str, Any]) -> dict[str, Any] | None:
    material = _clean_public_text(fact.get("material"))
    regime = _clean_public_text(fact.get("regime"))
    prop = _clean_public_text(fact.get("property"))
    value_label = _summary_value_label(fact) or _effect_label(fact.get("effect"))
    if not any([material, regime, prop, value_label]):
        return None

    source, quote = _public_evidence_for_fact(payload, fact)
    material = material or "материал не указан"
    regime = regime or "режим не указан"
    prop = prop or "свойство не указано"
    value_label = value_label or "значение не указано"
    original = _summary_original_value_label(fact)
    if _same_display_value(original, value_label):
        original = None
    return {
        "Факт": f"{material} · {regime} · {prop}: {value_label}",
        "Материал": material,
        "Режим": regime,
        "Свойство": prop,
        "Значение": value_label,
        "Исходное значение": original,
        "Источник": source,
        "Фрагмент": quote,
    }


def _source_metadata_input_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for collection in [payload.get("evidence") or [], payload.get("sources") or []]:
        rows.extend(item for item in collection if isinstance(item, dict))
    for fact in payload.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        rows.extend(item for item in fact.get("evidence") or [] if isinstance(item, dict))
    seen = set()
    result = []
    for row in rows:
        key = (
            row.get("document_id") or row.get("doc_id"),
            row.get("chunk_id") or row.get("source_chunk_id"),
            row.get("source_name") or row.get("title") or row.get("filename"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _source_metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else {}
    return {
        "source_type_detected": row.get("source_type_detected") or nested.get("source_type_detected"),
        "publication_year": row.get("publication_year") or nested.get("publication_year"),
        "geographies": row.get("geographies") or nested.get("geographies") or [],
        "practice_scope": row.get("practice_scope") or nested.get("practice_scope"),
        "reliability_level": row.get("reliability_level") or nested.get("reliability_level"),
    }


def _public_evidence_for_fact(payload: dict[str, Any], fact: dict[str, Any]) -> tuple[str, str]:
    candidates: list[dict[str, Any]] = []
    evidence = fact.get("evidence")
    if isinstance(evidence, list):
        candidates.extend(item for item in evidence if isinstance(item, dict))
    if fact.get("source_name") or fact.get("quote"):
        candidates.append(fact)

    top_level = [item for item in evidence_to_rows(payload) if isinstance(item, dict)]
    matching = [item for item in top_level if _fact_matches_evidence(fact, item)]
    candidates.extend(matching)
    candidates.extend(top_level)

    for item in candidates:
        source = _public_source_name(item, fact_material=fact.get("material"))
        quote = _public_quote(item)
        if quote or source != "источник корпуса":
            return source, quote
    return "источник корпуса", "цитата не сохранена в кратком виде"


def _fact_matches_evidence(fact: dict[str, Any], evidence: dict[str, Any]) -> bool:
    fact_ids = {
        str(fact.get("source_chunk_id") or ""),
        str(fact.get("chunk_id") or ""),
        str(fact.get("document_id") or fact.get("doc_id") or ""),
    }
    for item in fact.get("evidence") or []:
        if isinstance(item, dict):
            fact_ids.add(str(item.get("chunk_id") or ""))
            fact_ids.add(str(item.get("document_id") or item.get("doc_id") or ""))
    fact_ids.discard("")
    evidence_ids = {
        str(evidence.get("chunk_id") or ""),
        str(evidence.get("document_id") or evidence.get("doc_id") or ""),
    }
    evidence_ids.discard("")
    return bool(fact_ids and evidence_ids and fact_ids.intersection(evidence_ids))


def _public_source_name(row: dict[str, Any], *, fact_material: Any = None) -> str:
    raw = row.get("source_name") or row.get("title") or row.get("filename")
    return friendly_source_name(
        raw,
        material=fact_material,
        source_url=row.get("source_url"),
        source_type=row.get("source_type"),
        title=row.get("title"),
    )


def _practice_label(scope: Any, geographies: Any) -> str:
    if scope == "domestic":
        return "Отечественная практика"
    if scope == "foreign_or_global":
        return "Зарубежная/мировая практика"
    if scope == "domestic_and_foreign":
        return "Россия и зарубежная практика"
    values = [str(item) for item in geographies if item] if isinstance(geographies, list) else []
    if "Россия" in values and any(item != "Россия" for item in values):
        return "Россия и зарубежная практика"
    if "Россия" in values:
        return "Отечественная практика"
    if values:
        return ", ".join(values[:3])
    return "География не указана"


def _year_label(value: Any) -> str:
    if value in {None, "", 0}:
        return "Год не указан"
    return str(value)


def _source_type_label(value: Any) -> str:
    return {
        "publication": "Публикация/обзор",
        "internal_report": "Внутренний отчет",
        "patent": "Патент",
        "standard": "Нормативный источник",
        "catalog": "Каталог/таблица",
        "presentation": "Презентация/доклад",
        "web_page": "Веб-страница",
        "url": "Веб-страница",
        "file": "Файл корпуса",
        "unknown": "Тип не указан",
        None: "Тип не указан",
        "": "Тип не указан",
    }.get(value, str(value))


def _reliability_label(value: Any) -> str:
    return {
        "high": "высокая",
        "medium": "средняя",
        "low": "низкая",
        "unknown": "не указана",
        None: "не указана",
        "": "не указана",
    }.get(value, str(value))


def _source_group_rank(item: dict[str, Any]) -> int:
    reliability_rank = {"высокая": 0, "средняя": 1, "низкая": 2, "не указана": 3}.get(str(item.get("Достоверность")), 3)
    year_rank = 0 if item.get("Год") != "Год не указан" else 1
    geo_rank = 0 if item.get("Практика") != "География не указана" else 1
    return reliability_rank * 10 + year_rank * 3 + geo_rank


def friendly_source_name(raw: Any, *, material: Any = None, source_url: Any = None, source_type: Any = None, title: Any = None) -> str:
    """Return a user-facing source label while keeping raw provenance elsewhere."""

    raw_text = str(raw or "").strip()
    url_text = str(source_url or "").strip()
    is_url = str(source_type or "").lower() == "url" or _looks_like_url(raw_text) or _looks_like_url(url_text)
    if is_url:
        title_text = _clean_web_title(title or (raw_text if not _looks_like_url(raw_text) else ""))
        if title_text:
            return _shorten(title_text, 80)
        url_label = _friendly_url_label(url_text or raw_text)
        if url_label:
            return url_label
        return "Источник из веб-страницы"

    if not raw_text:
        return "Источник из корпуса"
    filename = raw_text.replace("\\", "/").rsplit("/", 1)[-1]
    filename = re.sub(r"^doc_[0-9a-fA-F]{8,64}_", "", filename)
    if _looks_like_public_source_filename(filename):
        return _shorten(filename, 90)
    stem = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", filename)
    stem = INTERNAL_ID_RE.sub("", stem)
    normalized = _normalize_source_stem(stem)
    material_label = _source_material_label(_normalize_source_stem(str(material or ""))) or _source_material_label(normalized)
    tokens = _source_tokens(normalized)
    meaningful_tokens = [token for token in tokens if token not in _SOURCE_NOISE_TOKENS]

    if any(token in tokens for token in {"article", "paper", "publication", "статья"}):
        return f"Статья по {material_label}" if material_label else "Статья из корпуса"
    if _has_heat_treatment_signal(tokens):
        return f"Данные по термообработке {material_label}" if material_label else "Данные по термообработке"
    if any(token in tokens for token in {"experiment", "experiments", "эксперимент", "эксперименты"}):
        return f"Экспериментальные данные по {material_label}" if material_label else "Данные экспериментов"
    if material_label:
        return f"Материал по {material_label}"
    if not meaningful_tokens:
        return "Источник из корпуса"
    return "Источник из корпуса"


def _looks_like_public_source_filename(filename: str) -> bool:
    text = str(filename or "").strip()
    if not text or INTERNAL_ID_RE.search(text):
        return False
    lowered = text.lower()
    if any(token in lowered for token in ["synthetic", "fixture", "smoke", "technical_name", "source_technical"]):
        return False
    return bool(re.search(r"[А-Яа-яЁё]", text) and re.search(r"\.(pdf|docx|pptx|xlsx|csv|txt|md|html?)$", text, flags=re.IGNORECASE))


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(str(value or ""))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _clean_web_title(value: Any) -> str | None:
    text = _clean_public_text(value)
    if not text or _looks_like_url(text) or INTERNAL_ID_RE.search(text):
        return None
    lowered = text.lower()
    if lowered in {"online_resource", "online resource", "index", "page"}:
        return None
    return text


def _friendly_url_label(value: Any) -> str | None:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    path_parts = [unquote(part) for part in parsed.path.split("/") if part]
    label = ""
    if path_parts:
        stem = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", path_parts[-1])
        stem = re.sub(r"[_\\-]+", " ", stem)
        stem = re.sub(r"\s+", " ", stem).strip()
        label = stem
    return _shorten(f"{domain} · {label}" if label else domain, 80)


def _public_quote(row: dict[str, Any]) -> str:
    quote = _clean_public_text(_compact_table_quote(row.get("quote")))
    return _shorten(quote, 180) if quote else "цитата не сохранена в кратком виде"


_SOURCE_NOISE_TOKENS = {
    "api",
    "corpus",
    "data",
    "demo",
    "doc",
    "file",
    "neo4j",
    "sample",
    "source",
    "synthetic",
    "test",
    "txt",
    "csv",
    "html",
    "htm",
    "xlsx",
    "md",
}


def _normalize_source_stem(value: str) -> str:
    text = value.lower().replace("ё", "е")
    text = text.replace("ti-6al-4v", "ti6al4v")
    text = text.replace("ti_6al_4v", "ti6al4v")
    text = text.replace("7075-t6", "7075t6")
    text = text.replace("7075_t6", "7075t6")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _source_tokens(value: str) -> list[str]:
    return [token for token in value.split() if token]


def _source_material_label(value: str) -> str | None:
    compact = re.sub(r"\s+", "", value.lower())
    if "вт6" in compact or "vt6" in compact:
        return "ВТ6"
    if "ti6al4v" in compact:
        return "Ti-6Al-4V"
    if "7075t6" in compact or "7075" in compact:
        return "7075-T6"
    if "12х18н10т" in compact or "12x18h10t" in compact or "12x18n10t" in compact:
        return "12Х18Н10Т"
    if "alloy825" in compact or "825" in compact:
        return "Alloy 825"
    return None


def _has_heat_treatment_signal(tokens: list[str]) -> bool:
    token_set = set(tokens)
    return bool(
        {"heat", "thermal", "treatment", "thermo", "термообработка", "термообработке", "отжиг", "annealing"}
        & token_set
    ) or ("heat" in token_set and "treatment" in token_set)


def _compact_table_quote(value: Any) -> str:
    text = str(value or "")
    if "Table columns:" not in text and "experiment_id:" not in text:
        return text
    data_line = next((line for line in text.splitlines() if "material:" in line and "property:" in line), text)
    fields = []
    allowed_prefixes = ("material:", "process_regime:", "property:", "value:", "unit:", "effect:", "conclusion:", "data_gap:")
    for part in data_line.split("|"):
        cleaned = part.strip()
        if cleaned.lower().startswith(allowed_prefixes):
            fields.append(cleaned)
    return "; ".join(fields) if fields else text


def _summary_value_label(row: dict[str, Any]) -> str | None:
    value = row.get("value_normalized")
    unit = row.get("unit_normalized")
    if value is not None and unit:
        return f"{_format_display_number(value)} {unit}"
    value = row.get("value") if row.get("value") is not None else row.get("raw_value")
    unit = row.get("unit")
    if value is not None and unit:
        return f"{_format_display_number(value)} {unit}"
    return None


def _summary_original_value_label(row: dict[str, Any]) -> str | None:
    value = row.get("value_original")
    unit = row.get("unit_original")
    if value is None:
        value = row.get("value") if row.get("value") is not None else row.get("raw_value")
    if not unit:
        unit = row.get("unit")
    if value is None or not unit:
        return None
    return f"{_format_display_number(value)} {unit}"


def _format_display_number(value: Any) -> str:
    if isinstance(value, int | float):
        if math.isfinite(float(value)) and abs(float(value) - round(float(value))) < 1e-9:
            return str(int(round(float(value))))
        return f"{float(value):.1f}".rstrip("0").rstrip(".")
    return _clean_public_text(value) or str(value)


def _same_display_value(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return re.sub(r"\s+", " ", left).strip().lower() == re.sub(r"\s+", " ", right).strip().lower()


def _fact_basis(row: dict[str, Any]) -> str | None:
    evidence = row.get("evidence")
    if isinstance(evidence, list) and evidence:
        first = next((item for item in evidence if isinstance(item, dict)), None)
        if first:
            source = first.get("source_name") or first.get("title") or first.get("filename") or "источник"
            page = first.get("page") or first.get("page_start")
            quote = _clean_public_text(first.get("quote"))
            location = f", стр. {page}" if page else ""
            if quote:
                return _clean_public_text(f"{source}{location}: {_shorten(quote, 140)}")
            return _clean_public_text(f"{source}{location}")
    source = row.get("source_name") or row.get("title") or row.get("filename")
    cleaned = _clean_public_text(source)
    return cleaned or None


def _effect_label(effect: Any) -> str | None:
    return {
        "increase": "рост",
        "decrease": "снижение",
        "no_change": "без заметного изменения",
        "unchanged": "без заметного изменения",
        "mixed": "смешанный эффект",
        "unknown": "не указан явно",
        None: None,
        "": None,
    }.get(str(effect), str(effect))


def _normalized_value_label(row: dict[str, Any]) -> str | None:
    value = row.get("value_normalized")
    unit = row.get("unit_normalized")
    if value is None or not unit:
        return None
    original_unit = row.get("unit_original") or row.get("unit")
    original_value = row.get("value_original") if row.get("value_original") is not None else row.get("value")
    if original_unit == unit and original_value == value:
        return None
    if isinstance(value, int | float):
        return f"{value:.1f} {unit}"
    return f"{value} {unit}"


def _conflict_values_label(conflict: dict[str, Any]) -> str:
    values = conflict.get("values") or []
    if isinstance(values, dict):
        values = [values]
    if isinstance(values, str):
        values = [values]
    labels: list[str] = []
    for item in values:
        if isinstance(item, dict):
            label = _conflict_value_item_label(item)
        else:
            label = _clean_public_text(item)
        if label and label not in labels:
            labels.append(label)
    return " и ".join(labels[:4])


def _conflict_value_item_label(item: dict[str, Any]) -> str:
    value = item.get("value")
    unit = item.get("unit")
    if value is not None and unit:
        label = f"{_format_display_number(value)} {unit}"
    else:
        label = _effect_label(item.get("effect")) or ""
    original = None
    if item.get("value_original") is not None and item.get("unit_original"):
        original = f"{_format_display_number(item.get('value_original'))} {item.get('unit_original')}"
    if original and not _same_display_value(original, label):
        label = f"{label} (исходно {original})" if label else original
    return _clean_public_text(label)


def _conflict_reason_label(reason: Any) -> str:
    value = str(reason or "").strip().lower()
    mapping = {
        "values reported in different source units; normalized values are shown for comparison": (
            "значения приведены в разных исходных единицах и нормализованы для сравнения"
        ),
        "sources report different qualitative effects": "источники по-разному описывают качественный эффект",
        "sources report different numeric values for the same material/regime/property; check source conditions": (
            "различаются параметры режима, источники или исходное состояние материала"
        ),
    }
    return mapping.get(value, "различаются параметры режима, источники или исходное состояние материала")


def _property_for_sentence(value: str) -> str:
    mapping = {
        "прочность": "прочности",
        "предел прочности": "предела прочности",
        "tensile strength": "прочности",
        "ultimate tensile strength": "прочности",
        "коррозионная стойкость": "коррозионной стойкости",
    }
    return mapping.get(value.lower(), value)


def _regime_for_sentence(value: str) -> str:
    mapping = {
        "отжиг": "отжига",
        "annealing": "отжига",
        "старение": "старения",
        "aging": "старения",
        "термообработка": "термообработки",
        "heat treatment": "термообработки",
    }
    return mapping.get(value.lower(), value)


def _clean_public_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = INTERNAL_ID_RE.sub("", text)
    text = re.sub(r"\b[A-Za-z]+_from_chunk\b", "", text)
    text = re.sub(r"\bg\d{4,}\b", "", text)
    text = re.sub(r"\bстали\s+в\b", "", text, flags=re.IGNORECASE)
    replacements = {
        "PropertyValue": "",
        "SourceChunk": "источник",
        "Experiment": "эксперимент",
        "MEASURES": "",
        "OF_PROPERTY": "",
        "STUDIES": "",
        "USES_REGIME": "",
        "process_regime:": "режим:",
        "property:": "свойство:",
        "value:": "значение:",
        "unit:": "единица:",
        "material:": "материал:",
        "effect:": "эффект:",
        "conclusion:": "вывод:",
        "data_gap:": "пробел:",
    }
    for raw, replacement in replacements.items():
        text = text.replace(raw, replacement)
    text = re.sub(r"\bincreased\b", "повышена", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdecreased\b", "снижена", text, flags=re.IGNORECASE)
    text = re.sub(r"\bincrease\b", "рост", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdecrease\b", "снижение", text, flags=re.IGNORECASE)
    text = re.sub(r"\bunknown\b", "не указано", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    cleaned = translate_system_message(text.strip(" _：:;:-"))
    return "" if _looks_like_damaged_entity(cleaned) else cleaned


def _strip_markdown_sections(markdown: str) -> str:
    lines = []
    skip_prefixes = ("**Что найдено:**", "**Ограничения:**", "**Вывод:**", "**Уверенность:**")
    for raw in str(markdown or "").splitlines():
        line = raw.strip()
        if not line or any(line.startswith(prefix) for prefix in skip_prefixes):
            continue
        if re.match(r"^(?:\d+\.|-)\s+", line):
            continue
        lines.append(line.lstrip("#").strip())
    return "\n\n".join(lines).strip()


def _evidence_type_label(value: Any) -> str:
    mapping = {
        "retrieval": "Поиск по документам",
        "graph_fact": "Факт графа",
        "source": "Источник",
        "vector": "Поиск по документам",
        "hybrid": "Гибридный поиск",
        "neo4j": "Граф знаний",
    }
    text = str(value or "").strip()
    return mapping.get(text.lower(), _clean_public_text(text) or "Источник")


def _looks_like_damaged_entity(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return bool(
        re.fullmatch(r"g\d{4,}", text)
        or re.fullmatch(r"[a-z]+_from_chunk", text)
        or text in {"material_from_chunk", "стали в"}
    )


def _translate_domain_terms(value: str) -> str:
    replacements = {
        "corrosion": "коррозионной стойкости",
        "corrosion resistance": "коррозионной стойкости",
        "strength": "прочности",
        "hardness": "твердости",
        "plasticity": "пластичности",
        "viscosity": "вязкости",
        "evidence": "источники",
        "source": "источник",
        "sources": "источники",
    }
    text = value
    for raw, translated in replacements.items():
        text = re.sub(rf"\b{re.escape(raw)}\b", translated, text, flags=re.IGNORECASE)
    return text


def _payload_question(payload: dict[str, Any]) -> str:
    for key in ("question", "query", "user_question", "original_question"):
        value = _clean_public_text(payload.get(key))
        if value:
            return value
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    return _clean_public_text(request.get("question"))


def _markdown_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = INTERNAL_ID_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "Нет данных."


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    header = "| " + " | ".join(_escape_markdown_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_escape_markdown_cell(row.get(column)) for column in columns) + " |")
    return "\n".join([header, separator, *body])


def _escape_markdown_cell(value: Any) -> str:
    text = _clean_public_text(value)
    text = text.replace("|", "\\|")
    return text.replace("\n", " ").strip()


def _markdown_to_plain_text(markdown: str) -> list[str]:
    lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if line.startswith("#"):
            lines.append(line.lstrip("#").strip())
        elif line.startswith("|"):
            cells = [cell.strip().strip("-") for cell in line.strip("|").split("|")]
            if any(cells):
                lines.append(" | ".join(cell for cell in cells if cell))
        else:
            lines.append(line.replace("**", ""))
    return lines


def _wrap_pdf_lines(lines: list[str], *, width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        words = line.split()
        current = ""
        for word in words:
            if current and len(current) + len(word) + 1 > width:
                wrapped.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            wrapped.append(current)
    return wrapped


PDF_FONT_CANDIDATES = [
    str(Path(__file__).resolve().parent / "assets" / "fonts" / "NotoSans-Regular.ttf"),
    "/code/hackathon_project/app/assets/fonts/NotoSans-Regular.ttf",
    "/models/fonts/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/noto_sans_regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
]


def _pillow_text_pdf(lines: list[str], *, title: str) -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - exercised only when deployment deps are broken
        raise RuntimeError("PDF export requires Pillow, which is already installed with Streamlit.") from exc

    font_path = _find_pdf_font_path()
    if not font_path:
        raise RuntimeError("PDF export requires a bundled Cyrillic TTF font.")

    page_width, page_height = 1240, 1754  # A4 at roughly 150 DPI.
    margin_x, margin_y = 88, 76
    body_font = ImageFont.truetype(font_path, 22)
    title_font = ImageFont.truetype(font_path, 34)
    heading_font = ImageFont.truetype(font_path, 27)
    line_gap = 8
    paragraph_gap = 14

    pages: list[Any] = []
    image = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(image)
    y = margin_y

    def new_page() -> None:
        nonlocal image, draw, y
        pages.append(image)
        image = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(image)
        y = margin_y

    def draw_line(text: str, font: Any, *, gap: int = line_gap) -> None:
        nonlocal y
        if y > page_height - margin_y - 36:
            new_page()
        draw.text((margin_x, y), text, fill=(24, 24, 24), font=font)
        bbox = draw.textbbox((margin_x, y), text or " ", font=font)
        y += max(24, bbox[3] - bbox[1]) + gap

    for raw_line in lines:
        line = _clean_public_text(raw_line)
        if not line:
            y += paragraph_gap
            if y > page_height - margin_y:
                new_page()
            continue
        if line == title:
            draw_line(line, title_font, gap=18)
        elif line in {
            "Вопрос",
            "Краткий вывод",
            "Что найдено",
            "Подтвержденные факты",
            "Найденные противоречия",
            "Ограничения анализа",
            "Использованные источники",
            "Служебная информация",
        }:
            y += 6
            draw_line(line, heading_font, gap=10)
        else:
            draw_line(line, body_font)

    pages.append(image)
    buffer = BytesIO()
    first, rest = pages[0], pages[1:]
    first.save(
        buffer,
        format="PDF",
        save_all=True,
        append_images=rest,
        resolution=150.0,
        title=title,
        subject="\n".join(_clean_public_text(line) for line in lines[:120]),
    )
    return buffer.getvalue()


def _find_pdf_font_path() -> str:
    candidates: list[str] = []
    env_path = os.getenv("PDF_FONT_PATH", "").strip()
    if env_path:
        candidates.append(env_path)
    candidates.extend(PDF_FONT_CANDIDATES)
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return str(path)
    return ""

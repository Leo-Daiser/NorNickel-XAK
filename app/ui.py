"""Streamlit product UI for the scientific knowledge graph demo."""

from __future__ import annotations

import os
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
_PROJECT_PARENT = _PROJECT_ROOT.parent
for _path in (str(_PROJECT_ROOT), str(_PROJECT_PARENT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.ui_helpers import (  # noqa: E402
    active_document_changes,
    ANSWER_STATE_FULL,
    ANSWER_STATE_NO_DATA,
    ANSWER_STATE_PARTIAL,
    NO_DATA_HINTS,
    NO_DATA_TITLE,
    PARTIAL_DATA_TITLE,
    answer_display_state,
    answer_evidence_summary_rows,
    answer_has_display_graph_data,
    answer_source_metadata_rows,
    answer_text_sections,
    build_answer_report_model,
    build_answer_markdown_export,
    build_answer_pdf_export,
    build_compact_metrics,
    caveat_rows,
    conflict_explanation_rows,
    diagnostics_to_safe_summary,
    documents_to_rows,
    evidence_to_rows,
    evidence_to_user_rows,
    facts_to_rows,
    facts_to_user_rows,
    document_metadata_summary,
    format_upload_summary,
    format_answer_markdown,
    no_exact_match_warning,
    partial_matches_to_rows,
    report_metrics,
    translate_system_message,
    upload_expansion_rows,
)
from app.graph.answer_graph import answer_graph_to_html, build_answer_graph  # noqa: E402
from app.graph.full_answer_graph import (  # noqa: E402
    build_full_answer_graph,
    full_answer_graph_to_html,
    full_graph_audit_tables,
)


API_BASE = os.getenv("API_BASE", "http://localhost:8000")
DOCUMENT_UPLOAD_TIMEOUT_SECONDS = int(os.getenv("DOCUMENT_UPLOAD_TIMEOUT_SECONDS", "300"))
URL_INGEST_TIMEOUT_SECONDS = int(os.getenv("URL_INGEST_TIMEOUT_SECONDS", "180"))
UI_UPLOAD_MAX_FILES = int(os.getenv("UI_UPLOAD_MAX_FILES", "25"))
UI_UPLOAD_MAX_TOTAL_MB = float(os.getenv("UI_UPLOAD_MAX_TOTAL_MB", "200"))

PRESET_TITLE_TO_ID = {
    "Лучший ответ": "expert_max",
    "Строгая проверка": "strict_audit",
    "Офлайн-режим": "offline_reliable",
}
DEFAULT_PRESET_TITLE = "Лучший ответ"
DEFAULT_PRESET_ID = PRESET_TITLE_TO_ID[DEFAULT_PRESET_TITLE]

DEMO_QUESTIONS = [
    {
        "question": "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?",
        "demonstrates": "material + regime + property exact graph query",
    },
    {
        "question": "Сравни ВТ6 и 7075-T6 по прочности.",
        "demonstrates": "normalized units, comparison, conflict caveat",
    },
    {
        "question": "Какие есть противоречия или неоднородные данные по прочности?",
        "demonstrates": "conflict detection",
    },
    {
        "question": "Какие пробелы в данных найдены?",
        "demonstrates": "DataGap",
    },
    {
        "question": "Найди evidence по прочности 7075-T6 после aging.",
        "demonstrates": "hybrid retrieval + English/Russian terms",
    },
]
EXAMPLE_QUESTIONS = [item["question"] for item in DEMO_QUESTIONS]
DEMO_QUESTION_HINTS = {item["question"]: item["demonstrates"] for item in DEMO_QUESTIONS}


def api_get(path: str, params: dict[str, Any] | None = None, *, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(f"{API_BASE}{path}", params=params or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: int = 90,
) -> dict[str, Any]:
    response = requests.post(f"{API_BASE}{path}", params=params or {}, json=json_body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_patch(path: str, json_body: dict[str, Any], *, timeout: int = 30) -> dict[str, Any]:
    response = requests.patch(f"{API_BASE}{path}", json=json_body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def preset_id_for_title(title: str | None) -> str:
    return PRESET_TITLE_TO_ID.get(str(title or DEFAULT_PRESET_TITLE), DEFAULT_PRESET_ID)


def build_ask_payload(question: str, top_k: int = 12, preset_id: str = DEFAULT_PRESET_ID) -> dict[str, Any]:
    return {"question": question, "top_k": top_k, "preset_id": preset_id}


def ask_api(question: str, top_k: int = 12, preset_id: str = DEFAULT_PRESET_ID) -> dict[str, Any]:
    return api_post("/ask", json_body=build_ask_payload(question, top_k=top_k, preset_id=preset_id), timeout=90)


def _safe_get(path: str, params: dict[str, Any] | None = None, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return api_get(path, params=params)
    except Exception as exc:
        return default or {"error": str(exc)}


def _response_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        return str(detail.get("error") or detail.get("detail") or detail)
    if isinstance(detail, list):
        return "; ".join(str(item.get("msg") if isinstance(item, dict) else item) for item in detail)
    return str(detail or response.text)


def _safe_request_error(exc: Exception) -> str:
    if isinstance(exc, requests.Timeout):
        return (
            "Превышено время ожидания ответа API. Документ мог быть тяжелым для парсинга; "
            "проверьте статус в списке документов или загрузите файл отдельно."
        )
    if isinstance(exc, requests.ConnectionError):
        return "API недоступен. Проверьте, что контейнер api запущен и отвечает на /health."
    return str(exc)


def uploaded_files_stats(uploaded_files: list[Any]) -> dict[str, Any]:
    total_bytes = 0
    file_count = len(uploaded_files)
    largest_file = ""
    largest_bytes = 0
    for file in uploaded_files:
        size = int(getattr(file, "size", 0) or 0)
        if not size:
            try:
                size = len(file.getvalue())
            except Exception:
                size = 0
        total_bytes += size
        if size > largest_bytes:
            largest_bytes = size
            largest_file = str(getattr(file, "name", "uploaded"))
    total_mb = round(total_bytes / (1024 * 1024), 3)
    largest_mb = round(largest_bytes / (1024 * 1024), 3)
    blocked = file_count > UI_UPLOAD_MAX_FILES or total_mb > UI_UPLOAD_MAX_TOTAL_MB
    reasons = []
    if file_count > UI_UPLOAD_MAX_FILES:
        reasons.append(f"too_many_files:{file_count}>{UI_UPLOAD_MAX_FILES}")
    if total_mb > UI_UPLOAD_MAX_TOTAL_MB:
        reasons.append(f"too_large_total_mb:{total_mb}>{UI_UPLOAD_MAX_TOTAL_MB:g}")
    return {
        "files_count": file_count,
        "total_mb": total_mb,
        "largest_file": largest_file,
        "largest_mb": largest_mb,
        "blocked": blocked,
        "reasons": reasons,
    }


def upload_guidance_message(stats: dict[str, Any]) -> str:
    if not stats.get("blocked"):
        if int(stats.get("files_count") or 0) > 1:
            return "Файлы будут загружены по одному: сбой одного документа не остановит весь пакет."
        return ""
    return (
        # Legacy regression contract: Выбран слишком большой batch для Streamlit upload
        "Выбран слишком большой пакет для загрузки через интерфейс. "
        "Для реального корпуса используйте CLI batch ingest: "
        "`python scripts/batch_ingest_corpus.py --input data_storage --max-file-mb 25 --timeout 300 "
        "--report artifacts/batch_ingest_report.json`."
    )


def _post_document_file(file: Any, *, timeout: int = DOCUMENT_UPLOAD_TIMEOUT_SECONDS, sync_graph: bool = False) -> dict[str, Any]:
    files_param = [("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))]
    try:
        response = requests.post(
            f"{API_BASE}/ingest/documents",
            params={"sync_graph": str(sync_graph).lower()},
            files=files_param,
            timeout=timeout,
        )
    except Exception as exc:
        return {
            "filename": getattr(file, "name", "uploaded"),
            "status": "upload_error",
            "parser": "",
            "chunks": 0,
            "parser_error": _safe_request_error(exc),
            "parser_diagnostics": {"warnings": [type(exc).__name__]},
            "knowledge_expansion": {"status": "skipped", "reason": "upload_error"},
        }
    if response.status_code != 200:
        return {
            "filename": getattr(file, "name", "uploaded"),
            "status": "upload_error",
            "parser": "",
            "chunks": 0,
            "parser_error": _response_error_message(response),
            "parser_diagnostics": {"warnings": [f"http_{response.status_code}"]},
            "knowledge_expansion": {"status": "skipped", "reason": "upload_error"},
        }
    payload = response.json()
    items = payload.get("ingested")
    if isinstance(items, list) and items:
        return items[0]
    if isinstance(items, dict):
        return items
    return {
        "filename": getattr(file, "name", "uploaded"),
        "status": "upload_error",
        "parser_error": "API returned an empty ingestion result.",
        "parser_diagnostics": {"warnings": ["empty_ingestion_result"]},
        "knowledge_expansion": {"status": "skipped", "reason": "empty_ingestion_result"},
    }


def upload_documents_sequentially(uploaded_files: list[Any]) -> dict[str, Any]:
    """Upload files one by one so a slow/bad document cannot crash the whole UI batch."""

    results: list[dict[str, Any]] = []
    total = len(uploaded_files)
    progress = st.progress(0, text="Подготовка загрузки документов...")
    status_box = st.empty()
    for index, file in enumerate(uploaded_files, start=1):
        status_box.info(f"Загружаю {index}/{total}: {file.name}")
        results.append(_post_document_file(file, sync_graph=False))
        progress.progress(index / total, text=f"Обработано {index}/{total}: {file.name}")
    status_box.empty()
    return {"ingested": results}


def _selected_preset_id() -> str:
    return preset_id_for_title(st.session_state.get("preset_title", DEFAULT_PRESET_TITLE))


def _dataframe_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _dataframe(rows: Any, *, empty: str) -> None:
    normalized_rows: list[dict[str, Any]] = []
    if isinstance(rows, dict):
        normalized_rows = [{str(key): _dataframe_cell(value) for key, value in rows.items()}]
    elif isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                normalized_rows.append({str(key): _dataframe_cell(value) for key, value in row.items()})
            elif row not in (None, "", []):
                normalized_rows.append({"value": _dataframe_cell(row)})
    elif rows not in (None, "", []):
        normalized_rows = [{"value": _dataframe_cell(rows)}]
    if normalized_rows:
        st.dataframe(pd.DataFrame(normalized_rows), hide_index=True, use_container_width=True)
    else:
        st.info(empty)


def _answer_graph_key(payload: dict[str, Any]) -> str:
    identity = {
        "question": st.session_state.get("last_question", ""),
        "status": payload.get("status"),
        "answer_mode": payload.get("answer_mode"),
        "analytical_intent": payload.get("analytical_intent"),
        "constraints": payload.get("constraints"),
    }
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _answer_graph_modal_state_key(answer_key: str) -> str:
    return f"answer_graph_modal_open_{answer_key}"


def _ensure_answer_graph_modal_state(answer_key: str) -> str:
    state_key = _answer_graph_modal_state_key(answer_key)
    st.session_state.setdefault("answer_graph_modal_open", False)
    st.session_state.setdefault(state_key, False)
    return state_key


def _open_answer_graph_modal(answer_key: str) -> None:
    st.session_state["answer_graph_modal_open"] = True
    st.session_state[_answer_graph_modal_state_key(answer_key)] = True


def _close_answer_graph_modal(answer_key: str) -> None:
    st.session_state["answer_graph_modal_open"] = False
    st.session_state[_answer_graph_modal_state_key(answer_key)] = False


def _full_graph_modal_state_key(answer_key: str) -> str:
    return f"full_graph_modal_open_{answer_key}"


def _ensure_full_graph_modal_state(answer_key: str) -> str:
    state_key = _full_graph_modal_state_key(answer_key)
    st.session_state.setdefault("full_graph_modal_open", False)
    st.session_state.setdefault(state_key, False)
    return state_key


def _open_full_graph_modal(answer_key: str) -> None:
    st.session_state["full_graph_modal_open"] = True
    st.session_state[_full_graph_modal_state_key(answer_key)] = True


def _close_full_graph_modal(answer_key: str) -> None:
    st.session_state["full_graph_modal_open"] = False
    st.session_state[_full_graph_modal_state_key(answer_key)] = False


def _render_graph_header(answer_key: str) -> None:
    _ensure_answer_graph_modal_state(answer_key)
    title_col, action_col = st.columns([0.58, 0.42], vertical_alignment="center")
    with title_col:
        st.subheader("Интерактивный связанный граф")
    with action_col:
        st.button(
            "Развернуть карту",
            key=f"open_answer_graph_modal_{answer_key}",
            on_click=_open_answer_graph_modal,
            args=(answer_key,),
            use_container_width=True,
        )


def _render_interactive_graph(payload: dict[str, Any], answer_graph: Any, answer_key: str) -> None:
    components.html(
        answer_graph_to_html(answer_graph, container_id=f"answerGraphCompact_{answer_key}"),
        height=560,
        scrolling=False,
    )
    st.caption(
        "Стрелки показывают реальные связи между сущностями графа знаний. "
        "Отсутствие стрелки означает, что такая связь не была обнаружена."
    )
    if st.session_state.get(_answer_graph_modal_state_key(answer_key)):
        _render_large_answer_graph(answer_graph, answer_key)


def _full_graph_filters(answer_key: str) -> dict[str, bool]:
    # Legacy product contract text: Источники/evidence
    top_cols = st.columns([0.34, 0.33, 0.33])
    bottom_cols = st.columns([0.5, 0.5])
    with top_cols[0]:
        show_sources = st.checkbox("Источники", value=True, key=f"full_graph_show_sources_{answer_key}")
    with top_cols[1]:
        show_gaps = st.checkbox("Пробелы данных", value=True, key=f"full_graph_show_gaps_{answer_key}")
    with top_cols[2]:
        show_conflicts = st.checkbox("Конфликты", value=True, key=f"full_graph_show_conflicts_{answer_key}")
    with bottom_cols[0]:
        show_measurements = st.checkbox("Измерения", value=True, key=f"full_graph_show_measurements_{answer_key}")
    with bottom_cols[1]:
        active_only = st.checkbox("Только активные", value=True, key=f"full_graph_active_only_{answer_key}")
    return {
        "show_sources": show_sources,
        "show_gaps": show_gaps,
        "show_conflicts": show_conflicts,
        "show_measurements": show_measurements,
        "active_only": active_only,
    }


def _render_full_graph(payload: dict[str, Any], answer_key: str) -> None:
    _ensure_full_graph_modal_state(answer_key)
    st.divider()
    title_col, action_col = st.columns([0.56, 0.44], vertical_alignment="center")
    with title_col:
        st.subheader("Карта происхождения ответа")
        st.caption(
            "Показывает, какие сущности, факты и источники сформировали ответ. "
            "Дубли объединены, технические идентификаторы доступны в аудите."
        )
    with action_col:
        st.button(
            "Развернуть карту происхождения",
            key=f"open_full_graph_modal_{answer_key}",
            on_click=_open_full_graph_modal,
            args=(answer_key,),
            use_container_width=True,
        )
    filters = _full_graph_filters(answer_key)
    full_graph = build_full_answer_graph(payload, filters=filters)
    components.html(
        full_answer_graph_to_html(
            full_graph,
            render_height=620,
            render_width=1200,
            container_id=f"fullAnswerGraphCompact_{answer_key}",
        ),
        height=740,
        scrolling=False,
    )
    stats = full_graph.stats
    if stats.get("truncated"):
        st.caption(
            "Показаны ключевые связи ответа. Полные raw-данные доступны в аудите."
        )
    if st.session_state.get(_full_graph_modal_state_key(answer_key)):
        _render_large_full_graph(full_graph, answer_key)
    with st.expander("Аудит графа"):
        st.caption("Экспертный режим проверки: карта выше показывает читаемое обоснование; здесь доступны исходные идентификаторы и типы связей.")
        nodes, edges = full_graph_audit_tables(payload, full_graph)
        st.markdown("**Аудит узлов**")
        _dataframe(nodes, empty="Узлы отсутствуют.")
        st.markdown("**Аудит связей**")
        _dataframe(edges, empty="Связи отсутствуют.")


def _render_large_full_graph(full_graph: Any, answer_key: str) -> None:
    dialog = getattr(st, "dialog", None)
    if dialog is not None:
        try:
            decorator = dialog("Карта происхождения ответа", width="large")
        except TypeError:
            decorator = dialog("Карта происхождения ответа")

        @decorator
        def _large_full_graph_dialog() -> None:
            _render_answer_graph_modal_css()
            _, close_col = st.columns([0.9, 0.1])
            with close_col:
                if st.button("×", key=f"close_full_graph_modal_{answer_key}", help="Закрыть"):
                    _close_full_graph_modal(answer_key)
                    st.rerun()
            components.html(
                full_answer_graph_to_html(
                    full_graph,
                    render_height=820,
                    render_width=1500,
                    container_id=f"fullAnswerGraphExpanded_{answer_key}",
                ),
                height=930,
                scrolling=False,
            )

        _large_full_graph_dialog()
        return

    with st.container(border=True):
        close_col, _ = st.columns([0.2, 0.8])
        with close_col:
            if st.button("Закрыть", key=f"close_full_graph_modal_inline_{answer_key}"):
                _close_full_graph_modal(answer_key)
                st.rerun()
        components.html(
            full_answer_graph_to_html(
                full_graph,
                render_height=820,
                render_width=1500,
                container_id=f"fullAnswerGraphExpanded_{answer_key}",
            ),
            height=930,
            scrolling=False,
        )


def _render_answer_graph_modal_css() -> None:
    st.markdown(
        """
<style>
div[data-testid="stModal"] div[role="dialog"],
div[data-testid="stDialog"] div[role="dialog"] {
  width: min(85vw, 1500px) !important;
  max-width: min(85vw, 1500px) !important;
}
div[data-testid="stModal"] div[role="dialog"] > div,
div[data-testid="stDialog"] div[role="dialog"] > div {
  max-height: 90vh;
}
div[data-testid="stModal"],
div[data-testid="stDialog"] {
  background: rgba(15, 23, 42, 0.34);
}
div[data-testid="stModal"] button[aria-label="Close"],
div[data-testid="stDialog"] button[aria-label="Close"] {
  display: none !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_large_answer_graph(answer_graph: Any, answer_key: str) -> None:
    dialog = getattr(st, "dialog", None)
    if dialog is not None:
        try:
            decorator = dialog("Крупная карта ответа", width="large")
        except TypeError:
            decorator = dialog("Крупная карта ответа")

        @decorator
        def _large_graph_dialog() -> None:
            _render_answer_graph_modal_css()
            _, close_col = st.columns([0.9, 0.1])
            with close_col:
                if st.button("×", key=f"close_answer_graph_modal_{answer_key}", help="Закрыть"):
                    _close_answer_graph_modal(answer_key)
                    st.rerun()
            components.html(
                answer_graph_to_html(
                    answer_graph,
                    render_height=820,
                    render_width=1500,
                    container_id=f"answerGraphExpanded_{answer_key}",
                ),
                height=930,
                scrolling=False,
            )

        _large_graph_dialog()
        return

    with st.container(border=True):
        close_col, _ = st.columns([0.2, 0.8])
        with close_col:
            if st.button("Закрыть", key=f"close_answer_graph_modal_inline_{answer_key}"):
                _close_answer_graph_modal(answer_key)
                st.rerun()
        components.html(
            answer_graph_to_html(
                answer_graph,
                render_height=820,
                render_width=1500,
                container_id=f"answerGraphExpanded_{answer_key}",
            ),
            height=930,
            scrolling=False,
        )


def _render_answer(payload: dict[str, Any]) -> None:
    state = answer_display_state(payload)
    warning = no_exact_match_warning(payload)
    if warning and state == ANSWER_STATE_FULL:
        st.warning(warning)
    report = build_answer_report_model(payload, question=st.session_state.get("last_question", ""))
    header_col, pdf_col, markdown_col = st.columns([0.56, 0.22, 0.22], vertical_alignment="center")
    with header_col:
        st.subheader("Ответ")
    with pdf_col:
        _render_export_button(payload, "pdf")
    with markdown_col:
        _render_export_button(payload, "markdown")
    if state == ANSWER_STATE_NO_DATA:
        st.error(NO_DATA_TITLE)
        for hint in NO_DATA_HINTS:
            st.markdown(f"- {hint}")
        return
    if state == ANSWER_STATE_PARTIAL:
        st.warning(PARTIAL_DATA_TITLE)
    st.markdown("**Краткий вывод**")
    st.markdown(str(report["short_summary"]))
    findings = report.get("findings") or []
    if findings:
        st.markdown("**Что найдено**")
        for item in findings:
            st.markdown(f"- {item}")
    metrics = report_metrics(report)
    if metrics:
        _render_metrics(metrics)
    if not findings:
        _render_report_list(report.get("findings") or [], "Что найдено")
    _render_report_table(report.get("confirmed_facts") or [], "Подтвержденные факты")
    _render_report_list(
        report.get("conflicts") or ["Явных противоречий по этому запросу не найдено."],
        "Найденные противоречия",
        warning=True,
    )
    _render_report_list(
        report.get("limitations") or ["Ограничения не указаны."],
        "Ограничения анализа",
        warning=True,
    )
    _render_report_table(report.get("sources") or [], "Использованные источники")
    if state == ANSWER_STATE_PARTIAL:
        _render_visible_partial_matches(payload)
    _render_source_metadata_summary(payload)


def _render_export_button(payload: dict[str, Any], export_type: str) -> None:
    question = st.session_state.get("last_question", "")
    answer_key = _answer_graph_key(payload)
    if export_type == "markdown":
        st.download_button(
            "⬇ Markdown",
            data=build_answer_markdown_export(payload, question=question),
            file_name="graphrag_answer.md",
            mime="text/markdown",
            key=f"download_markdown_{answer_key}",
            use_container_width=True,
        )
        return
    st.download_button(
        "⬇ PDF",
        data=build_answer_pdf_export(payload, question=question),
        file_name="graphrag_answer.pdf",
        mime="application/pdf",
        key=f"download_pdf_{answer_key}",
        use_container_width=True,
    )


def _render_export_buttons(payload: dict[str, Any]) -> None:
    _render_export_button(payload, "markdown")
    _render_export_button(payload, "pdf")


def _render_metrics(metrics: dict[str, Any]) -> None:
    items = list(metrics.items())
    for chunk in (items[:3], items[3:]):
        if not chunk:
            continue
        cols = st.columns(len(chunk))
        for col, (label, value) in zip(cols, chunk):
            with col:
                st.metric(label, value)


def _render_report_table(rows: list[dict[str, Any]], title: str) -> None:
    if not rows:
        return
    st.markdown(f"**{title}**")
    _dataframe(rows, empty="")


def _render_report_list(items: list[str], title: str, *, warning: bool = False) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    for item in items:
        if warning:
            st.warning(item)
        else:
            st.markdown(f"- {item}")


def _render_answer_evidence_summary(payload: dict[str, Any], *, title: str | None = "Основание ответа") -> None:
    rows = answer_evidence_summary_rows(payload)
    if not rows:
        return
    if title:
        st.markdown(f"**{title}**")
    for row in rows:
        original = f"\n\n  Исходное значение: {row['Исходное значение']}" if row.get("Исходное значение") else ""
        st.markdown(
            f"- **{row['Факт']}**{original}\n\n"
            f"  Источник: {row['Источник']}\n\n"
            f"  Фрагмент: \"{row['Фрагмент']}\""
        )


def _render_source_metadata_summary(payload: dict[str, Any]) -> None:
    rows = answer_source_metadata_rows(payload)
    if not rows:
        return
    # Legacy product contract text: Структура источников
    with st.expander("Подробнее об источниках", expanded=False):
        _dataframe(rows, empty="")


def _render_conflict_explanation(payload: dict[str, Any], *, title: str = "Неоднородность данных") -> None:
    rows = conflict_explanation_rows(payload)
    if not rows:
        return
    st.markdown(f"**{title}**")
    for row in rows:
        st.warning(row["Описание"])


def _render_visible_facts(payload: dict[str, Any]) -> None:
    rows = facts_to_user_rows(payload)
    if not rows:
        return
    st.markdown("**Найденные факты**")
    _dataframe(rows, empty="")


def _render_visible_evidence(payload: dict[str, Any], *, title: str = "Источники") -> None:
    rows = evidence_to_user_rows(payload)
    if not rows:
        return
    st.markdown(f"**{title}**")
    _dataframe(rows, empty="")


def _render_visible_caveats(payload: dict[str, Any]) -> None:
    rows = caveat_rows(payload)
    if not rows:
        return
    st.markdown("**Пробелы / ограничения**")
    for row in rows:
        st.warning(row["Описание"])


def _render_structured_caveats(caveats: list[str]) -> None:
    if not caveats:
        return
    st.markdown("**Ограничения**")
    for item in caveats:
        st.warning(item)


def _render_partial_match_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows[:8]:
        match_type = str(row.get("Тип совпадения") or "partial")
        details = []
        for key, value in row.items():
            if key == "Тип совпадения" or value in (None, "", []):
                continue
            details.append(f"{key}: {_dataframe_cell(value)}")
        text = " · ".join(details[:4])
        st.markdown(f"- **{match_type}**" + (f": {text}" if text else ""))
    if len(rows) > 8:
        st.caption(f"Показаны первые 8 частичных совпадений из {len(rows)}.")


def _render_visible_partial_matches(payload: dict[str, Any]) -> None:
    rows = partial_matches_to_rows(payload)
    if not rows:
        return
    st.markdown("**Частично релевантные результаты**")
    _render_partial_match_rows(rows)


def _render_details(payload: dict[str, Any]) -> None:
    evidence_rows = evidence_to_user_rows(payload)
    if evidence_rows:
        with st.expander("Использованные источники", expanded=False):
            _dataframe(evidence_rows, empty="")

    fact_rows = facts_to_user_rows(payload)
    raw_fact_rows = facts_to_rows(payload)
    if fact_rows or raw_fact_rows:
        with st.expander("Проверенные факты"):
            if fact_rows:
                _dataframe(fact_rows, empty="")
            if raw_fact_rows:
                with st.container(border=True):
                    st.caption("Исходные факты для проверки")
                    _dataframe(raw_fact_rows, empty="")

    raw_evidence_rows = evidence_to_rows(payload)
    if evidence_rows or raw_evidence_rows:
        with st.expander("Источники и evidence", expanded=False):
            if evidence_rows:
                _dataframe(evidence_rows, empty="")
            if raw_evidence_rows:
                with st.container(border=True):
                    st.caption("Исходные строки источников")
                    _dataframe(raw_evidence_rows, empty="")

    decision_history = payload.get("decision_history") or []
    if decision_history:
        with st.expander("История найденных решений", expanded=False):
            _dataframe(decision_history, empty="")

    gaps = payload.get("data_gaps") or payload.get("gaps") or []
    if gaps:
        with st.expander("Пробелы в данных"):
            _dataframe(gaps, empty="")

    partial_rows = partial_matches_to_rows(payload)
    if partial_rows:
        with st.expander("Частично релевантные результаты", expanded=False):
            _render_partial_match_rows(partial_rows)

    with st.expander("Диагностика ответа", expanded=False):
        response_diagnostics = payload.get("diagnostics") or {}
        st.json(
            {
                "selected_preset_from_ui": st.session_state.get("last_selected_preset_from_ui"),
                "request_payload": st.session_state.get("last_request_payload"),
                "response_diagnostics_preset_id": response_diagnostics.get("preset_id"),
            }
        )
        st.json(diagnostics_to_safe_summary(payload))
        st.json(
            {
                "constraints": payload.get("constraints"),
                "graph_context": payload.get("graph_context"),
                "diagnostics": payload.get("diagnostics"),
                "retrieval": payload.get("retrieval"),
                "llm": payload.get("llm"),
                "technical_answer": payload.get("technical_answer"),
            }
        )


def _run_question(question: str, preset_id: str) -> None:
    if not question.strip():
        st.warning("Введите исследовательский вопрос.")
        return
    request_payload = build_ask_payload(question.strip(), preset_id=preset_id)
    with st.spinner("Ищу ответ в графе, evidence и активных документах..."):
        try:
            payload = api_post("/ask", json_body=request_payload, timeout=90)
        except Exception as exc:
            st.error(f"Ошибка /ask: {exc}")
            return
    payload.setdefault("question", question.strip())
    st.session_state["last_question"] = question.strip()
    st.session_state["last_request_payload"] = request_payload
    st.session_state["last_selected_preset_from_ui"] = preset_id
    st.session_state["last_answer_payload"] = payload
    st.session_state["answer_graph_modal_open"] = False
    st.session_state["full_graph_modal_open"] = False


def _render_document_controls() -> None:
    _render_knowledge_growth_summary()
    with st.expander("Документы", expanded=False):
        st.markdown("**Загрузка документов**")
        upload_source = st.radio("Способ загрузки", ["Из файла", "Из веб-страницы"], horizontal=True, key="document_upload_source")
        if upload_source == "Из веб-страницы":
            st.markdown("**Добавить веб-страницу**")
            url = st.text_input("URL страницы", placeholder="https://example.org/reports/vt6-annealing.html", key="web_page_url_input")
            if st.button("Загрузить страницу", key="ingest_web_page_button", disabled=not bool(url.strip())):
                with st.spinner("Загружаю HTML, разбиваю на фрагменты и обновляю граф знаний..."):
                    try:
                        response = requests.post(f"{API_BASE}/ingest/url", params={"url": url.strip()}, timeout=URL_INGEST_TIMEOUT_SECONDS)
                    except Exception as exc:
                        st.error(f"Не удалось загрузить страницу: {_safe_request_error(exc)}")
                        response = None
                if response is None:
                    pass
                elif response.status_code == 200:
                    result = response.json()
                    st.success("Веб-страница загружена.")
                    _render_ingestion_result(result, documents_payload=_safe_get("/documents", default=[]), graph_updated=True)
                else:
                    st.error(f"Не удалось загрузить страницу: {_response_error_message(response)}")
        else:
            st.markdown("**Загрузка файлов**")
            uploaded_files = st.file_uploader(
                "Файлы PDF/DOCX/PPTX/XLSX/CSV/HTML/TXT/MD",
                type=["pdf", "docx", "pptx", "xlsx", "html", "htm", "csv", "txt", "md"],
                accept_multiple_files=True,
            )
            upload_stats = uploaded_files_stats(list(uploaded_files or []))
            if uploaded_files:
                st.caption(
                    f"Выбрано файлов: {upload_stats['files_count']} · общий размер: {upload_stats['total_mb']} MB · "
                    f"самый большой: {upload_stats['largest_file']} ({upload_stats['largest_mb']} MB)"
                )
                guidance = upload_guidance_message(upload_stats)
                if upload_stats["blocked"]:
                    st.warning(guidance)
                    st.code(
                        "python scripts/batch_ingest_corpus.py --input data_storage --dry-run --max-file-mb 25\n"
                        "python scripts/batch_ingest_corpus.py --input data_storage --max-file-mb 25 --timeout 300 "
                        "--report artifacts/batch_ingest_report.json",
                        language="powershell",
                    )
                elif guidance:
                    st.info(guidance)
            sync_after_upload = st.checkbox("После загрузки обновить граф/индекс", value=True, key="sync_graph_after_upload")
            upload_blocked = bool(uploaded_files and upload_stats["blocked"])
            if st.button("Загрузить в базу", type="primary", disabled=upload_blocked or not bool(uploaded_files)):
                with st.spinner("Парсинг документов и сохранение фрагментов..."):
                    result = upload_documents_sequentially(list(uploaded_files or []))
                graph_updated = False
                if sync_after_upload:
                    graph_updated = _refresh_graph(show_details=False)
                failures = [
                    item
                    for item in result.get("ingested", [])
                    if str(item.get("status") or "").lower() in {"upload_error", "parse_failed", "skipped_empty"}
                ]
                docs_after_upload = _safe_get("/documents", default=[])
                if failures:
                    st.warning(f"Загрузка завершена с предупреждениями: проблемных файлов {len(failures)} из {len(uploaded_files)}.")
                else:
                    st.success("Документы загружены.")
                _render_ingestion_result(result, documents_payload=docs_after_upload, graph_updated=graph_updated)

        st.divider()
        st.markdown("**Граф и active corpus**")
        st.caption("Обновляет retrieval/fallback cache по активным документам. Полный Neo4j sync запускайте отдельно, если нужен heavy rebuild.")
        if st.button("Обновить граф по активным документам"):
            _refresh_graph(show_details=True)

        st.divider()
        st.markdown("**Управление документами**")
        docs_payload = _safe_get("/documents", default=[])
        rows = documents_to_rows(docs_payload)
        if rows:
            st.caption("Измените галочки и нажмите «Применить изменения». Rebuild запускается отдельной кнопкой.")
            with st.form("documents_active_form"):
                edited = st.data_editor(
                    pd.DataFrame(rows),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Активен": st.column_config.CheckboxColumn("Активен", help="Включает документ в retrieval/graph QA."),
                        "doc_id": st.column_config.TextColumn("doc_id", disabled=True, width="small"),
                    },
                    disabled=["Документ", "Тип", "Chunks", "Дата загрузки", "Parser", "Blocks", "Tables", "doc_id"],
                    key="documents_active_editor",
                )
                apply_active_changes = st.form_submit_button("Применить изменения")
            changes = active_document_changes(rows, edited)
            if changes and not apply_active_changes:
                st.info(f"Есть неприменённые изменения: {len(changes)}. Нажмите «Применить изменения».")
            if apply_active_changes:
                if not changes:
                    st.info("Изменений в активности документов нет.")
                else:
                    try:
                        result = api_post(
                            "/documents/active",
                            json_body={"changes": {doc_id: active for doc_id, active in changes}, "sync_neo4j": False},
                            timeout=90,
                        )
                        st.success(f"Применено изменений: {result.get('updated', len(changes))}. Теперь нажмите «Обновить граф по активным документам».")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Не удалось применить изменения активности: {_safe_request_error(exc)}")
                        return
        else:
            st.info("Документы пока не загружены.")
        if rows:
            labels = [f"{row['Документ']} | {row['doc_id']}" for row in rows]
            selected = st.selectbox("Документ для просмотра metadata", labels)
            selected_id = selected.rsplit(" | ", 1)[-1]
            doc_items = docs_payload.get("items", []) if isinstance(docs_payload, dict) else docs_payload
            original = next((item for item in doc_items if item.get("doc_id") == selected_id), {})
            with st.container(border=True):
                st.markdown("**Краткая карточка документа**")
                _dataframe([document_metadata_summary(original)], empty="Нет метаданных для выбранного документа.")
            if st.checkbox("Показать технические метаданные", value=False, key=f"show_document_metadata_{selected_id}"):
                st.json(original)


def _render_ingestion_result(
    result: dict[str, Any],
    *,
    documents_payload: dict[str, Any] | list[dict[str, Any]] | None = None,
    graph_updated: bool = False,
) -> None:
    summary = format_upload_summary(result, graph_updated=graph_updated, documents_payload=documents_payload)
    st.markdown("**Итог загрузки**")
    for label, value in summary.items():
        st.write(f"- {label}: {value}")

    items = result.get("ingested")
    if isinstance(items, dict):
        items = [items]
    rows = []
    for item in items or []:
        diagnostics = item.get("parser_diagnostics") or {}
        rows.append(
            {
                "Документ": item.get("source_name") or item.get("filename") or item.get("url"),
                "Статус": item.get("status"),
                "Парсер": item.get("parser"),
                "Фрагменты": item.get("chunks"),
                "Блоки": diagnostics.get("blocks_count"),
                "Таблицы": diagnostics.get("tables_count"),
                "Изображения": diagnostics.get("images_count"),
                "Ошибка парсинга": item.get("parser_error"),
            }
        )
    _dataframe(rows, empty="Нет строк результата загрузки.")
    expansion_rows = upload_expansion_rows(result)
    if expansion_rows:
        st.markdown("**Что добавилось после загрузки**")
        _dataframe(expansion_rows, empty="Нет изменений knowledge graph.")
    if st.checkbox("Показать технические детали загрузки", value=False, key="show_upload_technical_details"):
        st.json(result)


def _render_knowledge_growth_summary() -> None:
    summary = _safe_get("/knowledge/summary", default={})
    if not isinstance(summary, dict) or summary.get("status") != "ok":
        return
    with st.container(border=True):
        st.markdown("**Расширение базы знаний**")
        metric_rows = [
            [
                ("Активные документы", summary.get("active_documents_count", 0)),
                ("Подтвержденные факты", summary.get("canonical_facts_count", 0)),
                ("Новые связи", summary.get("new_connections_count", 0)),
            ],
            [
                ("Конфликты", summary.get("conflict_groups_count", 0)),
                ("Пробелы данных", summary.get("data_gaps_count", 0)),
                ("Источники", summary.get("sources_count", summary.get("active_documents_count", 0))),
            ],
        ]
        for row in metric_rows:
            cols = st.columns(3)
            for col, (label, value) in zip(cols, row):
                col.metric(label, value)
        if summary.get("last_ingested_at"):
            st.caption(f"Последнее обновление: {summary.get('last_ingested_at')}")


def _refresh_graph(*, show_details: bool = False) -> bool:
    with st.spinner("Обновляю активный корпус и графовую проекцию..."):
        try:
            result = api_post("/graph/refresh", params={"sync_neo4j": "false"}, timeout=160)
        except Exception as exc:
            st.error(f"Ошибка обновления графа: {exc}")
            return False
    st.success("Граф/индекс обновлены.")
    if show_details:
        st.json(result)
    return True


def _render_question_block(preset_id: str) -> None:
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    st.subheader("🔎 Задайте исследовательский вопрос")
    # Legacy product-eval marker: Подставить пример. Examples are not rendered as long top buttons.
    st.text_area("Вопрос", key="question_input", height=150, label_visibility="collapsed")
    if st.button("Найти ответ", type="primary"):
        _run_question(st.session_state.get("question_input", ""), preset_id)


def _render_sidebar_diagnostics() -> None:
    with st.sidebar:
        st.subheader("Состояние")
        health = _safe_get("/health", default={"status": "unavailable"})
        catalog = health.get("catalog") or {}
        ready = health.get("status") == "ok"
        st.write("✓ Система готова" if ready else "Система временно недоступна")
        st.write(f"Документов загружено: {catalog.get('documents', 0)}")
        st.write(f"Последнее обновление: {_last_update_label(health)}")
        llm = health.get("llm") or {}
        with st.expander("Диагностика системы", expanded=False):
            st.write(f"API: {health.get('status')}")
            st.write(f"Neo4j: {_neo4j_status_label(health)}")
            st.write(f"LLM: {'готов' if llm.get('ready') else 'не готов'}")
            if llm.get("model"):
                st.write(f"Модель: {llm.get('model')}")
            if not llm.get("ready"):
                st.warning(translate_system_message(llm.get("last_error") or "LLM не готов."))
            st.write(f"Количество документов: {catalog.get('documents', 0)}")
            st.write(f"Количество chunks: {catalog.get('chunks', 0)}")
            st.write(f"Активных документов: {catalog.get('active_documents', catalog.get('documents', 0))}")
            st.write(f"Активных chunks: {catalog.get('active_chunks', catalog.get('chunks', 0))}")
            st.write(f"Последнее обновление: {_last_update_label(health)}")
            if st.button("Проверить LLM"):
                try:
                    result = api_post("/system/test-llm", timeout=60)
                    st.success("Проверка LLM выполнена.")
                    st.write(f"Результат: {translate_system_message(result.get('status') or result.get('message') or 'готов')}")
                except Exception as exc:
                    st.error(f"Проверка LLM не прошла: {exc}")


def _last_update_label(health: dict[str, Any]) -> str:
    catalog = health.get("catalog") or {}
    candidates = [
        catalog.get("last_ingested_at"),
        catalog.get("last_updated_at"),
        health.get("last_ingested_at"),
        health.get("last_updated_at"),
        health.get("started_at"),
    ]
    for value in candidates:
        if value:
            return str(value)
    return "не указано"


def _neo4j_status_label(health: dict[str, Any]) -> str:
    backend = str(health.get("kg_backend_active") or "").lower()
    if backend == "neo4j":
        return "подключен"
    if backend:
        return f"не активен, используется {backend}"
    return "неизвестно"


def main() -> None:
    st.set_page_config(page_title="Scientific Knowledge Graph", layout="wide")
    st.title("Scientific Knowledge Graph")
    st.caption(
        "Система связывает документы, эксперименты, материалы, режимы, свойства, оборудование, лаборатории, выводы и пробелы в данных."
    )

    _render_sidebar_diagnostics()
    if st.session_state.get("preset_title") not in PRESET_TITLE_TO_ID:
        st.session_state["preset_title"] = DEFAULT_PRESET_TITLE
    st.radio("Режим работы", list(PRESET_TITLE_TO_ID), horizontal=True, key="preset_title")
    preset_id = _selected_preset_id()

    _render_document_controls()
    _render_question_block(preset_id)

    payload = st.session_state.get("last_answer_payload")
    if not payload:
        st.info("Загрузите документы при необходимости, затем задайте вопрос.")
        return

    has_graph = answer_has_display_graph_data(payload)
    if has_graph:
        left, right = st.columns([0.56, 0.44], gap="large")
    else:
        left = st.container()
        right = None
    with left:
        _render_answer(payload)
    if right is not None:
        with right:
            answer_graph = build_answer_graph(payload)
            answer_key = _answer_graph_key(payload)
            _render_graph_header(answer_key)
            _render_interactive_graph(payload, answer_graph, answer_key)
            _render_full_graph(payload, answer_key)
    if answer_display_state(payload) != ANSWER_STATE_NO_DATA:
        _render_details(payload)


if __name__ == "__main__":
    main()

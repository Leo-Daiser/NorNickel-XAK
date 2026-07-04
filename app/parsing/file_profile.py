"""Resource-efficient file and corpus readiness profiler."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..domain.fact_normalization import build_conflict_summary, dedupe_fact_rows, fact_rows_from_experiments
from ..extraction.pipeline import ExtractionPipeline
from ..extraction.to_graph_models import bundle_to_data_gaps, bundle_to_experiment_facts
from ..ingestion.parser_router import ParserRouter
from .source_metadata import infer_source_metadata
from .text_quality import SUPPORTED_FILE_EXTENSIONS, text_quality_metrics

ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".001", ".002"}
LEGACY_OFFICE_EXTENSIONS = {".doc", ".xls", ".docm", ".ppt"}
IMAGE_EXTENSIONS = {".gif", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_MAX_PARSE_MB = 25.0


def profile_file(
    path: str | Path,
    *,
    parser: ParserRouter | None = None,
    pipeline: ExtractionPipeline | None = None,
    profile_mode: str = "full",
    max_parse_mb: float | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Profile one document without LLM, embeddings, Neo4j or API state."""

    file_path = Path(path)
    ext = file_path.suffix.lower()
    mode = _profile_mode(profile_mode)
    max_mb = DEFAULT_MAX_PARSE_MB if max_parse_mb is None else float(max_parse_mb)
    base = _base_profile(file_path, root=root)
    unsupported = _unsupported_profile(base, file_path)
    if unsupported:
        return unsupported
    if mode == "inventory":
        return _inventory_profile(base, file_path, warnings=["inventory_only_parse_not_run"])
    if mode == "auto" and _file_size_mb(file_path) > max_mb:
        return _inventory_profile(base, file_path, warnings=["parse_skipped_large_file"], max_parse_mb=max_mb)

    parser = parser or ParserRouter()
    pipeline = pipeline or ExtractionPipeline(mode="deterministic", enable_llm=False, audit_enabled=False)
    try:
        parsed = parser.parse_document_intelligence(
            str(file_path),
            doc_id=_stable_profile_doc_id(file_path),
            source_type="file",
        )
    except Exception as exc:
        return {
            **base,
            "parser_backend": "unknown",
            "parse_status": "failed",
            "warnings": ["parser_failed"],
            "parser_error": _safe_error(exc),
        }

    pages = _pages_estimated(parsed)
    quality = text_quality_metrics(parsed.text or "\n".join(chunk.text for chunk in parsed.chunks), pages_estimated=pages)
    raw_facts, data_gaps, rejected_count = _extract_rows(parsed.chunks, pipeline)
    canonical_facts = dedupe_fact_rows(raw_facts)
    conflicts = build_conflict_summary(canonical_facts)
    warnings = _profile_warnings(parsed, quality, canonical_facts, data_gaps)
    parse_status = _parse_status(parsed, quality, warnings)
    facts_without_evidence = sum(1 for row in canonical_facts if not row.get("evidence"))
    source_metadata = infer_source_metadata(
        source_name=file_path.name,
        source_type="file",
        parser_name=parsed.parser_name,
        text=parsed.text or "\n".join(chunk.text for chunk in parsed.chunks),
        diagnostics=parsed.diagnostics,
    )["source_metadata"]
    if base.get("source_group") and source_metadata.get("source_type_detected") in {None, "unknown"}:
        source_metadata["source_type_detected"] = _source_type_from_group(str(base.get("source_group")), ext)
        source_metadata["type_basis"] = "data_storage_group"
    return {
        **base,
        "parser_backend": parsed.parser_name,
        "parser_backend_requested": parsed.diagnostics.get("parser_backend_requested"),
        "parse_status": parse_status,
        "text_chars": quality["text_chars"],
        "text_density": quality["text_density"],
        "pages_estimated": pages,
        "tables_detected": len(parsed.tables),
        "images_detected": len(parsed.images),
        "blocks_count": len(parsed.blocks),
        "chunks_count": len(parsed.chunks),
        "facts_extracted": len(raw_facts),
        "canonical_facts": len(canonical_facts),
        "facts_without_evidence": facts_without_evidence,
        "conflict_groups": len(conflicts),
        "data_gaps": len(data_gaps),
        "source_metadata": source_metadata,
        "rejected_or_low_confidence_candidates": rejected_count,
        "warnings": warnings,
        "parser_diagnostics": _public_parser_diagnostics(parsed.diagnostics),
        "text_quality": quality,
        "canonical_fact_keys": [row.get("canonical_fact_key") for row in canonical_facts if row.get("canonical_fact_key")],
        "fact_preview": _fact_preview(canonical_facts),
        "data_gap_preview": _gap_preview(data_gaps),
    }


def profile_corpus(
    input_path: str | Path,
    *,
    profile_mode: str = "auto",
    max_parse_mb: float | None = DEFAULT_MAX_PARSE_MB,
    max_files: int | None = None,
    sample_per_group: int | None = None,
) -> dict[str, Any]:
    """Profile all files under a corpus path."""

    root = Path(input_path)
    files, selection = _corpus_files(root, max_files=max_files, sample_per_group=sample_per_group)
    parser = ParserRouter()
    pipeline = ExtractionPipeline(mode="deterministic", enable_llm=False, audit_enabled=False)
    profiles = [
        profile_file(path, parser=parser, pipeline=pipeline, profile_mode=profile_mode, max_parse_mb=max_parse_mb, root=root)
        for path in files
    ]
    all_keys = []
    for item in profiles:
        all_keys.extend(item.get("canonical_fact_keys") or [])
    summary = _corpus_summary(profiles, all_keys)
    summary["documents_skipped_by_selection"] = selection.get("skipped_files", 0)
    summary["documents_found_total"] = selection.get("total_files_found", len(profiles))
    return {
        "status": "ok",
        "input": str(root),
        "profile_mode": _profile_mode(profile_mode),
        "selection": selection,
        "summary": summary,
        "files": profiles,
        "high_risk_documents": _high_risk_documents(profiles),
        "resource_profile": {
            "runtime_profile": "economy_core_compatible",
            "llm_required": False,
            "embeddings_required": False,
            "ocr_executed": False,
            "note": "OCR is detected as required but not executed by this profiler.",
        },
    }


def write_corpus_report(report: dict[str, Any], *, json_path: str | Path, markdown_path: str | Path | None = None) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_path:
        md_path = Path(markdown_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Corpus Readiness Report",
        "",
        "## Summary",
        f"- documents_total: {summary.get('documents_total', 0)}",
        f"- successfully_parsed: {summary.get('parse_status_counts', {}).get('ok', 0)}",
        f"- partial: {summary.get('parse_status_counts', {}).get('partial', 0)}",
        f"- failed: {summary.get('parse_status_counts', {}).get('failed', 0)}",
        f"- ocr_required: {summary.get('ocr_required_count', 0)}",
        f"- zero_fact_documents: {summary.get('zero_fact_documents_count', 0)}",
        f"- facts_without_evidence: {summary.get('facts_without_evidence', 0)}",
        f"- files_profiled: {summary.get('documents_profiled', 0)}",
        f"- files_skipped_by_selection: {summary.get('documents_skipped_by_selection', 0)}",
        f"- total_size_mb_profiled: {summary.get('total_size_mb_profiled', 0)}",
        "",
        "## File Type Coverage",
        "| extension | files |",
        "|---|---:|",
    ]
    for ext, count in sorted((summary.get("files_by_extension") or {}).items()):
        lines.append(f"| {ext or '<none>'} | {count} |")
    lines.extend(["", "## Source Group Coverage", "| source group | files |", "|---|---:|"])
    for group, count in sorted((summary.get("source_group_counts") or {}).items()):
        lines.append(f"| {group or 'root'} | {count} |")
    lines.extend(["", "## High-Risk Documents"])
    high_risk = report.get("high_risk_documents") or []
    if not high_risk:
        lines.append("- none")
    for item in high_risk:
        warnings = ", ".join(item.get("warnings") or [])
        lines.append(f"- {item.get('filename')}: status={item.get('parse_status')}; warnings={warnings}")
    lines.extend(
        [
            "",
            "## Extraction Coverage",
            f"- total_chunks: {summary.get('total_chunks', 0)}",
            f"- total_raw_facts: {summary.get('total_raw_facts', 0)}",
            f"- total_canonical_facts: {summary.get('total_canonical_facts', 0)}",
            f"- zero_fact_documents_count: {summary.get('zero_fact_documents_count', 0)}",
            "",
        "## Conflicts And Gaps",
        f"- conflict_groups: {summary.get('conflict_groups', 0)}",
        f"- data_gaps: {summary.get('data_gaps', 0)}",
        "",
        "## Source Metadata Coverage",
        f"- documents_with_year: {summary.get('documents_with_year', 0)}",
        f"- documents_with_geography: {summary.get('documents_with_geography', 0)}",
        f"- documents_with_source_group: {summary.get('documents_with_source_group', 0)}",
        f"- source_type_counts: {summary.get('source_type_counts', {})}",
        f"- reliability_counts: {summary.get('reliability_counts', {})}",
        "",
        "## Parser Risk Coverage",
        f"- archive_files_count: {summary.get('archive_files_count', 0)}",
        f"- legacy_office_files_count: {summary.get('legacy_office_files_count', 0)}",
        f"- image_files_count: {summary.get('image_files_count', 0)}",
        f"- large_files_skipped_count: {summary.get('large_files_skipped_count', 0)}",
        f"- inventory_only_count: {summary.get('inventory_only_count', 0)}",
        "",
        "## Resource Profile",
            "- economy_core: LLM disabled, embeddings disabled, deterministic extraction only.",
            "- OCR is not executed by this report. Image-only PDFs are marked as ocr_required.",
        ]
    )
    return "\n".join(lines) + "\n"


def _extract_rows(chunks: Iterable[Any], pipeline: ExtractionPipeline) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    raw_facts: list[dict[str, Any]] = []
    data_gaps: list[dict[str, Any]] = []
    rejected_count = 0
    for chunk in chunks:
        bundle = pipeline.extract_from_chunk(chunk)
        rejected_count += len(bundle.rejected_items)
        raw_facts.extend(fact_rows_from_experiments(bundle_to_experiment_facts(bundle)))
        for gap in bundle_to_data_gaps(bundle):
            data_gaps.append(
                {
                    "gap_id": gap.gap_id,
                    "material": gap.material,
                    "regime": gap.regime,
                    "property": gap.property,
                    "reason": gap.reason,
                    "evidence": [item.model_dump() for item in gap.evidence],
                }
            )
    return raw_facts, data_gaps, rejected_count


def _profile_warnings(parsed: Any, quality: dict[str, Any], canonical_facts: list[dict[str, Any]], data_gaps: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    diagnostics = parsed.diagnostics or {}
    ext = Path(str(getattr(parsed, "source_name", ""))).suffix.lower()
    if diagnostics.get("scanned_pdf_detected"):
        warnings.append("ocr_required")
    if quality.get("text_density") == "empty" or (ext == ".pdf" and quality.get("text_density") in {"very_low", "low"}):
        warnings.append("low_text_density")
    if quality.get("text_chars", 0) == 0:
        warnings.append("empty_text")
    if parsed.tables and (len(parsed.tables) >= 2 or len(parsed.tables) >= max(1, len(parsed.blocks) // 2)):
        warnings.append("table_heavy_document")
    if quality.get("dirty_ocr_score", 0) >= 2:
        warnings.append("dirty_ocr_text")
    if not canonical_facts and not data_gaps and parsed.chunks:
        warnings.append("zero_facts")
    if any("OCR" in str(item) or "scanned" in str(item).lower() for item in diagnostics.get("warnings") or []):
        warnings.append("ocr_required")
    return sorted(set(warnings))


def _parse_status(parsed: Any, quality: dict[str, Any], warnings: list[str]) -> str:
    if "ocr_required" in warnings and quality.get("text_density") in {"empty", "very_low", "low"}:
        return "ocr_required"
    if not parsed.chunks or quality.get("text_density") == "empty":
        return "partial"
    if any(item in warnings for item in ["low_text_density", "dirty_ocr_text"]):
        return "partial"
    return "ok"


def _corpus_summary(profiles: list[dict[str, Any]], all_keys: list[str]) -> dict[str, Any]:
    status_counts = Counter(item.get("parse_status") for item in profiles)
    extension_counts = Counter(item.get("extension") for item in profiles)
    source_group_counts = Counter(item.get("source_group") or "root" for item in profiles)
    warning_counts = Counter(warning for item in profiles for warning in item.get("warnings") or [])
    source_type_counts = Counter(
        (item.get("source_metadata") or {}).get("source_type_detected") or "unknown"
        for item in profiles
    )
    reliability_counts = Counter(
        (item.get("source_metadata") or {}).get("reliability_level") or "unknown"
        for item in profiles
    )
    canonical_unique = len({key for key in all_keys if key})
    return {
        "documents_total": len(profiles),
        "documents_profiled": len(profiles),
        "documents_skipped_by_selection": sum(int(item.get("selection_skipped_count") or 0) for item in profiles),
        "total_size_mb_profiled": round(sum(float(item.get("file_size_mb") or 0.0) for item in profiles), 3),
        "files_by_extension": dict(sorted(extension_counts.items())),
        "source_group_counts": dict(sorted(source_group_counts.items())),
        "parse_status_counts": dict(status_counts),
        "parser_failures_count": status_counts.get("failed", 0),
        "ocr_required_count": warning_counts.get("ocr_required", 0),
        "zero_fact_documents_count": warning_counts.get("zero_facts", 0),
        "empty_text_documents_count": warning_counts.get("empty_text", 0),
        "table_heavy_documents_count": warning_counts.get("table_heavy_document", 0),
        "dirty_ocr_documents_count": warning_counts.get("dirty_ocr_text", 0),
        "total_chunks": sum(int(item.get("chunks_count") or 0) for item in profiles),
        "total_raw_facts": sum(int(item.get("facts_extracted") or 0) for item in profiles),
        "total_canonical_facts": canonical_unique,
        "per_file_canonical_facts_sum": sum(int(item.get("canonical_facts") or 0) for item in profiles),
        "facts_without_evidence": sum(int(item.get("facts_without_evidence") or 0) for item in profiles),
        "conflict_groups": sum(int(item.get("conflict_groups") or 0) for item in profiles),
        "data_gaps": sum(int(item.get("data_gaps") or 0) for item in profiles),
        "documents_with_year": sum(1 for item in profiles if (item.get("source_metadata") or {}).get("publication_year")),
        "documents_with_geography": sum(1 for item in profiles if (item.get("source_metadata") or {}).get("geographies")),
        "documents_with_source_group": sum(1 for item in profiles if item.get("source_group")),
        "source_type_counts": dict(source_type_counts),
        "reliability_counts": dict(reliability_counts),
        "archive_files_count": warning_counts.get("archive_needs_extraction", 0),
        "legacy_office_files_count": warning_counts.get("legacy_format_needs_conversion", 0),
        "image_files_count": warning_counts.get("image_ocr_required", 0),
        "large_files_skipped_count": warning_counts.get("parse_skipped_large_file", 0),
        "inventory_only_count": warning_counts.get("inventory_only_parse_not_run", 0),
        "economy_core_compatible": True,
        "warning_counts": dict(warning_counts),
    }


def _high_risk_documents(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risk_warnings = {"parser_failed", "ocr_required", "zero_facts", "empty_text", "dirty_ocr_text", "low_text_density"}
    result = []
    for item in profiles:
        warnings = set(item.get("warnings") or [])
        if item.get("parse_status") in {"failed", "ocr_required", "unsupported"} or warnings.intersection(risk_warnings):
            result.append(
                {
                    "path": item.get("path"),
                    "filename": item.get("filename"),
                    "extension": item.get("extension"),
                    "parse_status": item.get("parse_status"),
                    "warnings": item.get("warnings") or [],
                    "text_chars": item.get("text_chars", 0),
                    "chunks_count": item.get("chunks_count", 0),
                    "canonical_facts": item.get("canonical_facts", 0),
                }
            )
    return result


def _base_profile(path: Path, *, root: str | Path | None = None) -> dict[str, Any]:
    source_group = _source_group(path, root=root)
    return {
        "path": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "source_type": "file",
        "source_group": source_group,
        "file_size_bytes": _file_size(path),
        "file_size_mb": _file_size_mb(path),
        "parser_backend": None,
        "parse_status": "unknown",
        "text_chars": 0,
        "text_density": "empty",
        "pages_estimated": 0,
        "tables_detected": 0,
        "images_detected": 0,
        "chunks_count": 0,
        "facts_extracted": 0,
        "canonical_facts": 0,
        "facts_without_evidence": 0,
        "conflict_groups": 0,
        "data_gaps": 0,
        "warnings": [],
    }


def _pages_estimated(parsed: Any) -> int:
    diagnostics = parsed.diagnostics or {}
    if diagnostics.get("scanned_pdf_page_count"):
        return max(1, int(diagnostics["scanned_pdf_page_count"]))
    pages = [int(chunk.page_end or chunk.page_start or 1) for chunk in parsed.chunks or [] if chunk.page_end or chunk.page_start]
    return max(pages or [1])


def _public_parser_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in (diagnostics or {}).items()
        if "password" not in key.lower() and "key" not in key.lower() and "authorization" not in key.lower()
    }


def _fact_preview(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "material": row.get("material"),
            "regime": row.get("regime"),
            "property": row.get("property"),
            "value_normalized": row.get("value_normalized"),
            "unit_normalized": row.get("unit_normalized"),
            "evidence_count": len(row.get("evidence") or []),
        }
        for row in rows[:limit]
    ]


def _gap_preview(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "material": row.get("material"),
            "regime": row.get("regime"),
            "property": row.get("property"),
            "reason": row.get("reason"),
            "evidence_count": len(row.get("evidence") or []),
        }
        for row in rows[:limit]
    ]


def _stable_profile_doc_id(path: Path) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    except Exception:
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
    return f"profile_doc_{digest}"


def _corpus_files(
    root: Path,
    *,
    max_files: int | None = None,
    sample_per_group: int | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    if root.is_file():
        return [root], {"total_files_found": 1, "selected_files": 1, "max_files": max_files, "sample_per_group": sample_per_group}
    all_files = sorted(path for path in root.rglob("*") if path.is_file() and not path.name.startswith("."))
    selected = all_files
    if sample_per_group and sample_per_group > 0:
        grouped: dict[str, list[Path]] = {}
        for path in all_files:
            grouped.setdefault(_source_group(path, root=root) or "root", []).append(path)
        selected = []
        for group_paths in grouped.values():
            selected.extend(group_paths[:sample_per_group])
        selected = sorted(selected)
    if max_files and max_files > 0:
        selected = selected[:max_files]
    return selected, {
        "total_files_found": len(all_files),
        "selected_files": len(selected),
        "skipped_files": max(0, len(all_files) - len(selected)),
        "max_files": max_files,
        "sample_per_group": sample_per_group,
    }


def _safe_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:500]


def _profile_mode(value: str) -> str:
    lowered = str(value or "auto").strip().lower()
    return lowered if lowered in {"auto", "full", "inventory"} else "auto"


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _file_size_mb(path: Path) -> float:
    return round(_file_size(path) / (1024 * 1024), 3)


def _source_group(path: Path, *, root: str | Path | None = None) -> str | None:
    try:
        if root:
            relative = path.resolve().relative_to(Path(root).resolve())
            return relative.parts[0] if len(relative.parts) > 1 else None
        parts = path.parts
        if "data_storage" in parts:
            idx = parts.index("data_storage")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception:
        return None
    return None


def _unsupported_profile(base: dict[str, Any], path: Path) -> dict[str, Any] | None:
    ext = path.suffix.lower()
    if ext in ARCHIVE_EXTENSIONS:
        return {
            **base,
            "parse_status": "unsupported",
            "warnings": ["unsupported_format", "archive_needs_extraction"],
            "parser_error": f"Archive format requires controlled extraction before parsing: {ext}",
            "source_metadata": _filename_source_metadata(path, base),
        }
    if ext in LEGACY_OFFICE_EXTENSIONS:
        return {
            **base,
            "parse_status": "unsupported",
            "warnings": ["unsupported_format", "legacy_format_needs_conversion"],
            "parser_error": f"Legacy Office format requires conversion to docx/xlsx/pptx or parser adapter: {ext}",
            "source_metadata": _filename_source_metadata(path, base),
        }
    if ext in IMAGE_EXTENSIONS:
        return {
            **base,
            "parse_status": "ocr_required",
            "warnings": ["unsupported_format", "image_ocr_required", "ocr_required"],
            "parser_error": f"Image file requires OCR before extraction: {ext}",
            "source_metadata": _filename_source_metadata(path, base),
        }
    if ext not in SUPPORTED_FILE_EXTENSIONS:
        return {
            **base,
            "parse_status": "unsupported",
            "warnings": ["unsupported_format"],
            "parser_error": f"Unsupported file extension: {ext or '<none>'}",
            "source_metadata": _filename_source_metadata(path, base),
        }
    return None


def _inventory_profile(
    base: dict[str, Any],
    path: Path,
    *,
    warnings: list[str],
    max_parse_mb: float | None = None,
) -> dict[str, Any]:
    parser_error = None
    if "parse_skipped_large_file" in warnings:
        parser_error = f"Full parser skipped because file_size_mb={base.get('file_size_mb')} exceeds max_parse_mb={max_parse_mb}."
    return {
        **base,
        "parser_backend": "inventory",
        "parse_status": "partial",
        "warnings": sorted(set(warnings)),
        "parser_error": parser_error,
        "source_metadata": _filename_source_metadata(path, base),
        "parser_diagnostics": {
            "inventory_only": True,
            "full_parse_not_run": True,
            "reason": ",".join(sorted(set(warnings))),
        },
    }


def _filename_source_metadata(path: Path, base: dict[str, Any]) -> dict[str, Any]:
    metadata = infer_source_metadata(
        source_name=path.name,
        source_type="file",
        text=" ".join(str(part) for part in path.parts[-5:]),
        diagnostics={"source_group": base.get("source_group")},
    )["source_metadata"]
    group = base.get("source_group")
    if group and metadata.get("source_type_detected") in {None, "unknown"}:
        metadata["source_type_detected"] = _source_type_from_group(str(group), path.suffix.lower())
        metadata["type_basis"] = "data_storage_group"
    return metadata


def _source_type_from_group(group: str, ext: str) -> str:
    lowered = group.lower()
    if "доклад" in lowered or ext == ".pptx":
        return "presentation"
    if "журнал" in lowered or "стать" in lowered:
        return "publication"
    if "конференц" in lowered:
        return "conference_material"
    if "обзор" in lowered:
        return "review"
    return "unknown"

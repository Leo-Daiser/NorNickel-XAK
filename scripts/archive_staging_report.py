"""Inspect and optionally stage ZIP archive contents safely.

The script does not process RAR or multipart archives directly. Those are
reported as requiring an external controlled extractor. ZIP extraction is
opt-in via --extract-zip and protected against path traversal and size limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

os.environ.setdefault("RUNTIME_PROFILE", "economy_core")
os.environ.setdefault("ENABLE_LLM", "false")
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("ENABLE_LOCAL_EMBEDDINGS", "false")
os.environ.setdefault("RETRIEVAL_MODE", "bm25")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parsing.text_quality import SUPPORTED_FILE_EXTENSIONS  # noqa: E402
from scripts.batch_ingest_corpus import build_ingest_plan  # noqa: E402


DEFAULT_JSON = ROOT / "artifacts" / "archive_staging_report.json"
DEFAULT_MD = ROOT / "artifacts" / "archive_staging_report.md"
DEFAULT_STAGING = ROOT / "artifacts" / "archive_staging"


def build_archive_staging_report(
    input_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_STAGING,
    max_file_mb: float = 25.0,
    max_archive_mb: float = 250.0,
    max_total_uncompressed_mb: float = 500.0,
    max_members: int = 500,
    extract_zip: bool = False,
    max_files: int | None = None,
    sample_per_group: int | None = None,
) -> dict[str, Any]:
    plan, selection = build_ingest_plan(
        input_path,
        max_file_mb=max_file_mb,
        max_files=max_files,
        sample_per_group=sample_per_group,
    )
    archives = [item for item in plan if item.planned_reason == "archive_needs_extraction"]
    rows = [
        inspect_archive(
            item.path,
            relative_path=item.relative_path,
            source_group=item.source_group,
            output_dir=Path(output_dir),
            max_archive_mb=max_archive_mb,
            max_total_uncompressed_mb=max_total_uncompressed_mb,
            max_members=max_members,
            extract_zip=extract_zip,
        )
        for item in archives
    ]
    summary = summarize(rows, selection=selection, extract_zip=extract_zip)
    return {
        "status": "ok",
        "input": str(input_path),
        "output_dir": str(output_dir),
        "summary": summary,
        "rows": rows,
        "resource_profile": {
            "economy_core_compatible": True,
            "llm_required": False,
            "embeddings_required": False,
            "zip_extraction_executed": bool(extract_zip),
            "rar_extraction_executed": False,
            "note": "RAR and multipart archives are inventory-only and require external controlled extraction.",
        },
    }


def inspect_archive(
    path: str | Path,
    *,
    relative_path: str | None = None,
    source_group: str | None = None,
    output_dir: Path = DEFAULT_STAGING,
    max_archive_mb: float = 250.0,
    max_total_uncompressed_mb: float = 500.0,
    max_members: int = 500,
    extract_zip: bool = False,
) -> dict[str, Any]:
    archive_path = Path(path)
    extension = archive_path.suffix.lower()
    size_mb = round(archive_path.stat().st_size / (1024 * 1024), 3)
    base = {
        "path": relative_path or str(archive_path),
        "extension": extension,
        "source_group": source_group,
        "file_size_mb": size_mb,
        "archive_id": archive_id(archive_path),
        "extracted": False,
        "extracted_files": [],
        "warnings": [],
    }
    if extension != ".zip":
        return {
            **base,
            "archive_status": "external_extractor_required",
            "reason": "rar_or_multipart_archive",
            "recommended_action": "extract_with_controlled_external_tool",
        }
    if size_mb > max_archive_mb:
        return {
            **base,
            "archive_status": "skipped_large_archive",
            "reason": "archive_file_too_large",
            "max_archive_mb": max_archive_mb,
            "recommended_action": "inspect_large_archive_separately",
        }
    if not zipfile.is_zipfile(archive_path):
        return {
            **base,
            "archive_status": "invalid_archive",
            "reason": "not_a_valid_zip",
            "recommended_action": "manual_source_review",
        }
    return inspect_zip(
        archive_path,
        base=base,
        output_dir=output_dir,
        max_total_uncompressed_mb=max_total_uncompressed_mb,
        max_members=max_members,
        extract_zip=extract_zip,
    )


def inspect_zip(
    archive_path: Path,
    *,
    base: dict[str, Any],
    output_dir: Path,
    max_total_uncompressed_mb: float,
    max_members: int,
    extract_zip: bool,
) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        members = [info for info in infos if not info.is_dir()]
        unsafe = [info.filename for info in members if not safe_zip_member(info.filename)]
        encrypted = [info.filename for info in members if info.flag_bits & 0x1]
        total_uncompressed = sum(int(info.file_size or 0) for info in members)
        total_uncompressed_mb = round(total_uncompressed / (1024 * 1024), 3)
        supported = [info for info in members if Path(PurePosixPath(normalize_member_name(info.filename)).name).suffix.lower() in SUPPORTED_FILE_EXTENSIONS]
        row = {
            **base,
            "archive_status": "zip_inventory_ok",
            "members_count": len(members),
            "supported_members_count": len(supported),
            "unsafe_members_count": len(unsafe),
            "encrypted_members_count": len(encrypted),
            "total_uncompressed_mb": total_uncompressed_mb,
            "supported_member_preview": [info.filename for info in supported[:20]],
            "unsafe_member_preview": unsafe[:20],
            "recommended_action": "extract_zip_to_staging" if supported else "manual_source_review",
        }
        warnings: list[str] = []
        if unsafe:
            warnings.append("unsafe_zip_members")
        if encrypted:
            warnings.append("encrypted_zip_members")
        if len(members) > max_members:
            warnings.append("too_many_zip_members")
        if total_uncompressed_mb > max_total_uncompressed_mb:
            warnings.append("zip_uncompressed_size_too_large")
        row["warnings"] = warnings
        if not extract_zip:
            return row
        if warnings:
            return {**row, "archive_status": "zip_extraction_blocked", "reason": ",".join(warnings)}
        extracted = extract_supported_zip_members(archive, supported, output_dir / str(base["archive_id"]))
        return {
            **row,
            "archive_status": "zip_extracted",
            "extracted": True,
            "extracted_files": extracted,
            "extracted_files_count": len(extracted),
        }


def extract_supported_zip_members(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo], target_root: Path) -> list[dict[str, Any]]:
    target_root.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    resolved_root = target_root.resolve()
    for info in members:
        member_name = normalize_member_name(info.filename)
        destination = target_root / member_name
        resolved_destination = destination.resolve()
        if not str(resolved_destination).startswith(str(resolved_root)):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, destination.open("wb") as target:
            target.write(source.read())
        extracted.append(
            {
                "archive_member": info.filename,
                "staged_path": str(destination),
                "file_size_bytes": int(info.file_size or 0),
                "provenance_source": str(archive.filename),
            }
        )
    return extracted


def summarize(rows: list[dict[str, Any]], *, selection: dict[str, Any], extract_zip: bool) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("archive_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        for warning in row.get("warnings") or []:
            warning_counts[str(warning)] = warning_counts.get(str(warning), 0) + 1
    return {
        "documents_total_found": selection.get("total_files_found"),
        "archives_selected": len(rows),
        "extract_zip": bool(extract_zip),
        "archive_status_counts": dict(sorted(status_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "zip_supported_members_count": sum(int(row.get("supported_members_count") or 0) for row in rows),
        "zip_extracted_files_count": sum(int(row.get("extracted_files_count") or 0) for row in rows),
        "external_extractor_required_count": status_counts.get("external_extractor_required", 0),
    }


def archive_id(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}"
    return f"archive_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def normalize_member_name(name: str) -> str:
    return str(PurePosixPath(str(name).replace("\\", "/")))


def safe_zip_member(name: str) -> bool:
    normalized = normalize_member_name(name)
    if not normalized or normalized.startswith("/"):
        return False
    if ":" in normalized:
        return False
    parts = PurePosixPath(normalized).parts
    return ".." not in parts


def write_report(report: dict[str, Any], *, json_path: str | Path, markdown_path: str | Path | None = None) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_path:
        md = Path(markdown_path)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Archive Staging Report",
        "",
        "## Summary",
        f"- documents_total_found: {summary.get('documents_total_found')}",
        f"- archives_selected: {summary.get('archives_selected')}",
        f"- extract_zip: {summary.get('extract_zip')}",
        f"- zip_supported_members_count: {summary.get('zip_supported_members_count')}",
        f"- zip_extracted_files_count: {summary.get('zip_extracted_files_count')}",
        f"- external_extractor_required_count: {summary.get('external_extractor_required_count')}",
        "",
        "## Archive Status",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in (summary.get("archive_status_counts") or {}).items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Warnings", "| warning | count |", "|---|---:|"])
    warnings = summary.get("warning_counts") or {}
    if warnings:
        for warning, count in warnings.items():
            lines.append(f"| {warning} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(["", "## Notes", "- ZIP extraction is opt-in via `--extract-zip`.", "- RAR and multipart archives require controlled external extraction.", "- Staged files keep archive/member provenance in JSON."])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and optionally stage safe ZIP archive contents.")
    parser.add_argument("--input", default="data_storage", help="Input corpus directory or file.")
    parser.add_argument("--output", default=str(DEFAULT_JSON), help="Output JSON report path.")
    parser.add_argument("--markdown", default=str(DEFAULT_MD), help="Output Markdown report path.")
    parser.add_argument("--staging-dir", default=str(DEFAULT_STAGING), help="Directory for optional ZIP extraction.")
    parser.add_argument("--max-file-mb", type=float, default=25.0, help="Direct-ingest size threshold used to identify archive backlog.")
    parser.add_argument("--max-archive-mb", type=float, default=250.0, help="Skip ZIP extraction for archives larger than this.")
    parser.add_argument("--max-total-uncompressed-mb", type=float, default=500.0, help="Block ZIP extraction above this uncompressed size.")
    parser.add_argument("--max-members", type=int, default=500, help="Block ZIP extraction above this member count.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional selected file cap.")
    parser.add_argument("--sample-per-group", type=int, default=None, help="Optional sample cap per top-level source group.")
    parser.add_argument("--extract-zip", action="store_true", help="Actually extract safe supported ZIP members to --staging-dir.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_archive_staging_report(
        args.input,
        output_dir=args.staging_dir,
        max_file_mb=args.max_file_mb,
        max_archive_mb=args.max_archive_mb,
        max_total_uncompressed_mb=args.max_total_uncompressed_mb,
        max_members=args.max_members,
        extract_zip=args.extract_zip,
        max_files=args.max_files,
        sample_per_group=args.sample_per_group,
    )
    write_report(report, json_path=args.output, markdown_path=args.markdown)
    summary = report["summary"]
    print("Archive Staging Report")
    print(f"archives_selected: {summary.get('archives_selected')}")
    print(f"archive_status_counts: {json.dumps(summary.get('archive_status_counts') or {}, ensure_ascii=False)}")
    print(f"zip_supported_members_count: {summary.get('zip_supported_members_count')}")
    print(f"zip_extracted_files_count: {summary.get('zip_extracted_files_count')}")
    print(f"external_extractor_required_count: {summary.get('external_extractor_required_count')}")
    print(f"json_report: {args.output}")
    print(f"markdown_report: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

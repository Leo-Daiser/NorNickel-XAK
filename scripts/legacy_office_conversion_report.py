"""Plan and optionally run legacy Office conversion into ignored staging.

The script targets old `.doc`, `.xls`, `.docm`, `.ppt` files. Conversion is
opt-in via --convert and requires LibreOffice/soffice. It preserves provenance
by writing a manifest that links every staged output to the original source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("RUNTIME_PROFILE", "economy_core")
os.environ.setdefault("ENABLE_LLM", "false")
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("ENABLE_LOCAL_EMBEDDINGS", "false")
os.environ.setdefault("RETRIEVAL_MODE", "bm25")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.batch_ingest_corpus import PlannedFile, build_ingest_plan  # noqa: E402


DEFAULT_JSON = ROOT / "artifacts" / "legacy_office_conversion_report.json"
DEFAULT_MD = ROOT / "artifacts" / "legacy_office_conversion_report.md"
DEFAULT_STAGING = ROOT / "artifacts" / "legacy_office_staging"


CONVERSION_TARGETS = {
    ".doc": ("docx", ".docx"),
    ".docm": ("docx", ".docx"),
    ".xls": ("xlsx", ".xlsx"),
    ".ppt": ("pptx", ".pptx"),
}


def build_legacy_conversion_report(
    input_path: str | Path,
    *,
    staging_dir: str | Path = DEFAULT_STAGING,
    soffice_path: str | None = None,
    max_file_mb: float = 25.0,
    convert: bool = False,
    timeout: int = 180,
    max_files: int | None = None,
    sample_per_group: int | None = None,
) -> dict[str, Any]:
    plan, selection = build_ingest_plan(
        input_path,
        max_file_mb=max_file_mb,
        max_files=max_files,
        sample_per_group=sample_per_group,
    )
    legacy = [item for item in plan if item.planned_reason == "legacy_format_needs_conversion"]
    detected = find_soffice(soffice_path)
    rows = [
        process_legacy_file(
            item,
            staging_dir=Path(staging_dir),
            soffice_path=detected,
            convert=convert,
            timeout=timeout,
        )
        for item in legacy
    ]
    summary = summarize(rows, selection=selection, convert=convert, soffice_path=detected)
    return {
        "status": "ok",
        "input": str(input_path),
        "staging_dir": str(staging_dir),
        "summary": summary,
        "rows": rows,
        "resource_profile": {
            "economy_core_compatible": True,
            "llm_required": False,
            "embeddings_required": False,
            "conversion_executed": bool(convert and detected),
            "conversion_tool": detected,
            "note": "Conversion is optional and provenance-preserving; original files remain source of truth.",
        },
    }


def process_legacy_file(
    item: PlannedFile,
    *,
    staging_dir: Path,
    soffice_path: str | None,
    convert: bool,
    timeout: int,
) -> dict[str, Any]:
    convert_to, target_ext = conversion_target(item.extension)
    target_dir = staging_dir / legacy_id(item.path)
    target_path = target_dir / f"{item.path.stem}{target_ext}"
    base = {
        "path": item.relative_path,
        "extension": item.extension,
        "source_group": item.source_group,
        "file_size_mb": item.file_size_mb,
        "conversion_id": legacy_id(item.path),
        "target_format": convert_to,
        "target_extension": target_ext,
        "staged_path": str(target_path),
        "provenance_source": str(item.path),
        "provenance_policy": "Keep original file path/content hash as provenance; converted file is a derived parsing artifact.",
    }
    if soffice_path is None:
        return {
            **base,
            "conversion_status": "conversion_tool_missing",
            "recommended_action": "install_or_configure_libreoffice_soffice",
        }
    if not convert:
        return {
            **base,
            "conversion_status": "planned",
            "soffice_path": soffice_path,
            "command": conversion_command(soffice_path, item.path, target_dir, convert_to),
        }
    target_dir.mkdir(parents=True, exist_ok=True)
    command = conversion_command(soffice_path, item.path, target_dir, convert_to)
    result = run_conversion(command, timeout=timeout)
    converted = target_path.exists()
    status = "converted" if result["returncode"] == 0 and converted else "conversion_failed"
    return {
        **base,
        "conversion_status": status,
        "soffice_path": soffice_path,
        "command": command,
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "converted_exists": converted,
    }


def run_conversion(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "returncode": completed.returncode,
            "stdout": safe_output(completed.stdout),
            "stderr": safe_output(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout": safe_output(exc.stdout or ""),
            "stderr": "conversion_timeout",
        }


def summarize(rows: list[dict[str, Any]], *, selection: dict[str, Any], convert: bool, soffice_path: str | None) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("conversion_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        group = str(row.get("source_group") or "root")
        group_counts[group] = group_counts.get(group, 0) + 1
    return {
        "documents_total_found": selection.get("total_files_found"),
        "legacy_files_selected": len(rows),
        "convert_requested": bool(convert),
        "soffice_available": bool(soffice_path),
        "soffice_path": soffice_path,
        "conversion_status_counts": dict(sorted(status_counts.items())),
        "source_group_counts": dict(sorted(group_counts.items())),
        "converted_count": status_counts.get("converted", 0),
        "tool_missing_count": status_counts.get("conversion_tool_missing", 0),
        "conversion_failed_count": status_counts.get("conversion_failed", 0),
    }


def find_soffice(explicit: str | None = None) -> str | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    for name in ["soffice", "libreoffice"]:
        found = shutil.which(name)
        if found:
            candidates.append(found)
    candidates.extend(
        [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return None


def conversion_target(extension: str) -> tuple[str, str]:
    lowered = str(extension or "").lower()
    if lowered not in CONVERSION_TARGETS:
        raise ValueError(f"Unsupported legacy Office extension for conversion: {extension}")
    return CONVERSION_TARGETS[lowered]


def conversion_command(soffice_path: str, source: Path, target_dir: Path, convert_to: str) -> list[str]:
    return [
        soffice_path,
        "--headless",
        "--convert-to",
        convert_to,
        "--outdir",
        str(target_dir),
        str(source),
    ]


def legacy_id(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}"
    return f"legacy_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def safe_output(value: Any) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
    return text[:1000]


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
        "# Legacy Office Conversion Report",
        "",
        "## Summary",
        f"- documents_total_found: {summary.get('documents_total_found')}",
        f"- legacy_files_selected: {summary.get('legacy_files_selected')}",
        f"- convert_requested: {summary.get('convert_requested')}",
        f"- soffice_available: {summary.get('soffice_available')}",
        f"- converted_count: {summary.get('converted_count')}",
        f"- tool_missing_count: {summary.get('tool_missing_count')}",
        f"- conversion_failed_count: {summary.get('conversion_failed_count')}",
        "",
        "## Conversion Status",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in (summary.get("conversion_status_counts") or {}).items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Source Groups", "| group | count |", "|---|---:|"])
    for group, count in (summary.get("source_group_counts") or {}).items():
        lines.append(f"| {group} | {count} |")
    lines.extend(
        [
            "",
            "## Notes",
            "- Conversion is opt-in via `--convert`.",
            "- The original legacy file remains the provenance source.",
            "- Converted files are staging artifacts and must not be committed.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan and optionally run legacy Office conversion with provenance.")
    parser.add_argument("--input", default="data_storage", help="Input corpus directory or file.")
    parser.add_argument("--output", default=str(DEFAULT_JSON), help="Output JSON report path.")
    parser.add_argument("--markdown", default=str(DEFAULT_MD), help="Output Markdown report path.")
    parser.add_argument("--staging-dir", default=str(DEFAULT_STAGING), help="Directory for converted files.")
    parser.add_argument("--soffice-path", default=None, help="Explicit path to LibreOffice soffice executable.")
    parser.add_argument("--max-file-mb", type=float, default=25.0, help="Direct-ingest size threshold used by the shared plan.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-file conversion timeout in seconds.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional selected file cap.")
    parser.add_argument("--sample-per-group", type=int, default=None, help="Optional sample cap per top-level source group.")
    parser.add_argument("--convert", action="store_true", help="Actually run LibreOffice conversion.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_legacy_conversion_report(
        args.input,
        staging_dir=args.staging_dir,
        soffice_path=args.soffice_path,
        max_file_mb=args.max_file_mb,
        convert=args.convert,
        timeout=args.timeout,
        max_files=args.max_files,
        sample_per_group=args.sample_per_group,
    )
    write_report(report, json_path=args.output, markdown_path=args.markdown)
    summary = report["summary"]
    print("Legacy Office Conversion Report")
    print(f"legacy_files_selected: {summary.get('legacy_files_selected')}")
    print(f"convert_requested: {summary.get('convert_requested')}")
    print(f"soffice_available: {summary.get('soffice_available')}")
    print(f"conversion_status_counts: {json.dumps(summary.get('conversion_status_counts') or {}, ensure_ascii=False)}")
    print(f"json_report: {args.output}")
    print(f"markdown_report: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

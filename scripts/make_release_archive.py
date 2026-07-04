from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
RELEASE_DIR = DIST_DIR / "release_unpacked"
ZIP_PATH = DIST_DIR / "hackathon_project_release.zip"

INCLUDE_DIRS = ["app", "scripts", "evaluation", "tests", "demo_data", "docs", "multipart"]
INCLUDE_FILES = [
    "README.md",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "pytest.ini",
    "sitecustomize.py",
    ".env.example",
    ".gitignore",
    ".dockerignore",
]

EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "data",
    "dist",
    "volumes",
}
EXCLUDE_FILE_PATTERNS = [
    ".env",
    "*.sqlite3",
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.jsonl",
]


def remove_runtime_artifacts(root: Path) -> None:
    """Remove bytecode/cache files that Python may create while checking release output."""

    if not root.exists():
        return
    for directory in root.rglob("__pycache__"):
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
    for pattern in ("*.pyc", "*.pyo"):
        for file_path in root.rglob(pattern):
            if file_path.is_file():
                file_path.unlink(missing_ok=True)


def should_exclude(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIR_NAMES:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDE_FILE_PATTERNS)


def copy_tree_filtered(source: Path, target: Path) -> None:
    for item in source.rglob("*"):
        rel = item.relative_to(source)
        if should_exclude(rel):
            continue
        destination = target / rel
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)


def build_release() -> dict[str, object]:
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    included_top_level: list[str] = []
    for dirname in INCLUDE_DIRS:
        source = ROOT / dirname
        if not source.exists():
            continue
        copy_tree_filtered(source, RELEASE_DIR / dirname)
        included_top_level.append(dirname)
    for filename in INCLUDE_FILES:
        source = ROOT / filename
        if not source.exists() or should_exclude(Path(filename)):
            continue
        shutil.copy2(source, RELEASE_DIR / filename)
        included_top_level.append(filename)

    remove_runtime_artifacts(RELEASE_DIR)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in RELEASE_DIR.rglob("*"):
            if item.is_file():
                archive.write(item, item.relative_to(RELEASE_DIR))

    check_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_release_package.py"), "--path", str(RELEASE_DIR)],
        cwd=ROOT,
        env=check_env,
        text=True,
        capture_output=True,
        check=False,
    )
    remove_runtime_artifacts(RELEASE_DIR)
    if check.returncode != 0:
        print(check.stdout)
        print(check.stderr, file=sys.stderr)
        raise SystemExit(check.returncode)

    files_count = sum(1 for item in RELEASE_DIR.rglob("*") if item.is_file())
    return {
        "release_dir": str(RELEASE_DIR),
        "zip_path": str(ZIP_PATH),
        "files_count": files_count,
        "zip_size": ZIP_PATH.stat().st_size,
        "forbidden_files_count": 0,
        "included_top_level": sorted(set(included_top_level)),
    }


def main() -> int:
    summary = build_release()
    print("RELEASE ARCHIVE CREATED")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

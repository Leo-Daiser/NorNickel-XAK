from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path


FORBIDDEN_PATTERNS = [
    ".env",
    "*.sqlite3",
    "data/extraction_audit/*",
    "data/parser_audit/*",
    "data/parser_audit*/*",
    "__pycache__/",
    "*.pyc",
    "volumes/",
    "*.log",
]

SKIP_DIRS = {".git", ".venv", "node_modules", ".pytest_cache"}


def _is_forbidden(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    name = path.name
    if name == ".env":
        return True
    if path.is_dir() and name in {"__pycache__", "volumes"}:
        return True
    if path.is_file() and any(fnmatch.fnmatch(name, pattern) for pattern in ["*.sqlite3", "*.pyc", "*.log"]):
        return True
    return any(fnmatch.fnmatch(rel, pattern) for pattern in FORBIDDEN_PATTERNS)


def find_forbidden(root: Path) -> list[str]:
    forbidden: list[str] = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if _is_forbidden(path, root):
            forbidden.append(path.relative_to(root).as_posix() + ("/" if path.is_dir() else ""))
    return sorted(set(forbidden))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check release package for runtime/secrets artifacts.")
    parser.add_argument("--path", default=".", help="Path to release directory/archive unpacked root. Default: current directory.")
    args = parser.parse_args()
    root = Path(args.path).resolve()
    if not root.exists():
        print(f"Release path does not exist: {root}", file=sys.stderr)
        return 2
    forbidden = find_forbidden(root)
    if forbidden:
        print("FORBIDDEN RELEASE FILES FOUND:")
        for item in forbidden:
            print(f"- {item}")
        print("\nDo not include these files in the hackathon archive.")
        return 1
    print("RELEASE PACKAGE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

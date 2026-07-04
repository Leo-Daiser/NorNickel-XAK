"""Project self-check script.

Runs fast checks that do not require Neo4j, Qdrant or heavy parsing models.
"""

from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print("[1/2] compileall")
    if not compileall.compile_dir(str(root), quiet=1):
        return 1
    print("[2/2] smoke_test")
    proc = subprocess.run([sys.executable, "-m", "tests.smoke_test"], cwd=root)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

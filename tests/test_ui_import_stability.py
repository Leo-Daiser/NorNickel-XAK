from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_streamlit_ui_imports_when_run_as_file() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import runpy; runpy.run_path('app/ui.py', run_name='ui_import_test')"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

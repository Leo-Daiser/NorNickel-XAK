from __future__ import annotations

import zipfile
from pathlib import Path


def test_make_release_archive_excludes_runtime_artifacts() -> None:
    from scripts.make_release_archive import RELEASE_DIR, ZIP_PATH, build_release
    from scripts.check_release_package import find_forbidden

    summary = build_release()
    assert Path(summary["zip_path"]).exists()
    assert ZIP_PATH.exists()
    assert RELEASE_DIR.exists()
    assert not find_forbidden(RELEASE_DIR)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = set(archive.namelist())
    assert "app/api.py" in names
    assert "evaluation/test_corpus/clean/vt6_annealing_strength.txt" in names
    assert ".env" not in names
    assert all(not name.startswith("data/") for name in names)
    assert all(not name.startswith("data_storage/") for name in names)
    assert all("__pycache__" not in name for name in names)

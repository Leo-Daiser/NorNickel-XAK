from __future__ import annotations

from pathlib import Path

import app.ui as ui


ROOT = Path(__file__).resolve().parents[1]


class DummyUpload:
    def __init__(self, name: str, size: int) -> None:
        self.name = name
        self.size = size
        self.type = "text/plain"

    def getvalue(self) -> bytes:
        return b"x" * self.size


def test_uploaded_files_stats_blocks_too_many_files(monkeypatch) -> None:
    monkeypatch.setattr(ui, "UI_UPLOAD_MAX_FILES", 2)
    monkeypatch.setattr(ui, "UI_UPLOAD_MAX_TOTAL_MB", 100.0)

    stats = ui.uploaded_files_stats([DummyUpload("a.txt", 10), DummyUpload("b.txt", 10), DummyUpload("c.txt", 10)])

    assert stats["blocked"] is True
    assert stats["files_count"] == 3
    assert any(str(reason).startswith("too_many_files") for reason in stats["reasons"])
    assert "CLI batch ingest" in ui.upload_guidance_message(stats)


def test_uploaded_files_stats_blocks_too_large_total(monkeypatch) -> None:
    monkeypatch.setattr(ui, "UI_UPLOAD_MAX_FILES", 10)
    monkeypatch.setattr(ui, "UI_UPLOAD_MAX_TOTAL_MB", 0.001)

    stats = ui.uploaded_files_stats([DummyUpload("large.txt", 4096)])

    assert stats["blocked"] is True
    assert stats["largest_file"] == "large.txt"
    assert any(str(reason).startswith("too_large_total_mb") for reason in stats["reasons"])


def test_streamlit_ui_contains_batch_ingest_guidance_for_large_uploads() -> None:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")

    assert "UI_UPLOAD_MAX_FILES" in ui_text
    assert "UI_UPLOAD_MAX_TOTAL_MB" in ui_text
    assert "Выбран слишком большой batch для Streamlit upload" in ui_text
    assert "scripts/batch_ingest_corpus.py --input data_storage" in ui_text
    assert "disabled=upload_blocked" in ui_text


def test_streamlit_ui_uses_explicit_apply_for_document_activity() -> None:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")

    assert "documents_active_form" in ui_text
    assert "Применить изменения" in ui_text
    assert 'api_post(\n                            "/documents/active"' in ui_text
    assert "Обновить граф по активным документам" in ui_text

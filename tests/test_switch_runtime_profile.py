from __future__ import annotations

from pathlib import Path

from scripts.switch_runtime_profile import switch_profile


def test_switch_runtime_profile_dry_run_does_not_overwrite_env(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    original = "RUNTIME_PROFILE=quality_full\nMISTRAL_API_KEY=secret-value\n"
    env_path.write_text(original, encoding="utf-8")

    result = switch_profile("economy", write=False, backup=False, env_path=env_path)

    assert result["written"] is False
    assert env_path.read_text(encoding="utf-8") == original
    assert "secret-value" not in str(result["diff"])
    assert "[redacted]" in str(result["diff"])


def test_switch_runtime_profile_write_creates_backup_and_preserves_api_key(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("RUNTIME_PROFILE=quality_full\nMISTRAL_API_KEY=secret-value\n", encoding="utf-8")

    result = switch_profile("balanced", write=True, backup=True, env_path=env_path)

    assert result["written"] is True
    assert result["backup_path"]
    assert Path(str(result["backup_path"])).exists()
    written = env_path.read_text(encoding="utf-8")
    assert "RUNTIME_PROFILE=balanced_hybrid" in written
    assert "MISTRAL_API_KEY=secret-value" in written
    assert "secret-value" not in str(result["diff"])

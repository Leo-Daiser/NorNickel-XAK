from __future__ import annotations

import argparse
import difflib
import shutil
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

PROFILE_TEMPLATES = {
    "economy": ROOT / ".env.economy.example",
    "balanced": ROOT / ".env.balanced.example",
    "quality": ROOT / ".env.quality.example",
}

PRESERVE_KEYS = {
    "API_BASE",
    "NEO4J_URI",
    "NEO4J_DOCKER_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "MISTRAL_API_KEY",
    "MISTRAL_BASE_URL",
    "MISTRAL_MODEL",
    "MISTRAL_TIMEOUT_SECONDS",
    "MISTRAL_MAX_TOKENS",
    "MISTRAL_TEMPERATURE",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
}


def parse_env_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip()
    return values


def is_sensitive_key(key: str) -> bool:
    upper = key.upper()
    return upper.endswith("_API_KEY") or any(token in upper for token in ["PASSWORD", "SECRET", "TOKEN"])


def sanitize_env_text(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if "=" not in stripped or stripped.startswith("#"):
            lines.append(raw)
            continue
        key, value = raw.split("=", 1)
        if is_sensitive_key(key.strip()) and value.strip():
            lines.append(f"{key}=[redacted]")
        else:
            lines.append(raw)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def render_profile_env(profile: str, current_text: str = "") -> str:
    template_path = PROFILE_TEMPLATES[profile]
    template_text = template_path.read_text(encoding="utf-8")
    current_values = parse_env_values(current_text)
    template_values = parse_env_values(template_text)
    preserved = {
        key: value
        for key, value in current_values.items()
        if value and (key in PRESERVE_KEYS or is_sensitive_key(key)) and key not in template_values
    }
    if not preserved:
        return _ensure_trailing_newline(template_text)
    lines = [_ensure_trailing_newline(template_text).rstrip(), "", "# Preserved local secrets/settings from previous .env"]
    for key in sorted(preserved):
        lines.append(f"{key}={preserved[key]}")
    return "\n".join(lines).rstrip() + "\n"


def make_sanitized_diff(current_text: str, next_text: str) -> str:
    before = sanitize_env_text(current_text).splitlines(keepends=True)
    after = sanitize_env_text(next_text).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile=".env current",
            tofile=".env proposed",
        )
    )


def backup_env(env_path: Path = ENV_PATH) -> Path | None:
    if not env_path.exists():
        return None
    backup_path = env_path.with_name(f".env.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}")
    shutil.copy2(env_path, backup_path)
    return backup_path


def switch_profile(profile: str, *, write: bool = False, backup: bool = False, env_path: Path = ENV_PATH) -> dict[str, str | bool | None]:
    if profile not in PROFILE_TEMPLATES:
        raise ValueError(f"Unknown profile: {profile}")
    current_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    next_text = render_profile_env(profile, current_text)
    diff = make_sanitized_diff(current_text, next_text)
    backup_path: Path | None = None
    if write:
        if backup:
            backup_path = backup_env(env_path)
        env_path.write_text(next_text, encoding="utf-8")
    return {
        "profile": profile,
        "written": write,
        "backup_path": str(backup_path) if backup_path else None,
        "diff": diff,
    }


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Switch local .env between reproducible runtime profiles.")
    parser.add_argument("profile", choices=sorted(PROFILE_TEMPLATES), help="Profile template to apply.")
    parser.add_argument("--write", action="store_true", help="Write proposed profile into .env. Without this flag, only prints sanitized diff.")
    parser.add_argument("--backup", action="store_true", help="Backup existing .env before writing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = switch_profile(args.profile, write=args.write, backup=args.backup)
    if result["written"]:
        print(f"Profile {args.profile} written to .env")
        if result["backup_path"]:
            print(f"Backup created: {result['backup_path']}")
    else:
        print("Dry run only. Re-run with --write to update .env.")
    diff = str(result["diff"] or "").strip()
    if diff:
        print("\nSanitized diff:")
        print(diff)
    else:
        print("No changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

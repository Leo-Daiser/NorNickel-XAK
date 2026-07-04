# Codex Project Setup Report

## What was found

- Repository is a git repo.
- Initial `git status --short` was clean.
- `.env` exists in the project root. Values were not printed or copied. Treat this file as private local configuration and do not commit it to public repositories.
- `.gitignore` already excludes `.env`, `data/`, `volumes/`, `dist/`, Python caches, logs, SQLite files, and zip archives.
- Docker Compose defines `api`, `ui`, `neo4j`, and `qdrant` services.
- Current Docker Compose status showed `api`, `ui`, `neo4j`, and `qdrant` running.
- Default API environment uses `RETRIEVAL_MODE=bm25`; optional compose can switch retrieval to hybrid.
- Required project files for Neo4j, answer quality, Streamlit demo, and release checks are present.

## MCP servers

### Available before changes

- `codex mcp --help` could not be executed: WindowsApps returned `Отказано в доступе`.
- `codex mcp list` could not be executed: WindowsApps returned `Отказано в доступе`.
- Because CLI listing failed, existing globally registered MCP servers could not be verified from this shell.

### Added to project config

- `context7` in `.codex/config.toml`
  - Command: `npx -y @upstash/context7-mcp`
  - Purpose: current documentation for FastAPI, Streamlit, Neo4j Python driver, Qdrant, Pydantic, Docker Compose, pytest.
  - Verification: configured, manual verification required.

- `playwright` in `.codex/config.toml`
  - Command: `npx -y @playwright/mcp@latest --headless`
  - Purpose: Streamlit UI checks.
  - Verification: configured, manual verification required.

- `neo4j` in `.codex/config.toml`
  - Command: Docker MCP image `mcp/neo4j`
  - URI: `bolt://localhost:7687`
  - Read-only: `NEO4J_READ_ONLY=true`
  - Password: `${NEO4J_PASSWORD}` placeholder only.
  - Reason for localhost URI: Codex runs on the Windows host and the Neo4j container publishes port `7687`.
  - Verification: configured, manual verification required.

### Not added

- `github`
  - Not configured.
  - Reason: no `GITHUB_TOKEN`/`GH_TOKEN` was detected in environment, and this task must not request or hardcode a token.

- `qdrant`
  - Postponed.
  - Reason: although a Qdrant container exists, the default demo path is currently BM25/fallback and no read-only official/vendor-backed MCP server was configured in this setup.
  - Required condition: Qdrant MCP postponed until `qdrant_ready=true` and vector retrieval is part of demo path.

## Skills

Curated OpenAI skills were checked with `skill-installer`.

Found curated skills include general tools such as `playwright`, `security-best-practices`, `pdf`, and GitHub skills, but no specific curated skill for scientific KG / Neo4j GraphRAG / Streamlit demo QA. No global curated skill was installed because the task is project-scoped and generic skills were explicitly disallowed.

Created project playbook skills:

- `.agents/skills/neo4j-backend-triage/SKILL.md`
- `.agents/skills/answer-quality-regression/SKILL.md`
- `.agents/skills/streamlit-demo-regression/SKILL.md`
- `.agents/skills/release-security-gate/SKILL.md`

Updated project instructions:

- `AGENTS.md`

## Checks run

- `git status --short`
- repository file discovery with `rg --files`
- `codex mcp --help`
- `codex mcp list`
- curated skills listing via `skill-installer`
- Docker Compose service inspection
- `python .../skill-creator/scripts/quick_validate.py` for all four project skills: all valid
- `.codex/config.toml` and `.codex/config.example.toml` parsed successfully with Python `tomllib`
- native PowerShell equivalent of `find .agents/skills -maxdepth 2 -name "SKILL.md" -print`
- `python scripts/check_project.py`: `SMOKE TEST PASSED`

Notes:

- Unix-style `find` through WSL-bash failed on this machine because `/bin/bash` is unavailable. The SKILL.md list was verified with PowerShell instead.
- Post-change MCP checks are recorded separately in `.codex/mcp_checklist.md` and final assistant response.

## Manual actions required

1. Restart Codex or reload the project so `.codex/config.toml` is picked up.
2. Ensure `NEO4J_PASSWORD` is set in the environment before using the Neo4j MCP server.
3. Run `codex mcp list` from a shell where `codex.exe` is executable. In this session, WindowsApps denied execution.
4. If Context7 starts requiring an API key in the installed version, configure only an environment placeholder such as `${CONTEXT7_API_KEY}`, not a real key in the repo.
5. If Playwright MCP fails in headless mode on this machine, remove `--headless` from the project MCP config and document that change.

## Security notes

- No real API keys, GitHub tokens, Neo4j passwords, or OpenRouter keys were added to Codex config, AGENTS.md, reports, or skills.
- `.env` was not copied or printed.
- `.env` is already ignored by git.

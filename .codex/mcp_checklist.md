# MCP Checklist

context7:
- configured: yes
- command: `npx -y @upstash/context7-mcp`
- verification: configured, manual verification required; `codex mcp list` could not run in this WindowsApps environment due access denied
- notes: no API key configured; add only an environment placeholder if a future Context7 version requires one

playwright:
- configured: yes
- command: `npx -y @playwright/mcp@latest --headless`
- verification: configured, manual verification required; `codex mcp list` could not run in this WindowsApps environment due access denied
- notes: headless mode is configured because no local headless failure was observed during configuration

neo4j:
- configured: yes
- read_only: true
- uri: `bolt://localhost:7687`
- password_hardcoded: false
- verification: configured, manual verification required; `codex mcp list` could not run in this WindowsApps environment due access denied
- notes: Codex runs on the Windows host while Neo4j publishes `7687`, so host URI is `bolt://localhost:7687`; password uses `${NEO4J_PASSWORD}` placeholder only

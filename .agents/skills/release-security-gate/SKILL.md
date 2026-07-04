---
name: release-security-gate
description: Use before creating, checking, submitting, or sharing a release archive; use when secrets, .env, API keys, package contents, or hackathon delivery are involved.
---

# Release Security Gate

## Purpose

Prevent secrets, local state, caches, and bulky generated data from entering the deliverable archive.

## Read first

- `scripts/make_release_archive.py`
- `scripts/check_release_package.py`
- `.gitignore`
- `README.md`
- `.env.example`

## Never include

- `.env`
- API keys
- tokens
- passwords
- `data/`
- `db/`
- `volumes/`
- `.pytest_cache/`
- `__pycache__/`
- old `dist/`
- local Docker volumes
- raw private datasets unless explicitly intended

## Required checks

Run:

```bash
python scripts/make_release_archive.py
python scripts/check_release_package.py --path dist/release_unpacked
```

## Failure conditions

The release is invalid if:

- `.env` is present;
- any likely API key/token/password is present;
- generated database/vector storage is included;
- release cannot be unpacked and checked cleanly.

# Security Policy

## Scope

AI Trading Sandbox is a local research tool, **not** a real trading system,
and holds no real funds or personal data. It is meant to run on your machine.

## Supported Versions

| Version | Supported |
| --- | --- |
| 0.2.x | ✅ |
| < 0.2 | ❌ |

## Reporting a Vulnerability

Do **not** file a public issue. Use GitHub's private report:

<https://github.com/LXTimer/ai-trading-sandbox/security/advisories/new>

Include a description, a minimal repro, and the version. We'll respond within a
few days and ask for coordinated disclosure before any public details.

## Built-in Safeguards

- **No hard-coded secret.** Flask uses a random key each boot
  (`os.urandom(24)`), or `FLASK_SECRET_KEY` from the environment.
- **CSRF/DNS-rebinding protection.** `@app.before_request` rejects state-changing
  requests with a mismatched `origin` header (HTTP 403).
- **Server-side sessions.** The browser cookie holds only an id; the session
  lives on the server and is capped at 64 (LRU eviction).
- **Secrets stay local.** LLM API keys come only from `user_config.json` or
  `.env` (both gitignored) and are never logged. (Covered by
  `tests/test_bug_fixes.py`.)

## For Contributors & Users

Never commit secrets — not in code, tests, or `runs/` snapshots. Tests must mock
LLM calls: no network, no real credentials. If you expose the dashboard beyond
local host, put it behind a firewall you control — it has no auth layer.

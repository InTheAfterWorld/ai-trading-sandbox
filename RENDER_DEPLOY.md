# Render deployment

This repository has a dedicated `render-web` branch for the hosted version. The
local CLI / Flask entry point is unchanged; Render starts the protected wrapper
instead.

## Deploy

1. In Render, create a **Web Service** from this repository.
2. Select branch **`render-web`**.
3. If Render detects `render.yaml`, use the Blueprint configuration, or set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn -w 1 --threads 8 --timeout 180 -b 0.0.0.0:$PORT ai_trading_society.web.render_app:app`
4. Set these environment variables:
   - `ATS_WEB_PASSWORD`: a strong password for the hosted dashboard.
   - `ATS_ALLOWED_HOSTS`: the Render hostname, e.g. `ai-trading-sandbox-web.onrender.com`.
   - `FLASK_SECRET_KEY`: generate a random value (the Blueprint can generate it automatically).
   - `ATS_REDACT_CONFIG=1` (recommended; prevents stored API keys from being sent to the browser).

The `/health` endpoint is intentionally unauthenticated so Render can check that
the process is alive.

## Important security behavior

The hosted wrapper requires HTTP Basic Auth. This is deliberate: the dashboard
can start AI agents and make paid provider requests, so it should not be left
as an anonymous public API.

API keys are kept server-side by the existing config store. The hosted branch
also enables config redaction so `/api/config` does not send the actual keys to
the browser.

## Local version

Nothing here changes how the normal local app is started:

```text
python run.py
```

The hosted branch is separate so deployment work can continue without changing
the local codebase you use for development.

## Current limitation

`/api/step` still performs a complete simulation round synchronously. The
Gunicorn timeout is set to 180 seconds, but this does **not** remove any upstream
platform timeout. If rounds regularly exceed the host's request limit, the next
web-only improvement should be a background round job + polling endpoint. That
is intentionally not mixed into this deployment patch.

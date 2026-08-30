"""Render-only WSGI entry point for AI Trading Sandbox.

This is intentionally a thin deployment wrapper around the existing Flask
application. The Render deployment is same-origin, so the browser should not
need CORS at all. Local-only Host/CSRF request guards are disabled here because
this repository is dedicated to the public Render deployment.
"""

import os

from flask import request

# Import the Flask application object directly from the module that creates it.
# Do not import through ai_trading_society.web.__init__: that package exports
# `app` as a Flask object, not as a module, which is easy to confuse with the
# module itself and can produce AttributeError during Gunicorn startup.
from ai_trading_society.web.app import app

# Disable the local development Host/Origin guards for this Render-only repo.
# They are appropriate for a localhost-only deployment, but reject legitimate
# requests when Flask is behind Render's public reverse proxy.
app.before_request_funcs[None] = []

# Never expose saved API keys to browsers on the public deployment.
os.environ["ATS_REDACT_CONFIG"] = "1"

# Render terminates TLS before forwarding to Gunicorn. The site itself is
# same-origin, so Lax cookies are sufficient and avoid cross-site cookies.
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


@app.after_request
def render_headers(response):
    """Apply deployment headers and optional explicit external-origin CORS."""
    origin = request.headers.get("Origin", "").rstrip("/")
    host_origin = request.host_url.rstrip("/")
    external_origins = {
        value.strip().rstrip("/")
        for value in os.environ.get("ATS_CORS_ORIGINS", "").split(",")
        if value.strip()
    }

    if origin and (origin == host_origin or origin in external_origins):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, POST, OPTIONS"
        response.headers["Vary"] = "Origin"

    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


@app.route("/healthz", methods=["GET"])
def healthz():
    """Render health-check endpoint."""
    return {"status": "ok"}


# Gunicorn supports both `render_app:app` and `render_app:application`.
application = app

print("[render] WSGI app loaded successfully")

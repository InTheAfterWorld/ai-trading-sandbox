"""Render-only WSGI entry point for the AI Trading Sandbox.

This module is the production entry point for the dedicated Render repo.
It imports the Flask application object directly, disables localhost-only
request guards, and adds only deployment-specific behavior.
"""

import os

from flask import request

# IMPORTANT: import the Flask OBJECT directly from app.py.
# Do not import a module object through the package namespace and then access
# `.app`; the package may already export `app` as a Flask instance.
from ai_trading_society.web.app import app

# The source app is intentionally protected for localhost use. This repo is
# the separate public Render deployment, so those local Host/Origin guards
# must not run here.
app.before_request_funcs[None] = []

# Never return stored trader API keys to a browser in the public deployment.
os.environ["ATS_REDACT_CONFIG"] = "1"

# Render terminates TLS at its reverse proxy. Use secure, HTTP-only session
# cookies for the browser-facing deployment.
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


@app.after_request
def render_headers(response):
    """Apply conservative security headers and optional explicit CORS."""
    origin = request.headers.get("Origin", "").rstrip("/")
    host_origin = request.host_url.rstrip("/")

    # The dashboard is same-origin, so CORS is normally unnecessary. Keep an
    # explicit allow-list only for the case where a separate frontend is
    # intentionally configured later.
    allowed = {
        value.strip().rstrip("/")
        for value in os.environ.get("ATS_CORS_ORIGINS", "").split(",")
        if value.strip()
    }
    if origin and (origin == host_origin or origin in allowed):
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
    """Return a lightweight Render health-check response."""
    return {"status": "ok"}


# Also expose the conventional WSGI name for platforms that use it.
application = app

print("[render] WSGI app loaded successfully")

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
        debug=False,
    )

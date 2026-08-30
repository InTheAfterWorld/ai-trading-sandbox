"""Render-only WSGI entry point for the AI Trading Sandbox.

This module is the public web deployment entry point. It loads the existing
Flask routes, removes the localhost-only request guards from the local app,
and applies a small deployment policy without changing simulation behavior.
"""

import os

from flask import request

from ai_trading_society.web.app import app

# The local application has localhost-only Host/Origin protections. This repo
# is the dedicated Render deployment, so those local guards must not run here.
app.before_request_funcs[None] = []

# Never return stored trader API keys to a browser in the public deployment.
os.environ["ATS_REDACT_CONFIG"] = "1"

# Flask is behind Render's HTTPS proxy. Keep browser sessions secure and
# usable from the single deployed site.
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


@app.after_request
def render_headers(response):
    """Add safe defaults for the Render-hosted application."""
    origin = request.headers.get("Origin")
    host_origin = request.host_url.rstrip("/")

    # Same-origin requests do not require CORS. For an explicitly configured
    # external frontend, allow only that exact origin.
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
    return {"status": "ok"}


application = app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")), debug=False)

"""Render deployment entry point for AI Trading Sandbox.

This module wraps the existing Flask dashboard for public web deployment.
The original app contains local-development Host/CSRF guards that reject
Render/browser origins, so the deployment entry point replaces those guards
with deployment-safe CORS handling.
"""

import os

from flask import request

from ai_trading_society.web.app import app

# The Render service serves the UI and API from the same origin, so CORS is
# normally unnecessary. If the UI is hosted separately, set
# ATS_CORS_ORIGINS to a comma-separated list of exact origins.
_CORS_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.environ.get("ATS_CORS_ORIGINS", "").split(",")
    if origin.strip()
}

# The local version of app.py has strict localhost-only Host/Origin guards.
# The Render entry point is intentionally web-only, so replace those guards
# with the deployment policy below.
app.before_request_funcs[None] = []

# Never expose stored API keys through GET /api/config in the public service.
os.environ["ATS_REDACT_CONFIG"] = "1"

# Flask's default session cookie is fine for same-origin Render traffic. These
# settings also make an explicitly configured external frontend work with
# credentialed CORS requests.
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="None",
)


@app.after_request
def add_deployment_cors_headers(response):
    """Add CORS headers only for explicitly trusted external frontends."""
    origin = request.headers.get("Origin", "").rstrip("/")
    if origin and origin in _CORS_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, POST, OPTIONS"
        response.headers["Vary"] = "Origin"
    return response


@app.route("/healthz", methods=["GET"])
def healthz():
    """Simple Render health check."""
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")), debug=False)

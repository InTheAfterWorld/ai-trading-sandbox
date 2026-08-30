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
# ATS_CORS_ORIGINS to a comma-separated list of exact origins, e.g.
# https://my-frontend.example.com,https://www.example.com
_CORS_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.environ.get("ATS_CORS_ORIGINS", "").split(",")
    if origin.strip()
}

# The local version of app.py intentionally has strict localhost-only Host and
# Origin checks because it can expose saved API keys. Render is a separate,
# web-only deployment, so remove those local guards here.
app.before_request_funcs[None] = []

# Never expose stored API keys through GET /api/config in the public deployment.
os.environ["ATS_REDACT_CONFIG"] = "1"


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
    # Local invocation of this file is not the supported workflow; Render
    # starts it through gunicorn. This fallback is useful for a quick smoke
    # test without restoring the old local CLI entry points.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))

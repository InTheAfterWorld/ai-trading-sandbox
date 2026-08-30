"""Production wrapper for the Render deployment.

Keeps the local Flask app unchanged while adding the small amount of protection
needed when the dashboard is exposed on the public internet.

Local development continues to use ``ai_trading_society.web.app:app``.
Render uses ``ai_trading_society.web.render_app:app``.
"""

import hmac
import os

from flask import Response, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

from ai_trading_society.web import app as _app_module

app = _app_module.app

# Render sits behind a reverse proxy. Without ProxyFix Flask sees the internal
# HTTP/127.0.0.1 host instead of the public https://<service>.onrender.com host.
# That breaks both Host validation and the app's same-origin CSRF check.
# Trust exactly one proxy hop: Render's edge proxy.
app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_for=1,
    x_proto=1,
    x_host=1,
    x_port=1,
)

# Render provides RENDER_EXTERNAL_HOSTNAME automatically. Include it in the
# existing app allowlist so the hosted request passes Host validation.
_render_hostname = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "").strip().lower()
if _render_hostname:
    _app_module._ALLOWED_HOSTS.add(_render_hostname)

_web_password = os.environ.get("ATS_WEB_PASSWORD", "").strip()


@app.route("/health", methods=["GET"])
def health():
    """Unauthenticated health endpoint for Render's health checker."""
    return jsonify({"ok": True})


def _authorized() -> bool:
    """Validate HTTP Basic Auth using a shared deployment password."""
    if not _web_password:
        return False
    auth = request.authorization
    if auth is None or not auth.password:
        return False
    return hmac.compare_digest(auth.password, _web_password)


@app.before_request
def require_web_password():
    """Require the shared password before any hosted request is processed."""
    if request.path == "/health":
        return None
    if _authorized():
        return None
    return Response(
        "Authentication required.",
        401,
        {"WWW-Authenticate": 'Basic realm="AI Trading Sandbox"'},
    )

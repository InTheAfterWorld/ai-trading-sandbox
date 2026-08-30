"""Production wrapper for the Render deployment.

Keeps the local Flask app unchanged while adding the small amount of protection
needed when the dashboard is exposed on the public internet.

Local development continues to use ``ai_trading_society.web.app:app``.
Render uses ``ai_trading_society.web.render_app:app``.
"""

import hmac
import os

from flask import Response, request

from ai_trading_society.web import app as _app_module

app = _app_module.app

# The existing app intentionally protects against DNS rebinding with a local
# Host allowlist. Render provides RENDER_EXTERNAL_HOSTNAME automatically;
# include it here so the hosted copy works without changing local defaults.
_render_hostname = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "").strip().lower()
if _render_hostname:
    _app_module._ALLOWED_HOSTS.add(_render_hostname)

_web_password = os.environ.get("ATS_WEB_PASSWORD", "").strip()


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
    if _authorized():
        return None
    return Response(
        "Authentication required.",
        401,
        {"WWW-Authenticate": 'Basic realm="AI Trading Sandbox"'},
    )

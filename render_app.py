"""Compatibility WSGI entry point for Render.

The canonical deployment implementation lives in
``ai_trading_society.web.render_app``. This root-level shim intentionally keeps
``gunicorn render_app:app`` working for existing Render service settings.
"""

from ai_trading_society.web.render_app import app, application

__all__ = ["app", "application"]

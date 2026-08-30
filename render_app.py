"""Render deployment compatibility entry point.

Kept at repository root so the Render start command has a stable module path.
"""

from ai_trading_society.web.render_app import app, application

__all__ = ["app", "application"]

"""Web-only entry point for AI Trading Sandbox.

Render uses ``gunicorn render_app:app``. This file is kept as a minimal
production-style fallback for platforms that start the project with
``python run.py``.
"""

import os

from render_app import app


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
        debug=False,
    )

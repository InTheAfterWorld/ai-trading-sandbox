"""
Web UI for AI TRADING SANDBOX.

A Flask-based dashboard that runs the simulation step-by-step
with a modern visual interface.

Run:
    python run.py

Then open http://localhost:5000 in your browser.
"""

import json
import os
import re
import sys
import uuid
from collections import OrderedDict
from typing import Any

try:
    from flask import Flask, jsonify, render_template, request, send_file
    from flask import session as flask_session
    from werkzeug.exceptions import HTTPException
except ImportError:
    print("Flask is not installed. Install it with:")
    print("    pip install flask")
    sys.exit(1)

from ai_trading_society.agents.external_ai_agent import (
    _DEFAULT_MODELS,
    ExternalAIAgent,
)
from ai_trading_society.agents.player_agent import PlayerAgent
from ai_trading_society.agents.roster import build_agent_roster, resolve_social_map
from ai_trading_society.chat_context import build_chat_system_prompt
from ai_trading_society.config import MarketConfig, StockSpec
from ai_trading_society.config_store import (
    load_config,
    redact_config,
    save_config,
)
from ai_trading_society.console_utils import (
    agent_personality,
    agent_personality_desc,
    agent_type_label,
)
from ai_trading_society.market_env import MarketEnv
from ai_trading_society.market_events import EVENT_TEMPLATES
from ai_trading_society.prompt_version import (
    PROMPT_TEMPLATE_VERSION,
    shipped_fingerprints,
)
from ai_trading_society.report_export import generate_report_html, save_report
from ai_trading_society.simulator import Simulator, grade_performance, grade_wealth_curve
from ai_trading_society.usage import agent_usage, collect_usage

app = Flask(__name__, template_folder="../../templates", static_folder="../../static")
# Never fall back to a hard-coded secret: it would let an attacker on the
# local network forge session cookies. Generate a fresh random key per boot.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(24)

# --- DNS-rebinding protection -------------------------------------------------
# The dashboard binds to loopback and has no auth, so GET /api/config can hand
# the browser the stored trader API keys (that is the point of the shared
# config -- reload the page, keys are already there). The one way a remote
# page could still reach that endpoint is DNS rebinding: it resolves its own
# domain to 127.0.0.1 so the browser treats it as same-origin, and the
# request then carries `Host: attacker-domain`. Requiring a known-local Host
# closes that path. Override the list when fronting the app with a proxy.
_ALLOWED_HOSTS = {
    h.strip().lower()
    for h in os.environ.get(
        "ATS_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]"
    ).split(",")
    if h.strip()
}


def _request_hostname() -> str:
    """The Host header with any port stripped, lower-cased ('' if absent)."""
    raw = (request.host or "").strip().lower()
    if raw.startswith("["):            # IPv6 literal: [::1]:5000 -> [::1]
        return raw.split("]", 1)[0] + "]"
    return raw.split(":", 1)[0]         # 127.0.0.1:5000 -> 127.0.0.1


def _redact_config_responses() -> bool:
    """Whether GET/POST /api/config withhold the real trader API keys.

    Default False: this is a single-user local tool, the keys already sit in
    user_config.json, and having the browser reload them every visit is the
    convenience the shared config exists for. Set ATS_REDACT_CONFIG=1 to
    withhold them anyway (screen-sharing the dashboard, running it on a box
    other people can reach, etc.).
    """
    return os.environ.get("ATS_REDACT_CONFIG") == "1"


# Exported HTML reports live here and are served as read-only snapshots.
# Anchored to the project root so saving (CWD-relative open) and serving
# (Flask resolves relative paths against app.root_path) always agree.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(_PROJECT_ROOT, "runs", "reports")

# ---------------------------------------------------------------------------
# Session state is server-side; the browser cookie stores only its identifier.
# Sessions hold the full simulation state, so cap their count to keep a
# long-running server from leaking memory: least-recently-used sessions are
# evicted first.
# ---------------------------------------------------------------------------
_sessions: "OrderedDict[str, dict]" = OrderedDict()
_MAX_SESSIONS = 64


def _get_session() -> dict:
    """Return the simulation state belonging to the current browser."""
    session_id = flask_session.get("simulation_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        flask_session["simulation_id"] = session_id
    state = _sessions.get(session_id)
    if state is None:
        state = {}
        _sessions[session_id] = state
        while len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)
    else:
        _sessions.move_to_end(session_id)
    return state


def _parse_int(value, field: str, *, minimum: int = 0):
    """Parse a strict integer API value, returning a Flask error when invalid."""
    if isinstance(value, bool) or value is None:
        return None, jsonify({"error": f"{field} must be an integer."}), 400
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, jsonify({"error": f"{field} must be an integer."}), 400
    if isinstance(value, float) and not value.is_integer():
        return None, jsonify({"error": f"{field} must be an integer."}), 400
    if isinstance(value, str) and str(parsed) != value.strip():
        return None, jsonify({"error": f"{field} must be an integer."}), 400
    if parsed < minimum:
        return None, jsonify({"error": f"{field} must be at least {minimum}."}), 400
    return parsed, None, None


def _parse_float(value, field: str, *, minimum: float = 0.0, maximum=None):
    """Parse a numeric API value (int/float/numeric string) into a float."""
    if isinstance(value, bool) or value is None:
        return None, jsonify({"error": f"{field} must be a number."}), 400
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, jsonify({"error": f"{field} must be a number."}), 400
    if parsed < minimum:
        return None, jsonify({"error": f"{field} must be at least {minimum}."}), 400
    if maximum is not None and parsed > maximum:
        return None, jsonify({"error": f"{field} must be at most {maximum}."}), 400
    return parsed, None, None


def _aggregate_actions(stock_acts, first_symbol: str = "ATSX"):
    """
    Collapse one agent's nested per-stock actions into a compact summary.

    agent_actions is nested: {sym: {action, requested_qty, filled_qty,
    reasoning, error}}. Returns (action, requested, filled, reasoning,
    error, per_stock_list).
    """
    if isinstance(stock_acts, dict) and stock_acts:
        per_stock = []
        dominant_action = "hold"
        max_filled = 0
        total_req = 0
        total_filled = 0
        any_error = False
        reasoning_parts = []
        for sym, sa in stock_acts.items():
            if not isinstance(sa, dict):
                continue
            act = sa.get("action", "hold")
            req = sa.get("requested_qty", 0)
            filled = sa.get("filled_qty", 0)
            total_req += req
            total_filled += filled
            if filled > max_filled:
                max_filled = filled
                dominant_action = act
            if sa.get("error"):
                any_error = True
            if sa.get("reasoning"):
                reasoning_parts.append(f"{sym}: {sa['reasoning']}")
            per_stock.append({
                "symbol": sym,
                "action": act,
                "quantity": filled,
                "requested": req,
                "filled": filled,
                "reasoning": sa.get("reasoning", ""),
            })
        return (
            dominant_action,
            total_req,
            total_filled,
            " | ".join(reasoning_parts),
            any_error,
            per_stock,
        )
    # Legacy flat structure fallback.
    return (
        stock_acts.get("action", "hold"),
        stock_acts.get("requested_qty", 0),
        stock_acts.get("filled_qty", 0),
        stock_acts.get("reasoning", ""),
        bool(stock_acts.get("error", False)),
        [{
            "symbol": first_symbol,
            "action": stock_acts.get("action", "hold"),
            "quantity": stock_acts.get("filled_qty", 0),
            "requested": stock_acts.get("requested_qty", 0),
            "filled": stock_acts.get("filled_qty", 0),
            "reasoning": stock_acts.get("reasoning", ""),
        }],
    )


def _usage_payload(agent, step: int | None = None):
    """Compact token/cost snapshot for one agent, or None if it makes no calls.

    Deliberately smaller than ``UsageTracker.to_dict()``: this rides along
    with every round, so it carries the running total plus this round's
    slice and leaves the per-kind breakdown to ``/api/usage``.
    """
    tracker = agent_usage(agent)
    if tracker is None:
        return None
    payload: dict[str, Any] = {
        "model": tracker.model,
        "priced": tracker.total.unpriced_calls == 0 and tracker.total.calls > 0,
        "total": tracker.total.to_dict(),
    }
    if step is not None:
        payload["round"] = tracker.step_totals(step).to_dict()
    return payload


@app.before_request
def block_dns_rebinding():
    """Reject any request whose Host header is not a known local name.

    See _ALLOWED_HOSTS: this is what makes it safe for /api/config to return
    real API keys to the local browser. A request with no Host at all (some
    HTTP/1.0 clients) is allowed through; the loopback bind already covers it.
    """
    host = _request_hostname()
    if host and host not in _ALLOWED_HOSTS:
        return jsonify({"error": "Host not allowed."}), 403
    return None


@app.before_request
def csrf_protect():
    """Reject cross-origin state-changing requests (CSRF / DNS rebinding).

    Requests without an Origin header (curl, the test client, same-origin
    form posts) are allowed; requests carrying a mismatched Origin are not.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    origin = request.headers.get("Origin")
    if not origin:
        return None
    host = request.host_url.rstrip("/")
    if origin != host:
        return jsonify({"error": "Cross-origin request blocked."}), 403
    return None


@app.errorhandler(Exception)
def handle_exception(e):
    """Ensure API errors return JSON instead of HTML error pages.

    Aborts raised by Flask (404, 400, ...) keep their status. Genuine bugs
    are logged with a traceback instead of vanishing into a bare JSON body,
    and re-raised under --debug so the interactive debugger can show them.
    """
    if not isinstance(e, HTTPException):
        app.logger.exception("Unhandled error on %s %s", request.method, request.path)
        if app.debug:
            raise e
    # HTTPException.code is an int, but any object can reach this handler,
    # and jsonify(...), <non-int> is a TypeError inside the error path.
    code = getattr(e, "code", 500)
    if type(code) is not int or not (400 <= code <= 599):
        code = 500
    return jsonify({"error": str(e)}), code


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the home/configuration page."""
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """Serve the favicon icon."""
    return app.send_static_file("logo.png")



@app.route("/sim")
def sim():
    """Serve the simulation dashboard page."""
    return render_template("sim.html")



@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Return the saved user configuration (shared with CLI and the homepage).

    By default this includes the stored trader API keys so the homepage can
    repopulate itself on every load. The Host allowlist (see
    block_dns_rebinding) keeps that reachable only from a local browser. Set
    ATS_REDACT_CONFIG=1 to strip the keys and return a ``has_api_key`` flag
    instead.
    """
    cfg = load_config()
    return jsonify({"config": redact_config(cfg) if _redact_config_responses() else cfg})


@app.route("/api/config", methods=["POST"])
def api_save_config():
    """Persist the user configuration entered on the homepage.

    This file is the single source of truth: the CLI reads the same config
    so a simulation saved in the browser runs identically from the terminal.
    """
    data = request.get_json(silent=True) or {}
    saved = save_config(data)
    if _redact_config_responses():
        saved = redact_config(saved)
    return jsonify({"ok": True, "config": saved})


# A real user_config.json is a few KB; this is a generous ceiling that still
# rejects an accidental multi-megabyte upload before it is parsed.
_MAX_CONFIG_UPLOAD_BYTES = 256 * 1024


def _config_summary(cfg: dict) -> dict:
    """A short, safe-to-display recap of what an uploaded config resolved to."""
    traders = cfg.get("traders", []) or []
    return {
        "steps": cfg.get("steps"),
        "traders": len(traders),
        "traders_with_keys": sum(1 for t in traders if t.get("api_key")),
        "stocks": len(cfg.get("stocks", []) or []),
        "player_participates": cfg.get("player_participates"),
    }


@app.route("/api/config/upload", methods=["POST"])
def api_upload_config():
    """Replace the saved configuration with an uploaded user_config.json.

    Accepts a multipart file part (``file`` / ``config`` / the first file)
    or a raw JSON request body. The upload goes through the same
    ``save_config`` normalisation as a homepage edit -- unknown keys are
    dropped, out-of-range numbers clamped -- so a slightly-off file cannot
    break anything, and the previous config is kept as user_config.json.bak.

    A blanked trader ``api_key`` in the upload keeps whatever key is already
    stored for that trader name; the response ``summary`` reports how many
    traders ended up with a key so the outcome is visible.
    """
    raw = b""
    if request.files:
        storage = (
            request.files.get("file")
            or request.files.get("config")
            or next(iter(request.files.values()), None)
        )
        if storage is not None:
            raw = storage.read()
    if not raw:
        raw = request.get_data(cache=False) or b""

    if not raw.strip():
        return jsonify({"ok": False, "error": "No file or JSON body received."}), 400
    if len(raw) > _MAX_CONFIG_UPLOAD_BYTES:
        limit_kb = _MAX_CONFIG_UPLOAD_BYTES // 1024
        return jsonify({
            "ok": False,
            "error": f"Config file is too large (limit {limit_kb} KB).",
        }), 413

    try:
        parsed = json.loads(raw.decode("utf-8-sig"))
    except UnicodeDecodeError:
        return jsonify({"ok": False, "error": "File is not valid UTF-8 text."}), 400
    except json.JSONDecodeError as exc:
        return jsonify({
            "ok": False,
            "error": (
                f"Not valid JSON: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno})."
            ),
        }), 400

    if not isinstance(parsed, dict):
        return jsonify({
            "ok": False,
            "error": "Expected a JSON object (the contents of user_config.json).",
        }), 400

    saved = save_config(parsed)
    return jsonify({
        "ok": True,
        "config": redact_config(saved) if _redact_config_responses() else saved,
        "summary": _config_summary(saved),
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    """Initialize a new simulation session."""
    data = request.get_json(silent=True) or {}
    steps, error, status = _parse_int(data.get("steps", 25), "steps", minimum=1)
    if error:
        return error, status
    price = 100.0
    if "price" in data:
        price, error, status = _parse_float(data["price"], "price", minimum=0.01)
        if error:
            return error, status
    cash, error, status = _parse_float(
        data.get("cash", 10000.0), "cash", minimum=0.0
    )
    if error:
        return error, status
    hold, error, status = _parse_int(data.get("hold", 20), "hold", minimum=0)
    if error:
        return error, status
    fee, error, status = _parse_float(data.get("fee", 0.001), "fee", minimum=0.0, maximum=0.5)
    if error:
        return error, status
    slip, error, status = _parse_float(data.get("slip", 0.001), "slip", minimum=0.0, maximum=0.5)
    if error:
        return error, status
    social_influence, error, status = _parse_float(
        data.get("social_influence", 0.0), "social_influence", minimum=0.0, maximum=1.0
    )
    if error:
        return error, status
    # Whether the human player joins the market as a trader (default: yes).
    player_participates = data.get("player_participates", True)
    if not isinstance(player_participates, bool):
        player_participates = True
    # Deep personality simulation (default: off). Falls back to the saved
    # config so a launch that omits the field respects the homepage toggle.
    deep_persona = data.get("deep_persona")
    if not isinstance(deep_persona, bool):
        deep_persona = bool(load_config().get("deep_persona", False))
    provider = data.get("provider") or "openai"
    model = data.get("model") or _DEFAULT_MODELS.get(provider) or "gpt-4o"
    api_key = data.get("api_key", "")
    trader_configs = data.get("traders")
    if isinstance(trader_configs, list):
        # Safety net: if the launch payload has a trader with no api_key
        # (ATS_REDACT_CONFIG mode, or a stale browser), fill it from
        # user_config.json, matched by trader name.
        _stored_keys = {
            t.get("name"): t.get("api_key", "")
            for t in load_config().get("traders", [])
            if t.get("name")
        }
        for _t in trader_configs:
            if isinstance(_t, dict) and not _t.get("api_key"):
                _match = _stored_keys.get(_t.get("name"))
                if _match:
                    _t["api_key"] = _match
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            # int() raises TypeError for lists/dicts/None-likes and
            # ValueError for junk strings; neither is a 500.
            seed = None

    # Parse multi-stock configuration. Each entry: {name, price, hold}.
    # Falls back to the server-saved config's stocks, then to a single
    # default stock, so a request without "stocks" keeps multi-stock runs.
    raw_stocks = data.get("stocks")
    if not (isinstance(raw_stocks, list) and raw_stocks):
        saved_stocks = load_config().get("stocks")
        if isinstance(saved_stocks, list) and saved_stocks:
            raw_stocks = saved_stocks
    stock_specs: list = []
    if isinstance(raw_stocks, list) and raw_stocks:
        seen = set()
        for s in raw_stocks:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name") or s.get("symbol") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            try:
                s_price = float(s.get("price") or s.get("initial_price") or price)
            except (TypeError, ValueError):
                s_price = price
            try:
                s_hold = int(s.get("hold", s.get("initial_holdings", hold)))
            except (TypeError, ValueError):
                s_hold = hold
            stock_specs.append(StockSpec(
                name=name,
                initial_price=s_price,
                initial_holdings=s_hold,
                sector=str(s.get("sector") or "").strip(),
                blurb=str(s.get("blurb") or s.get("description") or "").strip(),
            ))
    if not stock_specs:
        stock_specs = [StockSpec(
            name="Stock 1",
            initial_price=price,
            initial_holdings=hold,
        )]

    config = MarketConfig(
        initial_price=price,
        price_sensitivity=0.02,
        max_price_change_ratio=0.10,
        fee_rate=fee,
        slippage_rate=slip,
        event_probability_multiplier=1.5,
        deep_persona=deep_persona,
        social_influence=social_influence,
        seed=seed,
        stocks=stock_specs,
    )

    try:
        agents, player_agent = build_agent_roster(
            provider=provider, model=model, api_key=api_key,
            trader_configs=trader_configs if isinstance(trader_configs, list) else None,
            cash=cash,
            holdings=hold,
            stocks=stock_specs,
            include_player=player_participates,
            deep_persona=deep_persona,
            mood_max_step=config.mood_max_step,
            mood_intensity=config.mood_intensity,
        )
    except RuntimeError as exc:
        print(f"[api/start] build_agent_roster failed: {exc}")
        return jsonify({"error": str(exc)}), 400

    # A market with neither AI traders nor the human player cannot run.
    ai_count = sum(
        1 for a in agents
        if not bool(getattr(a, "is_player", False))
    )
    if ai_count == 0 and player_agent is None:
        return jsonify({
            "error": "No participants: enable at least one AI trader "
                     "or rejoin as a trader yourself.",
        }), 400

    try:
        env = MarketEnv(config, agents)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    sim = Simulator(env)

    # Wire up the player agent so it can read buffered actions.
    if player_agent is not None:
        player_agent._env = env

    # Resolve social relationships (idol/friends/enemies) so peers' recent
    # trades flow into each agent's observation for herding behavior.
    env.social_map = resolve_social_map(list(env.agents.values()))
    env._social_influence = config.social_influence


    # Create run metadata
    from ai_trading_society.run_metadata import RunMetadata
    metadata = RunMetadata.create(config=config, agents=agents, seed=seed)
    sim.metadata = metadata

    # Capture initial wealths (portfolio-level: cash + all holdings * prices).
    initial_wealths = {aid: env.agent_wealth(a) for aid, a in env.agents.items()}

    state = _get_session()
    state.clear()
    state["env"] = env
    state["sim"] = sim
    state["steps"] = steps
    state["current_step"] = 0
    state["initial_wealths"] = initial_wealths
    state["prev_wealths"] = dict(initial_wealths)
    state["is_player_mode"] = True
    state["run_id"] = metadata.run_id
    state["seed"] = metadata.seed
    # Per-round snapshots for history replay, agent timelines, and reports.
    state["history"] = []
    # Wealth curve per agent, appended to once per round. Rebuilding these by
    # walking state["history"] on every step made /api/step cost O(steps^2).
    state["wealth_curves"] = {aid: [] for aid in env.agents}

    # Build agent roster for the frontend.
    roster = []
    for aid, agent in env.agents.items():
        h = agent.holdings if isinstance(agent.holdings, dict) else {}
        wealth = env.agent_wealth(agent)
        roster.append({
            "id": aid,
            "type": agent_type_label(agent),
            "personality": agent_personality(agent),
            "personality_desc": agent_personality_desc(agent),
            "is_player": isinstance(agent, PlayerAgent),
            "cash": round(agent.cash, 2),
            "holdings": h,
            "wealth": round(wealth, 2),
            "initial_wealth": round(initial_wealths[aid], 2),
        })

    # Stock list for the frontend.
    stocks_payload = [
        {
            "symbol": sm.symbol,
            "name": sm.name,
            "price": round(sm.price, 2),
            "initial_price": round(sm.initial_price, 2),
            "price_history": [round(p, 2) for p in sm.price_history],
            "volume": sm.volume_history[-1] if sm.volume_history else 0,
            "sector": sm.sector,
            "blurb": sm.blurb,
        }
        for sm in env.stocks.values()
    ]

    return jsonify({
        "roster": roster,
        "steps": steps,
        "initial_price": config.initial_price,
        "stocks": stocks_payload,
        "mode": "sandbox",
        "run_id": metadata.run_id,
        "seed": metadata.seed,
        "metadata": metadata.to_dict(),
    })


@app.route("/api/test_api", methods=["POST"])
def api_test_api():
    """Make a small real request to validate one trader's API settings."""
    data = request.get_json(silent=True) or {}
    api_key = str(data.get("api_key", "")).strip()
    provider = str(data.get("provider", "openai")).strip() or "openai"
    model = str(data.get("model", "gpt-4o")).strip() or "gpt-4o"
    base_url = str(data.get("base_url", "")).strip() or None
    if not api_key:
        # The browser may not hold the key; look it up by trader name.
        _name = str(data.get("name", "")).strip()
        if _name:
            for _t in load_config().get("traders", []):
                if _t.get("name") == _name and _t.get("api_key"):
                    api_key = _t["api_key"]
                    break
    if not api_key:
        return jsonify({"ok": False, "error": "API key is required."}), 400
    try:
        agent = ExternalAIAgent("API Test", api_provider=provider, model=model,
                                api_key=api_key, base_url=base_url,
                                enable_memory=False)
        agent._call_ai_api("Reply with exactly: OK")
        return jsonify({"ok": True, "message": "API connection works."})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400



@app.route("/api/step", methods=["POST"])
def api_step():
    """Run one simulation round and return the state."""
    state = _get_session()
    if "sim" not in state:
        return jsonify({"error": "No active simulation"}), 400

    env: MarketEnv = state["env"]
    sim = state["sim"]
    total_steps = state["steps"]

    if state["current_step"] >= total_steps:
        return jsonify({"done": True})

    prev_price = env.price
    step_data = env.step()
    sim.state_history.append(step_data)
    state["current_step"] += 1
    current_step = state["current_step"]

    price = step_data["price"]
    change_pct = (price - prev_price) / max(prev_price, 0.01) * 100
    actions = step_data.get("agent_actions", {})
    triggered_events = step_data.get("triggered_events", [])
    active_events = step_data.get("active_events_detail", [])
    total_buy = step_data.get("total_buy", 0)
    total_sell = step_data.get("total_sell", 0)

    # Build per-agent data.
    # agent_actions is nested: {aid: {sym: {action, requested_qty, filled_qty,
    # reasoning, error}}}. Aggregate per-stock actions into an `actions` list
    # for the frontend, and derive a dominant action for backward compat.
    agents_data = []
    for aid, agent in env.agents.items():
        stock_acts = actions.get(aid, {})
        action, req, filled, reasoning, error, per_stock = _aggregate_actions(
            stock_acts, first_symbol=next(iter(env.stocks.keys()), "ATSX")
        )

        wealth = env.agent_wealth(agent)
        init_w = state["initial_wealths"].get(aid, wealth)
        ret_pct = (wealth / init_w - 1) * 100 if init_w > 0 else 0.0
        prev_w = state["prev_wealths"].get(aid, wealth)
        delta = wealth - prev_w

        # Performance grade from this agent's wealth curve so far. The curve is
        # kept incrementally (see state["wealth_curves"]) and this round's point
        # is appended here, before grading, so the grade always includes it.
        wealth_curve = state.setdefault("wealth_curves", {}).setdefault(aid, [])
        wealth_curve.append(wealth)
        if init_w > 0:
            g = grade_wealth_curve(wealth_curve, init_w)
        else:
            g = {"score": 0, "grade": "D"}

        agents_data.append({
            "id": aid,
            "type": agent_type_label(agent),
            "personality": agent_personality(agent),
            "action": action,
            "requested": req,
            "filled": filled,
            "reasoning": reasoning,
            "error": bool(error),
            "actions": per_stock,
            "is_player": isinstance(agent, PlayerAgent),
            "cash": round(agent.cash, 2),
            "holdings": agent.holdings if isinstance(agent.holdings, dict) else {},
            "wealth": round(wealth, 2),
            "return_pct": round(ret_pct, 2),
            "delta": round(delta, 2),
            "score": g["score"],
            "grade": g["grade"],
            # Deep-mode persona mood; None in simple runs.
            "mood": getattr(agent, "mood", None) if getattr(agent, "deep", False) else None,
            # Tokens and cost: this round's slice plus the running total.
            "usage": _usage_payload(agent, current_step),
        })

    # Update prev_wealths.
    for aid, agent in env.agents.items():
        state["prev_wealths"][aid] = env.agent_wealth(agent)

    # Per-stock data for the frontend chart.
    stocks_payload = [
        {
            "symbol": sm.symbol,
            "name": sm.name,
            "price": round(sm.price, 2),
            "initial_price": round(sm.initial_price, 2),
            "price_history": [round(p, 2) for p in sm.price_history],
            "volume": sm.volume_history[-1] if sm.volume_history else 0,
            "sector": sm.sector,
            "blurb": sm.blurb,
        }
        for sm in env.stocks.values()
    ]

    events_payload = [
        {
            "name": e.get("name", ""),
            "description": e.get("description", ""),
            "type": e.get("type", ""),
            "price_impact": e.get("price_impact", 0.0),
            "scope": e.get("scope", "global"),
            "stock": e.get("stock"),
        }
        for e in triggered_events
    ]

    # Persist a round snapshot so the frontend can replay any past round
    # and so report export can include full decision logs.
    state.setdefault("history", []).append({
        "step": current_step,
        "stocks": stocks_payload,
        "agents": agents_data,
        "events": events_payload,
        "price": round(price, 2),
        "prev_price": round(prev_price, 2),
        "change_pct": round(change_pct, 2),
        "volume": step_data.get("matched_volume", 0),
        "total_buy": total_buy,
        "total_sell": total_sell,
    })

    return jsonify({
        "step": current_step,
        "total_steps": total_steps,
        "is_last": current_step >= total_steps,
        "price": round(price, 2),
        "prev_price": round(prev_price, 2),
        "change_pct": round(change_pct, 2),
        "volume": step_data.get("matched_volume", 0),
        "total_buy": total_buy,
        "total_sell": total_sell,
        "agents": agents_data,
        "stocks": stocks_payload,
        "events": events_payload,
        "active_events": [
            {
                "name": e.get("name", ""),
                "remaining": e.get("remaining_steps", 0),
                "total": e.get("total_steps", 1),
            }
            for e in active_events
        ],
        "price_history": [round(p, 2) for p in env.price_history],
    })


@app.route("/api/player_action", methods=["POST"])
def api_player_action():
    """Buffer the human player's action for the next step."""
    state = _get_session()
    if "sim" not in state:
        return jsonify({"error": "No active simulation"}), 400
    env = state["env"]
    if not any(getattr(a, "is_player", False) for a in env.agents.values()):
        return jsonify({"error": "You are spectating — rejoin as a trader to place orders."}), 400
    data = request.get_json(silent=True) or {}
    action = data.get("action", "hold")
    quantity, error, status = _parse_int(data.get("quantity", 0), "quantity")
    if error:
        return error, status

    if action not in ("buy", "sell", "hold"):
        return jsonify({"error": "Invalid action"}), 400

    symbol = data.get("name") or data.get("symbol")
    env.set_player_action(action, quantity, symbol=symbol)
    return jsonify({"ok": True})


@app.route("/api/results", methods=["GET"])
def api_results():
    """Return the final simulation report."""
    state = _get_session()
    if "sim" not in state:
        return jsonify({"error": "No active simulation"}), 400

    sim = state["sim"]
    env: MarketEnv = state["env"]
    initial_price = env.config.initial_price

    agents = list(env.agents.values())
    ranked = sorted(agents, key=env.agent_wealth, reverse=True)

    rankings = []
    for rank, agent in enumerate(ranked, 1):
        wealth = env.agent_wealth(agent)
        init_w = state["initial_wealths"].get(agent.agent_id, wealth)
        ret = (wealth / init_w - 1) * 100 if init_w > 0 else 0.0
        metrics = sim._compute_agent_metrics(agent.agent_id)

        rankings.append({
            "rank": rank,
            "id": agent.agent_id,
            "type": agent_type_label(agent),
            "personality": agent_personality(agent),
            "is_player": isinstance(agent, PlayerAgent),
            "cash": round(agent.cash, 2),
            "holdings": agent.holdings if isinstance(agent.holdings, dict) else {},
            "wealth": round(wealth, 2),
            "return_pct": round(ret, 2),
            "sharpe": round(metrics["sharpe"], 2),
            "max_drawdown": round(metrics["max_drawdown"] * 100, 2),
            "volatility": round(metrics["volatility"] * 100, 2),
            "win_rate": round(metrics["win_rate"] * 100, 1),
            **grade_performance(
                ret, metrics["sharpe"],
                metrics["max_drawdown"] * 100, metrics["win_rate"] * 100,
            ),
        })

    buy_trades = [t for t in env.trade_history if t.action == "buy"]
    sell_trades = [t for t in env.trade_history if t.action == "sell"]
    avg_vol = sum(env.volume_history) / len(env.volume_history) if env.volume_history else 0

    # Build wealth history for chart.
    wealth_history: dict[str, list[dict[str, float]]] = {}
    for aid in env.agents:
        wealth_history[aid] = []
    for snapshot in sim.state_history:
        step = snapshot["step"]
        for aid, data in snapshot.get("agents", {}).items():
            wealth_history.setdefault(aid, []).append({
                "step": step,
                "wealth": round(data.get("wealth", 0), 2),
            })

    return jsonify({
        "price_summary": {
            "initial": initial_price,
            "final": round(env.price, 2),
            "change_pct": round((env.price / initial_price - 1) * 100, 2),
            "min": round(min(env.price_history), 2),
            "max": round(max(env.price_history), 2),
        },
        "trade_summary": {
            "total": len(env.trade_history),
            "buys": len(buy_trades),
            "sells": len(sell_trades),
            "avg_volume": round(avg_vol, 1),
        },
        "rankings": rankings,
        "price_history": [round(p, 2) for p in env.price_history],
        "wealth_history": wealth_history,
    })


@app.route("/api/history", methods=["GET"])
def api_history():
    """Return per-round snapshots for timeline replay and agent timelines."""
    state = _get_session()
    if "env" not in state:
        return jsonify({"error": "No active simulation"}), 400
    return jsonify({
        "history": state.get("history", []),
        "current_step": state.get("current_step", 0),
        "total_steps": state.get("steps", 0),
    })


@app.route("/api/usage", methods=["GET"])
def api_usage():
    """Token and cost accounting for the run so far.

    ``cost_complete`` is false when some model had no price row, in which
    case the reported cost is a floor rather than the whole bill.
    """
    state = _get_session()
    if "env" not in state:
        return jsonify({"error": "No active simulation"}), 400
    env: MarketEnv = state["env"]

    agents = list(env.agents.values())
    summary = collect_usage(agents)

    # Per-round cost curve, so the dashboard can show spend accumulating.
    by_round: list[dict[str, Any]] = []
    for step in range(1, state.get("current_step", 0) + 1):
        cost = 0.0
        tokens = 0
        complete = True
        for agent in agents:
            tracker = agent_usage(agent)
            if tracker is None:
                continue
            totals = tracker.step_totals(step)
            cost += totals.cost_usd
            tokens += totals.total_tokens
            complete = complete and totals.unpriced_calls == 0
        by_round.append({
            "step": step,
            "cost_usd": round(cost, 6),
            "tokens": tokens,
            "cost_complete": complete,
        })

    return jsonify({
        "total": summary["total"],
        "agents": summary["agents"],
        "by_round": by_round,
        "prompt": {
            "template_version": PROMPT_TEMPLATE_VERSION,
            "fingerprints": shipped_fingerprints(),
        },
    })


@app.route("/api/report/export", methods=["POST"])
def api_report_export():
    """Generate a self-contained HTML report and return its share link."""
    state = _get_session()
    if "env" not in state:
        return jsonify({"error": "No active simulation"}), 400

    env: MarketEnv = state["env"]
    sim = state["sim"]
    metadata = getattr(sim, "metadata", None)
    run_id = state.get("run_id") or (metadata.run_id if metadata else uuid.uuid4().hex[:12])
    seed = state.get("seed")

    ranked = sorted(env.agents.values(), key=env.agent_wealth, reverse=True)
    rankings = []
    for rank, agent in enumerate(ranked, 1):
        wealth = env.agent_wealth(agent)
        init_w = state["initial_wealths"].get(agent.agent_id, wealth)
        metrics = sim._compute_agent_metrics(agent.agent_id)
        rankings.append({
            "rank": rank,
            "id": agent.agent_id,
            "type": agent_type_label(agent),
            "is_player": isinstance(agent, PlayerAgent),
            "cash": round(agent.cash, 2),
            "wealth": round(wealth, 2),
            "return_pct": round((wealth / init_w - 1) * 100, 2) if init_w > 0 else 0.0,
            "sharpe": round(metrics["sharpe"], 2),
            "max_drawdown": round(metrics["max_drawdown"] * 100, 2),
            "volatility": round(metrics["volatility"] * 100, 2),
            "win_rate": round(metrics["win_rate"] * 100, 1),
            **grade_performance(
                (wealth / init_w - 1) * 100 if init_w > 0 else 0.0,
                metrics["sharpe"],
                metrics["max_drawdown"] * 100, metrics["win_rate"] * 100,
            ),
        })

    stocks = [
        {
            "symbol": sm.symbol,
            "name": sm.name,
            "price_history": [round(p, 2) for p in sm.price_history],
        }
        for sm in env.stocks.values()
    ]

    # Wealth curves: round 0 is the initial wealth, then one point per step.
    wealth_history = {}
    for aid in env.agents:
        pts = [round(state["initial_wealths"].get(aid, 0.0), 2)]
        for snap in state.get("history", []):
            for a in snap.get("agents", []):
                if a.get("id") == aid:
                    pts.append(a.get("wealth", pts[-1]))
        wealth_history[aid] = pts

    # Decision log per agent: one row per round, plus the mood trace beside
    # it so the report can show what the trader felt as well as what it did.
    agent_logs: dict[str, list[dict[str, Any]]] = {}
    mood_history: dict[str, list[dict[str, Any]]] = {}
    for snap in state.get("history", []):
        for a in snap.get("agents", []):
            aid = a.get("id", "?")
            agent_logs.setdefault(aid, []).append({
                "round": snap.get("step", 0),
                "action": a.get("action", "hold"),
                "requested": a.get("requested", 0),
                "filled": a.get("filled", 0),
                "reasoning": a.get("reasoning", ""),
                # Per-stock breakdown so the report can show each stock's
                # own reasoning instead of the pipe-joined aggregate.
                "actions": a.get("actions", []),
                "wealth": a.get("wealth", 0),
                "delta": a.get("delta"),
                "mood": a.get("mood"),
            })
            mood = a.get("mood")
            if isinstance(mood, dict) and mood:
                mood_history.setdefault(aid, []).append({
                    "step": snap.get("step", 0),
                    **{k: float(v) for k, v in mood.items()},
                })

    buy_trades = [t for t in env.trade_history if t.action == "buy"]
    sell_trades = [t for t in env.trade_history if t.action == "sell"]
    trade_summary = {
        "total": len(env.trade_history),
        "buys": len(buy_trades),
        "sells": len(sell_trades),
    }

    event_history = (
        list(env.event_manager.event_history) if env.event_manager else []
    )

    usage = collect_usage(list(env.agents.values()))
    if metadata is not None:
        # Fold the final numbers into the run snapshot too, so a saved
        # metadata.json and its report agree on what the run cost.
        metadata.attach_usage(list(env.agents.values()))

    html_text = generate_report_html(
        run_id=run_id,
        seed=seed,
        total_steps=state.get("steps", 0),
        steps_completed=state.get("current_step", 0),
        stocks=stocks,
        rankings=rankings,
        wealth_history=wealth_history,
        event_history=event_history,
        agent_logs=agent_logs,
        trade_summary=trade_summary,
        mood_history=mood_history,
        usage=usage,
        prompt_info={
            "template_version": PROMPT_TEMPLATE_VERSION,
            "fingerprints": shipped_fingerprints(),
        },
    )
    save_report(html_text, run_id, reports_dir=REPORTS_DIR)
    return jsonify({
        "ok": True,
        "run_id": run_id,
        "url": f"/report/{run_id}",
        "download_url": f"/report/{run_id}?download=1",
    })


@app.route("/report/<run_id>")
def serve_report(run_id: str):
    """Serve an exported read-only report snapshot."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_id):
        return jsonify({"error": "Invalid report id"}), 400
    path = os.path.join(REPORTS_DIR, f"{run_id}.html")
    if not os.path.isfile(path):
        return jsonify({"error": "Report not found"}), 404
    as_attachment = request.args.get("download") == "1"
    return send_file(path, mimetype="text/html", as_attachment=as_attachment)


@app.route("/api/god/event", methods=["POST"])
def api_god_event():
    """Inject a forced market event or custom news (God Mode)."""
    state = _get_session()
    if "env" not in state:
        return jsonify({"error": "No active simulation"}), 400

    data = request.get_json(silent=True) or {}
    event_name = data.get("event_name", "custom_news")
    # None (or omitted) -> the template's own description is used.
    description = data.get("description")
    price_impact, error, status = _parse_float(
        data.get("price_impact", 0.05), "price_impact", minimum=-1.0, maximum=1.0
    )
    if error:
        return error, status

    env = state["env"]
    if env.event_manager is not None:
        event = env.event_manager.force_trigger_event(
            name=event_name,
            step=env.step_count,
            custom_desc=description,
            price_impact=price_impact,
        )
        return jsonify({
            "ok": True,
            "event": {
                "name": event.name,
                "description": event.description,
                "type": event.event_type.value,
                "price_impact": event.price_impact,
                "scope": event.scope,
                "stock": event.target_stock,
            },
        })
    return jsonify({"error": "Event manager not initialized"}), 400


@app.route("/api/god/config", methods=["POST"])
def api_god_config():
    """Dynamically update market config parameters (God Mode)."""
    state = _get_session()
    if "env" not in state:
        return jsonify({"error": "No active simulation"}), 400

    data = request.get_json(silent=True) or {}
    env = state["env"]
    if "price_sensitivity" in data:
        val, error, status = _parse_float(
            data["price_sensitivity"], "price_sensitivity", minimum=0.0, maximum=1.0
        )
        if error:
            return error, status
        env.config.price_sensitivity = val
    if "max_price_change_ratio" in data:
        val, error, status = _parse_float(
            data["max_price_change_ratio"], "max_price_change_ratio", minimum=0.0, maximum=1.0
        )
        if error:
            return error, status
        env.config.max_price_change_ratio = val
    # Control random events: event_multiplier 0 disables, >0 enables with scale
    if "event_multiplier" in data:
        val, error, status = _parse_float(
            data["event_multiplier"], "event_multiplier", minimum=0.0, maximum=10.0
        )
        if error:
            return error, status
        env.config.event_probability_multiplier = val
        if env.event_manager is not None:
            env.event_manager.multiplier = val
    # Persistent sentiment drift added to every agent observation.
    if "sentiment_drift" in data:
        try:
            env._sentiment_drift = max(-1.0, min(1.0, float(data["sentiment_drift"])))
        except (TypeError, ValueError):
            pass
    # Runtime social-influence strength override (0 = off, 1 = strong).
    if "social_influence" in data:
        val, error, status = _parse_float(
            data["social_influence"], "social_influence", minimum=0.0, maximum=1.0
        )
        if error:
            return error, status
        env._social_influence = val
    return jsonify({
        "ok": True,
        "price_sensitivity": env.config.price_sensitivity,
        "max_price_change_ratio": env.config.max_price_change_ratio,
        "event_multiplier": env.config.event_probability_multiplier,
        "sentiment_drift": getattr(env, "_sentiment_drift", 0.0),
        "social_influence": getattr(
            env, "_social_influence", getattr(env.config, "social_influence", 0.0)
        ),
    })


@app.route("/api/events/list", methods=["GET"])
def api_events_list():
    """Return all available market event templates for manual triggering."""
    events = []
    for t in EVENT_TEMPLATES:
        events.append({
            "name": t.name,
            "description": t.description,
            "type": t.event_type.value,
            "price_impact": t.price_impact,
            "duration": t.duration_steps,
        })
    return jsonify({"events": events})



@app.route("/api/agents/social", methods=["GET"])
def api_agents_social():
    """Return agent personality details and social relationships."""
    state = _get_session()
    if "env" not in state:
        return jsonify({"error": "No active simulation"}), 400

    env = state["env"]
    agents_info = []
    resolved = resolve_social_map(list(env.agents.values()))

    for aid, agent in env.agents.items():
        personality = agent_personality(agent)
        desc = agent_personality_desc(agent)

        # Persona state: the evolving mood and the fixed sensitivity dials.
        # Both are deep-mode only; simple runs report neither.
        deep = bool(getattr(agent, "deep", False))
        mood = (
            {k: round(float(v), 2) for k, v in getattr(agent, "mood", {}).items()}
            if deep else {}
        )
        dials = (
            {k: round(float(v), 2) for k, v in getattr(agent, "dials", {}).items()}
            if deep else {}
        )

        social = resolved.get(aid, {"idol": None, "friends": [], "enemies": []})

        agents_info.append({
            "id": aid,
            "type": agent_type_label(agent),
            "personality": personality,
            "personality_desc": desc,
            "mood": mood,
            "dials": dials,
            "idol": social.get("idol"),
            "friends": social.get("friends", []),
            "enemies": social.get("enemies", []),
        })

    return jsonify({"agents": agents_info})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chat with an agent in character (Web UI chat panel)."""
    state = _get_session()
    if "env" not in state:
        return jsonify({"error": "No active simulation"}), 400

    data = request.get_json(silent=True) or {}
    agent_id = data.get("agent_id", "")
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    env = state["env"]
    agent = env.agents.get(agent_id)
    if agent is None:
        return jsonify({"error": f"Unknown agent: {agent_id}"}), 404

    # Chat runs on the underlying ExternalAIAgent (TraitAgent wraps it).
    target = getattr(agent, "base_agent", agent)
    if not hasattr(target, "chat"):
        return jsonify({"error": "This agent has no AI backend to chat with"}), 400

    # Per-agent chat history kept server-side (last 20 turns).
    chats = state.setdefault("chats", {})
    history = list(chats.get(agent_id, []))

    try:
        # Rebuilt every message: the trader's standing, mood and memory are
        # only true for the round they were read in. Built inside the try so
        # a failure here returns the JSON error the frontend already handles
        # rather than a Flask HTML 500.
        persona = build_chat_system_prompt(
            env, agent_id, initial_wealths=state.get("initial_wealths") or {}
        )
        reply = target.chat(message, system_prompt=persona, history=history)
    except Exception as exc:  # missing key, network error, rate limit, etc.
        return jsonify({"error": str(exc)}), 500

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    chats[agent_id] = history[-20:]

    return jsonify({"reply": reply})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  AI TRADING SANDBOX — Web Dashboard")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    # Bind to loopback only: this app has no authentication and the
    # God-Mode / config endpoints are unprotected. Debug (the Werkzeug
    # debugger = RCE on any unhandled error) is opt-in via ATS_DEBUG=1.
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=os.environ.get("ATS_DEBUG") == "1",
    )


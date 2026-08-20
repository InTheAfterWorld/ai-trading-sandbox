"""
Web UI for AI Trading Society.

A Flask-based dashboard that runs the simulation step-by-step
with a modern visual interface.

Run:
    python run.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import uuid
from collections import OrderedDict

try:
    from flask import Flask, render_template, jsonify, request, session as flask_session
except ImportError:
    print("Flask is not installed. Install it with:")
    print("    pip install flask")
    sys.exit(1)

from ai_trading_society.config import MarketConfig
from ai_trading_society.config_store import load_config, save_config
from ai_trading_society.agents.roster import build_agent_roster, resolve_social_map
from ai_trading_society.agents.external_ai_agent import (
    ExternalAIAgent,
    _DEFAULT_MODELS,
)
from ai_trading_society.agents.traits import _PERSONALITY_DESCRIPTIONS
from ai_trading_society.market_env import MarketEnv
from ai_trading_society.market_events import EVENT_TEMPLATES, EventType
from ai_trading_society.simulator import Simulator
from ai_trading_society.console_utils import (
    agent_type_label,
    agent_personality,
    agent_personality_desc,
)

app = Flask(__name__, template_folder="../../templates", static_folder="../../static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ai-trading-society-local-secret")

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


def _parse_float(value, field: str, *, default: float = 0.0, minimum: float = 0.0):
    """Parse a numeric API value (int/float/numeric string) into a float."""
    if isinstance(value, bool) or value is None:
        return default, None, None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, jsonify({"error": f"{field} must be a number."}), 400
    if parsed < minimum:
        return None, jsonify({"error": f"{field} must be at least {minimum}."}), 400
    return parsed, None, None



@app.errorhandler(Exception)
def handle_exception(e):
    """Ensure API errors return JSON instead of HTML error pages."""
    code = getattr(e, "code", 500)
    return jsonify({"error": str(e)}), code


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the home/configuration page."""
    return render_template("index.html")


@app.route("/sim")
def sim():
    """Serve the simulation dashboard page."""
    return render_template("sim.html")



@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Return the saved user configuration (shared with CLI and the homepage)."""
    return jsonify({"config": load_config()})


@app.route("/api/config", methods=["POST"])
def api_save_config():
    """Persist the user configuration entered on the homepage.

    This file is the single source of truth: the CLI reads the same config
    so a simulation saved in the browser runs identically from the terminal.
    """
    data = request.get_json(silent=True) or {}
    saved = save_config(data)
    return jsonify({"ok": True, "config": saved})


@app.route("/api/start", methods=["POST"])
def api_start():
    """Initialize a new simulation session."""
    data = request.get_json(silent=True) or {}
    steps, error, status = _parse_int(data.get("steps", 25), "steps", minimum=1)
    if error:
        return error, status
    price, error, status = _parse_float(data.get("price", 100.0), "price", default=100.0, minimum=0.01)
    if error:
        return error, status
    cash, error, status = _parse_float(data.get("cash", 10000.0), "cash", default=10000.0, minimum=0.0)
    if error:
        return error, status
    hold, error, status = _parse_int(data.get("hold", 20), "hold", minimum=0)
    if error:
        return error, status
    fee, error, status = _parse_float(data.get("fee", 0.001), "fee", default=0.001, minimum=0.0)
    if error:
        return error, status
    slip, error, status = _parse_float(data.get("slip", 0.001), "slip", default=0.001, minimum=0.0)
    if error:
        return error, status
    provider = data.get("provider") or "openai"
    model = data.get("model") or _DEFAULT_MODELS.get(provider) or "gpt-4o"
    api_key = data.get("api_key", "")
    trader_configs = data.get("traders")
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except ValueError:
            seed = None

    config = MarketConfig(
        initial_price=price,
        price_sensitivity=0.02,
        max_price_change_ratio=0.10,
        fee_rate=fee,
        slippage_rate=slip,
        event_probability_multiplier=1.5,
        random_traits=True,
        seed=seed,
    )

    try:
        agents, player_agent = build_agent_roster(
            provider=provider, model=model, api_key=api_key,
            trader_configs=trader_configs if isinstance(trader_configs, list) else None,
            cash=cash,
            holdings=hold,
        )
    except RuntimeError as exc:
        print(f"[api/start] build_agent_roster failed: {exc}")
        return jsonify({"error": str(exc)}), 400

    env = MarketEnv(config, agents)
    sim = Simulator(env)

    # Wire up the always-present player agent so it can read buffered actions.
    player_agent._env = env


    # Create run metadata
    from ai_trading_society.run_metadata import RunMetadata
    metadata = RunMetadata.create(config=config, agents=agents, seed=seed)
    sim.metadata = metadata

    # Capture initial wealths.
    initial_wealths = {}
    for aid, a in env.agents.items():
        initial_wealths[aid] = a.cash + a.holdings * env.price

    state = _get_session()
    state.clear()
    state["env"] = env
    state["sim"] = sim
    state["steps"] = steps
    state["current_step"] = 0
    state["initial_wealths"] = initial_wealths
    state["prev_wealths"] = dict(initial_wealths)
    state["is_player_mode"] = True

    # Build agent roster for the frontend.
    roster = []
    for aid, agent in env.agents.items():
        roster.append({
            "id": aid,
            "type": agent_type_label(agent),
            "personality": agent_personality(agent),
            "personality_desc": agent_personality_desc(agent),
            "cash": round(agent.cash, 2),
            "holdings": agent.holdings,
            "wealth": round(agent.cash + agent.holdings * env.price, 2),
            "initial_wealth": round(initial_wealths[aid], 2),
        })

    return jsonify({
        "roster": roster,
        "steps": steps,
        "initial_price": config.initial_price,
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

    env = state["env"]
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
    agents_data = []
    for aid, agent in env.agents.items():
        act_info = actions.get(aid, {})
        action = act_info.get("action", "hold")
        req = act_info.get("requested_qty", 0)
        filled = act_info.get("filled_qty", 0)
        reasoning = act_info.get("reasoning", "")

        wealth = agent.cash + agent.holdings * price
        init_w = state["initial_wealths"].get(aid, wealth)
        ret_pct = (wealth / init_w - 1) * 100 if init_w > 0 else 0.0
        prev_w = state["prev_wealths"].get(aid, wealth)
        delta = wealth - prev_w

        agents_data.append({
            "id": aid,
            "type": agent_type_label(agent),
            "personality": agent_personality(agent),
            "action": action,
            "requested": req,
            "filled": filled,
            "reasoning": reasoning,
            "error": bool(act_info.get("error", False)),
            "cash": round(agent.cash, 2),
            "holdings": agent.holdings,
            "wealth": round(wealth, 2),
            "return_pct": round(ret_pct, 2),
            "delta": round(delta, 2),
        })

    # Update prev_wealths.
    for aid, agent in env.agents.items():
        state["prev_wealths"][aid] = agent.cash + agent.holdings * price

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
        "events": [
            {
                "name": e.get("name", ""),
                "description": e.get("description", ""),
                "price_impact": e.get("price_impact", 0.0),
            }
            for e in triggered_events
        ],
        "active_events": [
            {"name": e.get("name", ""), "remaining": e.get("remaining_steps", 0), "total": e.get("total_steps", 1)}
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
    data = request.get_json(silent=True) or {}
    action = data.get("action", "hold")
    quantity, error, status = _parse_int(data.get("quantity", 0), "quantity")
    if error:
        return error, status

    if action not in ("buy", "sell", "hold"):
        return jsonify({"error": "Invalid action"}), 400

    env = state["env"]
    env.set_player_action(action, quantity)
    return jsonify({"ok": True})


@app.route("/api/results", methods=["GET"])
def api_results():
    """Return the final simulation report."""
    state = _get_session()
    if "sim" not in state:
        return jsonify({"error": "No active simulation"}), 400

    sim = state["sim"]
    env = state["env"]
    initial_price = env.config.initial_price

    agents = list(env.agents.values())
    ranked = sorted(agents, key=lambda a: a.cash + a.holdings * env.price, reverse=True)

    rankings = []
    for rank, agent in enumerate(ranked, 1):
        wealth = agent.cash + agent.holdings * env.price
        init_w = state["initial_wealths"].get(agent.agent_id, wealth)
        ret = (wealth / init_w - 1) * 100 if init_w > 0 else 0.0
        metrics = sim._compute_agent_metrics(agent.agent_id)

        rankings.append({
            "rank": rank,
            "id": agent.agent_id,
            "type": agent_type_label(agent),
            "personality": agent_personality(agent),
            "cash": round(agent.cash, 2),
            "holdings": agent.holdings,
            "wealth": round(wealth, 2),
            "return_pct": round(ret, 2),
            "sharpe": round(metrics["sharpe"], 2),
            "max_drawdown": round(metrics["max_drawdown"] * 100, 2),
            "volatility": round(metrics["volatility"] * 100, 2),
            "win_rate": round(metrics["win_rate"] * 100, 1),
        })

    buy_trades = [t for t in env.trade_history if t.action == "buy"]
    sell_trades = [t for t in env.trade_history if t.action == "sell"]
    avg_vol = sum(env.volume_history) / len(env.volume_history) if env.volume_history else 0

    # Build wealth history for chart.
    wealth_history = {}
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
    price_impact = float(data.get("price_impact", 0.05))

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
        env.config.price_sensitivity = float(data["price_sensitivity"])
    if "max_price_change_ratio" in data:
        env.config.max_price_change_ratio = float(data["max_price_change_ratio"])
    # Control random events: event_multiplier 0 disables, >0 enables with scale
    if "event_multiplier" in data:
        mult = max(0.0, float(data["event_multiplier"]))
        env.config.event_probability_multiplier = mult
        if env.event_manager is not None:
            env.event_manager.multiplier = mult
    # Persistent sentiment drift added to every agent observation.
    if "sentiment_drift" in data:
        try:
            env._sentiment_drift = max(-1.0, min(1.0, float(data["sentiment_drift"])))
        except (TypeError, ValueError):
            pass
    return jsonify({
        "ok": True,
        "price_sensitivity": env.config.price_sensitivity,
        "max_price_change_ratio": env.config.max_price_change_ratio,
        "event_multiplier": env.config.event_probability_multiplier,
        "sentiment_drift": getattr(env, "_sentiment_drift", 0.0),
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

        # Extract trait values if available
        traits = {}
        for t in ("panic", "greed", "fomo", "stubbornness", "loss_aversion", "overconfidence", "regret_avoidance"):
            val = getattr(agent, t, None)
            if val is not None and val > 0:
                traits[t] = round(float(val), 2)

        social = resolved.get(aid, {"idol": None, "friends": [], "enemies": []})

        agents_info.append({
            "id": aid,
            "type": agent_type_label(agent),
            "personality": personality,
            "personality_desc": desc,
            "traits": traits,
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

    wealth = agent.cash + agent.holdings * env.price
    init_w = state["initial_wealths"].get(agent_id, wealth)
    ret = (wealth / init_w - 1) * 100 if init_w > 0 else 0.0

    persona = (
        f"You are '{agent_id}', a trader in the AI Trading Society market sandbox. "
        f"Your trading personality: {agent_personality(agent)}. "
        f"{agent_personality_desc(agent)} "
        f"Current situation — price: ${env.price:.2f}, your cash: ${agent.cash:.0f}, "
        f"holdings: {agent.holdings} shares, total wealth: ${wealth:.0f} "
        f"({ret:+.1f}% return so far). "
        "Reply in character, conversationally, in 2-4 sentences. "
        "Do not output JSON."
    )

    # Per-agent chat history kept server-side (last 20 turns).
    chats = state.setdefault("chats", {})
    history = list(chats.get(agent_id, []))

    try:
        reply = target.chat(message, system_prompt=persona, history=history)
    except Exception as exc:  # missing key, network error, rate limit, etc.
        return jsonify({"error": str(exc)}), 500

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    chats[agent_id] = history[-20:]

    return jsonify({"reply": reply})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  AI Trading Society — Web Dashboard")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)


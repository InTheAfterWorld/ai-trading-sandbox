"""The background a trader carries into a chat with a human.

The dashboard's chat panel used to hand the model four facts: an id, a
personality name, a canned preset label and a balance sheet. So the trader
could discuss its cash and nothing else -- not the character it was written
with, not the traders it idolises or resents, not what it just did or why.

This module builds that missing background, fresh on every message, because
standing, mood and memory are only true for the round they were read in.

Read-only, deliberately. The briefing flows *into* the model as a system
prompt and nothing flows back: no chat turn ever reaches
``_conversation_history``, ``_short_term_memory``, ``_lessons`` or
``_position_plans``. A human cannot talk a trader into a position, so a run
stays explainable from its config, seed and prompt version alone -- the
black-box property ``MarketEnv.get_observation`` documents.

The renderers are the same ones the decision prompt uses, so the character a
trader shows in chat is provably the character it trades on. One exception
is load-bearing: ``ExternalAIAgent._build_prompt`` must never be called from
here. It appends to ``_market_history`` and records key events, so rendering
a briefing through it would quietly corrupt the agent's long-horizon view of
the market with every message the human sent.
"""

from typing import Any, Dict, List, Optional, Tuple

from .agents.roster import resolve_social_map
from .agents.traits import character_text
from .console_utils import agent_personality, agent_personality_desc

# Size caps. The briefing is rebuilt and re-sent on every message, on top of
# up to 20 stored turns, and chat spend is metered separately (agents tag
# these calls "chat"), so the cost of being generous here is visible and
# recurring. A typical build lands around 2-3k characters.
_MAX_PEERS = 5
_MAX_PEER_QUOTE_CHARS = 120
_MAX_STOCK_LINES = 8
_MAX_EVENTS = 4
_MAX_LESSONS = 3
_MAX_BRIEFING_CHARS = 6000

# Section priorities. 0 never gets trimmed -- identity, standing and the
# social graph are the whole point of the feature. The rest are dropped
# whole, highest number first, until the briefing fits.
_P_KEEP = 0
_P_MARKET_STATE = 1
_P_MEMORY = 2
_P_LESSONS = 3
_P_PLANS = 4
_P_MARKET_SUMMARY = 5

_SCALE_HINT = "0 = none at all, 10 = as strong as it gets"

# Peer lines describe a round that has already happened.
_PAST_TENSE = {"buy": "bought", "sell": "sold"}

# Replaces the decision prompt's closing check, which talks about matching a
# reasoning to a trade and means nothing in a conversation.
_CHAT_STYLE_RULES = (
    "=== HOW TO REPLY ===\n"
    "You are talking to a human visitor watching this market on a "
    "dashboard. Stay in character as the trader described above. You may "
    "talk about the other traders by name and about how you feel toward "
    "them. Answer conversationally in 2-4 sentences. Do not output JSON.\n"
    "Earlier turns in this conversation may date from earlier rounds -- the "
    "numbers above are the current ones."
)


def _base(agent: Any) -> Any:
    """The ExternalAIAgent under a persona wrapper, or the agent itself."""
    return getattr(agent, "base_agent", agent)


def _render(agent: Any, name: str, *args: Any) -> str:
    """Call one of the agent's prompt renderers, tolerating its absence.

    A roster can hold a PlayerAgent or a bare BaseAgent test double, neither
    of which has the ExternalAIAgent renderers. Chat should render less for
    those, not fail.
    """
    fn = getattr(_base(agent), name, None)
    if not callable(fn):
        return ""
    try:
        out = fn(*args)
    except Exception:
        return ""
    if isinstance(out, list):
        return "\n".join(str(line) for line in out).strip("\n")
    return str(out or "")


def agent_relations(env: Any, agent_id: str) -> Dict[str, Any]:
    """This agent's idol / friends / enemies, as resolved roster ids.

    ``env.social_map`` is populated by the web and CLI entry points but
    defaults to empty, so an env built directly -- in a test, or by a
    library user -- falls back to resolving the map on the spot. It is
    O(n^2) over a roster of at most a handful of agents.
    """
    social_map = getattr(env, "social_map", None) or {}
    relations = social_map.get(agent_id)
    if relations is None:
        try:
            relations = resolve_social_map(
                list(env.agents.values())
            ).get(agent_id)
        except Exception:
            relations = None
    if not isinstance(relations, dict):
        return {"idol": None, "friends": [], "enemies": []}
    return relations


def _standing(env: Any, agent_id: str) -> Dict[str, Any]:
    """Rank and leader gap for one agent, in both simple and deep mode.

    Read directly rather than from the observation: ``get_observation``
    attaches ``standing`` only in deep mode, but a trader should be able to
    say where it is placed whichever mode the run is in.
    """
    fn = getattr(env, "standing_for", None) or getattr(env, "_standing_for", None)
    if not callable(fn):
        return {}
    try:
        out = fn(agent_id)
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def describe_peer(
    env: Any,
    peer_id: str,
    relation: str,
    initial_wealths: Optional[Dict[str, float]] = None,
) -> str:
    """One line about a trader this agent has a relationship with.

    Carries enough for the agent to actually talk about them: who they are,
    how they are doing, and what they last did and said. The quote is
    included in both modes, unlike the decision prompt which gates peer
    reasoning on deep mode -- chat is roleplay, not a decision input, so the
    token economy that motivates the gate does not apply.
    """
    peer = (getattr(env, "agents", None) or {}).get(peer_id)
    if peer is None:
        return ""

    label = agent_personality_desc(peer) or agent_personality(peer) or "trader"
    lead = f"- {relation}: {peer_id} ({label})."

    standing = _standing(env, peer_id)
    if standing.get("of"):
        lead += (
            f" {standing.get('rank')} of {standing.get('of')} at "
            f"{float(standing.get('my_return_pct', 0.0)):+.1f}%."
        )

    recent = (getattr(env, "_recent_actions", None) or {}).get(peer_id)
    if not isinstance(recent, dict):
        return lead

    action = str(recent.get("action", "hold"))
    if action == "hold":
        tail = "  Last round they sat on their hands."
    else:
        verb = _PAST_TENSE.get(action, action)
        tail = f"  Last round they {verb} {int(recent.get('filled', 0) or 0)} shares."
    quote = str(recent.get("reasoning", "") or "").strip()
    if quote:
        if len(quote) > _MAX_PEER_QUOTE_CHARS:
            quote = quote[:_MAX_PEER_QUOTE_CHARS].rstrip() + "..."
        tail += f' They said: "{quote}"'
    return f"{lead}\n{tail}"


def _peer_section(
    env: Any,
    agent_id: str,
    initial_wealths: Optional[Dict[str, float]],
    max_peers: int,
) -> str:
    """The social graph, by name. Dropped entirely when there is none.

    Built from the relationship map rather than the observation's
    ``social_peers``, which is filtered to peers that traded last round --
    an agent would forget its idol on any quiet round.
    """
    relations = agent_relations(env, agent_id)
    ordered: List[Tuple[str, str]] = []
    idol = relations.get("idol")
    if idol:
        ordered.append(("Your idol", str(idol)))
    for friend in relations.get("friends") or []:
        ordered.append(("Friend", str(friend)))
    for enemy in relations.get("enemies") or []:
        ordered.append(("Enemy", str(enemy)))

    lines: List[str] = []
    seen: set = {agent_id}
    for relation, peer_id in ordered:
        if peer_id in seen or len(lines) >= max_peers:
            continue
        seen.add(peer_id)
        rendered = describe_peer(env, peer_id, relation, initial_wealths)
        if rendered:
            lines.append(rendered)
    if not lines:
        return ""
    return "=== THE PEOPLE AROUND YOU ===\n" + "\n".join(lines)


def _identity_section(agent: Any, agent_id: str) -> str:
    """WHO YOU ARE, and HOW YOU FEEL when the run tracks mood.

    Rendered through the decision prompt's own ``_persona_lines`` so the two
    prompts cannot drift apart. Mood is passed only in deep mode, which is
    how ``TraitAgent.act`` gates it -- omitting the key makes the renderer
    skip the block.
    """
    # The name first: _persona_lines renders the disposition only, and the
    # disposition never says what the trader is called, so without this the
    # agent cannot answer "who are you?" with its own name.
    label = agent_personality_desc(agent) or agent_personality(agent)
    name_line = f"You are '{agent_id}', a trader in this market sandbox."
    if label:
        name_line += f" {label}"

    body = character_text(str(getattr(agent, "disposition", "") or ""))
    # No persona layer (a bare ExternalAIAgent, or a directly built
    # TraitAgent) leaves only the name line -- what chat used to send.
    character = f"{name_line}\n\n{body}" if body else name_line

    persona: Dict[str, Any] = {
        "name": agent_id,
        "disposition": character,
        "scale_hint": _SCALE_HINT,
    }
    if getattr(agent, "deep", False):
        mood = getattr(agent, "mood", None)
        if isinstance(mood, dict) and mood:
            persona["mood"] = dict(mood)

    rendered = _render(agent, "_persona_lines", {"persona": persona})
    if rendered:
        return rendered
    # An agent with no ExternalAIAgent behind it (the human player, a test
    # double) has no renderer to borrow. Say who it is anyway rather than
    # opening the briefing on a balance sheet.
    return f"=== WHO YOU ARE ===\n{character}"


def _portfolio_line(obs: Dict[str, Any]) -> str:
    """Cash and per-stock position, compact enough to sit inside chat."""
    parts = [f"${float(obs.get('my_cash', 0.0)):,.0f} cash"]
    for stock in (obs.get("stocks") or [])[:_MAX_STOCK_LINES]:
        held = float(stock.get("my_holdings", 0) or 0)
        if held <= 0:
            continue
        parts.append(
            f"{stock.get('name') or stock.get('symbol')} {held:g} @ "
            f"${float(stock.get('price', 0) or 0):,.2f}"
        )
    return "- Portfolio: " + "; ".join(parts)


def _standing_section(agent: Any, obs: Dict[str, Any]) -> str:
    """Rank, stakes, concentration, exposure, floor mood, portfolio.

    Every renderer here already returns "" when its data is missing, so a
    round-zero or simple-mode run simply produces fewer lines.
    """
    lines = [
        _render(agent, "_standing_line", obs),
        _render(agent, "_stakes_line", obs),
        _render(agent, "_concentration_line", obs),
        _render(agent, "_exposure_line", obs),
        _render(agent, "_floor_mood_line", obs),
        _portfolio_line(obs),
    ]
    body = "\n".join(line for line in lines if line)
    if not body:
        return ""
    return "=== WHERE YOU STAND ===\n" + body


def _market_section(obs: Dict[str, Any]) -> str:
    """Prices, sentiment and live events, one compact line each.

    Not the decision prompt's stock table: that carries sector, blurb,
    volume and a price window per stock, which is most of a chat's budget
    spent on something the human can already see on screen.
    """
    lines = []
    bits = []
    for stock in (obs.get("stocks") or [])[:_MAX_STOCK_LINES]:
        bits.append(
            f"{stock.get('name') or stock.get('symbol')} "
            f"${float(stock.get('price', 0) or 0):,.2f} "
            f"({float(stock.get('move_since_last_pct', 0.0) or 0.0):+.1f}%)"
        )
    if bits:
        lines.append("- " + " · ".join(bits))

    sentiment = float(obs.get("market_sentiment", 0.0) or 0.0)
    if sentiment:
        lines.append(f"- Sentiment: {sentiment:+.2f} (-1 bearish, +1 bullish)")

    events = obs.get("active_events") or []
    if events:
        named = [
            f"{e.get('name', '?')} ({e.get('stock') or 'market-wide'})"
            for e in events[:_MAX_EVENTS]
            if isinstance(e, dict)
        ]
        if named:
            lines.append("- Active: " + ", ".join(named))

    if not lines:
        return ""
    return (
        f"=== THE MARKET RIGHT NOW (round {int(obs.get('step', 0) or 0)}) ===\n"
        + "\n".join(lines)
    )


def _lessons_section(agent: Any) -> str:
    """The trader's own lessons, capped to the most recent few."""
    rendered = _render(agent, "_lesson_lines")
    if not rendered:
        return ""
    lines = [line for line in rendered.split("\n") if line.strip()]
    if len(lines) <= 1:
        return ""
    header, entries = lines[0], lines[1:]
    return "\n".join([header, *entries[-_MAX_LESSONS:]])


def build_chat_system_prompt(
    env: Any,
    agent_id: str,
    *,
    initial_wealths: Optional[Dict[str, float]] = None,
    max_peers: int = _MAX_PEERS,
) -> str:
    """Assemble the system prompt for one chat message.

    Rebuilt per message on purpose: a briefing cached across rounds would
    describe a trader that no longer exists, and God Mode can change the
    market mid-round.

    Raises
    ------
    ValueError
        If ``agent_id`` is not in the roster. ``get_observation`` would
        otherwise raise a bare ``KeyError`` from deep inside the engine.
    """
    agents = getattr(env, "agents", None) or {}
    if agent_id not in agents:
        raise ValueError(f"unknown agent: {agent_id}")
    agent = agents[agent_id]

    # The observation is freshly built and never handed to the trading loop,
    # so this copy is ours to annotate. _concentration_line renders only
    # when a persona is present, and _standing_line needs a standing that
    # simple mode does not attach.
    obs: Dict[str, Any] = dict(env.get_observation(agent_id))
    obs["persona"] = {"name": agent_id}
    standing = _standing(env, agent_id)
    if standing:
        obs["standing"] = standing
    if initial_wealths and agent_id in initial_wealths:
        obs["initial_wealth"] = initial_wealths[agent_id]

    sections: List[Tuple[int, str]] = [
        (_P_KEEP, _identity_section(agent, agent_id)),
        (_P_KEEP, _standing_section(agent, obs)),
        (_P_KEEP, _peer_section(env, agent_id, initial_wealths, max_peers)),
        (_P_MARKET_STATE, _market_section(obs)),
        (_P_MEMORY, _render(agent, "_build_memory_context", {})),
        (_P_LESSONS, _lessons_section(agent)),
        (_P_PLANS, _render(agent, "_plan_lines", obs)),
        (_P_MARKET_SUMMARY, _render(agent, "_build_market_summary")),
    ]
    kept = [(p, text.strip()) for p, text in sections if text and text.strip()]

    # Trim whole sections, least important first, until the body fits. The
    # reply rules are appended afterwards so they can never be trimmed away.
    def _body(blocks: List[Tuple[int, str]]) -> str:
        return "\n\n".join(text for _, text in blocks)

    for priority in (
        _P_MARKET_SUMMARY, _P_PLANS, _P_LESSONS, _P_MEMORY, _P_MARKET_STATE,
    ):
        if len(_body(kept)) <= _MAX_BRIEFING_CHARS:
            break
        kept = [(p, text) for p, text in kept if p != priority]

    return f"{_body(kept)}\n\n{_CHAT_STYLE_RULES}"

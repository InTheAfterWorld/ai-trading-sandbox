"""
Unified Agent Roster Factory.

Centralizes the creation and setup of trading agents for both Web UI and CLI modes.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..base_agent import BaseAgent
from ..config import StockSpec
from .external_ai_agent import _DEFAULT_MODELS, ExternalAIAgent
from .player_agent import PlayerAgent
from .traits import create_personality_agent

# Standard AI model roster definitions
DEFAULT_AI_MODELS = [
    # (model_name,             personality)
    ("gpt-oss-20b",            "aggressive"),
    ("gemini-3.1-flash-lite",  "conservative"),
    ("gemini-3.6-flash",      "panicky"),
    ("gemini-3.5-flash-lite", "greedy"),
    ("gemini-3.1-pro",        "fomo_driven"),
    ("gemini-3-flash",        "stubborn"),
    ("deepseek-r1",           "emotional"),
]

# Agent IDs in order of DEFAULT_AI_MODELS (derived, never manually duplicated)
AGENT_IDS = [model for model, _ in DEFAULT_AI_MODELS]

# Social relationships between agents: each agent has idol, friends, enemies
# Keys are personality names, values are dicts with 'idol', 'friends', 'enemies' (agent IDs)
SOCIAL_MAP = {
    "aggressive": {
        "idol": "deepseek-r1",
        "friends": ["gemini-3.5-flash-lite"],
        "enemies": ["gemini-3.1-flash-lite"],
    },
    "conservative": {
        "idol": "gemini-3.6-flash",
        "friends": ["gemini-3.6-flash", "gemini-3-flash"],
        "enemies": ["gpt-oss-20b"],
    },
    "panicky": {
        "idol": "gemini-3.1-flash-lite",
        "friends": ["gemini-3.1-flash-lite", "deepseek-r1"],
        "enemies": ["gpt-oss-20b", "gemini-3.1-pro"],
    },
    "greedy": {
        "idol": "gpt-oss-20b",
        "friends": ["gpt-oss-20b", "gemini-3.1-pro"],
        "enemies": ["gemini-3.6-flash"],
    },
    "fomo_driven": {
        "idol": "gemini-3.5-flash-lite",
        "friends": ["gpt-oss-20b", "gemini-3.5-flash-lite"],
        "enemies": ["gemini-3-flash"],
    },
    "stubborn": {
        "idol": "deepseek-r1",
        "friends": ["gemini-3.1-flash-lite"],
        "enemies": ["gemini-3.1-pro", "gemini-3-flash"],
    },
    "emotional": {
        "idol": "gemini-3.6-flash",
        "friends": ["gemini-3.6-flash", "gemini-3-flash", "deepseek-r1"],
        "enemies": [],
    },
}

# Maps the default roster model IDs (as referenced by SOCIAL_MAP) to the
# personality each agent had in the default lineup. Used to resolve social
# relationships when traders are custom-named on the homepage.
_DEFAULT_ID_PERSONALITY = {
    "gpt-oss-20b": "aggressive",
    "gemini-3.1-flash-lite": "conservative",
    "gemini-3.6-flash": "panicky",
    "gemini-3.5-flash-lite": "greedy",
    "gemini-3.1-pro": "fomo_driven",
    "gemini-3-flash": "stubborn",
    "deepseek-r1": "emotional",
}


def resolve_social_map(agents) -> dict:
    """
    Resolve SOCIAL_MAP relationships against the actual agent roster.

    SOCIAL_MAP references default model IDs (e.g. "gpt-oss-20b") that do not
    exist once traders are given custom names on the homepage. Each referenced
    default ID is mapped to its personality and then to a real agent in the
    roster that shares it. Relations that point at nobody are dropped, and a
    relation never points back at the agent itself.

    Returns a dict keyed by agent_id:
        {agent_id: {"idol": str | None, "friends": [...], "enemies": [...]}}
    """
    # personality -> real agent IDs in roster order
    by_personality: dict = {}
    for agent in agents:
        by_personality.setdefault(_agent_personality(agent), []).append(agent.agent_id)

    resolved: dict = {}
    for agent in agents:
        aid = agent.agent_id
        pkey = _agent_personality(agent) or "balanced"
        rel: dict[str, Any] = SOCIAL_MAP.get(
            pkey, {"idol": None, "friends": [], "enemies": []}
        )

        def _find(target_id, exclude=aid):
            if not target_id:
                return None
            target_personality = _DEFAULT_ID_PERSONALITY.get(target_id)
            for candidate in by_personality.get(target_personality, []):
                if candidate != exclude:
                    return candidate
            return None

        friends = []
        for fid in rel.get("friends", []):
            hit = _find(fid)
            if hit and hit not in friends:
                friends.append(hit)

        enemies = []
        for eid in rel.get("enemies", []):
            hit = _find(eid)
            if hit and hit not in enemies:
                enemies.append(hit)

        resolved[aid] = {
            "idol": _find(rel.get("idol")),
            "friends": friends,
            "enemies": enemies,
        }

    return resolved


def _agent_personality(agent) -> str:
    """Return the personality name for an agent, or empty string."""
    return getattr(agent, "personality_name", "") or ""


def _build_holdings_dict(
    stocks: Optional[List[StockSpec]], holdings: int
) -> Dict[str, float]:
    """Build a per-stock holdings dict from the stock specs.

    When ``stocks`` is None or empty, returns an empty dict so MarketEnv
    falls back to its legacy single-stock migration path.
    """
    if not stocks:
        return {}
    return {spec.name: float(holdings) for spec in stocks}


def build_agent_roster(
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    trader_configs: Optional[List[dict]] = None,
    cash: float = 10000.0,
    holdings: int = 20,
    stocks: Optional[List[StockSpec]] = None,
) -> Tuple[List[BaseAgent], PlayerAgent]:
    """
    Build and return the list of participating agents for a market simulation run.

    Parameters
    ----------
    provider : str
        API provider name (e.g. "openai", "google", "openrouter", etc.).
    model : str
        Model identifier string.
    api_key : str, optional
        Default API key used for traders that don't specify their own. Keys
        always come from the user's configuration, never the environment.
    cash : float
        Starting cash balance for every trader and the player.
    holdings : int
        Starting share holdings per stock for every trader and the player.
    stocks : list of StockSpec, optional
        Stock specifications. When provided, each agent's holdings is
        initialized as a ``{symbol: holdings}`` dict. When None, holdings
        stays as a legacy int that MarketEnv migrates to the first stock.
    Returns
    -------
    agents : list of BaseAgent
        All participating AI agents and the always-present PlayerAgent.
    player_agent : PlayerAgent
        Reference to the player agent, which may trade or simply observe.
    """
    holdings_dict = _build_holdings_dict(stocks, holdings)
    agents: List[BaseAgent] = []
    # Only fall back to the default roster when no trader list was provided
    # at all (None). An explicit empty list [] means "no AI traders" and must
    # be respected (e.g. a pure-observer / player-only run).
    if trader_configs is None:
        trader_configs = [
            {"name": f"Trade {index}", "provider": provider, "model": model, "api_key": api_key}
            for index, _ in enumerate(DEFAULT_AI_MODELS, 1)
        ]
    configs = trader_configs
    # Build exactly one AI agent per configured trader (defaulting to the
    # full default roster when no trader list is provided), so traders can be
    # added or removed from the homepage and the roster follows along.
    count = len(configs)
    for index in range(count):
        trader = configs[index] if index < len(configs) else {}
        model_name, default_personality = (
            DEFAULT_AI_MODELS[index]
            if index < len(DEFAULT_AI_MODELS)
            else ("gpt-4o", "balanced")
        )
        trader_provider = trader.get("provider") or provider
        trader_model = (
            trader.get("model")
            or _DEFAULT_MODELS.get(trader_provider)
            or model
            or model_name
        )
        trader_key = trader.get("api_key") or api_key
        trader_name = trader.get("name") or f"Trade {index + 1}"
        personality = trader.get("personality") or default_personality
        base = ExternalAIAgent(
            trader_name,
            cash=cash,
            holdings=holdings_dict,
            api_provider=trader_provider,
            model=trader_model,
            api_key=trader_key or None,
            base_url=trader.get("base_url") or None,
        )
        trait_agent = create_personality_agent(base, personality=personality)
        agents.append(trait_agent)

    player_agent = PlayerAgent(
        agent_id="Player (You)", cash=cash, holdings=holdings_dict
    )
    agents.append(player_agent)

    return agents, player_agent

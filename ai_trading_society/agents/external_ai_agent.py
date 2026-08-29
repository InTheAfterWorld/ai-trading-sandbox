"""
ExternalAIAgent provides an interface for external AI trading models.

Supports any OpenAI-compatible API (OpenAI, OpenRouter, ChatAnywhere,
Groq, Google Gemini via OpenAI-compat endpoint) as well as native
Anthropic and Google Gemini SDKs.

If no API key is available, or if an API call fails (rate limit, network
error, bad credentials), the agent raises an exception so the caller
can mark it as failed and display it accordingly.
"""

import json
import math
import re
import time
from typing import Any, Dict, List, Optional, cast

from ..base_agent import BaseAgent

# Provider presets: maps provider name → base_url.
# base_url=None means use the SDK's default endpoint.
# Providers with a base_url use the OpenAI-compatible chat completions API.
_PROVIDER_PRESETS: Dict[str, Optional[str]] = {
    # --- Native SDK providers (no base_url needed) ---
    "openai": None,
    "anthropic": None,
    "google": None,  # Uses google-generativeai SDK

    # --- OpenAI-compatible providers ---
    "openrouter": "https://openrouter.ai/api/v1",
    "chatanywhere": "https://api.chatanywhere.tech/v1",
    "groq": "https://api.groq.com/openai/v1",
    "arliai": "https://api.arliai.com/v1",
    # Google Gemini via OpenAI-compatible endpoint (recommended for newer models).
    "google_compat": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

# Provider presets: maps provider name → default model id.
# Used when a model is not explicitly configured so traders always resolve
# to a model the provider actually serves.
_DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
    "google": "gemini-3.5-flash-lite",
    "openrouter": "google/gemma-4-31b-it:free",
    "chatanywhere": "gpt-4o",
    "groq": "groq/compound-mini",
    "arliai": "Fastest",
    "google_compat": "gemini-3.5-flash-lite",
}

# Per-provider max output tokens. Reasoning models (Groq gpt-oss, ArliAI
# Qwen) consume tokens on "thinking", so they need a generous budget or the
# final content comes back empty. OpenRouter free tier caps output low and
# the account balance is often small, so keep it modest there.
_DEFAULT_MAX_TOKENS: Dict[str, int] = {
    "openrouter": 512,
    "groq": 2048,
    "arliai": 2048,
}
# Default output cap for all other providers: enough room for a multi-stock
# JSON decision list (each stock ≈ 80 tokens), small enough to stop a chatty
# model from writing essays. 2048 comfortably fits 5 stocks + reasoning.
_DEFAULT_MAX_TOKENS_FALLBACK = 2048

# How much reasoning to ask for per stock. The simple wording is the rule
# this project has always used; deep mode trades tokens for character.
_REASONING_DETAIL_SIMPLE = "1-2 short sentences, under 30 words"
_REASONING_DETAIL_DEEP = (
    "2-4 sentences when you buy or sell -- explain your thinking AND how you "
    "feel about it, in character; 1 sentence is enough for a hold"
)

# Deep mode invites a few extra OPTIONAL fields. Every one may be omitted:
# the parser drops anything missing or malformed, so a model that ignores
# all of them still produces a valid decision.
_DEEP_OPTIONAL_FIELDS = (
    "\n"
    "\n"
    "Optional extras (include them only if you want to; omit freely):\n"
    '- "mood": {"confidence": <0-10>, "stress": <0-10>, '
    '"frustration": <0-10>} alongside "decisions", saying how you feel '
    "AFTER deciding. Move each number by a few points at most.\n"
    '- "lesson": one short sentence you want to remember next round.\n'
    '- "stop_loss" / "target": prices on a buy or sell decision, if you are '
    "committing to an exit. Nothing executes them automatically -- you will "
    "be reminded next round and it is up to you to act."
)

# HTTP statuses worth exactly one automatic retry: rate limiting and
# transient server-side failures. Auth (401/403) and malformed requests
# (400) are excluded - they fail identically the second time.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Corrective re-asks (see _retry_with_escalation) each cost an extra API
# call. Cap how many a single agent may spend across a whole run so one
# badly-behaved model cannot multiply the cost and latency of every round.
_DEFAULT_REPAIR_BUDGET = 6


def _coerce_price(value: Any) -> Optional[float]:
    """Normalize an optional price ("$290", 290, "290.5") to a float.

    Returns None for anything uninterpretable, so a malformed stop or
    target is simply dropped rather than affecting the decision.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) and value > 0 else None
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            parsed = float(match.group())
            return parsed if parsed > 0 else None
    return None


def _error_status(exc: BaseException) -> Optional[int]:
    """Best-effort HTTP status code from a provider exception."""
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_transient_error(exc: BaseException) -> bool:
    """Whether an API failure is worth exactly one retry.

    A timeout, rate limit or 5xx is usually gone a second later. Without a
    retry it costs the agent its whole round: MarketEnv records an AI
    failure and forces a hold on every stock.
    """
    status = _error_status(exc)
    if status is not None and status in _RETRYABLE_STATUS:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
    }


# Phrases a provider uses when it rejects the structured-output parameter.
# Deliberately specific to response_format / JSON mode: generic wording like
# "unsupported parameter" also covers rejections that have nothing to do with
# structured output, and matching those would silently disable JSON mode (and
# burn a second request) over an unrelated BadRequestError.
_JSON_MODE_REJECTION_TOKENS = (
    "response_format",
    "json_object",
    "json_schema",
    "json mode",
    "structured output",
)


def _is_json_mode_rejection(exc: BaseException, openai_mod: Any) -> bool:
    """Whether an exception means the provider rejected JSON output mode.

    Providers signal this as a 400 naming the offending parameter. Every
    other failure -- 401, 429, timeout, 5xx -- must propagate instead: a
    blanket retry would double the load at the worst possible moment and
    permanently disable JSON mode on the strength of an unrelated error.
    """
    bad_request = getattr(openai_mod, "BadRequestError", None)
    is_bad_request = (
        (isinstance(bad_request, type) and isinstance(exc, bad_request))
        or _error_status(exc) == 400
        or type(exc).__name__ in {"BadRequestError", "UnprocessableEntityError"}
    )
    if not is_bad_request:
        return False
    text = str(exc).lower()
    return any(token in text for token in _JSON_MODE_REJECTION_TOKENS)


class ExternalAIAgent(BaseAgent):
    """
    External AI trading agent.

    Supports any OpenAI-compatible API (OpenAI, OpenRouter, ChatAnywhere,
    Groq, Google Gemini via OpenAI-compat endpoint) as well as native
    Anthropic and Google Gemini SDKs. The agent builds a natural-language
    prompt from the market observation, calls the provider API, and parses
    the JSON response into a trading action.

    If no API key is available, or if an API call fails, the agent
    raises an exception so the caller can mark it as failed.

    Parameters
    ----------
    agent_id : str
        Agent identifier.
    api_provider : str
        API provider. Built-in presets:
        - "openai"         : OpenAI (default endpoint)
        - "anthropic"      : Anthropic Claude (native SDK)
        - "google"         : Google Gemini (native google-generativeai SDK)
        - "openrouter"     : OpenRouter (OpenAI-compatible)
        - "chatanywhere"   : ChatAnywhere (OpenAI-compatible)
        - "groq"           : Groq (OpenAI-compatible)
        - "google_compat"  : Google Gemini via OpenAI-compatible endpoint
        Any other string is treated as a custom OpenAI-compatible provider
        when base_url is also provided.
    model : str
        Model name, such as "gpt-4o", "openai/gpt-oss-20b:free",
        "deepseek-r1", or "gemini-3.5-flash-lite".
    api_key : str, optional
        API key. Must be provided explicitly; it is never read from the
        environment or a `.env` file.
    base_url : str, optional
        Custom API base URL for OpenAI-compatible providers. When provided,
        overrides the preset base_url and forces the OpenAI-compatible code
        path regardless of api_provider.
    system_prompt : str, optional
        Custom system prompt. A default prompt is used when omitted.
    temperature : float
        Model temperature. 0.0 is deterministic, 1.0 is more exploratory.
    position_ratio : float
        Fraction of cash/holdings used per trade (0.0-1.0).
    """

    def __init__(
        self,
        agent_id: str,
        cash: float = 10000.0,
        holdings: int | Dict[str, float] = 0,
        api_provider: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        position_ratio: float = 0.2,
        memory_window: int = 6,
        enable_memory: bool = True,
        max_tokens: Optional[int] = None,
        repair_budget: int = _DEFAULT_REPAIR_BUDGET,
    ):
        super().__init__(agent_id, cash, holdings)
        self.api_provider = api_provider
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.position_ratio = position_ratio
        self.max_tokens = max_tokens or _DEFAULT_MAX_TOKENS.get(
            api_provider, _DEFAULT_MAX_TOKENS_FALLBACK
        )

        # --- Memory system ---
        # Conversation history stores dicts: {"role": "user"|"assistant",
        # "content": str}. Only the most recent `memory_window` entries
        # are kept to control token consumption.
        self.enable_memory = enable_memory
        self.memory_window = memory_window
        self._conversation_history: list[Dict[str, str]] = []
        # Track the last action we took, for inclusion in the next prompt.
        self._last_action: Optional[Dict[str, Any]] = None
        # Rolling market summary: short list of (step, price) tuples for
        # trend detection across the full simulation, not just the 10-step
        # window provided in the observation.
        self._market_history: list[tuple[int, float]] = []
        # SHORT-TERM memory: one-line summaries of the agent's decisions in
        # the most recent rounds (capped at memory_window entries).
        self._short_term_memory: List[str] = []
        # Lessons the model wrote for itself, replayed in later prompts.
        self._lessons: List[str] = []
        # Exit levels the model committed to per stock: {sym: {stop_loss,
        # target}}. Replayed as a reminder next round -- never auto-executed.
        self._position_plans: Dict[str, Dict[str, float]] = {}
        # LONG-TERM memory: significant market events the agent has lived
        # through (|price impact| >= 5%), kept for the whole run.
        self._key_events: List[Dict[str, Any]] = []

        # Resolve base_url from preset if not explicitly provided.
        self.base_url: Optional[str] = None
        preset = _PROVIDER_PRESETS.get(api_provider)
        if base_url is not None:
            self.base_url = base_url
        elif preset is not None:
            self.base_url = preset

        # API keys come only from the user's configuration; never from
        # environment variables or a .env file.
        self.api_key = api_key

        # Whether the provider accepted JSON structured-output mode. Toggled
        # off automatically the first time a provider rejects the parameter.
        self._json_mode_supported: bool = True

        # Remaining corrective re-ask calls for this agent's whole run.
        self._repair_calls_remaining: int = max(0, int(repair_budget))
        # Pause before retrying a transient API failure, in seconds.
        self.retry_backoff: float = 1.0

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _default_system_prompt(deep: bool = False) -> str:
        """Build the JSON-contract system prompt.

        ``deep`` asks for longer, in-character reasoning; the default is the
        lean rule this project has always used.
        """
        reasoning_detail = (
            _REASONING_DETAIL_DEEP if deep else _REASONING_DETAIL_SIMPLE
        )
        return (
            "You are a professional stock trader managing a portfolio in a simulated market. "
            "Your goal is to maximize total wealth (cash + sum of all holdings * their prices).\n"
            "\n"
            "Rules:\n"
            "- You can trade multiple stocks, each identified by its exact name as given "
            "in the market data.\n"
            "- You cannot short sell (holdings >= 0 for each stock).\n"
            "- You cannot borrow money (cash >= 0, shared across all stocks).\n"
            "- Each step, you MUST decide for EVERY listed stock: buy, sell, or hold.\n"
            "- You may trade multiple stocks in one step (e.g. sell one stock to buy another).\n"
            "\n"
            "OUTPUT FORMAT — THIS IS THE MOST IMPORTANT RULE:\n"
            "Your ENTIRE response must be ONE raw JSON object, starting with '{' and ending "
            "with '}'. Nothing else. Specifically:\n"
            "- NO markdown code fences (no ```json ... ```).\n"
            "- NO explanation, commentary, or text before or after the JSON.\n"
            "- NO trailing commas.\n"
            "- Use double quotes for all keys and string values.\n"
            "- 'action' must be exactly one of: \"buy\", \"sell\", \"hold\" (lowercase).\n"
            "- 'quantity' must be an integer (0 for hold).\n"
            f"- 'reasoning': {reasoning_detail}.\n"
            "- Include one decision object for EVERY stock listed in the market data,\n"
            "  using each stock's exact name.\n"
            "\n"
            "Exact schema:\n"
            '{"decisions": [{"name": "<stock name>", "action": "buy" | "sell" | "hold", '
            f'"quantity": <integer>, "reasoning": "<{reasoning_detail}>"}}, ...]}}\n'
            "\n"
            "Example response for two stocks:\n"
            '{"decisions": [{"name": "Stock 1", "action": "buy", "quantity": 10, '
            '"reasoning": "Momentum is strong after the earnings beat."}, '
            '{"name": "Stock 2", "action": "hold", "quantity": 0, '
            '"reasoning": "Sideways trend; waiting for a clearer signal."}]}\n'
            "\n"
            "You remember past decisions; learn from them."
            + (_DEEP_OPTIONAL_FIELDS if deep else "")
        )

    def _build_market_summary(self) -> str:
        """
        Build a concise summary of the market trajectory across all
        remembered steps.  This gives the AI a long-horizon view beyond
        the 10-step price window in the observation.

        Tracks the first (primary) stock for backward-compat summary.
        """
        if len(self._market_history) < 2:
            return ""

        first_step, first_price = self._market_history[0]
        last_step, last_price = self._market_history[-1]
        total_return = (last_price - first_price) / max(first_price, 0.01)

        # Find max and min prices.
        prices = [p for _, p in self._market_history]
        max_price = max(prices)
        min_price = min(prices)

        # Determine trend direction over the last few steps.
        recent = self._market_history[-5:]
        if len(recent) >= 2:
            short_trend = (recent[-1][1] - recent[0][1]) / max(recent[0][1], 0.01)
        else:
            short_trend = 0.0

        parts = [
            f"Market since Step {first_step}: "
            f"opened at ${first_price:.2f}, now ${last_price:.2f} "
            f"({total_return:+.1%} total return). "
            f"Range: ${min_price:.2f}-${max_price:.2f}. "
            f"Short-term trend: {short_trend:+.1%}.",
            self._describe_regime(prices, short_trend),
        ]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _describe_regime(prices: List[float], short_trend: float) -> str:
        """Name the market regime in words, not just a percentage.

        A number tells the model how much moved; a word tells it what kind
        of market it is trading in.
        """
        window = prices[-8:]
        if len(window) < 2:
            return ""
        # Mean absolute step size, as a fraction of price.
        steps = [
            abs(window[i] - window[i - 1]) / max(window[i - 1], 0.01)
            for i in range(1, len(window))
        ]
        churn = sum(steps) / len(steps)
        direction = abs(short_trend)

        if churn >= 0.03:
            regime = "volatile -- prices are swinging hard round to round"
        elif direction >= 0.05:
            regime = (
                "a strong uptrend" if short_trend > 0 else "a strong downtrend"
            )
        elif direction >= 0.02:
            regime = "a mild uptrend" if short_trend > 0 else "a mild downtrend"
        elif churn <= 0.005:
            regime = "calm and quiet -- barely moving"
        else:
            regime = "choppy and directionless -- no clear trend"
        return f"Regime: the market is {regime}."

    def _record_key_events(self, step: int, active_events: List[Dict[str, Any]]) -> None:
        """
        Store significant events (|price impact| >= 5%) in long-term memory.

        Deduplicated by event name: a multi-round event is recorded once,
        when first observed.
        """
        if not self.enable_memory:
            return
        known = {e["name"] for e in self._key_events}
        for evt in active_events:
            try:
                impact = abs(float(evt.get("price_impact") or 0.0))
            except (TypeError, ValueError):
                continue
            if impact < 0.05:
                continue
            name = str(evt.get("name") or "?")
            if name in known:
                continue
            known.add(name)
            self._key_events.append({
                "step": step,
                "name": name,
                "stock": evt.get("stock"),
                "impact": evt.get("price_impact", 0.0),
            })
        # Keep the most recent 10 key events to bound prompt size.
        if len(self._key_events) > 10:
            self._key_events = self._key_events[-10:]

    def _summarize_decisions(self, step: int, result: Dict[str, Any]) -> str:
        """Compress one round's decision into a single summary line."""
        parts: List[str] = []
        decisions = result.get("decisions") if isinstance(result, dict) else None
        if isinstance(decisions, list):
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                act = str(d.get("action", "hold")).lower()
                if act not in ("buy", "sell"):
                    continue
                qty = d.get("quantity", 0)
                sym = d.get("name") or d.get("symbol") or "?"
                parts.append(f"{act.upper()} {qty} {sym}")
        elif isinstance(result, dict):
            act = str(result.get("action", "hold")).lower()
            if act in ("buy", "sell"):
                parts.append(f"{act.upper()} {result.get('quantity', 0)}")
        summary = "; ".join(parts) if parts else "HOLD all positions"
        return f"Step {step}: {summary}"

    def _build_memory_context(self, observation: Dict[str, Any]) -> str:
        """
        Build the 'memory' section of the prompt.

        Combines:
        - SHORT-TERM memory: the agent's own decisions over the last
          ``memory_window`` rounds, plus the market outcome since the
          latest decision.
        - LONG-TERM memory: key market events (>= 5% impact) the agent has
          lived through, enabling cross-round learning and adaptation.
        """
        if not self.enable_memory:
            return ""

        sections: List[str] = []

        # --- Short-term: recent decision summaries ---
        if self._short_term_memory or self._last_action:
            lines = ["=== YOUR RECENT DECISIONS (Short-Term Memory) ==="]
            lines.extend(f"  {s}" for s in self._short_term_memory)
            # Market outcome since the most recent decision.
            if self._last_action and len(self._market_history) >= 2:
                prev_price = self._market_history[-2][1]
                curr_price = self._market_history[-1][1]
                price_pct = (curr_price - prev_price) / max(prev_price, 0.01) * 100
                lines.append(
                    f"  [Market since last action: ${prev_price:.2f} -> "
                    f"${curr_price:.2f} ({price_pct:+.1f}%)]"
                )
            lines.append("=== END SHORT-TERM MEMORY ===")
            sections.append("\n".join(lines))

        # --- Long-term: key events lived through ---
        if self._key_events:
            lines = ["=== KEY MARKET EVENTS YOU LIVED THROUGH (Long-Term Memory) ==="]
            for evt in self._key_events:
                target = evt.get("stock") or "market-wide"
                impact = evt.get("impact", 0.0)
                lines.append(
                    f"  Step {evt.get('step', '?')}: {evt.get('name')} "
                    f"({target}, {impact * 100:+.0f}%)"
                )
            lines.append("=== END LONG-TERM MEMORY ===")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    @staticmethod
    def _persona_lines(observation: Dict[str, Any]) -> List[str]:
        """WHO YOU ARE / HOW YOU FEEL, when the persona layer supplied one."""
        persona = observation.get("persona")
        if not isinstance(persona, dict):
            return []
        lines = []
        disposition = persona.get("disposition")
        if disposition:
            lines.append("=== WHO YOU ARE ===")
            lines.append(str(disposition))
        mood = persona.get("mood")
        if isinstance(mood, dict) and mood:
            hint = persona.get("scale_hint", "")
            lines.append("")
            lines.append("=== HOW YOU FEEL RIGHT NOW ===")
            lines.append(
                "Confidence {c:.0f}/10 · Stress {s:.0f}/10 · "
                "Frustration {f:.0f}/10{hint}".format(
                    c=float(mood.get("confidence", 0)),
                    s=float(mood.get("stress", 0)),
                    f=float(mood.get("frustration", 0)),
                    hint=f"   ({hint})" if hint else "",
                )
            )
            pressure = persona.get("pressure")
            if pressure:
                lines.append(str(pressure))
        return lines

    @staticmethod
    def _stakes_line(observation: Dict[str, Any]) -> str:
        """Frame the P&L as real money against the starting stake."""
        start = observation.get("initial_wealth")
        if start is None:
            return ""
        try:
            start_f = float(start)
            wealth = float(observation.get("my_wealth", 0.0))
        except (TypeError, ValueError):
            return ""
        if start_f <= 0:
            return ""
        delta = wealth - start_f
        word = "up" if delta >= 0 else "down"
        return (
            f"- Stakes: you started with ${start_f:,.0f}. You are at "
            f"${wealth:,.0f} -- {word} ${abs(delta):,.0f} "
            f"({delta / start_f * 100:+.1f}%)."
        )

    @staticmethod
    def _concentration_line(observation: Dict[str, Any]) -> str:
        """Flag when most of the agent's wealth sits in one name.

        Deep-only: the data this line needs is always in the observation,
        so the persona block is what gates it (MarketEnv supplies
        ``persona`` only in deep mode -- the same contract the other
        deep-only lines in _build_prompt follow).
        """
        if not observation.get("persona"):
            return ""
        stocks = observation.get("stocks") or []
        try:
            wealth = float(observation.get("my_wealth", 0.0))
        except (TypeError, ValueError):
            return ""
        if wealth <= 0 or not stocks:
            return ""
        biggest, biggest_value = "", 0.0
        for s in stocks:
            value = float(s.get("my_holdings", 0) or 0) * float(s.get("price", 0) or 0)
            if value > biggest_value:
                biggest_value, biggest = value, (
                    s.get("name") or s.get("symbol") or "a stock"
                )
        share = biggest_value / wealth
        if share < 0.4:
            return ""
        return (
            f"- Concentration: {share * 100:.0f}% of your wealth is in "
            f"{biggest} -- that is a big single bet."
        )

    @staticmethod
    def _standing_line(observation: Dict[str, Any]) -> str:
        """Rank plus a named leader, so the gap has a face on it."""
        standing = observation.get("standing")
        if not isinstance(standing, dict) or not standing.get("of"):
            return ""
        leader = standing.get("leader_name")
        gap = standing.get("gap_to_leader_pct", 0.0)
        line = (
            f"- Standing: you are {standing.get('rank')} of "
            f"{standing.get('of')} at {standing.get('my_return_pct', 0.0):+.1f}%."
        )
        if leader and gap and float(gap) > 0.01:
            line += (
                f" {leader} is leading at "
                f"{standing.get('leader_return_pct', 0.0):+.1f}% -- "
                f"{float(gap):.1f} points ahead of you."
            )
        return line

    @staticmethod
    def _exposure_line(observation: Dict[str, Any]) -> str:
        """What moved last round, and whether the agent was in it."""
        stocks = observation.get("stocks") or []
        held = observation.get("held_at_round_start")
        if not stocks or not isinstance(held, dict):
            return ""
        bits = []
        for s in stocks:
            name = s.get("name") or s.get("symbol") or "?"
            try:
                move = float(s.get("move_since_last_pct", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if abs(move) < 1.0:
                continue
            was_in = float(held.get(name, 0) or 0) > 0
            if was_in:
                bits.append(f"{name} {move:+.1f}% (you held it)")
            else:
                bits.append(f"{name} {move:+.1f}% (you weren't in it)")
        if not bits:
            return ""
        return "- Since last round: " + "; ".join(bits) + "."

    @staticmethod
    def _floor_mood_line(observation: Dict[str, Any]) -> str:
        """How the rest of the floor behaved, distinct from the price move."""
        floor = observation.get("floor_mood")
        if not isinstance(floor, dict) or not floor.get("sentence"):
            return ""
        return f"- {floor['sentence']}"

    @staticmethod
    def _peer_talk_lines(observation: Dict[str, Any]) -> List[str]:
        """Quote what the peers who traded actually said."""
        peers = observation.get("social_peers") or []
        quotes = [
            f"  {p.get('id', '?')} ({p.get('relation', 'peer')}, "
            f"{p.get('action', 'hold')}): \"{p['reasoning']}\""
            for p in peers
            if isinstance(p, dict) and p.get("reasoning")
        ]
        if not quotes:
            return []
        return ["", "=== WHAT OTHERS ARE SAYING ===", *quotes]

    def _lesson_lines(self) -> List[str]:
        """Replay the lessons the model wrote for itself."""
        lessons = getattr(self, "_lessons", None)
        if not lessons:
            return []
        return ["", "=== LESSONS YOU'VE LEARNED ===",
                *(f"  - {lesson}" for lesson in lessons)]

    def _plan_lines(self, observation: Dict[str, Any]) -> List[str]:
        """Remind the model of exit levels it committed to.

        Purely a reminder: nothing here executes, so the agent has to act on
        its own plan or explain why it is not.
        """
        plans = getattr(self, "_position_plans", None)
        if not plans:
            return []
        prices = {
            (s.get("name") or s.get("symbol")): s.get("price")
            for s in (observation.get("stocks") or [])
        }
        out = []
        for sym, plan in plans.items():
            price = prices.get(sym)
            if price is None:
                continue
            bits = []
            if plan.get("stop_loss") is not None:
                bits.append(f"stop at ${plan['stop_loss']:.2f}")
            if plan.get("target") is not None:
                bits.append(f"target ${plan['target']:.2f}")
            if bits:
                out.append(
                    f"  {sym}: you set {' and '.join(bits)}; it is now "
                    f"${float(price):.2f}."
                )
        if not out:
            return []
        return ["", "=== PLANS YOU COMMITTED TO ===", *out]

    def _build_prompt(self, observation: Dict[str, Any]) -> str:
        """Convert a market observation into a natural-language prompt."""
        # Track market history for long-horizon summary (primary stock).
        step = observation["step"]
        price = observation["price"]
        self._market_history.append((step, price))
        # Cap the stored history to avoid unbounded growth.
        if len(self._market_history) > 100:
            self._market_history = self._market_history[-100:]

        # Character first, so the model reads who it is before the numbers.
        # Present only when the persona layer supplied one (deep mode).
        lines = list(self._persona_lines(observation))
        if lines:
            lines.append("")

        lines += [
            f"Market Data (Step {observation['step']}):",
            f"- Your Cash: ${observation['my_cash']:.2f}",
            f"- Your Total Wealth: ${observation['my_wealth']:.2f}",
        ]

        # Per-stock market data.
        stocks = observation.get("stocks", [])
        if stocks:
            lines.append(f"- Stocks ({len(stocks)} total):")
            for s in stocks:
                stk_name = s.get("name") or s.get("symbol") or "Stock"
                hist = s.get("price_history", [])
                price_str = ", ".join(f"${p:.2f}" for p in hist[-5:])
                meta = ""
                if s.get("sector"):
                    meta += f" | Sector: {s['sector']}"
                if s.get("blurb"):
                    meta += f" | About: {s['blurb']}"
                lines.append(
                    f"  * {stk_name}: "
                    f"${s['price']:.2f} | Holdings: {s.get('my_holdings', 0)} | "
                    f"Recent Prices: [{price_str}] | Volume: {s.get('last_volume', 0)}"
                    f"{meta}"
                )
        else:
            # Fallback for legacy single-stock observations.
            prices = observation.get("price_history", [])
            price_str = ", ".join(f"${p:.2f}" for p in prices[-5:])
            lines.append(f"- Current Price: ${observation.get('price', 0):.2f}")
            lines.append(f"- Recent Prices: [{price_str}]")
            lines.append(f"- Your Holdings: {observation.get('my_holdings', 0)}")

        # Factual context. Each block renders only when its data is in the
        # observation: the stakes line is always there, the rest arrive only
        # in deep mode (MarketEnv gates them).
        for line in (
            self._stakes_line(observation),
            self._concentration_line(observation),
            self._standing_line(observation),
            self._exposure_line(observation),
            self._floor_mood_line(observation),
        ):
            if line:
                lines.append(line)

        # Include market sentiment when available.
        sentiment = observation.get("market_sentiment", 0.0)
        if sentiment != 0.0:
            lines.append(f"- Market Sentiment: {sentiment:+.2f} (-1=bearish, +1=bullish)")

        active_events = observation.get("active_events", [])
        if active_events:
            # Tag each event with its target so the agent knows whether it
            # hits the whole market or one specific stock.
            event_strs = []
            for e in active_events:
                target = e.get("stock") or "market-wide"
                event_strs.append(f"{e.get('name', '?')} ({target})")
            lines.append(f"- Active Events: {', '.join(event_strs)}")
            # Significant events become long-term memories.
            self._record_key_events(step, active_events)

        last_round = observation.get("last_round")
        if last_round:
            lines.append("- Last Round Outcome (learn from it):")
            for f in last_round.get("trades", []):
                cur = f.get("current_price")
                if cur is not None:
                    lines.append(
                        f"    * {f['action']} {f['quantity']} {f['symbol']} @ "
                        f"${f['fill_price']:.2f} -> now ${cur:.2f} "
                        f"({f['move_since_fill_pct']:+.1f}% since your fill)"
                    )
            lines.append(
                f"    Net cash flow: {last_round.get('net_cash_flow', 0):+,.2f}"
            )

        # --- Memory context ---
        if self.enable_memory:
            market_summary = self._build_market_summary()
            if market_summary:
                lines.append(f"\n[Market Summary] {market_summary}")

            memory_ctx = self._build_memory_context(observation)
            if memory_ctx:
                lines.append(f"\n{memory_ctx}")

        # Deep-mode blocks that live on the agent rather than the observation.
        lines += self._peer_talk_lines(observation)
        lines += self._lesson_lines()
        lines += self._plan_lines(observation)

        lines.append(
            "\nWhat actions do you take for each stock? Output a decision object for "
            "EVERY listed stock in the 'decisions' array."
        )
        # Models weight the final tokens heavily — repeat the format contract
        # right where generation starts.
        lines.append(
            "Respond with ONLY the raw JSON object now. Start your reply with '{' "
            "and end it with '}'. No markdown fences, no commentary."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _call_ai_api(
        self,
        prompt: str,
        messages: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Call the external AI API and return a text response.

        Routing logic:
        - "anthropic" without base_url → native Anthropic SDK
        - "google" without base_url → native google-generativeai SDK
        - Everything else (including custom base_url) → OpenAI-compatible API

        When `messages` is provided (chat mode), it is used directly instead
        of the trading prompt/history, and `system_prompt` overrides the
        default trading system prompt.

        Raises
        ------
        RuntimeError
            If the required provider library is not installed.
        ValueError
            If no API key is available.
        """
        if not self.api_key:
            raise ValueError(
                f"No API key for provider '{self.api_provider}'. "
                f"Enter the API key in the trader configuration (homepage -> "
                f"Configure Traders) or pass api_key directly."
            )

        # Native SDK providers (only when no custom base_url is set).
        if self.api_provider == "anthropic" and not self.base_url:
            return self._call_anthropic(
                prompt, messages=messages, system_prompt=system_prompt
            )
        if self.api_provider == "google" and not self.base_url:
            return self._call_google(
                prompt, messages=messages, system_prompt=system_prompt
            )

        # All other providers use the OpenAI-compatible chat completions API.
        return self._call_openai_compat(
            prompt, messages=messages, system_prompt=system_prompt
        )

    def _call_ai_api_with_retry(
        self,
        prompt: str,
        messages: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call the provider, retrying once on a transient failure.

        Rate limits, timeouts and 5xx responses are the common failure mode
        when several agents call the same provider concurrently, and they
        usually clear within a second. Retrying once here is much cheaper
        than losing the agent's entire round. Permanent failures (bad key,
        unknown model, malformed request) propagate immediately.
        """
        try:
            return self._call_ai_api(
                prompt, messages=messages, system_prompt=system_prompt
            )
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            if self.retry_backoff > 0:
                time.sleep(self.retry_backoff)
            return self._call_ai_api(
                prompt, messages=messages, system_prompt=system_prompt
            )

    _HTML_START_RE = re.compile(r"\s*(?:<!doctype\s+html|<html[\s>])", re.IGNORECASE)

    def _reject_html(self, content: str) -> str:
        """
        Guard against providers answering with an HTML web page.

        Some openai SDK versions hand back the raw body as a string when
        the server returns an HTML page (e.g. a misconfigured base_url
        pointing at a website instead of the API endpoint). Surface a
        clear configuration error instead of letting the HTML reach the
        JSON parser.
        """
        if isinstance(content, str) and self._HTML_START_RE.match(content):
            raise RuntimeError(
                f"Provider '{self.api_provider}' returned an HTML page, not an "
                f"API response. The base_url '{self.base_url}' probably points "
                f"to a website instead of an OpenAI-compatible chat/completions "
                f"endpoint (it usually must end with /v1)."
            )
        return content

    def _call_openai_compat(
        self,
        prompt: str,
        messages: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Call an OpenAI-compatible chat completions API.

        Works with OpenAI, OpenRouter, ChatAnywhere, Groq, Google Gemini
        (OpenAI-compat endpoint), and any other provider that implements
        the /chat/completions endpoint.

        When memory is enabled, the full conversation history is sent
        so the model can reason across past decisions. When `messages`
        is provided (chat mode), it is used directly instead.
        """
        try:
            import openai
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Install with: pip install openai"
            )

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        # Bound the HTTP request so a slow / unresponsive provider cannot
        # hang an entire /api/step for the SDK's default 600s. 100s keeps
        # it inside the frontend's 120s step timeout so the UI recovers.
        client_kwargs["timeout"] = 100

        # client_kwargs is built dynamically for OpenAI-compatible providers,
        # so mypy cannot validate the expanded kwargs here.
        client = openai.OpenAI(**client_kwargs)  # type: ignore[arg-type]

        if messages is None:
            # Trading mode: system prompt + memory + new user message.
            messages = [{"role": "system", "content": self.system_prompt}]
            if self.enable_memory and self._conversation_history:
                messages.extend(self._conversation_history)
            messages.append({"role": "user", "content": prompt})
            trading_mode = True
        else:
            # Chat mode: explicit message list with an overridden system prompt.
            system = system_prompt if system_prompt is not None else self.system_prompt
            messages = [{"role": "system", "content": system}] + list(messages)
            trading_mode = False

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        # Structured-output mode guarantees syntactically valid JSON on
        # providers that support it. Not all OpenAI-compatible endpoints do,
        # so fall back to a plain request when the parameter is rejected.
        # Only used for trading decisions — chat replies stay free-form.
        use_json_mode = trading_mode and self._json_mode_supported
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not use_json_mode or not _is_json_mode_rejection(exc, openai):
                raise
            # Provider rejected response_format — remember and retry plain.
            self._json_mode_supported = False
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)

        # Some providers return a plain string instead of a ChatCompletion object.
        if isinstance(response, str):
            return self._reject_html(response)

        # Some older SDKs / proxies also expose .json() style dict payloads.
        if isinstance(response, dict):
            try:
                return self._reject_html(response["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(
                    "API returned a dict without a valid 'choices' field: "
                    f"{str(response)[:200]}"
                )

        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError(
                "API response has no 'choices' field — the provider may be "
                "returning an unexpected format. Inspect the response manually."
            )
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None:
            # Some providers put content directly on the choice object.
            content = getattr(first, "text", None) or getattr(first, "content", None)
            if content is None:
                raise RuntimeError(
                    "API response choice has no message/text/content field: "
                    f"{str(first)[:200]}"
                )
            return self._reject_html(content)
        content = getattr(message, "content", None)
        if content is None:
            # Reasoning models (e.g. ArliAI's Derestricted) sometimes return
            # the answer in a reasoning/reasoning_content/thinking field
            # instead of content. Fall back to those before giving up.
            for field in ("reasoning", "reasoning_content", "thinking"):
                val = getattr(message, field, None)
                if val:
                    content = val
                    break
        if content is None:
            content = str(message)
        return self._reject_html(content)

    def _call_anthropic(
        self,
        prompt: str,
        messages: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call Anthropic Messages API with conversation history."""
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self.api_key)

        system = system_prompt if system_prompt is not None else self.system_prompt

        if messages is None:
            # Build messages from conversation history + new prompt.
            if self.enable_memory and self._conversation_history:
                messages = list(self._conversation_history)
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=messages,
        )
        return str(getattr(response.content[0], "text", ""))

    def _call_google(
        self,
        prompt: str,
        messages: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call Google Gemini API with conversation history."""
        try:
            import warnings
            with warnings.catch_warnings():
                # google.generativeai is deprecated but still functional;
                # silence the module-level FutureWarning so it does not spam stderr.
                warnings.simplefilter("ignore")
                import google.generativeai as genai
        except ImportError:
            raise RuntimeError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        genai.configure(api_key=self.api_key)

        system = system_prompt if system_prompt is not None else self.system_prompt

        # Build a chat session with history for multi-turn conversation.
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
        )

        if messages is not None:
            # Chat mode: explicit message list (user/assistant only).
            conv = [
                {"role": "user" if m["role"] == "user" else "model",
                 "parts": m["content"]}
                for m in messages
            ]
            if not conv:
                response = model.generate_content(prompt)
            else:
                chat = model.start_chat(history=cast(Any, conv[:-1]))
                response = chat.send_message(conv[-1]["parts"])
        elif self.enable_memory and self._conversation_history:
            chat = model.start_chat(history=cast(
                Any,
                [
                    {"role": "user" if msg["role"] == "user" else "model",
                     "parts": msg["content"]}
                    for msg in self._conversation_history
                ],
            ))
            response = chat.send_message(prompt)
        else:
            response = model.generate_content(prompt)
        return str(response.text)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_quantity(value: Any) -> Optional[int]:
        """
        Normalize a model-provided quantity to an int.

        Accepts ints/floats, numeric strings ("10", "10.0", "10 shares",
        "50%"), null (treated as 0) and words like "all"/"max" (mapped to
        a huge int that the market env clips to cash/holdings). Returns
        None when the value cannot be interpreted as a quantity.
        """
        if value is None:
            return 0
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            # Reject NaN/Infinity, which json.loads accepts by default and
            # which int() would otherwise turn into a ValueError/OverflowError.
            if not math.isfinite(value):
                return None
            return int(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if match:
                return int(float(match.group()))
            if any(word in text for word in ("all", "max", "full", "everything")):
                return 10**9
            return None
        return None

    @staticmethod
    def _repair_json_candidates(response: str) -> List[str]:
        """
        Produce progressively repaired variants of a malformed JSON response.

        Covers the most common LLM formatting mistakes:
        1. Markdown code fences (```json ... ```)
        2. Trailing commas before } or ]
        3. Python-style literals (True / False / None)
        4. Single-quoted strings instead of double quotes (last resort —
           may mangle apostrophes inside reasoning text)
        """
        variants: List[str] = []
        current = response

        stripped = re.sub(r"```[a-zA-Z]*", "", current)
        if stripped != current:
            variants.append(stripped)
            current = stripped

        no_trailing = re.sub(r",\s*(?=[}\]])", "", current)
        if no_trailing != current:
            variants.append(no_trailing)
            current = no_trailing

        py_fixed = re.sub(
            r"\bNone\b", "null",
            re.sub(r"\bFalse\b", "false",
                   re.sub(r"\bTrue\b", "true", current)),
        )
        if py_fixed != current:
            variants.append(py_fixed)
            current = py_fixed

        squoted = re.sub(
            r"'([^'\n]*)'",
            lambda m: '"' + m.group(1).replace('"', '\\"') + '"',
            current,
        )
        if squoted != current:
            variants.append(squoted)
        return variants

    def _scan_json(
        self, text: str
    ) -> tuple[
        Optional[List[Dict[str, Any]]],
        Optional[Dict[str, Any]],
        bool,
        Dict[str, Any],
    ]:
        """
        Scan ``text`` for trading-decision JSON at any position.

        Returns (best_decisions, best_legacy, saw_invalid, extras). Mirrors
        the historical parsing semantics: for multi-stock format all valid
        decision entries are collected; for legacy format the LAST valid
        single action wins. ``extras`` carries the optional top-level
        "mood" / "lesson" fields when the winning object had them.
        """
        decoder = json.JSONDecoder()
        best_decisions: Optional[List[Dict[str, Any]]] = None
        best_legacy: Optional[Dict[str, Any]] = None
        best_extras: Dict[str, Any] = {}
        saw_invalid = False

        for match in re.finditer(r"[\[{]", text):
            try:
                data, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue

            if not isinstance(data, dict):
                continue

            # Multi-stock format: {"decisions": [...]}
            if "decisions" in data:
                raw = data["decisions"]
                if not isinstance(raw, list):
                    continue
                parsed_list: List[Dict[str, Any]] = []
                for d in raw:
                    if not isinstance(d, dict):
                        break
                    raw_action = d.get("action")
                    if not isinstance(raw_action, str) or not raw_action.strip():
                        break
                    quantity = self._coerce_quantity(d.get("quantity", 0))
                    if quantity is None:
                        saw_invalid = True
                        continue
                    action = raw_action.strip().lower()
                    if action not in ("buy", "sell", "hold") or quantity < 0:
                        saw_invalid = True
                        continue
                    stk_name = str(d.get("name") or d.get("symbol") or "").strip()
                    entry = {
                        "name": stk_name,
                        "symbol": stk_name,
                        "action": action,
                        "quantity": quantity,
                        "reasoning": str(d.get("reasoning", "")).strip(),
                    }
                    # Optional commitments; dropped silently when absent or
                    # unparseable, so the decision itself is never at risk.
                    for field in ("stop_loss", "target"):
                        price = _coerce_price(d.get(field))
                        if price is not None:
                            entry[field] = price
                    parsed_list.append(entry)
                if parsed_list:
                    best_decisions = parsed_list
                    best_extras = {
                        k: data[k] for k in ("mood", "lesson") if k in data
                    }
                    saw_invalid = False
                continue

            # Legacy single-stock format: {"action": "...", "quantity": ...}
            raw_action = data.get("action")
            if not isinstance(raw_action, str) or not raw_action.strip():
                continue

            quantity = self._coerce_quantity(data.get("quantity", 0))
            if quantity is None:
                continue

            action = raw_action.strip().lower()
            if action not in ("buy", "sell", "hold") or quantity < 0:
                saw_invalid = True
                continue

            best_legacy = {
                "action": action,
                "quantity": quantity,
                "reasoning": str(data.get("reasoning", "")).strip(),
            }

        return best_decisions, best_legacy, saw_invalid, best_extras

    # Reasoning-model thought blocks: <think>...</think> (DeepSeek-R1, QwQ)
    # and <|begin_of_thought|>...<|end_of_thought|>. Some providers strip the
    # opening tag and leave an orphan closing tag, so everything before the
    # last </think> is reasoning.
    _THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
    _THOUGHT_TOKENS_RE = re.compile(
        r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>\s*", re.DOTALL
    )
    # Anchored at the start: an unclosed <think> only means "truncated
    # mid-thought" when the reasoning opens the response. Unanchored, this
    # deleted everything after a literal "<think>" appearing inside a JSON
    # string value, destroying an otherwise valid decision payload.
    _THINK_UNCLOSED_RE = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)
    _THINK_ORPHAN_CLOSE_RE = re.compile(r"^.*</think>\s*", re.DOTALL | re.IGNORECASE)

    def _strip_reasoning(self, response: str) -> str:
        """
        Remove reasoning-model thought blocks from a response.

        Complete <think>...</think> blocks are dropped; an unclosed <think>
        (response truncated mid-thought) drops everything after it; an orphan
        </think> keeps only the text after the last closing tag. Returns the
        original response when nothing recognizable was stripped.
        """
        cleaned = self._THINK_BLOCK_RE.sub("", response)
        cleaned = self._THOUGHT_TOKENS_RE.sub("", cleaned)
        cleaned = self._THINK_UNCLOSED_RE.sub("", cleaned)
        cleaned = self._THINK_ORPHAN_CLOSE_RE.sub("", cleaned, count=1)
        if cleaned.strip():
            return cleaned
        return response

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Extract a JSON action object from the AI response.

        Supports both multi-stock and legacy single-stock formats:
        - Multi-stock: ``{"decisions": [{"symbol", "action", "quantity",
          "reasoning"}, ...]}``
        - Legacy single-stock: ``{"action", "quantity", "reasoning"}``

        Handles pure JSON, markdown code blocks, JSON embedded in text, and
        reasoning-model thought blocks (<think>...</think>). When several
        valid objects appear, the LAST one wins.

        For multi-stock format, all valid decision entries are collected.
        For legacy format, a single action object is returned.
        """
        if not isinstance(response, str) or not response.strip():
            raise ValueError("AI response was empty or not text")

        # Reasoning models spend their output budget on thinking; strip the
        # thought blocks so only the final answer is scanned.
        response = self._strip_reasoning(response)

        # Pass 1: raw response. Pass 2+: progressively repaired variants
        # (code fences, trailing commas, Python literals, single quotes).
        best_decisions: Optional[List[Dict[str, Any]]] = None
        best_legacy: Optional[Dict[str, Any]] = None
        saw_invalid = False
        extras: Dict[str, Any] = {}

        for text in [response] + self._repair_json_candidates(response):
            best_decisions, best_legacy, saw_invalid, extras = self._scan_json(text)
            if best_decisions is not None or best_legacy is not None:
                break

        if best_decisions is not None:
            result: Dict[str, Any] = {"decisions": best_decisions}
            # Optional deep-mode extras ride alongside the decisions.
            mood = extras.get("mood")
            if isinstance(mood, dict):
                result["mood"] = mood
            # Must actually be text: a JSON null would otherwise become the
            # string "None" and be filed as a lesson forever.
            raw_lesson = extras.get("lesson")
            lesson = raw_lesson.strip() if isinstance(raw_lesson, str) else ""
            if lesson:
                result["lesson"] = lesson
            return result

        # Salvage pass for TRUNCATED responses (max_tokens cut-off): the
        # wrapper {"decisions": [...]} cannot be parsed, but the individual
        # complete decision objects inside can still be recovered.
        if best_decisions is None and '"decisions"' in response:
            salvaged: List[Dict[str, Any]] = []
            seen_items: List[Dict[str, Any]] = []
            salvage_decoder = json.JSONDecoder()
            for match in re.finditer(r"\{", response):
                try:
                    data, _ = salvage_decoder.raw_decode(response[match.start():])
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                raw_action = str(data.get("action", "")).strip().lower()
                if raw_action not in ("buy", "sell", "hold"):
                    continue
                quantity = self._coerce_quantity(data.get("quantity", 0))
                if quantity is None:
                    continue
                stk_name = str(data.get("name") or data.get("symbol") or "").strip()
                item = {
                    "name": stk_name,
                    "symbol": stk_name,
                    "action": raw_action,
                    "quantity": quantity,
                    "reasoning": str(data.get("reasoning", "")).strip(),
                }
                if item not in seen_items:
                    seen_items.append(item)
                    salvaged.append(item)
            if salvaged:
                print(
                    f"[WARN] Salvaged {len(salvaged)} decision(s) from a "
                    "truncated AI response (consider raising max_tokens)."
                )
                return {"decisions": salvaged}
        if best_legacy is not None:
            return best_legacy
        if saw_invalid:
            raise ValueError("AI response contained an invalid action or quantity")
        raise ValueError(
            "AI response did not contain a valid JSON action object: "
            f"{response[:200]!r}"
        )

    _MAX_LESSONS = 8

    def _record_lesson(self, result: Dict[str, Any]) -> None:
        """File the one-liner the model wrote for itself, if any."""
        lesson = str(result.get("lesson", "")).strip() if isinstance(result, dict) else ""
        if not lesson:
            return
        if lesson in self._lessons:
            return
        self._lessons.append(lesson)
        if len(self._lessons) > self._MAX_LESSONS:
            self._lessons = self._lessons[-self._MAX_LESSONS:]

    def _record_position_plans(
        self, observation: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Store any exit levels the model committed to this round.

        A plan is dropped once the position is gone, so the agent is never
        reminded about a stock it no longer holds.
        """
        decisions = result.get("decisions") if isinstance(result, dict) else None
        if isinstance(decisions, list):
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                sym = str(d.get("name") or d.get("symbol") or "").strip()
                if not sym:
                    continue
                plan = {
                    field: d[field]
                    for field in ("stop_loss", "target")
                    if d.get(field) is not None
                }
                if plan:
                    self._position_plans[sym] = plan

        # Clear plans for positions the agent no longer has.
        holdings = observation.get("my_holdings")
        if isinstance(holdings, dict):
            for sym in list(self._position_plans):
                if float(holdings.get(sym, 0) or 0) <= 0:
                    self._position_plans.pop(sym, None)

    def _retry_with_escalation(
        self,
        observation: Dict[str, Any],
        prompt: str,
        response: str,
        parse_error: ValueError,
    ) -> tuple[Dict[str, Any], str]:
        """
        Re-ask the model after an unparseable response, escalating strategies.

        Attempt 1 keeps the historical behavior: show the model its own
        invalid output and demand pure JSON. Attempt 2 uses a minimal prompt
        (stock data only, no memory context) with explicit anti-thinking
        constraints. Attempt 3 offers a fill-in JSON template that leaves
        almost no room for format errors.

        Each attempt spends one call from ``_repair_calls_remaining``, a
        per-run budget: without it a model that never emits valid JSON would
        cost 4 API calls per agent per round for the whole simulation.

        Returns (parsed_result, last_response) on the first success. Raises
        the last parse error when every attempt fails or the budget runs out.
        Errors raised by the API itself are NOT caught here -- an unreachable
        provider is a transport failure, not a formatting one, and re-asking
        it with a stricter prompt cannot help.
        """
        stocks = observation.get("stocks", [])
        if stocks:
            stock_lines = "\n".join(
                f"- {s.get('name') or s.get('symbol') or 'Stock'}: "
                f"price ${s.get('price', 0):.2f}, "
                f"holdings {s.get('my_holdings', 0)}"
                for s in stocks
            )
            template = (
                '{"decisions": ['
                + ", ".join(
                    f'{{"name": "{s.get("name") or s.get("symbol") or "Stock"}", '
                    '"action": "buy|sell|hold", "quantity": <integer>, '
                    '"reasoning": "<short>"}'
                    for s in stocks
                )
                + "]}"
            )
        else:
            stock_lines = (
                f"- Current price: ${observation.get('price', 0):.2f}, "
                f"holdings {observation.get('my_holdings', 0)}"
            )
            template = (
                '{"action": "buy|sell|hold", "quantity": <integer>, '
                '"reasoning": "<short>"}'
            )

        repair_instructions = [
            (
                "Your previous response was NOT valid JSON and could not be "
                "parsed. Respond again, corrected: output ONLY one raw JSON "
                "object with a 'decisions' array covering every stock, each "
                "entry having 'name', 'action' (buy/sell/hold), an integer "
                "'quantity', and short 'reasoning'. No markdown fences, no "
                "commentary before or after."
            ),
            (
                "Do NOT include any reasoning, thinking, or explanation. Do "
                "NOT use thinking tags. Start your response with '{' and end "
                "it with '}'. Output ONLY the JSON object. Decide on these "
                f"stocks:\n{stock_lines}"
            ),
            (
                "Your previous responses were not parseable JSON. Copy this "
                "template EXACTLY, replacing buy|sell|hold with one action "
                "and <integer>/<short> with your values. Output nothing "
                f"else, no thinking:\n{template}"
            ),
        ]

        last_error = parse_error
        for attempt, instruction in enumerate(repair_instructions):
            if self._repair_calls_remaining <= 0:
                break
            self._repair_calls_remaining -= 1
            if attempt == 0:
                retry_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": instruction},
                ]
            else:
                # Fresh, minimal prompts for attempts 2/3: repeating the
                # original prompt and the failed responses only biases the
                # model toward repeating its mistake.
                retry_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": instruction},
                ]
            response2 = self._call_ai_api("", messages=retry_messages)
            try:
                result = self._parse_response(response2)
                return result, response2
            except ValueError as exc:
                last_error = exc
        raise last_error

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observation -> prompt -> AI call -> parsing -> action.

        If the API is unavailable (no key, network error, rate limit, etc.),
        raises an exception so the caller can mark this agent as failed.

        When memory is enabled, each (prompt, response) pair is stored
        in _conversation_history so the agent can reason across past
        decisions in future rounds.
        """
        if not self.api_key:
            raise RuntimeError(f"Agent {self.agent_id}: no API key configured")

        prompt = self._build_prompt(observation)
        response = self._call_ai_api_with_retry(prompt)
        try:
            result = self._parse_response(response)
        except ValueError as parse_error:
            if self._repair_calls_remaining <= 0:
                # Budget spent: this model has already proved it cannot
                # produce parseable JSON, so stop paying for re-asks.
                raise
            # Corrective retries with escalating strategies: (1) show the
            # model its own invalid output and demand pure JSON, (2) retry
            # with a minimal prompt free of memory context that may distract
            # a weak or reasoning-heavy model, (3) offer a fill-in template
            # that leaves almost no room for format errors. Each attempt
            # costs one extra API call, drawn from _repair_calls_remaining.
            result, response = self._retry_with_escalation(
                observation, prompt, response, parse_error
            )

        # Store conversation turn for memory.
        if self.enable_memory:
            self._conversation_history.append(
                {"role": "user", "content": prompt}
            )
            self._conversation_history.append(
                {"role": "assistant", "content": response}
            )
            # Cap history to memory_window * 2 messages
            # (each turn = 1 user + 1 assistant).
            cap = self.memory_window * 2
            if len(self._conversation_history) > cap:
                self._conversation_history = (
                    self._conversation_history[-cap:]
                )

            # Short-term memory: keep a compact summary of this round's
            # decisions so later prompts stay small yet informative.
            self._short_term_memory.append(
                self._summarize_decisions(observation.get("step", 0), result)
            )
            if len(self._short_term_memory) > self.memory_window:
                self._short_term_memory = self._short_term_memory[-self.memory_window:]

        self._record_lesson(result)
        self._record_position_plans(observation, result)

        self._last_action = result
        return result

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list] = None,
    ) -> str:
        """
        Free-form conversation with the agent (Web UI chat panel).

        Reuses the same provider routing as act(), but keeps chat messages
        separate from the trading conversation history so chatting does not
        pollute the agent's trading memory.

        Parameters
        ----------
        message : str
            The user's message to this agent.
        system_prompt : str, optional
            In-character persona prompt. Falls back to the trading system
            prompt when omitted.
        history : list, optional
            Prior chat turns as {"role": "user"|"assistant", "content": str}.

        Returns
        -------
        str
            The agent's reply.
        """
        if not self.api_key:
            raise RuntimeError(f"Agent {self.agent_id}: no API key configured")

        messages: list = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        return self._call_ai_api_with_retry(
            message, messages=messages, system_prompt=system_prompt
        )



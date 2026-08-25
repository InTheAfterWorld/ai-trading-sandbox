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
        holdings: int = 0,
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

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _default_system_prompt() -> str:
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
            "- 'reasoning': 1-2 short sentences, under 30 words.\n"
            "- Include one decision object for EVERY stock listed in the market data,\n"
            "  using each stock's exact name.\n"
            "\n"
            "Exact schema:\n"
            '{"decisions": [{"name": "<stock name>", "action": "buy" | "sell" | "hold", '
            '"quantity": <integer>, "reasoning": "<1-2 short sentences>"}, ...]}\n'
            "\n"
            "Example response for two stocks:\n"
            '{"decisions": [{"name": "Stock 1", "action": "buy", "quantity": 10, '
            '"reasoning": "Momentum is strong after the earnings beat."}, '
            '{"name": "Stock 2", "action": "hold", "quantity": 0, '
            '"reasoning": "Sideways trend; waiting for a clearer signal."}]}\n'
            "\n"
            "You remember past decisions; learn from them."
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
        ]
        return " ".join(parts)

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

    def _build_prompt(self, observation: Dict[str, Any]) -> str:
        """Convert a market observation into a natural-language prompt."""
        # Track market history for long-horizon summary (primary stock).
        step = observation["step"]
        price = observation["price"]
        self._market_history.append((step, price))
        # Cap the stored history to avoid unbounded growth.
        if len(self._market_history) > 100:
            self._market_history = self._market_history[-100:]

        lines = [
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

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

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
        except Exception:
            if not use_json_mode:
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

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Extract a JSON action object from the AI response.

        Supports both multi-stock and legacy single-stock formats:
        - Multi-stock: ``{"decisions": [{"symbol", "action", "quantity",
          "reasoning"}, ...]}``
        - Legacy single-stock: ``{"action", "quantity", "reasoning"}``

        Handles pure JSON, markdown code blocks, and JSON embedded in text.
        When several valid objects appear, the LAST one wins.

        For multi-stock format, all valid decision entries are collected.
        For legacy format, a single action object is returned.
        """
        if not isinstance(response, str) or not response.strip():
            raise ValueError("AI response was empty or not text")

        decoder = json.JSONDecoder()

        # Try to find a multi-stock "decisions" list first.
        best_decisions: Optional[List[Dict[str, Any]]] = None
        # Fallback: legacy single action.
        best_legacy: Optional[Dict[str, Any]] = None
        saw_invalid = False

        for match in re.finditer(r"[\[{]", response):
            try:
                data, _ = decoder.raw_decode(response[match.start():])
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
                    parsed_list.append({
                        "name": stk_name,
                        "symbol": stk_name,
                        "action": action,
                        "quantity": quantity,
                        "reasoning": str(d.get("reasoning", "")).strip(),
                    })
                if parsed_list:
                    best_decisions = parsed_list
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

        if best_decisions is not None:
            return {"decisions": best_decisions}
        # Salvage pass for TRUNCATED responses (max_tokens cut-off): the
        # wrapper {"decisions": [...]} cannot be parsed, but the individual
        # complete decision objects inside can still be recovered.
        if best_decisions is None and '"decisions"' in response:
            salvaged: List[Dict[str, Any]] = []
            seen_items: List[Dict[str, Any]] = []
            for match in re.finditer(r"\{", response):
                try:
                    data, _ = decoder.raw_decode(response[match.start():])
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
        response = self._call_ai_api(prompt)
        result = self._parse_response(response)

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

        return self._call_ai_api(message, messages=messages, system_prompt=system_prompt)



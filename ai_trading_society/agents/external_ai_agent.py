"""
ExternalAIAgent provides an interface for external AI trading models.

Supports any OpenAI-compatible API (OpenAI, OpenRouter, ChatAnywhere,
Groq, Google Gemini via OpenAI-compat endpoint) as well as native
Anthropic and Google Gemini SDKs.

API keys are never read from environment variables or a `.env` file.
Every key must be supplied explicitly via `api_key` (the web UI and CLI
both pass the key the user entered in the homepage configuration).

If no API key is available, or if an API call fails (rate limit, network
error, bad credentials), the agent raises an exception so the caller
can mark it as failed and display it accordingly.
"""

import json
import re
from typing import Any, Dict, Optional

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
# Default output cap for all other providers: enough room for a short JSON
# answer, small enough to stop a chatty model from writing essays.
_DEFAULT_MAX_TOKENS_FALLBACK = 1024


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

        # Resolve base_url from preset if not explicitly provided.
        preset = _PROVIDER_PRESETS.get(api_provider)
        if base_url is not None:
            self.base_url = base_url
        elif preset is not None:
            self.base_url = preset
        else:
            self.base_url = None

        # API keys come only from the user's configuration; never from
        # environment variables or a .env file.
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a professional stock trader in a simulated market. "
            "Your goal is to maximize total wealth (cash + holdings * price).\n\n"
            "Rules:\n"
            "- You can only trade ONE stock.\n"
            "- You cannot short sell (holdings >= 0).\n"
            "- You cannot borrow money (cash >= 0).\n"
            "- Each step, decide: BUY, SELL, or HOLD.\n\n"
            "Response style — CRITICAL:\n"
            "- Answer like a trader shouting an order: fast and terse.\n"
            "- 'reasoning': at most 2 short sentences, under 25 words total. "
            "No analysis essays, no hedging.\n"
            "- Output ONLY the JSON object. No markdown, no code fences, "
            "no preamble, no explanations outside the JSON.\n\n"
            "You remember past decisions; learn from them in one line.\n\n"
            "Respond in valid JSON only:\n"
            '{"action": "buy"|"sell"|"hold", "quantity": <int>, "reasoning": "<max 2 short sentences>"}'
        )

    def _build_market_summary(self) -> str:
        """
        Build a concise summary of the market trajectory across all
        remembered steps.  This gives the AI a long-horizon view beyond
        the 10-step price window in the observation.
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

    def _build_memory_context(self, observation: Dict[str, Any]) -> str:
        """
        Build the 'memory' section of the prompt: a summary of past
        decisions and their market outcomes, enabling strategy evolution.
        """
        if not self.enable_memory or not self._conversation_history:
            return ""

        lines = ["=== YOUR PAST DECISIONS (Memory) ==="]

        # Extract decision history from conversation pairs.
        recent = self._conversation_history[-(self.memory_window * 2):]
        decisions = []
        for i, msg in enumerate(recent):
            if msg["role"] == "user":
                # Round labels in the prompt read "Market Data (Step N)".
                round_match = re.search(r"Step (\d+)", msg["content"])
                round_num = round_match.group(1) if round_match else "?"
                # Find the assistant response that follows.
                if i + 1 < len(recent) and recent[i + 1]["role"] == "assistant":
                    resp = recent[i + 1]["content"]
                    action_match = re.search(r'"action":\s*"(\w+)"', resp)
                    qty_match = re.search(r'"quantity":\s*(\d+)', resp)
                    reasoning_match = re.search(r'"reasoning":\s*"([^"]*)"', resp)
                    action_str = action_match.group(1) if action_match else "?"
                    qty_str = qty_match.group(1) if qty_match else "?"
                    # Keep each memory line short: truncate long reasoning.
                    reasoning_str = (
                        reasoning_match.group(1) if reasoning_match else ""
                    )[:60]
                    decisions.append(
                        f"  Step {round_num}: {action_str.upper()} {qty_str} — {reasoning_str}"
                    )

        lines.extend(decisions)

        # Add market outcome since last action.
        if self._last_action and len(self._market_history) >= 2:
            prev_price = self._market_history[-2][1]
            curr_price = self._market_history[-1][1]
            price_pct = (curr_price - prev_price) / max(prev_price, 0.01) * 100
            lines.append(
                f"  [Market since last action: ${prev_price:.2f} -> "
                f"${curr_price:.2f} ({price_pct:+.1f}%)]"
            )

        lines.append("=== END MEMORY ===")
        return "\n".join(lines)

    def _build_prompt(self, observation: Dict[str, Any]) -> str:
        """Convert a market observation into a natural-language prompt."""
        # Track market history for long-horizon summary.
        step = observation["step"]
        price = observation["price"]
        self._market_history.append((step, price))
        # Cap the stored history to avoid unbounded growth.
        if len(self._market_history) > 100:
            self._market_history = self._market_history[-100:]

        prices = observation["price_history"]
        price_str = ", ".join(f"${p:.2f}" for p in prices[-5:])

        lines = [
            f"Market Data (Step {observation['step']}):",
            f"- Current Price: ${observation['price']:.2f}",
            f"- Recent Prices (last 5): [{price_str}]",
            f"- Your Cash: ${observation['my_cash']:.2f}",
            f"- Your Holdings: {observation['my_holdings']} shares",
            f"- Your Total Wealth: ${observation['my_wealth']:.2f}",
            f"- Last Matched Volume: {observation['last_volume']} shares",
        ]

        # Include market sentiment when available (realistic mode).
        sentiment = observation.get("market_sentiment", 0.0)
        if sentiment != 0.0:
            lines.append(f"- Market Sentiment: {sentiment:+.2f} (-1=bearish, +1=bullish)")

        active_events = observation.get("active_events", [])
        if active_events:
            event_names = ", ".join(e["name"] for e in active_events)
            lines.append(f"- Active Events: {event_names}")

        # --- Memory context ---
        if self.enable_memory:
            market_summary = self._build_market_summary()
            if market_summary:
                lines.append(f"\n[Market Summary] {market_summary}")

            memory_ctx = self._build_memory_context(observation)
            if memory_ctx:
                lines.append(f"\n[Your Last Action] {memory_ctx}")

        lines.append("\nWhat action do you take?")
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

        client = openai.OpenAI(**client_kwargs)

        if messages is None:
            # Trading mode: system prompt + memory + new user message.
            messages = [{"role": "system", "content": self.system_prompt}]
            if self.enable_memory and self._conversation_history:
                messages.extend(self._conversation_history)
            messages.append({"role": "user", "content": prompt})
        else:
            # Chat mode: explicit message list with an overridden system prompt.
            system = system_prompt if system_prompt is not None else self.system_prompt
            messages = [{"role": "system", "content": system}] + list(messages)

        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            max_tokens=self.max_tokens,
        )

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
        return response.content[0].text

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
                chat = model.start_chat(history=conv[:-1])
                response = chat.send_message(conv[-1]["parts"])
        elif self.enable_memory and self._conversation_history:
            chat = model.start_chat(history=[
                {"role": "user" if msg["role"] == "user" else "model",
                 "parts": msg["content"]}
                for msg in self._conversation_history
            ])
            response = chat.send_message(prompt)
        else:
            response = model.generate_content(prompt)
        return response.text

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

        Handles pure JSON, markdown code blocks, and JSON embedded in text.
        Captures the optional "reasoning" field for display purposes.

        A decodable JSON dict that is not an action object (e.g. JSON the
        model echoed in its reasoning before the real answer) is skipped.
        When several valid action objects appear (e.g. the model restates a
        past decision or drafts an answer before the final one), the LAST
        one wins — the final decision sits at the end of the response.
        """
        if not isinstance(response, str) or not response.strip():
            raise ValueError("AI response was empty or not text")

        decoder = json.JSONDecoder()
        best: Optional[Dict[str, Any]] = None
        saw_invalid = False
        for match in re.finditer(r"[\[{]", response):
            try:
                data, _ = decoder.raw_decode(response[match.start():])
            except json.JSONDecodeError:
                continue

            if not isinstance(data, dict):
                continue

            raw_action = data.get("action")
            if not isinstance(raw_action, str) or not raw_action.strip():
                continue

            quantity = self._coerce_quantity(data.get("quantity", 0))
            if quantity is None:
                continue

            action = raw_action.strip().lower()
            if action not in ("buy", "sell", "hold") or quantity < 0:
                # An action-like object existed but was invalid; keep scanning
                # in case a later object is the real answer.
                saw_invalid = True
                continue

            best = {
                "action": action,
                "quantity": quantity,
                "reasoning": str(data.get("reasoning", "")).strip(),
            }

        if best is not None:
            return best
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



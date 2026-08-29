"""
Agent personality: a disposition written into the model's system prompt,
plus an evolving mood fed back into each round's prompt.

The agent's own reasoning is the ONLY thing that decides its trades. This
module never rewrites a decision -- it shapes the prompt so the model acts
in character, then passes whatever the model returns straight through. That
is why a decision can never contradict the reasoning printed beside it.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..base_agent import BaseAgent
from .mood import MOOD_AXES as MOOD_AXES  # re-exported for existing importers
from .mood import objective_pressure, settle_mood

# Fixed preamble: pulls the model out of calm-optimizer mode and ties the
# decision to the reasoning. Prepended to every persona, both depths.
_IDENTITY_LINES = (
    "You are a real trader, not a calculator. You have hunches, a temper, "
    "and pride. Trade the way YOU would, not the textbook-optimal way. "
    "Being wrong or emotional is fine; pretending you're a calm machine is not."
)

_IN_CHARACTER_CHECK = (
    "Before you answer, check: would the trader described above really do "
    "this, or are you being too reasonable? Stay in character. Your reasoning "
    "and your action must agree -- never explain a hold and then trade, or "
    "explain a buy and then sell."
)

# The three mood axes (0-10) and the rule-based mood dynamics live in
# ``mood``; ``MOOD_AXES`` is re-exported here for existing importers.

# The seven per-agent sensitivity dials, 0-10. They shape how fast mood
# moves and add a sentence to the disposition; they never touch a decision.
DIAL_NAMES = (
    "risk_appetite",
    "loss_sensitivity",
    "herd_pull",
    "patience",
    "resilience",
    "envy",
    "conviction",
)

# Value-bucketed sentences per dial: (low, mid, high). Rendered into the
# disposition so the model reads its dials as character, not numbers.
_DIAL_SENTENCES: Dict[str, Tuple[str, str, str]] = {
    "risk_appetite": (
        "You keep your bets small; a big position makes you uneasy.",
        "You size positions sensibly, neither timid nor reckless.",
        "You like big positions and you are comfortable betting heavily.",
    ),
    "loss_sensitivity": (
        "Losses roll off you; a red round barely registers.",
        "Losses bother you about as much as you would expect.",
        "Losses sting badly and stay with you long after the round ends.",
    ),
    "herd_pull": (
        "You barely care what other traders are doing.",
        "You notice what others do but you make your own call.",
        "You feel a strong pull to move with the crowd.",
    ),
    "patience": (
        "You are itchy; sitting still for long feels like wasted time.",
        "You can wait for a setup, but not forever.",
        "You are patient and happy to wait many rounds for the right moment.",
    ),
    "resilience": (
        "Once rattled you stay rattled for a long time.",
        "You recover from a bad round at a normal pace.",
        "You shake off bad rounds quickly and reset to your baseline.",
    ),
    "envy": (
        "Other traders getting ahead of you does not bother you.",
        "You notice when someone is beating you and it nags a little.",
        "It eats at you when another trader is ahead of you.",
    ),
    "conviction": (
        "You change your mind easily and often.",
        "You hold a view until there is a decent reason to drop it.",
        "Once you form a view you stick to it hard.",
    ),
}

# Per-personality profile fed to the model.
#   "short" -> simple mode: one line, cheap.
#   "full"  -> deep mode: a paragraph with enough texture to roleplay.
#   "mood"  -> starting confidence/stress/frustration.
#   "dials" -> starting sensitivity profile.
# Written in the second person. Display labels live in
# _PERSONALITY_DESCRIPTIONS below and are deliberately kept separate.
_PERSONALITY_DISPOSITIONS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "short": "You are a level-headed trader with no strong biases.",
        "full": (
            "You are a level-headed trader. You weigh what you see without "
            "letting fear or excitement run the show, and you are comfortable "
            "sitting on your hands when nothing looks compelling. You are not "
            "a robot -- you just don't rattle easily."
        ),
        "mood": {"confidence": 5.0, "stress": 3.0, "frustration": 2.0},
        "dials": {"risk_appetite": 5, "loss_sensitivity": 5, "herd_pull": 5,
                  "patience": 5, "resilience": 5, "envy": 4, "conviction": 5},
    },
    "aggressive": {
        "short": (
            "You are an aggressive, impatient trader who bets big and hates "
            "sitting still."
        ),
        "full": (
            "You are aggressive and impatient. You bet big when you like "
            "something, and you hate sitting on the sidelines watching other "
            "people make money. Half measures annoy you. You would rather be "
            "decisively wrong than timidly right, and you back your "
            "convictions with size."
        ),
        "mood": {"confidence": 7.0, "stress": 3.0, "frustration": 3.0},
        "dials": {"risk_appetite": 9, "loss_sensitivity": 3, "herd_pull": 2,
                  "patience": 3, "resilience": 7, "envy": 6, "conviction": 7},
    },
    "conservative": {
        "short": (
            "You are a cautious trader who protects capital before chasing "
            "gains."
        ),
        "full": (
            "You are cautious. Protecting what you have matters more to you "
            "than catching every move, and a loss bothers you far more than a "
            "missed gain pleases you. You want a reason before you commit, "
            "you keep positions modest, and you are quick to step back when "
            "things get murky."
        ),
        "mood": {"confidence": 4.0, "stress": 4.0, "frustration": 2.0},
        "dials": {"risk_appetite": 2, "loss_sensitivity": 7, "herd_pull": 4,
                  "patience": 8, "resilience": 5, "envy": 3, "conviction": 6},
    },
    "panicky": {
        "short": (
            "You are an anxious trader who feels losses sharply and wants out "
            "when things drop."
        ),
        "full": (
            "You are anxious and thin-skinned about losses. When the market "
            "drops or your portfolio slips you feel it in your stomach, and "
            "your instinct is to get out and make the discomfort stop -- even "
            "when part of you knows it is the wrong moment. You second-guess "
            "yourself constantly and your reasoning shows it."
        ),
        "mood": {"confidence": 3.0, "stress": 6.0, "frustration": 4.0},
        "dials": {"risk_appetite": 3, "loss_sensitivity": 9, "herd_pull": 6,
                  "patience": 4, "resilience": 2, "envy": 5, "conviction": 3},
    },
    "greedy": {
        "short": "You are a greedy trader who holds winners far too long.",
        "full": (
            "You are greedy. When a position is working you want more of it, "
            "and selling a winner feels like leaving money on the table, so "
            "you hold long past the point a calmer person would take profits. "
            "Gains make you want to press, not trim."
        ),
        "mood": {"confidence": 6.0, "stress": 3.0, "frustration": 3.0},
        "dials": {"risk_appetite": 7, "loss_sensitivity": 4, "herd_pull": 3,
                  "patience": 7, "resilience": 6, "envy": 7, "conviction": 8},
    },
    "fomo_driven": {
        "short": (
            "You are a FOMO-driven trader who chases anything already running."
        ),
        "full": (
            "You are driven by fear of missing out. A stock that is already "
            "running pulls at you hard -- the thought of watching it climb "
            "without you is worse than the risk of buying late. You chase "
            "strength, you buy after the move, and you rationalize it on the "
            "way in."
        ),
        "mood": {"confidence": 5.0, "stress": 5.0, "frustration": 5.0},
        "dials": {"risk_appetite": 8, "loss_sensitivity": 5, "herd_pull": 9,
                  "patience": 2, "resilience": 4, "envy": 9, "conviction": 3},
    },
    "stubborn": {
        "short": (
            "You are a stubborn trader who sticks to a call long after the "
            "evidence turns."
        ),
        "full": (
            "You are stubborn. Once you have made a call you dig in, and new "
            "information that contradicts you feels like noise rather than a "
            "signal. Admitting you were wrong costs you something, so you "
            "tend to repeat yourself and wait for the market to come around "
            "to your view."
        ),
        "mood": {"confidence": 6.0, "stress": 3.0, "frustration": 4.0},
        "dials": {"risk_appetite": 5, "loss_sensitivity": 4, "herd_pull": 1,
                  "patience": 8, "resilience": 6, "envy": 3, "conviction": 10},
    },
    "emotional": {
        "short": "You are a volatile trader whose mood swings drive your trades.",
        "full": (
            "You are emotional and streaky. Your mood drives your trading "
            "more than any plan does -- a couple of wins and you feel "
            "invincible, a couple of losses and you want out of everything. "
            "You know this about yourself and it still happens. Your "
            "reasoning swings with your feelings."
        ),
        "mood": {"confidence": 5.0, "stress": 5.0, "frustration": 5.0},
        "dials": {"risk_appetite": 6, "loss_sensitivity": 8, "herd_pull": 7,
                  "patience": 3, "resilience": 2, "envy": 7, "conviction": 3},
    },
    "custom": {
        "short": "You are a trader with your own particular way of doing things.",
        "full": (
            "You are a trader with your own particular way of doing things. "
            "You follow your own read of the market rather than any standard "
            "playbook."
        ),
        "mood": {"confidence": 5.0, "stress": 3.0, "frustration": 2.0},
        "dials": {"risk_appetite": 5, "loss_sensitivity": 5, "herd_pull": 5,
                  "patience": 5, "resilience": 5, "envy": 4, "conviction": 5},
    },
}

# Free-text persona fields are capped so a pasted essay cannot blow up
# every prompt (config_store applies the same limit on the way in).
_MAX_PERSONA_CHARS = 1024


def _preset(personality: str) -> Dict[str, Any]:
    """Return a personality preset, falling back to "balanced"."""
    return _PERSONALITY_DISPOSITIONS.get(
        personality, _PERSONALITY_DISPOSITIONS["balanced"]
    )


def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))


def preset_dials(personality: str = "balanced") -> Dict[str, float]:
    """Starting dial profile for a personality preset."""
    return {k: float(v) for k, v in _preset(personality)["dials"].items()}


def preset_mood(personality: str = "balanced") -> Dict[str, float]:
    """Starting mood for a personality preset."""
    return {k: float(v) for k, v in _preset(personality)["mood"].items()}


def resolve_dials(
    personality: str = "balanced", overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """Merge per-trader dial overrides over the preset profile."""
    dials = preset_dials(personality)
    for name in DIAL_NAMES:
        if not overrides:
            continue
        raw = overrides.get(name)
        if raw is None:
            continue
        try:
            dials[name] = _clamp(float(raw))
        except (TypeError, ValueError):
            continue
    return dials


def _dial_sentences(dials: Dict[str, float]) -> List[str]:
    """One value-bucketed sentence per dial (<=3 low, >=7 high)."""
    out = []
    for name in DIAL_NAMES:
        value = dials.get(name)
        if value is None:
            continue
        low, mid, high = _DIAL_SENTENCES[name]
        out.append(low if value <= 3 else (high if value >= 7 else mid))
    return out


def build_disposition(
    personality: str = "balanced",
    deep: bool = False,
    dials: Optional[Dict[str, float]] = None,
    trait_notes: str = "",
    persona: str = "",
) -> str:
    """Assemble the persona text for a personality preset.

    ``deep`` picks the full paragraph over the one-liner and adds the dial
    sentences plus any free-text. ``persona`` replaces the preset paragraph;
    ``trait_notes`` is appended verbatim after the dial sentences. Unknown
    personalities fall back to "balanced".
    """
    body = persona.strip()[:_MAX_PERSONA_CHARS] if (deep and persona) else ""
    if not body:
        body = _preset(personality)["full" if deep else "short"]

    parts = [_IDENTITY_LINES, body]
    if deep:
        sentences = _dial_sentences(
            dials if dials is not None else preset_dials(personality)
        )
        if sentences:
            parts.append(" ".join(sentences))
        notes = trait_notes.strip()[:_MAX_PERSONA_CHARS] if trait_notes else ""
        if notes:
            parts.append(notes)
    parts.append(_IN_CHARACTER_CHECK)
    return "\n\n".join(parts)


class TraitAgent(BaseAgent):
    """
    Wrapper that gives a base agent a personality and a mood.

    The personality is expressed entirely through the base agent's system
    prompt and the per-round persona block; ``act`` returns the base agent's
    decision untouched.

    Parameters
    ----------
    base_agent : BaseAgent
        The underlying agent whose prompt carries the personality.
    personality_name : str
        Display name of the personality preset.
    disposition : str, optional
        The assembled persona text, replayed in every deep-mode prompt.
    deep : bool
        Whether deep mode is on (mood tracking and the persona block).
    dials : dict, optional
        Sensitivity profile; defaults to the preset's.
    mood_max_step : float
        Largest per-round change allowed on any mood axis.
    mood_intensity : float
        Scales how strongly round events move mood.
    use_reported_mood : bool
        When True (default) the model's reported mood is used as a bounded
        adjustment around the rule-based mood; False gives a fully
        deterministic formula-only run.
    """

    # Temporary balances stored via the cash/holdings setters while
    # super().__init__() runs (before base_agent exists).
    _cash_tmp: float
    _holdings_tmp: Any

    def __init__(
        self,
        base_agent: BaseAgent,
        personality_name: str = "custom",
        disposition: str = "",
        deep: bool = False,
        dials: Optional[Dict[str, float]] = None,
        mood_max_step: float = 3.0,
        mood_intensity: float = 1.0,
        use_reported_mood: bool = True,
    ):
        # Copy base agent properties for BaseAgent.__init__
        super().__init__(base_agent.agent_id, base_agent.cash, base_agent.holdings)
        self.base_agent = base_agent

        # During super().__init__() the cash/holdings setters stored the
        # constructor values in _tmp attributes (base_agent did not exist yet).
        # Propagate those to the base agent so a different starting balance is
        # not silently discarded.
        if hasattr(self, "_cash_tmp"):
            self.base_agent.cash = self._cash_tmp
        if hasattr(self, "_holdings_tmp"):
            self.base_agent.holdings = self._holdings_tmp

        # Personality label for display
        self.personality_name = personality_name
        # Persona text handed to the model (empty when built directly).
        self.disposition = disposition
        self.deep = deep
        self.dials: Dict[str, float] = (
            dict(dials) if dials is not None else preset_dials(personality_name)
        )
        self.mood: Dict[str, float] = preset_mood(personality_name)
        self.mood_max_step = float(mood_max_step)
        self.mood_intensity = float(mood_intensity)
        self.use_reported_mood = bool(use_reported_mood)
        self._mood_baseline: Dict[str, float] = preset_mood(personality_name)

        # Wealth / streak tracking feeding the mood engine.
        self._initial_wealth: float = 0.0
        self._peak_wealth: Optional[float] = None
        self._prev_wealth: Optional[float] = None
        self._prev_rank: Optional[int] = None
        self._loss_streak: int = 0
        self._win_streak: int = 0
        self._first_act: bool = True

    @property
    def cash(self) -> float:
        """Delegate to base_agent so state stays in sync."""
        if hasattr(self, "base_agent"):
            return self.base_agent.cash
        return self._cash_tmp

    @cash.setter
    def cash(self, value: float) -> None:
        if hasattr(self, "base_agent"):
            self.base_agent.cash = value
        else:
            # During super().__init__, base_agent doesn't exist yet.
            object.__setattr__(self, "_cash_tmp", value)

    @property
    def holdings(self) -> Any:
        """Delegate to base_agent so state stays in sync.

        Multi-stock mode stores holdings as a ``{name: qty}`` dict; the
        loose annotation keeps legacy int holdings working too.
        """
        if hasattr(self, "base_agent"):
            return self.base_agent.holdings
        return self._holdings_tmp

    @holdings.setter
    def holdings(self, value: Any) -> None:
        if hasattr(self, "base_agent"):
            self.base_agent.holdings = value
        else:
            object.__setattr__(self, "_holdings_tmp", value)

    # ------------------------------------------------------------------
    # Mood engine (deep mode only) -- shapes the prompt, never the decision.
    # The dynamics live in agents/mood.py; this class only carries the state.
    # ------------------------------------------------------------------

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Return the base agent's decision, unmodified.

        In deep mode the persona and current mood are attached to the
        observation on the way in, and the model's reported mood is settled
        on the way out. The ``decisions`` list is never touched: anything
        that edited the action would put it at odds with the reasoning the
        model wrote to justify it.
        """
        # On first act, capture the real initial wealth from observation.
        if self._first_act:
            self._initial_wealth = observation.get("my_wealth", self._initial_wealth)
            self._first_act = False

        # Update peak wealth tracking
        current_wealth = observation.get("my_wealth", 0)
        if self._peak_wealth is None or current_wealth > self._peak_wealth:
            self._peak_wealth = current_wealth

        if not self.deep:
            self._prev_wealth = current_wealth
            return self.base_agent.act(observation)

        pressure, metrics = objective_pressure(
            observation,
            peak_wealth=self._peak_wealth,
            prev_wealth=self._prev_wealth,
            prev_rank=self._prev_rank,
            loss_streak=self._loss_streak,
            win_streak=self._win_streak,
        )
        observation["persona"] = {
            "name": self.agent_id,
            "disposition": self.disposition,
            "mood": dict(self.mood),
            "dials": dict(self.dials),
            "pressure": pressure,
            "scale_hint": "0 = none at all, 10 = as strong as it gets",
        }

        result = self.base_agent.act(observation)

        self.mood = settle_mood(
            self.mood,
            self._mood_baseline,
            self.dials,
            result.get("mood") if isinstance(result, dict) else None,
            metrics,
            use_reported_mood=self.use_reported_mood,
            mood_intensity=self.mood_intensity,
            mood_max_step=self.mood_max_step,
        )
        if isinstance(result, dict):
            result["mood"] = dict(self.mood)

        # Streaks for the next round's pressure line.
        if self._prev_wealth is not None:
            if current_wealth < self._prev_wealth:
                self._loss_streak += 1
                self._win_streak = 0
            elif current_wealth > self._prev_wealth:
                self._win_streak += 1
                self._loss_streak = 0
        self._prev_wealth = current_wealth
        self._prev_rank = metrics.get("rank") or self._prev_rank
        return result

    @property
    def wealth(self) -> float:
        return self.base_agent.wealth

    @property
    def personality_description(self) -> str:
        """Return a human-readable description of this agent's personality."""
        return _PERSONALITY_DESCRIPTIONS.get(
            self.personality_name, self.personality_name
        )

    def __repr__(self) -> str:
        return (
            f"TraitAgent({self.base_agent.__class__.__name__}, "
            f"{self.personality_name})"
        )


def create_personality_agent(
    base_agent: BaseAgent,
    personality: str = "balanced",
    deep: bool = False,
    dials: Optional[Dict[str, Any]] = None,
    trait_notes: str = "",
    persona: str = "",
    mood_max_step: float = 3.0,
    mood_intensity: float = 1.0,
    use_reported_mood: bool = True,
) -> TraitAgent:
    """
    Create an agent whose system prompt carries a named personality.

    Parameters
    ----------
    base_agent : BaseAgent
        The underlying agent.
    personality : str
        One of: "balanced", "aggressive", "conservative", "panicky",
        "greedy", "fomo_driven", "stubborn", "emotional"
    deep : bool
        Use the full disposition paragraph, the dial sentences and the free
        text, and ask for longer, in-character reasoning. Default False.
    dials : dict, optional
        Per-trader dial overrides merged over the preset profile.
    trait_notes : str
        Extra character notes appended to the disposition (deep mode).
    persona : str
        Replaces the preset disposition paragraph entirely (deep mode).
    use_reported_mood : bool
        When True (default) the model's reported mood is a bounded
        adjustment around the rule-based mood; False ignores it entirely
        (deterministic mood).

    Returns
    -------
    TraitAgent
        Agent whose base system prompt states who it is.
    """
    resolved_dials = resolve_dials(personality, dials)
    disposition = build_disposition(
        personality,
        deep=deep,
        dials=resolved_dials,
        trait_notes=trait_notes,
        persona=persona,
    )

    # Prepend the persona to the agent's own format rules so the JSON
    # contract survives. Agents without the ExternalAIAgent prompt API
    # (plain BaseAgent test doubles) simply keep their own behavior.
    build_rules = getattr(base_agent, "_default_system_prompt", None)
    if build_rules is not None and hasattr(base_agent, "system_prompt"):
        base_agent.system_prompt = (
            f"{disposition}\n\n{build_rules(deep=deep)}"
        )

    return TraitAgent(
        base_agent,
        personality_name=personality,
        disposition=disposition,
        deep=deep,
        dials=resolved_dials,
        mood_max_step=mood_max_step,
        mood_intensity=mood_intensity,
        use_reported_mood=use_reported_mood,
    )


# Human-readable descriptions for each personality preset (UI labels).
_PERSONALITY_DESCRIPTIONS = {
    "balanced": "Balanced — no strong biases",
    "aggressive": "Aggressive — overconfident, greedy, FOMO-driven",
    "conservative": "Conservative — loss-averse, regret-averse",
    "panicky": "Panicky — panic sells on drawdowns, quick to cut losses",
    "greedy": "Greedy — holds winners too long, overconfident",
    "fomo_driven": "FOMO-driven — buys impulsively on rallies",
    "stubborn": "Stubborn — repeats previous actions, resists new signals",
    "emotional": "Emotional — volatile mix of panic, greed, FOMO, and loss aversion",
    "custom": "Custom trait configuration",
}

"""
Agent personality traits for realistic simulation mode.

Traits modify agent behavior to simulate human-like biases and emotions:
panic selling, greed, FOMO, stubbornness, loss aversion, and more.
"""

import random
from typing import Any, Dict, Optional

from ..base_agent import BaseAgent


class TraitAgent(BaseAgent):
    """
    Wrapper that adds personality traits to any base agent.

    Parameters
    ----------
    base_agent : BaseAgent
        The underlying agent whose behavior will be modified.
    panic : float
        Probability of panic selling when drawdown exceeds threshold. (0.0-1.0)
    greed : float
        Tendency to hold winners too long. Higher = less likely to sell gains. (0.0-1.0)
    fomo : float
        Fear of missing out: buy impulsively on strong upward moves. (0.0-1.0)
    stubbornness : float
        Repeat previous action, resisting new information. (0.0-1.0)
    loss_aversion : float
        Quick to cut losses, selling on small drops. (0.0-1.0)
    overconfidence : float
        Trade larger sizes than strategy suggests. (0.0-1.0)
    regret_avoidance : float
        Hold losers to avoid realizing losses. (0.0-1.0)
    """

    def __init__(
        self,
        base_agent: BaseAgent,
        panic: float = 0.0,
        greed: float = 0.0,
        fomo: float = 0.0,
        stubbornness: float = 0.0,
        loss_aversion: float = 0.0,
        overconfidence: float = 0.0,
        regret_avoidance: float = 0.0,
        personality_name: str = "custom",
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

        # Trait strengths (0.0 to 1.0)
        self.panic = max(0.0, min(1.0, panic))
        self.greed = max(0.0, min(1.0, greed))
        self.fomo = max(0.0, min(1.0, fomo))
        self.stubbornness = max(0.0, min(1.0, stubbornness))
        self.loss_aversion = max(0.0, min(1.0, loss_aversion))
        self.overconfidence = max(0.0, min(1.0, overconfidence))
        self.regret_avoidance = max(0.0, min(1.0, regret_avoidance))

        # Track actual initial wealth for regret avoidance
        # Will be set on first act from observation['my_wealth'] for correctness
        self._initial_wealth: float = 0.0

        # Default RNG; may be overridden by MarketEnv injection
        import random as _random
        self.rng = _random

        # Track state for trait logic
        self._peak_wealth: Optional[float] = None
        self._last_action: Optional[str] = None
        self._last_quantity: int = 0
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
    def holdings(self) -> int:
        """Delegate to base_agent so state stays in sync."""
        if hasattr(self, "base_agent"):
            return self.base_agent.holdings
        return self._holdings_tmp

    @holdings.setter
    def holdings(self, value: int) -> None:
        if hasattr(self, "base_agent"):
            self.base_agent.holdings = value
        else:
            object.__setattr__(self, "_holdings_tmp", value)

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get base agent's action, then apply trait modifications.
        """
        # On first act, capture the real initial wealth from observation.
        if self._first_act:
            self._initial_wealth = observation.get("my_wealth", self._initial_wealth)
            self._first_act = False

        # Update peak wealth tracking
        current_wealth = observation.get("my_wealth", 0)
        if self._peak_wealth is None or current_wealth > self._peak_wealth:
            self._peak_wealth = current_wealth

        # Sync state: MarketEnv may have updated our cash/holdings externally.
        # Since we delegate via properties, base_agent is already in sync.

        # Get base agent's decision
        base_action = self.base_agent.act(observation)
        action_type = base_action.get("action", "hold")
        quantity = max(0, int(base_action.get("quantity", 0)))
        base_reasoning = base_action.get("reasoning", "")
        original_action = action_type

        # Apply traits in order
        action_type, quantity = self._apply_panic(observation, action_type, quantity)
        action_type, quantity = self._apply_fomo(observation, action_type, quantity)
        action_type, quantity = self._apply_loss_aversion(observation, action_type, quantity)
        action_type, quantity = self._apply_greed(observation, action_type, quantity)
        action_type, quantity = self._apply_regret_avoidance(observation, action_type, quantity)
        action_type, quantity = self._apply_stubbornness(action_type, quantity)
        action_type, quantity = self._apply_overconfidence(action_type, quantity)

        # Remember for next step
        self._last_action = action_type
        self._last_quantity = quantity

        result = {"action": action_type, "quantity": quantity}
        if base_reasoning:
            if action_type != original_action:
                result["reasoning"] = f"[trait override] {base_reasoning}"
            else:
                result["reasoning"] = base_reasoning
        return result

    def _apply_panic(
        self, obs: Dict[str, Any], action: str, quantity: int
    ) -> tuple[str, int]:
        """Panic sell on significant drawdown."""
        if self.panic <= 0:
            return action, quantity

        if self._peak_wealth is None or self._peak_wealth <= 0:
            return action, quantity

        # Calculate drawdown from peak
        current_wealth = obs.get("my_wealth", 0)
        drawdown = (self._peak_wealth - current_wealth) / self._peak_wealth

        # Trigger panic if drawdown > 10%
        if drawdown > 0.10:
            if self.rng.random() < self.panic:
                # Panic sell everything
                holdings = obs.get("my_holdings", 0)
                if holdings > 0:
                    return "sell", holdings

        return action, quantity

    def _apply_fomo(
        self, obs: Dict[str, Any], action: str, quantity: int
    ) -> tuple[str, int]:
        """FOMO buy on strong upward momentum."""
        if self.fomo <= 0:
            return action, quantity

        prices = obs.get("price_history", [])
        if len(prices) < 3:
            return action, quantity

        # Check for 3+ consecutive up moves
        recent = prices[-3:]
        if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
            if self.rng.random() < self.fomo:
                # FOMO buy with available cash
                price = obs.get("price", 100)
                cash = obs.get("my_cash", 0)
                if price > 0 and cash > 0:
                    fomo_qty = int(cash * 0.5 / price)  # Use 50% of cash
                    if fomo_qty > 0:
                        return "buy", fomo_qty

        return action, quantity

    def _apply_loss_aversion(
        self, obs: Dict[str, Any], action: str, quantity: int
    ) -> tuple[str, int]:
        """Quick to cut losses on small drops."""
        if self.loss_aversion <= 0:
            return action, quantity

        prices = obs.get("price_history", [])
        if len(prices) < 2:
            return action, quantity

        # Check for recent drop
        if prices[-1] < prices[-2] * 0.97:  # 3%+ drop
            if self.rng.random() < self.loss_aversion:
                holdings = obs.get("my_holdings", 0)
                if holdings > 0:
                    # Sell half of holdings
                    return "sell", max(1, holdings // 2)

        return action, quantity

    def _apply_greed(
        self, obs: Dict[str, Any], action: str, quantity: int
    ) -> tuple[str, int]:
        """Hold winners longer, resist selling gains."""
        if self.greed <= 0:
            return action, quantity

        if action != "sell":
            return action, quantity

        # Check if we're in a real profit position. Compare against the
        # initial wealth: _peak_wealth is updated to include the current
        # wealth every round, so "current > peak * 1.05" could never be true.
        if self._initial_wealth <= 0:
            return action, quantity

        current_wealth = obs.get("my_wealth", 0)
        if current_wealth > self._initial_wealth * 1.05:  # 5%+ gain
            if self.rng.random() < self.greed:
                # Reduce or cancel sell
                if self.rng.random() < 0.5:
                    return "hold", 0
                else:
                    return "sell", max(1, quantity // 2)

        return action, quantity

    def _apply_regret_avoidance(
        self, obs: Dict[str, Any], action: str, quantity: int
    ) -> tuple[str, int]:
        """Hold losers to avoid realizing losses."""
        if self.regret_avoidance <= 0:
            return action, quantity

        if action != "sell":
            return action, quantity

        # Check if we're losing money relative to actual initial wealth
        current_cash = obs.get("my_cash", 0)
        holdings = obs.get("my_holdings", 0)
        price = obs.get("price", 100)

        current_wealth = current_cash + holdings * price

        # If we're down and trying to sell holdings
        if holdings > 0 and current_wealth < self._initial_wealth * 0.95:
            if self.rng.random() < self.regret_avoidance:
                # Resist selling, hold onto loser
                return "hold", 0

        return action, quantity

    def _apply_stubbornness(self, action: str, quantity: int) -> tuple[str, int]:
        """Repeat a previous active trade decision, ignoring new signals.

        A stale "hold" is never repeated: repeating it would silently
        swallow the agent's fresh buy/sell decisions most of the time
        (e.g. turning this round's "buy" into "hold").
        """
        if self.stubbornness <= 0:
            return action, quantity

        # Only stubbornly repeat a real previous trade (buy/sell with size).
        if self._last_action not in ("buy", "sell") or self._last_quantity <= 0:
            return action, quantity

        if self.rng.random() < self.stubbornness:
            # Repeat last action
            return self._last_action, self._last_quantity

        return action, quantity

    def _apply_overconfidence(self, action: str, quantity: int) -> tuple[str, int]:
        """Trade larger sizes than strategy suggests."""
        if self.overconfidence <= 0:
            return action, quantity

        if action == "hold" or quantity <= 0:
            return action, quantity

        if self.rng.random() < self.overconfidence:
            # Increase trade size by 50-100%
            multiplier = 1.5 + self.rng.random() * 0.5
            return action, int(quantity * multiplier)

        return action, quantity

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
        traits = []
        if self.panic > 0:
            traits.append(f"panic={self.panic:.1f}")
        if self.greed > 0:
            traits.append(f"greed={self.greed:.1f}")
        if self.fomo > 0:
            traits.append(f"fomo={self.fomo:.1f}")
        if self.stubbornness > 0:
            traits.append(f"stubborn={self.stubbornness:.1f}")
        if self.loss_aversion > 0:
            traits.append(f"loss_averse={self.loss_aversion:.1f}")
        if self.overconfidence > 0:
            traits.append(f"overconfident={self.overconfidence:.1f}")
        if self.regret_avoidance > 0:
            traits.append(f"regret_averse={self.regret_avoidance:.1f}")

        trait_str = ", ".join(traits) if traits else "no traits"
        return f"TraitAgent({self.base_agent.__class__.__name__}, {trait_str})"


def create_personality_agent(
    base_agent: BaseAgent,
    personality: str = "balanced",
) -> TraitAgent:
    """
    Create a trait-enhanced agent with a named personality preset.

    Parameters
    ----------
    base_agent : BaseAgent
        The underlying agent.
    personality : str
        One of: "balanced", "aggressive", "conservative", "panicky",
        "greedy", "fomo_driven", "stubborn", "emotional"

    Returns
    -------
    TraitAgent
        Agent with personality traits applied.
    """
    presets = {
        "balanced": {},
        "aggressive": {
            "overconfidence": 0.6,
            "greed": 0.4,
            "fomo": 0.3,
        },
        "conservative": {
            "loss_aversion": 0.5,
            "regret_avoidance": 0.4,
        },
        "panicky": {
            "panic": 0.7,
            "loss_aversion": 0.6,
        },
        "greedy": {
            "greed": 0.8,
            "overconfidence": 0.5,
        },
        "fomo_driven": {
            "fomo": 0.8,
            "overconfidence": 0.4,
        },
        "stubborn": {
            "stubbornness": 0.7,
        },
        "emotional": {
            "panic": 0.4,
            "greed": 0.4,
            "fomo": 0.5,
            "loss_aversion": 0.4,
        },
    }

    traits = presets.get(personality, {})
    return TraitAgent(
        base_agent, personality_name=personality, **traits
    )


# Human-readable descriptions for each personality preset.
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
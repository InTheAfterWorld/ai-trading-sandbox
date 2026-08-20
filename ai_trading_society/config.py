"""
Configuration module for market and agent parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketConfig:
    """Global configuration for the market environment."""

    # --- Price parameters ---
    initial_price: float = 100.0
    """Initial stock price."""

    price_sensitivity: float = 0.02
    """Impact coefficient of net buying pressure on price movement."""

    max_price_change_ratio: float = 0.10
    """Maximum price change ratio per step to prevent extreme jumps."""

    min_price: float = 0.01
    """Lower price bound."""

    max_price: float = 1_000_000.0
    """Upper price bound."""

    # --- Observation parameters ---
    price_history_length: int = 20
    """Number of historical prices visible to each agent."""

    # --- Transaction costs ---
    fee_rate: float = 0.0
    """Transaction fee rate per trade (e.g., 0.001 = 0.1%). Applied to trade value."""

    slippage_rate: float = 0.0
    """Slippage rate applied to execution price (e.g., 0.001 = 0.1%).
    Buyers pay slightly more, sellers receive slightly less."""

    # --- Sandbox event settings ---
    event_probability_multiplier: float = 1.5
    """Multiplier for event probabilities in the unified sandbox."""

    random_traits: bool = True
    """Whether roster construction may assign randomized personality traits."""

    # --- Social influence ---
    social_influence: float = 0.0
    """Strength of social-relationship-driven behavior (0.0 = off, 1.0 = strong).
    Friends/idol trades are mimicked, enemies are faded, creating herding
    and bank-run cascades."""

    # --- Reproducibility / Random seed ---
    seed: Optional[int] = None
    """Random seed for reproducibility. If None, auto-generated at run start."""

    def __post_init__(self) -> None:
        """Clamp transaction-cost rates to a safe range.

        A ``fee_rate`` or ``slippage_rate`` >= 1 would make a seller receive
        negative revenue (revenue = value * (1 - slippage) * (1 - fee)),
        silently driving cash below zero with no way to recover. Clamp both
        to [0, 0.5] so any entry point (web, CLI, direct construction) is safe.
        """
        self.fee_rate = max(0.0, min(self.fee_rate, 0.5))
        self.slippage_rate = max(0.0, min(self.slippage_rate, 0.5))

    def to_dict(self) -> dict:
        """Serialize MarketConfig to dictionary."""
        return {
            "initial_price": self.initial_price,
            "price_sensitivity": self.price_sensitivity,
            "max_price_change_ratio": self.max_price_change_ratio,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "price_history_length": self.price_history_length,
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
            "event_probability_multiplier": self.event_probability_multiplier,
            "random_traits": self.random_traits,
            "social_influence": self.social_influence,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarketConfig":
        """Deserialize MarketConfig from dictionary."""
        data_copy = data.copy()
        data_copy.pop("mode", None)
        return cls(**data_copy)

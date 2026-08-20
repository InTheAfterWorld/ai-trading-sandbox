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

    # --- Reproducibility / Random seed ---
    seed: Optional[int] = None
    """Random seed for reproducibility. If None, auto-generated at run start."""


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
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarketConfig":
        """Deserialize MarketConfig from dictionary."""
        data_copy = data.copy()
        data_copy.pop("mode", None)
        return cls(**data_copy)



@dataclass
class AgentConfig:
    """Initial configuration for a single agent."""

    agent_id: str
    """Unique agent identifier."""

    agent_type: str = "external_ai"
    """External AI provider such as GPT, Claude, or Gemini."""

    initial_cash: float = 10000.0
    """Initial cash balance."""

    initial_holdings: int = 0
    """Initial share holdings."""

    # --- Strategy parameters ---
    risk_preference: float = 0.5
    """Risk preference: 0.0 = conservative, 1.0 = aggressive."""

    lookback: int = 5
    """Lookback window for trend or contrarian strategies."""

    threshold: float = 0.02
    """Trigger threshold for price-change-based strategies."""

    fair_value: float = 100.0
    """Intrinsic value used by value-investing strategies."""

    # --- External AI parameters ---
    api_provider: Optional[str] = None
    """API provider. Built-in presets:
    "openai", "anthropic", "google" (native SDKs);
    "openrouter", "chatanywhere", "groq", "google_compat" (OpenAI-compatible).
    """

    model: Optional[str] = None
    """Model name, e.g. "gpt-4o", "openai/gpt-oss-20b:free", "deepseek-r1"."""

    api_key: Optional[str] = None
    """API key. Must be provided explicitly; never read from the environment."""

    base_url: Optional[str] = None
    """Custom API base URL for OpenAI-compatible providers. Overrides preset."""

    system_prompt: Optional[str] = None
    """Custom system prompt."""

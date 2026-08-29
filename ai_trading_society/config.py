"""
Configuration module for market and agent parameters.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StockSpec:
    """Specification for a single stock in the multi-stock market.

    Each stock has an independent price series, order book, and
    per-agent holdings tracking.
    """
    name: str = "Stock 1"
    """Display name used as the primary identifier."""

    initial_price: float = 100.0
    """Initial price of this stock."""

    initial_holdings: int = 0
    """Default per-agent initial holdings (applied when an agent has no
    explicit holdings for this stock)."""

    sector: str = ""
    """Optional sector / tag shown to agents (e.g. "AI chips", "Mega Bank")."""

    blurb: str = ""
    """Optional one-line narrative about the company, fed to agent prompts."""

    @property
    def symbol(self) -> str:
        """Backward-compatibility alias returning name."""
        return self.name


@dataclass
class MarketConfig:
    """Global configuration for the market environment."""

    # --- Price parameters (legacy single-stock fallback when stocks is None) ---
    initial_price: float = 100.0
    """Initial stock price (single-stock fallback when ``stocks`` is None)."""

    price_sensitivity: float = 0.02
    """Impact coefficient of net buying pressure on price movement."""

    max_price_change_ratio: float = 0.10
    """Maximum price change ratio per step to prevent extreme jumps."""

    mean_reversion_strength: float = 0.0005
    """Pull back toward a stock's initial price each step, as a fraction
    of its current deviation. 0 disables mean reversion."""

    idle_price_noise: float = 0.003
    """Half-width of the uniform random drift applied to a stock in a step
    where nothing matched, so an untraded market still moves. 0 disables."""

    event_impact_scale: float = 0.3
    """Fraction of an active event's headline price/sentiment impact
    applied per step; the full impact is spread across its duration."""

    min_price: float = 0.01
    """Lower price bound."""

    max_price: float = 1_000_000.0
    """Upper price bound."""

    # --- Multi-stock parameters ---
    stocks: Optional[List[StockSpec]] = None
    """List of stock specifications. If None, a single default stock is used."""

    # --- Observation parameters ---
    price_history_length: int = 20
    """Number of historical prices visible to each agent."""

    history_backfill_steps: int = 30
    """Synthetic pre-history length: random-walk candles generated before
    round 1 (ending exactly at each stock's initial price) so agents can
    analyze trends from the very first step. 0 disables."""

    # --- Transaction costs ---
    fee_rate: float = 0.0
    """Transaction fee rate per trade (e.g., 0.001 = 0.1%). Applied to trade value."""

    slippage_rate: float = 0.0
    """Slippage rate applied to execution price (e.g., 0.001 = 0.1%).
    Buyers pay slightly more, sellers receive less."""

    # --- Sandbox event settings ---
    event_probability_multiplier: float = 1.5
    """Multiplier for event probabilities in the unified sandbox."""

    # --- Personality ---
    deep_persona: bool = False
    """Give each trader a full personality paragraph and ask for longer,
    in-character reasoning. Off by default: the lean prompt is faster and
    cheaper, and still carries a one-line disposition."""

    mood_max_step: float = 3.0
    """Largest change allowed on any mood axis in one round, so a single
    round cannot swing a personality end to end. Caps both the rule-based
    move and the model's reported adjustment."""

    mood_intensity: float = 1.0
    """Scales how strongly round events move mood (recovery toward the
    baseline is unaffected)."""

    # --- Social influence ---
    social_influence: float = 0.0
    """Retired: personality no longer overrides decisions, so this no longer
    scales anything. Kept so saved configs and the homepage slider still
    round-trip."""

    parallel_agents: bool = True
    """Collect agent actions concurrently in worker threads each round, so
    multiple LLM API calls overlap instead of running sequentially. Set
    False for strict sequential execution."""

    # --- Reproducibility / Random seed ---
    seed: Optional[int] = None
    """Random seed for reproducibility. If None, auto-generated at run start."""

    def __post_init__(self) -> None:
        """Clamp transaction-cost rates to a safe range."""
        self.fee_rate = max(0.0, min(self.fee_rate, 0.5))
        self.slippage_rate = max(0.0, min(self.slippage_rate, 0.5))

    def get_stock_specs(self) -> List[StockSpec]:
        """Return the list of stock specs, falling back to a single default.

        When ``stocks`` is None or empty, a single default stock ("Stock 1")
        priced at ``initial_price`` is used.
        """
        if self.stocks:
            return list(self.stocks)
        return [StockSpec(
            name="Stock 1",
            initial_price=self.initial_price,
        )]

    def to_dict(self) -> dict:
        """Serialize MarketConfig to dictionary."""
        return {
            "initial_price": self.initial_price,
            "price_sensitivity": self.price_sensitivity,
            "max_price_change_ratio": self.max_price_change_ratio,
            "mean_reversion_strength": self.mean_reversion_strength,
            "idle_price_noise": self.idle_price_noise,
            "event_impact_scale": self.event_impact_scale,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "stocks": [
                {
                    "name": s.name,
                    "initial_price": s.initial_price,
                    "initial_holdings": s.initial_holdings,
                    "sector": s.sector,
                    "blurb": s.blurb,
                }
                for s in self.get_stock_specs()
            ] if self.stocks else None,
            "price_history_length": self.price_history_length,
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
            "event_probability_multiplier": self.event_probability_multiplier,
            "deep_persona": self.deep_persona,
            "mood_max_step": self.mood_max_step,
            "mood_intensity": self.mood_intensity,
            "social_influence": self.social_influence,
            "parallel_agents": self.parallel_agents,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarketConfig":
        """Deserialize MarketConfig from dictionary."""
        data_copy = data.copy()
        data_copy.pop("mode", None)
        stocks_data = data_copy.pop("stocks", None)
        stocks: Optional[List[StockSpec]] = None
        if stocks_data and isinstance(stocks_data, list):
            stocks = []
            for s in stocks_data:
                if isinstance(s, StockSpec):
                    stocks.append(s)
                elif isinstance(s, dict):
                    price = s.get("initial_price", s.get("price", 100.0))
                    hold = s.get("initial_holdings", s.get("hold", 0))
                    stocks.append(StockSpec(
                        name=str(s.get("name") or s.get("symbol") or "Stock"),
                        initial_price=float(price) if price is not None else 100.0,
                        initial_holdings=int(hold) if hold is not None else 0,
                        sector=str(s.get("sector") or ""),
                        blurb=str(s.get("blurb") or ""),
                    ))
        return cls(stocks=stocks, **data_copy)

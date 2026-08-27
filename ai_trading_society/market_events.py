"""
Market events system for realistic simulation mode.

Random events that shake up the market: earnings surprises, analyst ratings,
Fed decisions, social media hype, black swans, and more.
"""

import random
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any, Dict, List, Optional, Union


class EventType(Enum):
    """Categories of market events."""
    EARNINGS = "earnings"
    ANALYST = "analyst"
    MACRO = "macro"
    SOCIAL = "social"
    REGULATORY = "regulatory"
    BLACK_SWAN = "black_swan"
    TECHNICAL = "technical"
    CRYPTO = "crypto"
    M_AND_A = "m_and_a"
    CORPORATE = "corporate"


# Event categories that affect the WHOLE market (every stock) rather than a
# single company. Everything else (earnings, analyst calls, M&A, ...) is
# company-specific and only hits one stock when triggered.
GLOBAL_EVENT_TYPES = {
    EventType.MACRO,
    EventType.BLACK_SWAN,
    EventType.CRYPTO,
}


def is_global_event_type(event_type: EventType) -> bool:
    """Return True when an event category impacts the whole market."""
    return event_type in GLOBAL_EVENT_TYPES


@dataclass
class MarketEvent:
    """
    A market event that affects price and agent sentiment.

    Parameters
    ----------
    name : str
        Short event name.
    description : str
        Human-readable description.
    event_type : EventType
        Category of event.
    price_impact : float
        Percentage price change (positive = bullish, negative = bearish).
        Example: -0.10 means 10% drop.
    sentiment_shift : float
        Change in market sentiment (-1 to 1).
        Affects how agents perceive the market.
    duration_steps : int
        How many steps the event's effects persist.
    probability : float
        Base probability of event occurring (0.0 to 1.0).
    target_stock : str, optional
        Stock this event applies to. None means the event is GLOBAL and
        affects every stock in the market.
    """
    name: str
    description: str
    event_type: EventType
    price_impact: float = 0.0
    sentiment_shift: float = 0.0
    duration_steps: int = 1
    probability: float = 0.01
    target_stock: Optional[str] = None

    def __post_init__(self) -> None:
        self.remaining_steps = 0
        self.triggered_step: Optional[int] = None

    @property
    def scope(self) -> str:
        """'global' when the event hits every stock, else 'stock'."""
        return "global" if self.target_stock is None else "stock"

    def affects_stock(self, symbol: Optional[str]) -> bool:
        """Whether this event's price/sentiment effects apply to `symbol`."""
        if self.target_stock is None:
            return True
        return symbol is not None and self.target_stock == symbol

    def is_active(self) -> bool:
        """Check if event is currently affecting the market."""
        return self.remaining_steps > 0

    def trigger(self, step: int) -> None:
        """Activate the event."""
        self.triggered_step = step
        self.remaining_steps = self.duration_steps

    def tick(self) -> None:
        """Advance one step, decrementing remaining duration."""
        if self.remaining_steps > 0:
            self.remaining_steps -= 1


# Pre-defined event templates
EVENT_TEMPLATES: List[MarketEvent] = [

    # === EARNINGS EVENTS ===
    MarketEvent(
        name="earnings_beat",
        description="Company beats earnings expectations by 15%",
        event_type=EventType.EARNINGS,
        price_impact=0.05,
        sentiment_shift=0.3,
        duration_steps=3,
        probability=0.02,
    ),
    MarketEvent(
        name="earnings_miss",
        description="Company misses earnings expectations by 12%",
        event_type=EventType.EARNINGS,
        price_impact=-0.07,
        sentiment_shift=-0.4,
        duration_steps=4,
        probability=0.02,
    ),
    MarketEvent(
        name="earnings_guidance_raised",
        description="Management raises forward guidance",
        event_type=EventType.EARNINGS,
        price_impact=0.04,
        sentiment_shift=0.25,
        duration_steps=2,
        probability=0.015,
    ),
    MarketEvent(
        name="earnings_guidance_lowered",
        description="Management lowers forward guidance amid uncertainty",
        event_type=EventType.EARNINGS,
        price_impact=-0.05,
        sentiment_shift=-0.3,
        duration_steps=3,
        probability=0.015,
    ),

    # === ANALYST EVENTS ===
    MarketEvent(
        name="analyst_upgrade",
        description="Major analyst upgrades stock to 'Strong Buy'",
        event_type=EventType.ANALYST,
        price_impact=0.03,
        sentiment_shift=0.2,
        duration_steps=2,
        probability=0.025,
    ),
    MarketEvent(
        name="analyst_downgrade",
        description="Major analyst downgrades stock to 'Sell'",
        event_type=EventType.ANALYST,
        price_impact=-0.04,
        sentiment_shift=-0.25,
        duration_steps=3,
        probability=0.025,
    ),
    MarketEvent(
        name="price_target_hike",
        description="Price target raised significantly",
        event_type=EventType.ANALYST,
        price_impact=0.02,
        sentiment_shift=0.15,
        duration_steps=1,
        probability=0.03,
    ),
    MarketEvent(
        name="price_target_cut",
        description="Price target slashed amid concerns",
        event_type=EventType.ANALYST,
        price_impact=-0.03,
        sentiment_shift=-0.2,
        duration_steps=2,
        probability=0.03,
    ),

    # === MACRO EVENTS ===
    MarketEvent(
        name="fed_rate_hike",
        description="Federal Reserve raises interest rates by 0.25%",
        event_type=EventType.MACRO,
        price_impact=-0.02,
        sentiment_shift=-0.15,
        duration_steps=5,
        probability=0.01,
    ),
    MarketEvent(
        name="fed_rate_cut",
        description="Federal Reserve cuts interest rates by 0.25%",
        event_type=EventType.MACRO,
        price_impact=0.03,
        sentiment_shift=0.2,
        duration_steps=5,
        probability=0.01,
    ),
    MarketEvent(
        name="inflation_data_good",
        description="Inflation data shows cooling prices",
        event_type=EventType.MACRO,
        price_impact=0.02,
        sentiment_shift=0.15,
        duration_steps=2,
        probability=0.02,
    ),
    MarketEvent(
        name="inflation_data_bad",
        description="Inflation runs hotter than expected",
        event_type=EventType.MACRO,
        price_impact=-0.03,
        sentiment_shift=-0.2,
        duration_steps=3,
        probability=0.02,
    ),
    MarketEvent(
        name="jobs_report_strong",
        description="Employment report shows robust hiring",
        event_type=EventType.MACRO,
        price_impact=0.02,
        sentiment_shift=0.1,
        duration_steps=2,
        probability=0.015,
    ),
    MarketEvent(
        name="jobs_report_weak",
        description="Employment report shows slowing labor market",
        event_type=EventType.MACRO,
        price_impact=-0.02,
        sentiment_shift=-0.15,
        duration_steps=2,
        probability=0.015,
    ),

    # === SOCIAL MEDIA / HYPE ===
    MarketEvent(
        name="social_media_hype",
        description="Stock trending on social media with bullish sentiment",
        event_type=EventType.SOCIAL,
        price_impact=0.04,
        sentiment_shift=0.35,
        duration_steps=2,
        probability=0.03,
    ),
    MarketEvent(
        name="social_media_backlash",
        description="Negative viral post sparks selling pressure",
        event_type=EventType.SOCIAL,
        price_impact=-0.05,
        sentiment_shift=-0.4,
        duration_steps=2,
        probability=0.025,
    ),
    MarketEvent(
        name="viral_short_report",
        description="Viral short-seller report circulates online",
        event_type=EventType.SOCIAL,
        price_impact=-0.08,
        sentiment_shift=-0.5,
        duration_steps=4,
        probability=0.01,
    ),
    MarketEvent(
        name="meme_stock_rally",
        description="Retail investors coordinate buying campaign",
        event_type=EventType.SOCIAL,
        price_impact=0.10,
        sentiment_shift=0.6,
        duration_steps=3,
        probability=0.008,
    ),

    # === REGULATORY ===
    MarketEvent(
        name="regulatory_probe",
        description="Regulators announce investigation into company",
        event_type=EventType.REGULATORY,
        price_impact=-0.06,
        sentiment_shift=-0.35,
        duration_steps=5,
        probability=0.012,
    ),
    MarketEvent(
        name="regulatory_approval",
        description="Key product receives regulatory approval",
        event_type=EventType.REGULATORY,
        price_impact=0.08,
        sentiment_shift=0.4,
        duration_steps=3,
        probability=0.01,
    ),
    MarketEvent(
        name="antitrust_concerns",
        description="Antitrust scrutiny increases",
        event_type=EventType.REGULATORY,
        price_impact=-0.04,
        sentiment_shift=-0.25,
        duration_steps=4,
        probability=0.015,
    ),

    # === BLACK SWANS ===
    MarketEvent(
        name="market_crash",
        description="Panic selling sweeps the broader market",
        event_type=EventType.BLACK_SWAN,
        price_impact=-0.15,
        sentiment_shift=-0.8,
        duration_steps=6,
        probability=0.005,
    ),
    MarketEvent(
        name="flash_crash",
        description="Flash crash triggers stop-loss cascades",
        event_type=EventType.BLACK_SWAN,
        price_impact=-0.12,
        sentiment_shift=-0.7,
        duration_steps=3,
        probability=0.008,
    ),
    MarketEvent(
        name="geopolitical_shock",
        description="Geopolitical tensions spike",
        event_type=EventType.BLACK_SWAN,
        price_impact=-0.08,
        sentiment_shift=-0.5,
        duration_steps=5,
        probability=0.008,
    ),
    MarketEvent(
        name="sector_rotation",
        description="Major sector rotation out of tech stocks",
        event_type=EventType.BLACK_SWAN,
        price_impact=-0.05,
        sentiment_shift=-0.3,
        duration_steps=4,
        probability=0.012,
    ),

    # === TECHNICAL EVENTS ===
    MarketEvent(
        name="short_squeeze",
        description="Short squeeze forces bears to cover",
        event_type=EventType.TECHNICAL,
        price_impact=0.12,
        sentiment_shift=0.5,
        duration_steps=2,
        probability=0.01,
    ),
    MarketEvent(
        name="breakout_resistance",
        description="Price breaks above key resistance level",
        event_type=EventType.TECHNICAL,
        price_impact=0.03,
        sentiment_shift=0.2,
        duration_steps=2,
        probability=0.025,
    ),
    MarketEvent(
        name="breakdown_support",
        description="Price breaks below key support level",
        event_type=EventType.TECHNICAL,
        price_impact=-0.04,
        sentiment_shift=-0.25,
        duration_steps=2,
        probability=0.025,
    ),

    # === CRYPTO MARKET EVENTS ===
    MarketEvent(
        name="crypto_crash",
        description="Major cryptocurrencies crash 25%, spilling into equities",
        event_type=EventType.CRYPTO,
        price_impact=-0.07,
        sentiment_shift=-0.45,
        duration_steps=4,
        probability=0.012,
    ),
    MarketEvent(
        name="crypto_rally",
        description="Crypto market surges, lifting risk assets broadly",
        event_type=EventType.CRYPTO,
        price_impact=0.05,
        sentiment_shift=0.3,
        duration_steps=3,
        probability=0.015,
    ),
    MarketEvent(
        name="crypto_etf_approval",
        description="Spot Bitcoin ETF approved, driving institutional inflows",
        event_type=EventType.CRYPTO,
        price_impact=0.06,
        sentiment_shift=0.4,
        duration_steps=4,
        probability=0.008,
    ),
    MarketEvent(
        name="crypto_exchange_collapse",
        description="Major crypto exchange collapses, freezing customer funds",
        event_type=EventType.CRYPTO,
        price_impact=-0.10,
        sentiment_shift=-0.6,
        duration_steps=5,
        probability=0.006,
    ),

    # === MERGERS & ACQUISITIONS ===
    MarketEvent(
        name="acquisition_announced",
        description="Company acquired at 30% premium",
        event_type=EventType.M_AND_A,
        price_impact=0.15,
        sentiment_shift=0.55,
        duration_steps=2,
        probability=0.008,
    ),
    MarketEvent(
        name="merger_rumor",
        description="Merger rumors circulate, driving speculative buying",
        event_type=EventType.M_AND_A,
        price_impact=0.06,
        sentiment_shift=0.35,
        duration_steps=3,
        probability=0.012,
    ),
    MarketEvent(
        name="hostile_takeover_bid",
        description="Hostile takeover bid launched at a premium",
        event_type=EventType.M_AND_A,
        price_impact=0.08,
        sentiment_shift=0.3,
        duration_steps=4,
        probability=0.007,
    ),
    MarketEvent(
        name="deal_breakup",
        description="M&A deal falls through amid regulatory pushback",
        event_type=EventType.M_AND_A,
        price_impact=-0.09,
        sentiment_shift=-0.4,
        duration_steps=3,
        probability=0.009,
    ),

    # === CORPORATE ACTIONS ===
    MarketEvent(
        name="stock_buyback",
        description="Company announces $10B stock buyback program",
        event_type=EventType.CORPORATE,
        price_impact=0.05,
        sentiment_shift=0.3,
        duration_steps=3,
        probability=0.014,
    ),
    MarketEvent(
        name="dividend_increase",
        description="Company raises dividend by 20%",
        event_type=EventType.CORPORATE,
        price_impact=0.03,
        sentiment_shift=0.2,
        duration_steps=2,
        probability=0.015,
    ),
    MarketEvent(
        name="stock_split",
        description="Company announces 3-for-1 stock split",
        event_type=EventType.CORPORATE,
        price_impact=0.04,
        sentiment_shift=0.25,
        duration_steps=2,
        probability=0.012,
    ),
    MarketEvent(
        name="spinoff_announced",
        description="Company announces spinoff of business unit",
        event_type=EventType.CORPORATE,
        price_impact=0.04,
        sentiment_shift=0.2,
        duration_steps=3,
        probability=0.01,
    ),
    MarketEvent(
        name="accounting_scandal",
        description="Accounting irregularities uncovered, SEC launches probe",
        event_type=EventType.CORPORATE,
        price_impact=-0.12,
        sentiment_shift=-0.65,
        duration_steps=6,
        probability=0.006,
    ),
]


class EventManager:
    """
    Manages market event triggering and lifecycle.

    Parameters
    ----------
    templates : list of MarketEvent, optional
        Event templates to draw from. Defaults to EVENT_TEMPLATES.
    event_probability_multiplier : float
        Multiplier for base event probabilities. Higher = more events.
        Default 1.0. Set to 0 to disable events.
    impact_scale : float
        Fraction of an active event's headline impact applied per step,
        so the full impact is spread across the event's duration.
        Mirrors ``MarketConfig.event_impact_scale``.
    """

    def __init__(
        self,
        templates: Optional[List[MarketEvent]] = None,
        event_probability_multiplier: float = 1.0,
        rng: Optional[Union[random.Random, ModuleType]] = None,
        stock_names: Optional[List[str]] = None,
        impact_scale: float = 0.3,
    ):
        self.templates = templates or EVENT_TEMPLATES.copy()
        self.multiplier = event_probability_multiplier
        self.impact_scale = impact_scale
        # RNG for event triggering; allow injection for reproducibility.
        # Default to the global random module so external seeding still works.
        self.rng = rng if rng is not None else random
        self.active_events: List[MarketEvent] = []
        self.event_history: List[Dict[str, Any]] = []
        # Known stock names, used to pick a target stock for stock-scoped
        # events. Updated by the market environment each step.
        self.stock_names: List[str] = list(stock_names or [])

    def _resolve_target_stock(self, template: MarketEvent) -> Optional[str]:
        """Pick the target stock for a newly triggered template event.

        Global categories (macro/black swan/crypto) return None so they hit
        every stock. Company-specific categories get a random stock.
        """
        if is_global_event_type(template.event_type):
            return None
        if not self.stock_names:
            return None
        return self.rng.choice(self.stock_names)

    def try_trigger_event(
        self, step: int, stock_names: Optional[List[str]] = None
    ) -> List[MarketEvent]:
        """
        Attempt to trigger random events this step.

        Every template is checked independently, so multiple events can fire
        in the same step. All triggered events are returned so callers can
        report every one of them.

        Parameters
        ----------
        step : int
            Current simulation step.
        stock_names : list of str, optional
            Stocks in the market. Company-specific events are assigned one
            of these at random; global events ignore it.

        Returns
        -------
        events : list of MarketEvent
            All events triggered this step (empty list if none).
        """
        if stock_names is not None:
            self.stock_names = list(stock_names)

        triggered: List[MarketEvent] = []

        for template in self.templates:
            prob = template.probability * self.multiplier
            if self.rng.random() < prob:
                event = MarketEvent(
                    name=template.name,
                    description=template.description,
                    event_type=template.event_type,
                    price_impact=template.price_impact,
                    sentiment_shift=template.sentiment_shift,
                    duration_steps=template.duration_steps,
                    probability=template.probability,
                    target_stock=self._resolve_target_stock(template),
                )
                event.trigger(step)
                self.active_events.append(event)
                self.event_history.append({
                    "step": step,
                    "name": event.name,
                    "description": event.description,
                    "type": event.event_type.value,
                    "price_impact": event.price_impact,
                    "scope": event.scope,
                    "stock": event.target_stock,
                })
                triggered.append(event)
        return triggered

    def force_trigger_event(
        self,
        name: str,
        step: int = 0,
        custom_desc: Optional[str] = None,
        price_impact: float = 0.05,
        sentiment_shift: float = 0.3,
        target_stock: Optional[str] = None,
        stock_names: Optional[List[str]] = None,
    ) -> MarketEvent:
        """Force trigger a custom or template event (God Mode).

        Parameters
        ----------
        target_stock : str, optional
            Explicit stock the event should hit. When omitted, a
            company-specific template gets a random stock from
            ``stock_names`` (or the manager's known stock list); global
            categories stay global.
        stock_names : list of str, optional
            Refresh the known stock list before resolving the target.
        """
        if stock_names is not None:
            self.stock_names = list(stock_names)
        # Find template by name if exists
        matched = [t for t in self.templates if t.name == name]
        if matched:
            t = matched[0]
            event = MarketEvent(
                name=t.name,
                description=custom_desc or t.description,
                event_type=t.event_type,
                price_impact=t.price_impact,
                sentiment_shift=t.sentiment_shift,
                duration_steps=t.duration_steps,
                target_stock=(
                    target_stock
                    if target_stock is not None
                    else self._resolve_target_stock(t)
                ),
            )
        else:
            # Custom event (not in the template library): global unless the
            # caller names a target stock.
            event = MarketEvent(
                name=name,
                description=custom_desc or name,
                event_type=EventType.SOCIAL,
                price_impact=price_impact,
                sentiment_shift=sentiment_shift,
                duration_steps=2,
                target_stock=target_stock,
            )
        event.trigger(step)
        self.active_events.append(event)
        self.event_history.append({
            "step": step,
            "name": event.name,
            "description": event.description,
            "type": event.event_type.value,
            "price_impact": event.price_impact,
            "scope": event.scope,
            "stock": event.target_stock,
            "forced": True,
        })
        return event


    def get_combined_effects(self, symbol: Optional[str] = None) -> Dict[str, float]:
        """
        Get combined price impact and sentiment shift from all active events.

        Parameters
        ----------
        symbol : str, optional
            Stock being priced. Global events always apply; stock-scoped
            events only apply when their target matches. None keeps only
            global events (backward-compatible aggregate view).

        Returns
        -------
        effects : dict
            {"price_impact": float, "sentiment_shift": float}
        """
        price_impact = 0.0
        sentiment_shift = 0.0

        for event in self.active_events:
            if event.is_active() and event.affects_stock(symbol):
                # Decay impact over duration
                decay = event.remaining_steps / event.duration_steps
                # Scale down per-step: the headline impact is spread
                # across the event's duration (see
                # MarketConfig.event_impact_scale).
                price_impact += event.price_impact * decay * self.impact_scale
                sentiment_shift += (
                    event.sentiment_shift * decay * self.impact_scale
                )

        return {
            "price_impact": price_impact,
            "sentiment_shift": max(-1.0, min(1.0, sentiment_shift)),
        }

    def tick(self) -> List[MarketEvent]:
        """
        Advance all active events by one step.

        Returns
        -------
        expired : list of MarketEvent
            Events that just expired this tick.
        """
        expired = []
        for event in self.active_events:
            event.tick()
            if not event.is_active():
                expired.append(event)

        # Remove expired events from active list
        self.active_events = [e for e in self.active_events if e.is_active()]

        return expired

    def get_active_events_detail(self) -> List[Dict[str, Any]]:
        """
        Return active events with remaining duration for display.

        Returns
        -------
        events : list of dict
            Each dict contains name, description, type, stock, scope,
            remaining_steps, and total_steps for an active event.
        """
        return [
            {
                "name": e.name,
                "description": e.description,
                "type": e.event_type.value,
                "stock": e.target_stock,
                "scope": e.scope,
                "remaining_steps": e.remaining_steps,
                "total_steps": e.duration_steps,
            }
            for e in self.active_events if e.is_active()
        ]

    def get_observation_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get event data to include in agent observations.

        Parameters
        ----------
        symbol : str, optional
            Stock the observing agent is being priced against. When given,
            only events affecting that stock (global + its own) are listed;
            global-only sentiment is still aggregated market-wide.

        Returns
        -------
        data : dict
            {"active_events": [...], "market_sentiment": float}
        """
        active = [
            {
                "name": e.name,
                "description": e.description,
                "type": e.event_type.value,
                "stock": e.target_stock,
                "price_impact": e.price_impact,
            }
            for e in self.active_events
            if e.is_active() and (symbol is None or e.affects_stock(symbol))
        ]
        effects = self.get_combined_effects()

        return {
            "active_events": active,
            "market_sentiment": effects["sentiment_shift"],
            "event_price_impact": effects["price_impact"],
        }

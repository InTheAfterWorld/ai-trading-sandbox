"""
Market events system for realistic simulation mode.

Random events that shake up the market: earnings surprises, analyst ratings,
Fed decisions, social media hype, black swans, and more.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(Enum):
    """Categories of market events."""
    EARNINGS = "earnings"
    ANALYST = "analyst"
    MACRO = "macro"
    SOCIAL = "social"
    REGULATORY = "regulatory"
    BLACK_SWAN = "black_swan"
    TECHNICAL = "technical"


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
    """
    name: str
    description: str
    event_type: EventType
    price_impact: float = 0.0
    sentiment_shift: float = 0.0
    duration_steps: int = 1
    probability: float = 0.01

    def __post_init__(self):
        self.remaining_steps = 0
        self.triggered_step: Optional[int] = None

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
    """

    def __init__(
        self,
        templates: Optional[List[MarketEvent]] = None,
        event_probability_multiplier: float = 1.0,
        rng: Optional[random.Random] = None,
    ):
        self.templates = templates or EVENT_TEMPLATES.copy()
        self.multiplier = event_probability_multiplier
        # RNG for event triggering; allow injection for reproducibility.
        # Default to the global random module so external seeding still works.
        self.rng = rng if rng is not None else random
        self.active_events: List[MarketEvent] = []
        self.event_history: List[Dict[str, Any]] = []

    def try_trigger_event(self, step: int) -> List[MarketEvent]:
        """
        Attempt to trigger random events this step.

        Every template is checked independently, so multiple events can fire
        in the same step. All triggered events are returned so callers can
        report every one of them.

        Returns
        -------
        events : list of MarketEvent
            All events triggered this step (empty list if none).
        """
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
                )
                event.trigger(step)
                self.active_events.append(event)
                self.event_history.append({
                    "step": step,
                    "name": event.name,
                    "description": event.description,
                    "type": event.event_type.value,
                    "price_impact": event.price_impact,
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
    ) -> MarketEvent:
        """Force trigger a custom or template event (God Mode)."""
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
            )
        else:
            event = MarketEvent(
                name=name,
                description=custom_desc or name,
                event_type=EventType.SOCIAL,
                price_impact=price_impact,
                sentiment_shift=sentiment_shift,
                duration_steps=2,
            )
        event.trigger(step)
        self.active_events.append(event)
        self.event_history.append({
            "step": step,
            "name": event.name,
            "description": event.description,
            "type": event.event_type.value,
            "price_impact": event.price_impact,
            "forced": True,
        })
        return event


    def get_combined_effects(self) -> Dict[str, float]:
        """
        Get combined price impact and sentiment shift from all active events.

        Returns
        -------
        effects : dict
            {"price_impact": float, "sentiment_shift": float}
        """
        price_impact = 0.0
        sentiment_shift = 0.0

        for event in self.active_events:
            if event.is_active():
                # Decay impact over duration
                decay = event.remaining_steps / event.duration_steps
                price_impact += event.price_impact * decay * 0.3  # Scale down per-step
                sentiment_shift += event.sentiment_shift * decay * 0.3

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
            Each dict contains name, description, type, remaining_steps,
            and total_steps for an active event.
        """
        return [
            {
                "name": e.name,
                "description": e.description,
                "type": e.event_type.value,
                "remaining_steps": e.remaining_steps,
                "total_steps": e.duration_steps,
            }
            for e in self.active_events if e.is_active()
        ]

    def get_observation_data(self) -> Dict[str, Any]:
        """
        Get event data to include in agent observations.

        Returns
        -------
        data : dict
            {"active_events": [...], "market_sentiment": float}
        """
        active = [
            {"name": e.name, "description": e.description, "type": e.event_type.value}
            for e in self.active_events if e.is_active()
        ]

        effects = self.get_combined_effects()

        return {
            "active_events": active,
            "market_sentiment": effects["sentiment_shift"],
            "event_price_impact": effects["price_impact"],
        }
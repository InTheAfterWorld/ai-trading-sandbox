"""
AI Trading Society.

An extensible multi-agent simulation framework for observing how different
AI decision makers behave in a virtual stock market.

Runs one unified sandbox where AI agents, market events, and an optional-action player coexist.
"""

from .base_agent import BaseAgent
from .config import MarketConfig
from .market_env import MarketEnv
from .market_events import EVENT_TEMPLATES, EventManager, MarketEvent
from .run_metadata import RunMetadata, load_run_snapshot, save_run_snapshot, set_seed
from .simulator import Simulator
from .visualization import (
    generate_all_charts,
    plot_final_rankings,
    plot_price_history,
    plot_wealth_timeline,
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "MarketConfig",
    "BaseAgent",
    "MarketEnv",
    "Simulator",
    # Metadata & Reproducibility
    "RunMetadata",
    "set_seed",
    "load_run_snapshot",
    "save_run_snapshot",
    # Events
    "EventManager",
    "MarketEvent",
    "EVENT_TEMPLATES",
    # Visualization
    "plot_price_history",
    "plot_wealth_timeline",
    "plot_final_rankings",
    "generate_all_charts",
]

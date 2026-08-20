"""
AI Trading Society.

An extensible multi-agent simulation framework for observing how different
AI decision makers behave in a virtual stock market.

Runs one unified sandbox where AI agents, market events, and an optional-action player coexist.
"""

from .config import MarketConfig, AgentConfig
from .base_agent import BaseAgent
from .market_env import MarketEnv
from .simulator import Simulator
from .run_metadata import RunMetadata, set_seed, load_run_snapshot, save_run_snapshot
from .market_events import EventManager, MarketEvent, EVENT_TEMPLATES
from .visualization import (
    plot_price_history,
    plot_wealth_timeline,
    plot_final_rankings,
    generate_all_charts,
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "MarketConfig",
    "AgentConfig",
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
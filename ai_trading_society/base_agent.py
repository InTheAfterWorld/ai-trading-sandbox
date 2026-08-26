"""
BaseAgent defines the abstract interface for all trading agents.

Each agent is treated as a black-box decision maker: observation in,
action out.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class BaseAgent(ABC):
    """
    Abstract base class for trading agents.

    Subclasses only need to implement act(observation) -> action.
    The framework updates cash and holdings after order execution.

    Parameters
    ----------
    agent_id : str
        Unique agent identifier.
    cash : float
        Initial cash balance.
    holdings : int or dict, optional
        Initial share holdings. In multi-stock mode this is a
        ``Dict[str, float]`` mapping stock symbol to share count.
        For backward compatibility, an int is accepted and mapped
        to the first stock at MarketEnv initialization time.
    """

    def __init__(
        self,
        agent_id: str,
        cash: float = 10000.0,
        holdings: Optional[Union[int, Dict[str, float]]] = None,
    ):
        self.agent_id = agent_id
        self.cash = cash
        # holdings: Dict[str, float] for multi-stock mode.
        # Backward compat: int input → {"_legacy": float(int)}; MarketEnv
        # will remap "_legacy" to the first stock's symbol at init time.
        if holdings is None:
            self.holdings: Dict[str, float] = {}
        elif isinstance(holdings, dict):
            self.holdings = {k: float(v) for k, v in holdings.items()}
        else:
            self.holdings = {"_legacy": float(holdings)}

    # ------------------------------------------------------------------
    # Core interface that subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return trading decisions for the given market observation.

        Parameters
        ----------
        observation : dict
            Market observation containing:
            - step            : int   Current simulation step
            - stocks          : list  Per-stock data: [{symbol, name, price,
                                      price_history, last_volume, my_holdings}]
            - my_cash         : float Current cash balance (global)
            - my_holdings     : dict  Per-stock holdings {symbol: float}
            - my_wealth       : float Current total wealth (cash + all holdings)
            - market_sentiment: float Current sentiment (-1 to +1)
            - price           : float (backward compat: first stock's price)
            - price_history   : list  (backward compat: first stock's history)
            - last_volume     : int   (backward compat: first stock's volume)

        Returns
        -------
        action : dict
            Multi-stock format (preferred):
            ``{"decisions": [{"symbol": str, "action": "buy"|"sell"|"hold",
            "quantity": int, "reasoning": str}, ...]}``

            Legacy single-stock format (backward compat, applied to first stock):
            ``{"action": "buy"|"sell"|"hold", "quantity": int}``

        Notes
        -----
        Concurrency contract: ``act()`` must be thread-safe and must NOT read or
        mutate the internal state of other agents or the environment. MarketEnv
        guarantees only observation snapshot consistency during parallel collection.
        """
        ...

    # ------------------------------------------------------------------
    # Helper properties
    # ------------------------------------------------------------------

    @property
    def wealth(self) -> float:
        """Cash-only placeholder. MarketEnv computes mark-to-market wealth."""
        return self.cash

    def __repr__(self) -> str:
        total = sum(self.holdings.values()) if isinstance(self.holdings, dict) else 0
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id}, "
            f"cash={self.cash:.2f}, "
            f"holdings={total})"
        )

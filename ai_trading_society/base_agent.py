"""
BaseAgent defines the abstract interface for all trading agents.

Each agent is treated as a black-box decision maker: observation in,
action out.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


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
    holdings : int
        Initial share holdings.
    """

    def __init__(
        self,
        agent_id: str,
        cash: float = 10000.0,
        holdings: int = 0,
    ):
        self.agent_id = agent_id
        self.cash = cash
        self.holdings = holdings

    # ------------------------------------------------------------------
    # Core interface that subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a trading action for the given market observation.

        Parameters
        ----------
        observation : dict
            Market observation containing:
            - step          : int   Current simulation step
            - price         : float Current stock price
            - price_history : list  Recent historical prices
            - my_cash       : float Current cash balance
            - my_holdings   : int   Current share holdings
            - my_wealth     : float Current total wealth
            - last_volume   : int   Previous matched volume

        Returns
        -------
        action : dict
            {"action": "buy"|"sell"|"hold", "quantity": int}
            The environment will clip quantity to legal bounds.
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
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id}, "
            f"cash={self.cash:.2f}, "
            f"holdings={self.holdings})"
        )

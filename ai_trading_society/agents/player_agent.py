"""Human player agent that reads actions from the MarketEnv buffer."""

from typing import TYPE_CHECKING, Any, Dict, Optional

from ..base_agent import BaseAgent

if TYPE_CHECKING:
    from ..market_env import MarketEnv


class PlayerAgent(BaseAgent):
    """Agent controlled by a human player via the web UI.

    The web API calls env.set_player_action() before each step.
    This agent reads that buffered action during act().
    """

    def __init__(
        self,
        agent_id: str = "Player (You)",
        cash: float = 10000.0,
        holdings: int | Dict[str, float] = 20,
    ):
        super().__init__(agent_id=agent_id, cash=cash, holdings=holdings)
        self._env: Optional["MarketEnv"] = None  # Set by MarketEnv or web_app after construction
        # MarketEnv routes player orders through an implicit market maker.
        self.is_player = True

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Return decisions for all stocks, reading buffered player actions."""
        # Get the list of stock names from the observation.
        stocks = observation.get("stocks", [])
        all_symbols = [
            s.get("name") or s.get("symbol")
            for s in stocks
            if s.get("name") or s.get("symbol")
        ]

        if self._env is not None:
            pending = self._env.pop_player_actions()
            if pending:
                decisions = []
                covered = set()
                for sym, act in pending.items():
                    if sym in all_symbols or not all_symbols:
                        decisions.append({
                            "name": sym,
                            "symbol": sym,
                            "action": act["action"],
                            "quantity": act["quantity"],
                            "reasoning": "Player decision",
                        })
                        covered.add(sym)
                # Fill remaining stocks with hold.
                for sym in all_symbols:
                    if sym not in covered:
                        decisions.append({
                            "name": sym,
                            "symbol": sym,
                            "action": "hold",
                            "quantity": 0,
                            "reasoning": "Player chose to hold",
                        })
                return {"decisions": decisions}

        # Default: hold all stocks.
        decisions = [
            {"name": sym, "symbol": sym, "action": "hold", "quantity": 0,
             "reasoning": "Player chose to hold"}
            for sym in all_symbols
        ]
        return {"decisions": decisions}

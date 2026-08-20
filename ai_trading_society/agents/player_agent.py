"""Human player agent that reads actions from the MarketEnv buffer."""

from typing import Any, Dict

from ..base_agent import BaseAgent


class PlayerAgent(BaseAgent):
    """Agent controlled by a human player via the web UI.

    The web API calls env.set_player_action() before each step.
    This agent reads that buffered action during act().
    """

    def __init__(self, agent_id: str = "Player (You)", cash: float = 10000.0, holdings: int = 20):
        super().__init__(agent_id=agent_id, cash=cash, holdings=holdings)
        self._env = None  # Set by MarketEnv or web_app after construction

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Return the action buffered by the web UI, or hold."""
        if self._env is not None:
            pending = self._env.pop_player_action()
            if pending:
                return {
                    "action": pending["action"],
                    "quantity": pending["quantity"],
                    "reasoning": "Player decision",
                }
        return {"action": "hold", "quantity": 0, "reasoning": "Player chose to hold"}

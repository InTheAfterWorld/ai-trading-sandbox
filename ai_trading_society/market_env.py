"""
MarketEnv implements the virtual stock market environment.

Core responsibilities:
1. Collect orders from all agents.
2. Match buy and sell interest using a simple proportional mechanism.
3. Update price from net buying pressure.
4. Record trades and state snapshots.
5. Trigger market events that affect price and sentiment.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import MarketConfig
from .base_agent import BaseAgent
from .market_events import EventManager


@dataclass
class TradeRecord:
    """Single trade execution record."""

    step: int
    agent_id: str
    action: str  # "buy" | "sell"
    quantity: int
    price: float
    cash_change: float  # Negative for buyers, positive for sellers.


class MarketEnv:
    """
    Virtual stock market environment.

    Per-step workflow:
    1. Generate an observation for each agent.
    2. Call Agent.act(observation) to collect orders.
    3. Match buy and sell orders proportionally.
    4. Update price from net buying pressure.
    5. Record the resulting state.

    Parameters
    ----------
    config : MarketConfig
        Market configuration parameters.
    agents : list of BaseAgent
        Agents participating in the market.
    """

    def __init__(self, config: MarketConfig, agents: List[BaseAgent], seed: int | None = None):
        self.config = config
        # RNG injected for reproducibility. An explicit seed argument wins;
        # otherwise fall back to config.seed so a seeded MarketConfig yields a
        # reproducible environment even when MarketEnv is constructed directly.
        # If both are None, use the global random module so external calls to
        # random.seed() still control behavior (helps tests that seed the
        # global RNG).
        if seed is None and config.seed is not None:
            seed = config.seed
        import random as _random
        self.rng = _random.Random(seed) if seed is not None else _random

        self.agents: Dict[str, BaseAgent] = {a.agent_id: a for a in agents}

        # Inject RNG into agents (and into wrapped base_agent if present)
        for a in agents:
            try:
                setattr(a, "rng", self.rng)
            except Exception:
                pass
            if hasattr(a, "base_agent"):
                try:
                    setattr(a.base_agent, "rng", self.rng)
                except Exception:
                    pass

        # --- Market state ---
        self.price: float = config.initial_price
        self.step_count: int = 0
        self.price_history: List[float] = [config.initial_price]
        self.trade_history: List[TradeRecord] = []
        self.volume_history: List[int] = []
        self._agent_error_counts: Dict[str, int] = {}

        # --- Event system (always active in the unified sandbox) ---
        self.event_manager: EventManager = EventManager(
            event_probability_multiplier=config.event_probability_multiplier,
            rng=self.rng,
        )

    def set_player_action(self, action: str, quantity: int):
        """Buffer the human player's action for the current step.

        Called by the web API before env.step() so the player agent
        can read it during the order-collection phase.
        """
        self._pending_player_action = {"action": action, "quantity": quantity}

    def pop_player_action(self):
        """Return and clear the buffered player action, or None."""
        action = getattr(self, "_pending_player_action", None)
        self._pending_player_action = None
        return action

    # ------------------------------------------------------------------
    # Observation generation
    # ------------------------------------------------------------------

    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """
        Generate the current market observation for one agent.

        The observation is the agent's only decision input, preserving the
        black-box setup: agents cannot inspect each other's internal state.
        """
        agent = self.agents[agent_id]
        hist_len = min(len(self.price_history), self.config.price_history_length)

        obs = {
            "step": self.step_count,
            "price": self.price,
            "price_history": self.price_history[-hist_len:],
            "my_cash": agent.cash,
            "my_holdings": agent.holdings,
            "my_wealth": agent.cash + agent.holdings * self.price,
            "last_volume": self.volume_history[-1] if self.volume_history else 0,
            "market_sentiment": 0.0,
        }

        # Add active event information.
        if self.event_manager:
            event_data = self.event_manager.get_observation_data()
            obs["active_events"] = event_data.get("active_events", [])
            obs["market_sentiment"] = event_data.get("market_sentiment", 0.0)

        # Optional persistent sentiment drift (God Mode setting).
        drift = getattr(self, "_sentiment_drift", 0.0)
        if drift:
            obs["market_sentiment"] = max(
                -1.0, min(1.0, obs["market_sentiment"] + drift)
            )

        return obs

    # ------------------------------------------------------------------
    # Main loop: one step
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """Run one simulation step."""
        self.step_count += 1

        # ---------- 0. Event system ----------
        triggered_events: list = []
        if self.event_manager:
            # Tick existing events
            self.event_manager.tick()
            # Try to trigger new events (may return multiple)
            triggered_events = self.event_manager.try_trigger_event(self.step_count)

        # ---------- 1. Collect orders ----------
        buy_orders: List[tuple] = []   # (agent_id, quantity)
        sell_orders: List[tuple] = []  # (agent_id, quantity)
        agent_actions: Dict[str, Dict[str, Any]] = {}

        for agent_id, agent in self.agents.items():
            obs = self.get_observation(agent_id)
            try:
                action = agent.act(obs)
                if not isinstance(action, dict):
                    raise ValueError("agent action must be a dictionary")
                action_type = action.get("action", "hold")
                if action_type not in ("buy", "sell", "hold"):
                    raise ValueError("agent action must be buy, sell, or hold")
                quantity = max(0, int(action.get("quantity", 0)))
                reasoning = str(action.get("reasoning", ""))
            except Exception as exc:
                count = self._agent_error_counts.get(agent_id, 0) + 1
                self._agent_error_counts[agent_id] = count
                if count == 1:
                    print(
                        f"[WARN] Agent '{agent_id}' failed to act: "
                        f"{type(exc).__name__}: {exc}. Recording an AI failure."
                    )
                action = {
                    "action": "hold",
                    "quantity": 0,
                    "reasoning": f"AI acquisition failed: {type(exc).__name__}: {exc}",
                    "error": True,
                }
                action_type = "hold"
                quantity = 0
                reasoning = action["reasoning"]

            # Record the agent's raw decision for this round.
            agent_actions[agent_id] = {
                "action": action_type,
                "requested_qty": quantity,
                "filled_qty": 0,
                "reasoning": reasoning,
                "error": bool(action.get("error", False)),
            }

            if action_type == "buy" and quantity > 0:
                # Clip quantity to the maximum affordable amount. Account for
                # slippage and fees so a full-size buy can never push cash
                # below zero (negative cash would silently cancel all of the
                # agent's future buy orders).
                effective_price = self.price * (
                    1.0 + self.config.slippage_rate
                ) * (1.0 + self.config.fee_rate)
                if self.price > 0 and agent.cash > 0 and effective_price > 0:
                    max_afford = math.floor(agent.cash / effective_price + 1e-9)
                else:
                    max_afford = 0
                quantity = min(quantity, max_afford)
                if quantity > 0:
                    buy_orders.append((agent_id, quantity))

            elif action_type == "sell" and quantity > 0:
                # Clip quantity to current holdings.
                quantity = min(quantity, agent.holdings)
                if quantity > 0:
                    sell_orders.append((agent_id, quantity))

        # ---------- 2. Match orders ----------
        total_buy = sum(q for _, q in buy_orders)
        total_sell = sum(q for _, q in sell_orders)
        matched_volume = min(total_buy, total_sell)
        self.volume_history.append(matched_volume)

        if matched_volume > 0:
            self._match_orders(buy_orders, sell_orders, total_buy, total_sell)

        # Update filled quantities from executed trades this step.
        for trade in self.trade_history:
            if trade.step == self.step_count:
                agent_actions[trade.agent_id]["filled_qty"] += trade.quantity

        # ---------- 3. Update price from net buying pressure and mean reversion ----------
        total_volume = total_buy + total_sell
        if total_volume > 0:
            net_pressure = (total_buy - total_sell) / total_volume
        else:
            net_pressure = 0.0

        price_change_ratio = self.config.price_sensitivity * net_pressure

        # Mean reversion: the farther price moves from its initial level,
        # the stronger the pullback force becomes.
        deviation = (self.price - self.config.initial_price) / max(self.config.initial_price, 0.01)
        mean_reversion = -0.0005 * deviation
        price_change_ratio += mean_reversion

        # Apply event-driven price impact.
        if self.event_manager:
            event_effects = self.event_manager.get_combined_effects()
            price_change_ratio += event_effects.get("price_impact", 0.0)

        # Add small noise when no shares are matched, simulating market hesitation.
        if matched_volume == 0:
            price_change_ratio += self.rng.uniform(-0.003, 0.003)

        # Clamp single-step price movement.
        price_change_ratio = max(
            -self.config.max_price_change_ratio,
            min(self.config.max_price_change_ratio, price_change_ratio),
        )

        self.price *= (1.0 + price_change_ratio)
        self.price = max(
            self.config.min_price,
            min(self.config.max_price, self.price),
        )
        self.price_history.append(self.price)

        state = self._get_state(agent_actions)

        # Add all triggered events info to state for logging
        if triggered_events:
            state["triggered_events"] = [
                {
                    "name": e.name,
                    "description": e.description,
                    "type": e.event_type.value,
                    "price_impact": e.price_impact,
                }
                for e in triggered_events
            ]

        # Add market pressure data for display
        state["total_buy"] = total_buy
        state["total_sell"] = total_sell

        # Add active event details for display.
        if self.event_manager:
            state["active_events_detail"] = self.event_manager.get_active_events_detail()

        return state

    # ------------------------------------------------------------------
    # Matching engine
    # ------------------------------------------------------------------

    def _match_orders(
        self,
        buy_orders: List[tuple],
        sell_orders: List[tuple],
        total_buy: int,
        total_sell: int,
    ):
        """
        Match buy and sell interest proportionally.

        Rules:
        - If buy demand >= sell supply: all sellers fill, buyers fill proportionally.
        - If sell supply > buy demand: all buyers fill, sellers fill proportionally.
        - Unmatched residual orders are discarded; there is no persistent order book.
        """
        if total_buy >= total_sell:
            # Sellers fully fill; buyers fill proportionally.
            # Sellers get full fills.
            for agent_id, qty in sell_orders:
                if qty > 0:
                    self._execute_trade(agent_id, "sell", qty)

            # Buyers receive proportional fills. Use floor allocation then
            # distribute remaining residuals by largest fractional part.
            fill_ratio = total_sell / max(total_buy, 1)
            ideal = [(agent_id, qty * fill_ratio, qty) for agent_id, qty in buy_orders]
            floors = [(aid, int(math.floor(val)), frac := val - math.floor(val), orig)
                      for aid, val, orig in ideal]
            allocated = {aid: fl for aid, fl, _, _ in floors}
            filled_total = sum(allocated.values())
            remaining = total_sell - filled_total

            if remaining > 0:
                # Sort by fractional part desc, break ties by original qty desc
                frac_list = sorted(
                    [(aid, frac, orig) for aid, _, frac, orig in floors],
                    key=lambda x: (x[1], x[2]),
                    reverse=True,
                )
                idx = 0
                while remaining > 0 and idx < len(frac_list):
                    aid, _, orig = frac_list[idx]
                    # Don't allocate more than originally requested
                    if allocated[aid] < orig:
                        allocated[aid] += 1
                        remaining -= 1
                    idx += 1

            # Execute buyer trades
            for aid, qty in allocated.items():
                if qty > 0:
                    self._execute_trade(aid, "buy", qty)

        else:
            # Buyers fully fill; sellers fill proportionally.
            for agent_id, qty in buy_orders:
                if qty > 0:
                    self._execute_trade(agent_id, "buy", qty)

            fill_ratio = total_buy / max(total_sell, 1)
            ideal = [(agent_id, qty * fill_ratio, qty) for agent_id, qty in sell_orders]
            floors = [(aid, int(math.floor(val)), frac := val - math.floor(val), orig)
                      for aid, val, orig in ideal]
            allocated = {aid: fl for aid, fl, _, _ in floors}
            filled_total = sum(allocated.values())
            remaining = total_buy - filled_total

            if remaining > 0:
                frac_list = sorted(
                    [(aid, frac, orig) for aid, _, frac, orig in floors],
                    key=lambda x: (x[1], x[2]),
                    reverse=True,
                )
                idx = 0
                while remaining > 0 and idx < len(frac_list):
                    aid, _, orig = frac_list[idx]
                    if allocated[aid] < orig:
                        allocated[aid] += 1
                        remaining -= 1
                    idx += 1

            for aid, qty in allocated.items():
                if qty > 0:
                    self._execute_trade(aid, "sell", qty)

    def _execute_trade(self, agent_id: str, action: str, quantity: int):
        """Execute one trade and update agent cash and holdings.

        Transaction costs (fee_rate and slippage_rate) are applied when
        configured. Slippage shifts the execution price against the trader:
        buyers pay more, sellers receive less.
        """
        agent = self.agents[agent_id]

        # Apply slippage to execution price (buyers pay up, sellers receive down).
        if self.config.slippage_rate > 0:
            if action == "buy":
                trade_price = self.price * (1.0 + self.config.slippage_rate)
            else:
                trade_price = self.price * (1.0 - self.config.slippage_rate)
        else:
            trade_price = self.price

        trade_value = quantity * trade_price
        fee = trade_value * self.config.fee_rate if self.config.fee_rate > 0 else 0.0

        if action == "buy":
            cost = trade_value + fee
            agent.cash -= cost
            agent.holdings += quantity
            cash_change = -cost
        else:  # sell
            revenue = trade_value - fee
            agent.cash += revenue
            agent.holdings -= quantity
            cash_change = revenue

        self.trade_history.append(TradeRecord(
            step=self.step_count,
            agent_id=agent_id,
            action=action,
            quantity=quantity,
            price=trade_price,
            cash_change=cash_change,
        ))

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def _get_state(
        self, agent_actions: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "price": self.price,
            "agents": {
                aid: {
                    "cash": a.cash,
                    "holdings": a.holdings,
                    "wealth": a.cash + a.holdings * self.price,
                }
                for aid, a in self.agents.items()
            },
            "matched_volume": (
                self.volume_history[-1] if self.volume_history else 0
            ),
            "agent_actions": agent_actions or {},
        }

"""
MarketEnv implements the virtual stock market environment.

Core responsibilities:
1. Collect orders from all agents (per-stock decisions).
2. Match buy and sell interest using a simple proportional mechanism
   independently for each stock.
3. Update each stock's price from net buying pressure.
4. Record trades and state snapshots.
5. Trigger market events that affect price and sentiment (global).
"""

import logging
import math
import random
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union

from .agents.external_ai_agent import ExternalAIAgent
from .base_agent import BaseAgent
from .config import MarketConfig
from .market_events import EventManager

logger = logging.getLogger(__name__)


def _coerce_quantity(value: Any) -> int:
    """Normalize a decision quantity to a non-negative int.

    Delegates to the agent-side parser so a model answering "10 shares",
    "all" or null is interpreted identically whether the value is read here
    or in ExternalAIAgent. A bare int() call raised ValueError/TypeError on
    such input, which cost the agent its entire round.  Anything genuinely
    uninterpretable becomes 0, i.e. a hold.
    """
    quantity = ExternalAIAgent._coerce_quantity(value)
    return max(0, quantity if quantity is not None else 0)


@dataclass
class TradeRecord:
    """Single trade execution record."""

    step: int
    agent_id: str
    action: str  # "buy" | "sell"
    quantity: int
    price: float
    cash_change: float  # Negative for buyers, positive for sellers.
    name: str = ""
    """Stock name this trade was executed on."""

    @property
    def symbol(self) -> str:
        """Backward-compatibility alias returning name."""
        return self.name


@dataclass
class StockMarket:
    """Per-stock market state: price, history, and volume.

    Each stock in a multi-stock environment maintains its own independent
    price series, trade volume, and order book for the current step.
    """

    name: str
    initial_price: float
    price: float
    price_history: List[float] = field(default_factory=list)
    volume_history: List[int] = field(default_factory=list)
    sector: str = ""
    blurb: str = ""
    pre_history: List[float] = field(default_factory=list)
    """Synthetic candles generated before round 1 (observation-only; never
    part of ``price_history`` so reports/snapshots stay round-accurate)."""

    @property
    def symbol(self) -> str:
        """Backward-compatibility alias returning name."""
        return self.name


class MarketEnv:
    """
    Virtual stock market environment supporting multiple stocks.

    Per-step workflow:
    1. Generate an observation for each agent (aggregated multi-stock data).
    2. Call Agent.act(observation) to collect per-stock orders.
    3. Match buy and sell orders proportionally for each stock independently.
    4. Update each stock's price from net buying pressure.
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
        self._pool: Optional[ThreadPoolExecutor] = None
        # RNG injected for reproducibility.
        if seed is None and config.seed is not None:
            seed = config.seed
        self._base_seed: Optional[int] = seed
        self.rng: Union[random.Random, ModuleType] = (
            random.Random(seed) if seed is not None else random
        )

        # Reject duplicate agent IDs up front.
        seen: set = set()
        for a in agents:
            if a.agent_id in seen:
                raise ValueError(f"Duplicate agent_id: {a.agent_id!r}")
            seen.add(a.agent_id)
        self.agents: Dict[str, BaseAgent] = {a.agent_id: a for a in agents}

        # Inject RNG into agents.
        #
        # When the environment is seeded, each agent receives its own
        # deterministic RNG stream derived from (seed, agent_id) instead of
        # sharing self.rng. Agent act() calls run concurrently in worker
        # threads (parallel_agents config), so per-agent streams keep seeded
        # runs reproducible regardless of thread scheduling order. The
        # environment's own stream keeps driving events / price noise as
        # before. When unseeded (self.rng = the random module), agents keep
        # sharing it to preserve legacy external-seeding behavior.
        for index, a in enumerate(agents):
            agent_rng = self.rng
            if isinstance(self.rng, random.Random):
                digest = zlib.crc32(str(a.agent_id).encode("utf-8"))
                agent_rng = random.Random(((self._base_seed or 0) << 32) ^ digest ^ index)
            try:
                setattr(a, "rng", agent_rng)
            except Exception:
                pass
            if hasattr(a, "base_agent"):
                try:
                    setattr(a.base_agent, "rng", agent_rng)
                except Exception:
                    pass

        # --- Multi-stock market state ---
        stock_specs = config.get_stock_specs()
        self.stocks: Dict[str, StockMarket] = {}
        for spec in stock_specs:
            self.stocks[spec.name] = StockMarket(
                name=spec.name,
                initial_price=spec.initial_price,
                price=spec.initial_price,
                price_history=[spec.initial_price],
                volume_history=[],
                sector=getattr(spec, "sector", "") or "",
                blurb=getattr(spec, "blurb", "") or "",
            )

        # --- Synthetic pre-history -------------------------------------
        # A short random walk per stock ending exactly at its current price,
        # visible to agents from the first observation so they can analyze
        # trends before any real rounds happen.
        backfill = int(getattr(config, "history_backfill_steps", 0) or 0)
        if backfill > 0:
            for sm in self.stocks.values():
                self._backfill_pre_history(sm, backfill)

        # Initialize each agent's holdings dict.
        first_name = stock_specs[0].name if stock_specs else "Stock 1"
        for a in agents:
            if not isinstance(a.holdings, dict):
                a.holdings = {}
            # Remap legacy int holdings ("_legacy" key) to the first stock.
            legacy = a.holdings.pop("_legacy", 0.0)
            if legacy:
                a.holdings[first_name] = a.holdings.get(first_name, 0.0) + legacy
            # Ensure every stock has an entry (default 0 or config value).
            for spec in stock_specs:
                if spec.symbol in a.holdings and spec.name not in a.holdings:
                    a.holdings[spec.name] = a.holdings.pop(spec.symbol)
                if spec.name not in a.holdings:
                    a.holdings[spec.name] = float(spec.initial_holdings)

        self.step_count: int = 0
        self.trade_history: List[TradeRecord] = []
        self._agent_error_counts: Dict[str, int] = {}

        # --- Fill indexes -------------------------------------------------
        # Maintained incrementally by _execute_trade so neither step() nor
        # get_observation() ever has to rescan the full trade history. Without
        # them a run costs O(agents * steps * total_trades) just to answer
        # "what did I fill last round?".
        # Fills executed during the CURRENT step: {(agent_id, symbol): qty}.
        self._step_fills: Dict[Tuple[str, str], int] = {}
        # Trades from the most recent step that produced any fill, grouped by
        # agent, for last-round observation feedback.
        self._last_fill_step: int = 0
        self._last_fills_by_agent: Dict[str, List[TradeRecord]] = {}

        # --- Event system ---
        self.event_manager: EventManager = EventManager(
            event_probability_multiplier=config.event_probability_multiplier,
            rng=self.rng,
            stock_names=list(self.stocks.keys()),
            impact_scale=config.event_impact_scale,
        )

        # --- Social influence ---
        self.social_map: Dict[str, dict] = {}
        self._recent_actions: Dict[str, dict] = {}
        self._social_influence: Optional[float] = None

        # --- Player action buffer ---
        self._pending_player_actions: Dict[str, dict] = {}

        # --- God Mode: persistent sentiment drift (-1..1), 0 by default ---
        self._sentiment_drift: float = 0.0

        # --- Reusable thread pool for parallel agent action collection ---
        # (initialized at the top of __init__)

    def _get_pool(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor for agent action collection."""
        target_workers = min(len(self.agents), 16)
        if self._pool is None or getattr(self._pool, "_max_workers", 0) != target_workers:
            self._shutdown_pool()
            self._pool = ThreadPoolExecutor(
                max_workers=target_workers,
                thread_name_prefix="ats-agent",
            )
        return self._pool

    def _shutdown_pool(self) -> None:
        """Shut down the background thread pool if active."""
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)
            self._pool = None

    def close(self) -> None:
        """Release resources; call after simulation finishes."""
        self._shutdown_pool()

    def __del__(self) -> None:
        self._shutdown_pool()

    # ------------------------------------------------------------------
    # Backward-compat properties (delegate to first stock)
    # ------------------------------------------------------------------

    @property
    def price(self) -> float:
        """Return the primary (first) stock's price for backward compat."""
        if self.stocks:
            return next(iter(self.stocks.values())).price
        return self.config.initial_price

    @price.setter
    def price(self, value: float) -> None:
        """Set the primary stock's price (backward compat)."""
        if self.stocks:
            first_key = next(iter(self.stocks.keys()))
            self.stocks[first_key].price = value

    @property
    def price_history(self) -> List[float]:
        """Return the primary stock's price history for backward compat."""
        if self.stocks:
            return next(iter(self.stocks.values())).price_history
        return []

    @property
    def volume_history(self) -> List[int]:
        """Return the primary stock's volume history for backward compat."""
        if self.stocks:
            return next(iter(self.stocks.values())).volume_history
        return []

    # ------------------------------------------------------------------
    # Portfolio valuation
    # ------------------------------------------------------------------

    def agent_wealth(self, agent: Union[str, BaseAgent]) -> float:
        """Mark-to-market portfolio wealth: cash + sum(holdings * price).

        Accepts an agent id or the agent object itself. This is the single
        definition of "wealth" in the project: the CLI simulator, the web API
        and the report exporter all call it, so a change to how a portfolio is
        valued cannot drift between them.
        """
        obj = self.agents[agent] if isinstance(agent, str) else agent
        holdings = obj.holdings if isinstance(obj.holdings, dict) else {}
        return obj.cash + sum(
            holdings.get(sym, 0) * sm.price for sym, sm in self.stocks.items()
        )

    def _backfill_pre_history(self, sm: "StockMarket", steps: int) -> None:
        """Generate ``steps`` synthetic prices ending exactly at the stock's
        current price (a de-drifted random walk, always positive)."""
        if steps <= 0 or sm.pre_history:
            return
        vol = min(0.03, max(0.005, self.config.price_sensitivity * 1.5))
        walk = [sm.initial_price]
        for _ in range(steps):
            walk.append(max(walk[-1] * (1.0 + self.rng.gauss(0.0, vol)), 0.01))
        # Rescale so the final synthetic point IS the anchor price —
        # the pre-history then joins round-0 with no gap.
        scale = sm.initial_price / walk[-1] if walk[-1] > 0 else 1.0
        sm.pre_history = [round(p * scale, 4) for p in walk]

    def set_player_action(self, action: str, quantity: int, symbol: Optional[str] = None):
        """Buffer the human player's action for the current step."""
        if symbol is None:
            if self.stocks:
                symbol = next(iter(self.stocks.keys()))
            else:
                symbol = "Stock 1"
        self._pending_player_actions[symbol] = {"action": action, "quantity": quantity}

    def pop_player_action(self):
        """Return and clear the first buffered player action, or None."""
        if self._pending_player_actions:
            sym = next(iter(self._pending_player_actions.keys()))
            return self._pending_player_actions.pop(sym)
        return None

    def pop_player_actions(self) -> Dict[str, dict]:
        """Return and clear all buffered player actions as a dict."""
        actions = dict(self._pending_player_actions)
        self._pending_player_actions = {}
        return actions

    # ------------------------------------------------------------------
    # Decision parsing (multi-stock + legacy compat)
    # ------------------------------------------------------------------

    def _parse_decisions(
        self, action: Dict[str, Any], stock_symbols: List[str]
    ) -> List[Dict[str, Any]]:
        """Parse an agent's action into a list of per-stock decisions.

        Supports two formats:
        - Multi-stock: ``{"decisions": [{"name", "action", "quantity",
          "reasoning"}, ...]}``
        - Legacy single-stock: ``{"action", "quantity", "reasoning"}`` —
          applied to the first stock.
        """
        if "decisions" in action:
            raw_decisions = action["decisions"]
            if not isinstance(raw_decisions, list):
                raise ValueError("'decisions' must be a list")
            result: List[Dict[str, Any]] = []
            for d in raw_decisions:
                if not isinstance(d, dict):
                    continue
                stk_name = str(d.get("name") or d.get("symbol") or "")
                act = str(d.get("action", "hold")).lower()
                if act not in ("buy", "sell", "hold"):
                    raise ValueError(f"invalid action: {act!r}")
                qty = _coerce_quantity(d.get("quantity", 0))
                result.append({
                    "name": stk_name,
                    "symbol": stk_name,
                    "action": act,
                    "quantity": qty,
                    "reasoning": str(d.get("reasoning", "")),
                    "error": bool(d.get("error", False)),
                })
            # Fill in missing stocks with hold.
            covered = {d["name"] for d in result}
            for sym in stock_symbols:
                if sym not in covered:
                    result.append({
                        "name": sym,
                        "symbol": sym,
                        "action": "hold",
                        "quantity": 0,
                        "reasoning": "",
                        "error": False,
                    })
            return result

        # Legacy single-stock format.
        act = str(action.get("action", "hold")).lower()
        if act not in ("buy", "sell", "hold"):
            raise ValueError(f"invalid action: {act!r}")
        qty = _coerce_quantity(action.get("quantity", 0))
        reasoning = str(action.get("reasoning", ""))
        error = bool(action.get("error", False))
        first_sym = stock_symbols[0] if stock_symbols else "Stock 1"
        return [
            {"name": sym, "symbol": sym, "action": "hold" if sym != first_sym else act,
             "quantity": qty if sym == first_sym else 0,
             "reasoning": reasoning if sym == first_sym else "",
             "error": error}
            for sym in stock_symbols
        ]

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

        # Build per-stock data.
        stocks_data: List[Dict[str, Any]] = []
        total_holdings_value = 0.0
        for sym, sm in self.stocks.items():
            h = agent.holdings.get(sym, 0) if isinstance(agent.holdings, dict) else 0
            # Observation window: synthetic pre-history first, then real
            # rounds — agents see a continuous trend from the very start.
            full_history = list(sm.pre_history) + list(sm.price_history)
            hist_len = min(len(full_history), self.config.price_history_length)
            stocks_data.append({
                "symbol": sym,
                "name": sm.name,
                "price": sm.price,
                "price_history": full_history[-hist_len:],
                "last_volume": sm.volume_history[-1] if sm.volume_history else 0,
                "my_holdings": h,
                "sector": sm.sector,
                "blurb": sm.blurb,
            })
            total_holdings_value += h * sm.price

        total_holdings = sum(agent.holdings.values()) if isinstance(agent.holdings, dict) else 0
        my_wealth = agent.cash + total_holdings_value

        obs: Dict[str, Any] = {
            "step": self.step_count,
            "stocks": stocks_data,
            "my_cash": agent.cash,
            "my_holdings": agent.holdings if isinstance(agent.holdings, dict) else {},
            "my_total_holdings": total_holdings,
            "my_wealth": my_wealth,
            "market_sentiment": 0.0,
        }

        # Backward-compat: expose first stock's data as top-level fields.
        if stocks_data:
            first = stocks_data[0]
            obs["price"] = first["price"]
            obs["price_history"] = first["price_history"]
            obs["last_volume"] = first["last_volume"]

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

        # --- Social influence ---
        # Expose this agent's social peers' most recent resolved actions so the
        # TraitAgent layer can mimic friends/idol and fade enemies. Only agents
        # that actually traded (action != hold) carry a signal.
        rel_map = self.social_map.get(agent_id)
        if rel_map:
            groups = (
                ("idol", [rel_map["idol"]] if rel_map.get("idol") else []),
                ("friend", rel_map.get("friends", [])),
                ("enemy", rel_map.get("enemies", [])),
            )
            peers = []
            for relation, ids in groups:
                for pid in ids:
                    if not pid or pid == agent_id:
                        continue
                    r = self._recent_actions.get(pid)
                    if r and r.get("action", "hold") != "hold":
                        peers.append({
                            "id": pid,
                            "relation": relation,
                            "action": r.get("action", "hold"),
                            "quantity": int(r.get("filled", 0)),
                        })
            if peers:
                obs["social_peers"] = peers
        influence = self._social_influence
        if influence is None:
            influence = getattr(self.config, "social_influence", 0.0)
        if influence:
            obs["social_influence"] = float(influence)

        # --- Last-round outcome feedback (learning loop) ---
        # Feedback reflects the most recent step that produced fills, whether
        # the observation is built mid-round or queried afterwards.
        # Served from the fill index: these are exactly this agent's trades
        # from the most recent step that filled anything.
        prev_trades: list = self._last_fills_by_agent.get(agent_id, [])
        if prev_trades:
            fills = []
            net_cash = 0.0
            for t in prev_trades:
                stk = self.stocks.get(t.name)
                move = 0.0
                if stk is not None and t.price > 0:
                    move = round((stk.price / t.price - 1) * 100, 2)
                fills.append({
                    "action": t.action,
                    "quantity": t.quantity,
                    "symbol": t.name,
                    "fill_price": round(t.price, 2),
                    "current_price": round(stk.price, 2) if stk else None,
                    "move_since_fill_pct": move,
                })
                net_cash += t.cash_change
            obs["last_round"] = {
                "trades": fills,
                "net_cash_flow": round(net_cash, 2),
            }

        return obs

    # ------------------------------------------------------------------
    # Main loop: one step
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """Run one simulation step."""
        self.step_count += 1
        self._step_fills = {}

        # ---------- 0. Event system ----------
        # Global events hit every stock; company-specific events hit one
        # randomly chosen stock (resolved inside the event manager).
        triggered_events: list = []
        if self.event_manager:
            # Tick existing events
            self.event_manager.tick()
            # Try to trigger new events (may return multiple)
            triggered_events = self.event_manager.try_trigger_event(
                self.step_count, stock_names=list(self.stocks.keys())
            )

        stock_symbols = list(self.stocks.keys())

        # ---------- 1. Collect orders per stock ----------
        buy_orders: Dict[str, List[Tuple[str, int]]] = {sym: [] for sym in stock_symbols}
        sell_orders: Dict[str, List[Tuple[str, int]]] = {sym: [] for sym in stock_symbols}
        # Player orders bypass peer matching: the human trades against an
        # implicit market maker so orders always fill (paper-trading).
        player_orders: Dict[str, List[Tuple[str, str, int]]] = {}
        # Per-agent remaining cash budget for this step's buy orders (cash is
        # shared across stocks; prevents negative-cash overdrafts).
        cash_budget: Dict[str, float] = {}
        # agent_actions[aid][sym] = {action, requested_qty, filled_qty, reasoning, error}
        agent_actions: Dict[str, Dict[str, dict]] = {
            aid: {} for aid in self.agents
        }

        # Generate all observations up front. Observation generation only
        # READS market state (prices, holdings, previous-round feedback) and
        # no agent's act() mutates shared market state, so these are exactly
        # the same observations a sequential loop would build.
        observations = {aid: self.get_observation(aid) for aid in self.agents}

        def _collect(agent_id: str):
            """Run one agent's act(); returns (decisions, error_or_None)."""
            try:
                action = self.agents[agent_id].act(observations[agent_id])
                if not isinstance(action, dict):
                    raise ValueError("agent action must be a dictionary")
                return self._parse_decisions(action, stock_symbols), None
            except Exception as exc:
                return None, exc

        # Collect actions concurrently by default: each trader's LLM API call
        # (typically the slowest part of a round) overlaps with the others,
        # cutting wall-clock time from sum-of-latencies to max-of-latencies.
        # Deterministic per-agent RNG streams keep seeded runs reproducible
        # regardless of completion order; results are gathered back into
        # agent insertion order before any further processing.
        use_parallel = bool(getattr(self.config, "parallel_agents", True)) and (
            len(self.agents) > 1
        )
        outcomes: Dict[str, Tuple[Optional[List[Dict[str, Any]]], Optional[Exception]]] = {}
        if use_parallel:
            pool = self._get_pool()
            futures = {aid: pool.submit(_collect, aid) for aid in self.agents}
            for aid, fut in futures.items():
                outcomes[aid] = fut.result()
        else:
            for aid in self.agents:
                outcomes[aid] = _collect(aid)

        for agent_id, agent in self.agents.items():
            decisions, first_exc = outcomes[agent_id]
            if first_exc is not None:
                exc = first_exc
                count = self._agent_error_counts.get(agent_id, 0) + 1
                self._agent_error_counts[agent_id] = count
                if count == 1:
                    logger.warning(
                        "Agent '%s' failed to act: %s: %s. Recording an AI failure.",
                        agent_id,
                        type(exc).__name__,
                        exc,
                    )
                decisions = [
                    {
                        "name": sym,
                        "symbol": sym,
                        "action": "hold",
                        "quantity": 0,
                        "reasoning": f"AI acquisition failed: {type(exc).__name__}: {exc}",
                        "error": True,
                    }
                    for sym in stock_symbols
                ]

            if decisions is None:
                continue

            for d in decisions:
                sym = d["symbol"]
                if sym not in self.stocks:
                    continue  # ignore unknown symbols
                action_type = d["action"]
                quantity = d["quantity"]
                reasoning = d.get("reasoning", "")
                error = d.get("error", False)

                agent_actions[agent_id][sym] = {
                    "action": action_type,
                    "requested_qty": quantity,
                    "filled_qty": 0,
                    "reasoning": reasoning,
                    "error": error,
                }

                sm = self.stocks[sym]
                is_player = bool(getattr(agent, "is_player", False))
                if action_type == "buy" and quantity > 0:
                    # Clip quantity to the maximum affordable amount. Cash is
                    # SHARED across stocks: track each agent's remaining
                    # budget so multiple max-size buy orders in one round
                    # cannot overdraw the account.
                    effective_price = sm.price * (
                        1.0 + self.config.slippage_rate
                    ) * (1.0 + self.config.fee_rate)
                    if agent_id not in cash_budget:
                        cash_budget[agent_id] = agent.cash
                    if sm.price > 0 and cash_budget[agent_id] > 0 and effective_price > 0:
                        max_afford = math.floor(
                            cash_budget[agent_id] / effective_price + 1e-9
                        )
                    else:
                        max_afford = 0
                    quantity = min(quantity, max_afford)
                    if quantity > 0:
                        # Reserve the estimated cost for this order.
                        cash_budget[agent_id] -= quantity * effective_price
                        if is_player:
                            player_orders.setdefault(sym, []).append(
                                (agent_id, "buy", quantity)
                            )
                        else:
                            buy_orders[sym].append((agent_id, quantity))

                elif action_type == "sell" and quantity > 0:
                    # Clip quantity to current holdings of this stock.
                    h = agent.holdings.get(sym, 0) if isinstance(agent.holdings, dict) else 0
                    quantity = min(quantity, int(h))
                    if quantity > 0:
                        if is_player:
                            player_orders.setdefault(sym, []).append(
                                (agent_id, "sell", quantity)
                            )
                        else:
                            sell_orders[sym].append((agent_id, quantity))

        # ---------- 2. Match orders per stock ----------
        per_stock_totals: Dict[str, dict] = {}
        for sym, sm in self.stocks.items():
            buys = buy_orders[sym]
            sells = sell_orders[sym]
            total_buy = sum(q for _, q in buys)
            total_sell = sum(q for _, q in sells)
            matched_volume = min(total_buy, total_sell)

            if matched_volume > 0:
                self._match_orders(sym, buys, sells, total_buy, total_sell)

            # Market maker fills the human player's orders at the current
            # price (slippage/fees still apply). Player volume counts
            # toward price pressure like any other order.
            player_volume = 0
            for agent_id, action_type, qty in player_orders.get(sym, []):
                if qty > 0:
                    self._execute_trade(sym, agent_id, action_type, qty)
                    player_volume += qty
                    if action_type == "buy":
                        total_buy += qty
                    else:
                        total_sell += qty

            sm.volume_history.append(matched_volume + player_volume)
            per_stock_totals[sym] = {
                "total_buy": total_buy,
                "total_sell": total_sell,
                "matched": matched_volume + player_volume,
            }

        # Update filled quantities from this step's fill index, which
        # _execute_trade maintains as trades happen. Rescanning
        # self.trade_history here made every step cost O(total trades so far).
        for (aid, sym), filled in self._step_fills.items():
            if sym in agent_actions.get(aid, {}):
                agent_actions[aid][sym]["filled_qty"] += filled

        # Snapshot this round's resolved actions (aggregated for social).
        self._recent_actions = {}
        for aid, stock_acts in agent_actions.items():
            total_filled = sum(
                sa.get("filled_qty", 0) for sa in stock_acts.values()
            )
            # Dominant action = the one with most filled quantity.
            dominant_action = "hold"
            max_filled = 0
            for sa in stock_acts.values():
                f = sa.get("filled_qty", 0)
                if f > max_filled:
                    max_filled = f
                    dominant_action = sa.get("action", "hold")
            self._recent_actions[aid] = {
                "action": dominant_action,
                "filled": total_filled,
            }

        # ---------- 3. Update price per stock ----------
        for sym, sm in self.stocks.items():
            totals = per_stock_totals[sym]
            total_buy = totals["total_buy"]
            total_sell = totals["total_sell"]
            total_volume = total_buy + total_sell
            if total_volume > 0:
                net_pressure = (total_buy - total_sell) / total_volume
            else:
                net_pressure = 0.0

            price_change_ratio = self.config.price_sensitivity * net_pressure

            # Mean reversion: pull toward initial price.
            deviation = (sm.price - sm.initial_price) / max(sm.initial_price, 0.01)
            mean_reversion = -self.config.mean_reversion_strength * deviation
            price_change_ratio += mean_reversion

            # Apply event-driven price impact. Global events affect all
            # stocks; stock-scoped events only move their own stock.
            if self.event_manager:
                event_effects = self.event_manager.get_combined_effects(sym)
                price_change_ratio += event_effects.get("price_impact", 0.0)

            # Add small noise when no shares are matched.
            noise = self.config.idle_price_noise
            if totals["matched"] == 0 and noise > 0:
                price_change_ratio += self.rng.uniform(-noise, noise)

            # Clamp single-step price movement.
            price_change_ratio = max(
                -self.config.max_price_change_ratio,
                min(self.config.max_price_change_ratio, price_change_ratio),
            )

            sm.price *= (1.0 + price_change_ratio)
            sm.price = max(
                self.config.min_price,
                min(self.config.max_price, sm.price),
            )
            sm.price_history.append(sm.price)

        state = self._get_state(agent_actions, per_stock_totals)

        # Add all triggered events info to state for logging
        if triggered_events:
            state["triggered_events"] = [
                {
                    "name": e.name,
                    "description": e.description,
                    "type": e.event_type.value,
                    "price_impact": e.price_impact,
                    "scope": e.scope,
                    "stock": e.target_stock,
                }
                for e in triggered_events
            ]

        # Add active event details for display.
        if self.event_manager:
            state["active_events_detail"] = self.event_manager.get_active_events_detail()

        return state

    # ------------------------------------------------------------------
    # Matching engine (per-stock)
    # ------------------------------------------------------------------

    def _match_orders(
        self,
        symbol: str,
        buy_orders: List[tuple],
        sell_orders: List[tuple],
        total_buy: int,
        total_sell: int,
    ):
        """
        Match buy and sell interest proportionally for a single stock.

        Rules:
        - If buy demand >= sell supply: all sellers fill, buyers fill proportionally.
        - If sell supply > buy demand: all buyers fill, sellers fill proportionally.
        - Unmatched residual orders are discarded; there is no persistent order book.
        """
        if total_buy >= total_sell:
            # Sellers fully fill; buyers fill proportionally.
            for agent_id, qty in sell_orders:
                if qty > 0:
                    self._execute_trade(symbol, agent_id, "sell", qty)

            fill_ratio = total_sell / max(total_buy, 1)
            ideal = [(agent_id, qty * fill_ratio, qty) for agent_id, qty in buy_orders]
            floors = [(aid, int(math.floor(val)), val - math.floor(val), orig)
                      for aid, val, orig in ideal]
            allocated = {aid: fl for aid, fl, _, _ in floors}
            filled_total = sum(allocated.values())
            remaining = total_sell - filled_total

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
                    self._execute_trade(symbol, aid, "buy", qty)

        else:
            # Buyers fully fill; sellers fill proportionally.
            for agent_id, qty in buy_orders:
                if qty > 0:
                    self._execute_trade(symbol, agent_id, "buy", qty)

            fill_ratio = total_buy / max(total_sell, 1)
            ideal = [(agent_id, qty * fill_ratio, qty) for agent_id, qty in sell_orders]
            floors = [(aid, int(math.floor(val)), val - math.floor(val), orig)
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
                    self._execute_trade(symbol, aid, "sell", qty)

    def _execute_trade(self, symbol: str, agent_id: str, action: str, quantity: int):
        """Execute one trade and update agent cash and holdings for a stock.

        Transaction costs (fee_rate and slippage_rate) are applied when
        configured. Slippage shifts the execution price against the trader:
        buyers pay more, sellers receive less.
        """
        agent = self.agents[agent_id]
        sm = self.stocks[symbol]

        # Apply slippage to execution price.
        if self.config.slippage_rate > 0:
            if action == "buy":
                trade_price = sm.price * (1.0 + self.config.slippage_rate)
            else:
                trade_price = sm.price * (1.0 - self.config.slippage_rate)
        else:
            trade_price = sm.price

        trade_value = quantity * trade_price
        fee = trade_value * self.config.fee_rate if self.config.fee_rate > 0 else 0.0

        if action == "buy":
            cost = trade_value + fee
            agent.cash -= cost
            if not isinstance(agent.holdings, dict):
                agent.holdings = {}
            agent.holdings[symbol] = agent.holdings.get(symbol, 0) + quantity
            cash_change = -cost
        else:  # sell
            revenue = trade_value - fee
            agent.cash += revenue
            if not isinstance(agent.holdings, dict):
                agent.holdings = {}
            agent.holdings[symbol] = agent.holdings.get(symbol, 0) - quantity
            cash_change = revenue

        record = TradeRecord(
            step=self.step_count,
            agent_id=agent_id,
            action=action,
            quantity=quantity,
            price=trade_price,
            cash_change=cash_change,
            name=symbol,
        )
        self.trade_history.append(record)

        # Keep the fill indexes in sync (see __init__ for why they exist).
        key = (agent_id, symbol)
        self._step_fills[key] = self._step_fills.get(key, 0) + quantity
        if self.step_count != self._last_fill_step:
            self._last_fill_step = self.step_count
            self._last_fills_by_agent = {}
        self._last_fills_by_agent.setdefault(agent_id, []).append(record)

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def _get_state(
        self,
        agent_actions: Optional[Dict[str, Dict[str, dict]]] = None,
        per_stock_totals: Optional[Dict[str, dict]] = None,
    ) -> Dict[str, Any]:
        """Build a state snapshot with multi-stock data.

        The ``agent_actions`` dict is nested: ``{aid: {sym: {action, ...}}}``.
        """
        per_stock_totals = per_stock_totals or {}
        # Top-level aggregates across all stocks (CLI pressure bar & web UI).
        total_buy = sum(
            per_stock_totals.get(sym, {}).get("total_buy", 0)
            for sym in self.stocks
        )
        total_sell = sum(
            per_stock_totals.get(sym, {}).get("total_sell", 0)
            for sym in self.stocks
        )
        state: Dict[str, Any] = {
            "step": self.step_count,
            "price": self.price,  # backward compat (first stock)
            "total_buy": total_buy,
            "total_sell": total_sell,
            "stocks": {
                sym: {
                    "price": sm.price,
                    "name": sm.name,
                    "price_history": list(sm.price_history),
                    "volume": sm.volume_history[-1] if sm.volume_history else 0,
                    "total_buy": per_stock_totals.get(sym, {}).get("total_buy", 0),
                    "total_sell": per_stock_totals.get(sym, {}).get("total_sell", 0),
                }
                for sym, sm in self.stocks.items()
            },
            "agents": {
                aid: {
                    "cash": a.cash,
                    "holdings": dict(a.holdings) if isinstance(a.holdings, dict) else {},
                    "wealth": self.agent_wealth(a),
                }
                for aid, a in self.agents.items()
            },
            "matched_volume": (
                sum(sm.volume_history[-1] if sm.volume_history else 0
                    for sm in self.stocks.values())
            ),
            "agent_actions": agent_actions or {},
        }
        return state

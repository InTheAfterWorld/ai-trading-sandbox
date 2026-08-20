"""
Simulator coordinates the market loop and final reporting.

It drives MarketEnv for a configured number of steps and prints a summary
when the run completes.
"""

from typing import Any, Dict, List, Optional

from .market_env import MarketEnv
from .run_metadata import RunMetadata, save_run_snapshot
from .console_utils import (
    Colors,
    colorize,
    pressure_bar,
    trend_arrow,
    agent_type_label,
    agent_personality,
    agent_personality_desc,
    sparkline,
)


class Simulator:
    """
    Simulation controller that connects MarketEnv and all agents.

    Parameters
    ----------
    env : MarketEnv
        Market environment instance.
    """

    def __init__(self, env: MarketEnv):
        self.env = env
        self.state_history: List[Dict[str, Any]] = []
        self._ranking_interval: int = 5
        self._total_steps: int = 0
        self._initial_wealths: Dict[str, float] = {}
        self._prev_wealths: Dict[str, float] = {}
        self.metadata: Optional[RunMetadata] = None

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(
        self,
        steps: int,
        verbose: bool = True,
        round_by_round: bool = True,
        log_interval: Optional[int] = None,
        ranking_interval: int = 5,
        interactive: bool = False,
        seed: Optional[int] = None,
        run_id: Optional[str] = None,
        save_snapshot: bool = True,
        runs_dir: str = "runs",
    ) -> List[Dict[str, Any]]:
        """
        Run the simulation for N steps.

        Parameters
        ----------
        steps : int
            Number of simulation steps (each step = one round).
        verbose : bool
            Whether to print any output.
        round_by_round : bool
            If True, print detailed per-round results including each agent's
            decision. If False, use compact interval-based summary printing.
        log_interval : int, optional
            Used only when round_by_round is False. Defaults to ~10 updates.
        ranking_interval : int
            How often (in steps) to print a mini wealth ranking. Set to 0
            to disable. Default: 5.
        interactive : bool
            If True, pause after each round and let the user choose to
            continue or stop early. Only effective when round_by_round is
            also True. Default: False.
        seed : int, optional
            Random seed for reproducibility. If specified, overrides config seed.
        run_id : str, optional
            Unique ID for this simulation run. Auto-generated if None.
        save_snapshot : bool
            Whether to save metadata and run snapshot to disk. Default: True.
        runs_dir : str
            Directory to store run snapshots. Default: "runs".

        Returns
        -------
        state_history : list of dict
            State snapshots for each simulation step.
        """
        if log_interval is None:
            log_interval = max(1, steps // 10)

        self._ranking_interval = ranking_interval
        self._total_steps = steps

        # Initialize run metadata & seed
        eff_seed = seed if seed is not None else self.env.config.seed
        self.metadata = RunMetadata.create(
            config=self.env.config,
            agents=list(self.env.agents.values()),
            seed=eff_seed,
            run_id=run_id,
        )

        version_info = self.metadata.version
        ver_str = version_info.get("package_version", "")
        commit = version_info.get("git_commit")
        if commit:
            dirty_str = " (dirty)" if version_info.get("git_dirty") else ""
            ver_str += f" [git:{commit}{dirty_str}]"

        print(f"\n{'='*60}")
        print(f"  AI Market Sandbox — Simulation Start")
        print(f"{'='*60}")
        print(f"  Run ID         : {self.metadata.run_id}")
        print(f"  Version        : {ver_str}")
        print(f"  Seed           : {self.metadata.seed}")
        print(f"  Initial Price  : ${self.env.config.initial_price:.2f}")
        print(f"  Agents         : {len(self.env.agents)}")
        print(f"  Steps          : {steps}")
        print(f"  Output         : {'round-by-round' if round_by_round else 'summary'}")
        print(f"{'='*60}\n")

        # Print agent roster with personality info.
        self._print_agent_roster()

        if not round_by_round:
            print(f"{'Step':>5}  {'Price':>10}  {'Volume':>8}  {'Change':>8}")
            print(f"{'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}")

        prev_price = self.env.price

        # Capture initial wealths before any trading begins.
        for aid, a in self.env.agents.items():
            self._initial_wealths[aid] = a.cash + a.holdings * prev_price

        for i in range(steps):
            state = self.env.step()
            self.state_history.append(state)

            if verbose:
                if round_by_round:
                    self._print_round(state, prev_price)
                elif i % log_interval == 0 or i == steps - 1:
                    change = (state["price"] - prev_price) / max(prev_price, 0.01) * 100
                    print(
                        f"{state['step']:5d}  "
                        f"${state['price']:9.2f}  "
                        f"{state['matched_volume']:8d}  "
                        f"{change:7.2f}%"
                    )

            # Update prev_wealths for next round's delta display.
            current_price = state["price"]
            for aid, a in self.env.agents.items():
                self._prev_wealths[aid] = a.cash + a.holdings * current_price

            prev_price = state["price"]

            # Interactive prompt: let the user continue or stop early.
            if interactive and round_by_round and i < steps - 1:
                stop = self._prompt_continue()
                if stop:
                    print(colorize(
                        f"\n  Simulation stopped early at round {state['step']} "
                        f"of {steps}.",
                        Colors.YELLOW,
                    ))
                    break

        # Save snapshot automatically if enabled
        if save_snapshot and self.metadata:
            self.metadata.summary = {
                "final_price": self.env.price,
                "total_trades": len(self.env.trade_history),
                "steps_completed": len(self.state_history),
            }
            snapshot_dir = save_run_snapshot(
                metadata=self.metadata,
                state_history=self.state_history,
                trade_history=self.env.trade_history,
                output_dir=runs_dir,
            )
            if verbose:
                print(f"  Run snapshot saved to: {snapshot_dir}")

        print()
        return self.state_history

    @staticmethod
    def _prompt_continue() -> bool:
        """
        Ask the user whether to continue or stop.

        Returns
        -------
        stop : bool
            True if the user wants to stop, False to continue.
        """
        try:
            choice = input(
                colorize(
                    "\n  [Enter] continue  |  [q] stop & show results  > ",
                    Colors.DIM,
                )
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Treat EOF or Ctrl-C as "stop".
            return True

        return choice in ("q", "quit", "stop", "s", "exit")

    # ------------------------------------------------------------------
    # Round-by-round printing
    # ------------------------------------------------------------------

    def _print_agent_roster(self) -> None:
        """Print a roster of all agents with their type and personality."""
        print(f"{'─'*64}")
        print(colorize("  Agent Roster:", Colors.BOLD))
        print(f"{'─'*64}")

        for agent_id, agent in self.env.agents.items():
            type_label = agent_type_label(agent)
            type_str = colorize(f"[{type_label}]", Colors.DIM)

            personality = agent_personality(agent)
            desc = agent_personality_desc(agent)

            if personality and personality not in ("balanced", "custom", ""):
                pers_str = colorize(f"{personality:<14}", Colors.YELLOW)
                desc_str = colorize(f"— {desc}", Colors.DIM)
            elif personality == "balanced":
                pers_str = colorize(f"{'balanced':<14}", Colors.GRAY)
                desc_str = colorize(f"— {desc}", Colors.DIM)
            else:
                pers_str = colorize(f"{'(none)':<14}", Colors.GRAY)
                desc_str = ""

            print(f"  {agent_id:<22} {type_str:<16} {pers_str}  {desc_str}")

        print(f"{'─'*64}\n")

    def _print_round(self, state: Dict[str, Any], prev_price: float) -> None:
        """Print detailed results for a single round with visual enhancements."""
        step = state["step"]
        price = state["price"]
        change = (price - prev_price) / max(prev_price, 0.01) * 100
        volume = state["matched_volume"]
        actions = state.get("agent_actions", {})

        triggered_events = state.get("triggered_events", [])
        total_buy = state.get("total_buy", 0)
        total_sell = state.get("total_sell", 0)

        # --- Progress bar ---
        if self._total_steps > 0:
            progress = step / self._total_steps
            bar_width = 30
            filled = int(progress * bar_width)
            bar = colorize("#" * filled, Colors.GREEN) + "-" * (bar_width - filled)
            print(f"  [{bar}] {step}/{self._total_steps}")

        # --- Determine price color ---
        if change > 0.01:
            price_color = Colors.GREEN
        elif change < -0.01:
            price_color = Colors.RED
        else:
            price_color = ""

        arrow = trend_arrow(change)
        change_str = f"{change:+.2f}%"
        colored_change = colorize(f"{arrow} {change_str}", price_color)
        colored_price = colorize(f"${price:.2f}", price_color)

        print(f"{'─'*64}")
        print(
            f" Round {step}  |  ${prev_price:.2f} -> {colored_price}  "
            f"({colored_change})  |  Vol: {volume}"
        )

        # --- Price sparkline ---
        spark = sparkline(list(self.env.price_history[-12:]))
        if spark:
            print(f"  Trend: {colorize(spark, price_color)}")

        # --- Market pressure bar ---
        if total_buy + total_sell > 0:
            print(f"  Pressure: {pressure_bar(total_buy, total_sell)}")

        print(f"{'─'*64}")

        # --- Newly triggered events in yellow ---
        for evt in triggered_events:
            event_name = evt.get("name", "Unknown Event")
            event_desc = evt.get("description", "")
            print(colorize(
                f'  *** EVENT: {event_name} — "{event_desc}" ***',
                Colors.YELLOW,
            ))
        if triggered_events:
            print()

        # --- Ongoing active events with remaining duration ---
        triggered_names = {e.get("name") for e in triggered_events}
        ongoing = state.get("active_events_detail", [])
        ongoing = [e for e in ongoing if e.get("name") not in triggered_names]
        if ongoing:
            print(colorize("  Active Events:", Colors.YELLOW))
            for e in ongoing:
                remaining = e.get("remaining_steps", 0)
                total = e.get("total_steps", 1)
                bar_filled = "#" * remaining
                bar_empty = "-" * max(0, total - remaining)
                print(colorize(
                    f"    {e['name']} [{bar_filled}{bar_empty}] "
                    f"{remaining}/{total} steps left",
                    Colors.YELLOW,
                ))
            print()

        # --- Agent actions with type labels, colors, return %, and wealth delta ---
        for agent_id, agent in self.env.agents.items():
            act_info = actions.get(agent_id, {})
            action = act_info.get("action", "hold")
            req = act_info.get("requested_qty", 0)
            filled = act_info.get("filled_qty", 0)
            reasoning = act_info.get("reasoning", "")

            wealth = agent.cash + agent.holdings * price
            type_label = agent_type_label(agent)

            # Return percentage from initial wealth
            initial_w = self._initial_wealths.get(agent_id, wealth)
            ret_pct = (wealth / initial_w - 1) * 100 if initial_w > 0 else 0.0
            ret_color = Colors.GREEN if ret_pct >= 0 else Colors.RED
            ret_str = colorize(f"{ret_pct:+.1f}%", ret_color)

            # Wealth delta from previous round
            prev_w = self._prev_wealths.get(agent_id)
            if prev_w is not None:
                delta = wealth - prev_w
                delta_color = Colors.GREEN if delta >= 0 else Colors.RED
                delta_str = colorize(f"{delta:>+8.0f}", delta_color)
            else:
                delta_str = colorize(f"{'--':>8}", Colors.GRAY)

            # Determine action color
            if action == "buy":
                action_color = Colors.GREEN
            elif action == "sell":
                action_color = Colors.RED
            else:
                action_color = Colors.GRAY

            # Format detail string
            if action == "hold":
                detail = "-"
            elif filled == 0:
                detail = f"req:{req} filled:0 (unfilled)"
            elif filled < req:
                detail = f"req:{req} filled:{filled} (partial)"
            else:
                detail = f"req:{req} filled:{filled}"

            # Pad action text BEFORE coloring so visual width is correct
            action_padded = f"{action.upper():<5}"
            action_str = colorize(action_padded, action_color)
            type_str = colorize(f"[{type_label}]", Colors.DIM)

            print(
                f"  {agent_id:<22} {type_str:<18} {action_str} {detail:<28} "
                f"| ${agent.cash:>9.0f}  H:{agent.holdings:>6}  W:${wealth:>9.0f}  "
                f"R:{ret_str}  d:{delta_str}"
            )

            # Display reasoning (truncated for readability)
            if reasoning:
                display_reasoning = reasoning[:80]
                if len(reasoning) > 80:
                    display_reasoning += "..."
                print(colorize(f"    -> {display_reasoning}", Colors.DIM))

        # --- Round action summary ---
        buys = sum(1 for a in actions.values() if a.get("action") == "buy")
        sells = sum(1 for a in actions.values() if a.get("action") == "sell")
        holds = sum(1 for a in actions.values() if a.get("action") == "hold")
        print(
            f"  {colorize(f'{buys} BUY', Colors.GREEN)}  |  "
            f"{colorize(f'{sells} SELL', Colors.RED)}  |  "
            f"{colorize(f'{holds} HOLD', Colors.GRAY)}"
        )

        # Mini wealth ranking every N rounds
        if self._should_show_mini_ranking(step):
            self._print_mini_ranking(state)

        print()

    def _should_show_mini_ranking(self, step: int) -> bool:
        """Check if a mini wealth ranking should be displayed this round."""
        if self._ranking_interval <= 0:
            return False
        return step % self._ranking_interval == 0

    def _print_mini_ranking(self, state: Dict[str, Any]) -> None:
        """Print a compact wealth ranking: top 3 + bottom 3."""
        agents_data = state.get("agents", {})
        if len(agents_data) < 4:
            return

        ranked = sorted(
            agents_data.items(),
            key=lambda x: x[1].get("wealth", 0),
            reverse=True,
        )

        print(colorize("  -- Mini Ranking --", Colors.DIM))

        # Top 3
        for rank, (aid, data) in enumerate(ranked[:3], 1):
            wealth = data.get("wealth", 0)
            marker = colorize(f"#{rank}", Colors.GREEN if rank == 1 else "")
            print(f"    {marker} {aid:<20} ${wealth:>9.0f}")

        # Ellipsis if more than 6 agents
        if len(ranked) > 6:
            print(colorize(f"    ... ({len(ranked) - 6} others)", Colors.DIM))

        # Bottom 3
        start_rank = len(ranked) - 2
        for rank, (aid, data) in enumerate(ranked[-3:], start_rank):
            wealth = data.get("wealth", 0)
            marker = colorize(f"#{rank}", Colors.RED if rank == len(ranked) else "")
            print(f"    {marker} {aid:<20} ${wealth:>9.0f}")

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def _compute_agent_metrics(self, agent_id: str) -> Dict[str, float]:
        """Compute performance metrics for a single agent from state history."""
        wealths: List[float] = []
        for state in self.state_history:
            agent_data = state.get("agents", {}).get(agent_id, {})
            wealths.append(agent_data.get("wealth", 0))

        if len(wealths) < 2:
            return {"sharpe": 0.0, "max_drawdown": 0.0, "volatility": 0.0, "win_rate": 0.0}

        # Per-step returns
        returns: List[float] = []
        for i in range(1, len(wealths)):
            if wealths[i - 1] > 0:
                returns.append((wealths[i] - wealths[i - 1]) / wealths[i - 1])
            else:
                returns.append(0.0)

        # Annualized Sharpe with zero risk-free return per simulation step.
        mean_ret = sum(returns) / len(returns) if returns else 0.0
        var_ret = sum((r - mean_ret) ** 2 for r in returns) / len(returns) if returns else 0.0
        std_ret = var_ret ** 0.5
        sharpe = (mean_ret / std_ret) * (252 ** 0.5) if std_ret > 0 else 0.0

        # Max drawdown
        peak = wealths[0]
        max_dd = 0.0
        for w in wealths:
            if w > peak:
                peak = w
            dd = (peak - w) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # Volatility (std of returns)
        volatility = std_ret

        # Win rate (percentage of positive-return steps)
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(returns) if returns else 0.0

        return {
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "volatility": volatility,
            "win_rate": win_rate,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(
        self,
        generate_charts: bool = False,
        chart_output_dir: str = "charts",
    ) -> None:
        """Print the final simulation report.

        Parameters
        ----------
        generate_charts : bool
            If True, generate visualization charts after the report.
            Default: False.
        chart_output_dir : str
            Directory to save chart images. Default: "charts".
        """
        env = self.env
        agents = list(env.agents.values())
        initial_price = env.config.initial_price

        print(f"{'='*60}")
        print(f"  FINAL REPORT")
        print(f"{'='*60}")
        if self.metadata:
            print(f"  Run ID  : {self.metadata.run_id}")
            print(f"  Seed    : {self.metadata.seed}")

        # Price summary.
        print(f"\n  Price Summary:")
        print(f"    Initial : ${initial_price:.2f}")
        print(f"    Final   : ${env.price:.2f}")
        pct = (env.price / initial_price - 1) * 100
        print(f"    Change  : {pct:+.2f}%")
        print(f"    Min/Max : ${min(env.price_history):.2f} / ${max(env.price_history):.2f}")

        # Trade summary.
        buy_trades = [t for t in env.trade_history if t.action == "buy"]
        sell_trades = [t for t in env.trade_history if t.action == "sell"]
        print(f"\n  Trade Summary:")
        print(f"    Total Trades : {len(env.trade_history)}")
        print(f"    Buy Orders   : {len(buy_trades)}")
        print(f"    Sell Orders  : {len(sell_trades)}")
        if env.volume_history:
            print(f"    Avg Volume   : {sum(env.volume_history)/len(env.volume_history):.1f}")

        # Agent ranking.
        print(f"\n  Agent Rankings (by final wealth):")
        print(f"  {'Rank':>4}  {'Agent':<22}  {'Cash':>10}  {'Holdings':>10}  {'Wealth':>10}  {'Return':>8}")
        print(f"  {'-'*4}  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

        ranked = sorted(
            agents,
            key=lambda a: a.cash + a.holdings * env.price,
            reverse=True,
        )

        for rank, agent in enumerate(ranked, 1):
            wealth = agent.cash + agent.holdings * env.price
            # Use the exact initial wealth captured before trading began.
            # Estimating it from FINAL holdings at the initial price would
            # misstate returns (e.g. an agent that bought high and sold low
            # would look like it started with almost nothing).
            initial_wealth = self._initial_wealths.get(agent.agent_id)
            if initial_wealth is None:
                # Fallback for report() called without run().
                initial_wealth = agent.cash + agent.holdings * initial_price
            ret = (wealth / initial_wealth - 1) * 100 if initial_wealth > 0 else 0.0

            ret_color = Colors.GREEN if ret >= 0 else Colors.RED
            ret_str = colorize(f"{ret:>7.2f}%", ret_color)

            print(
                f"  {rank:4d}  "
                f"{agent.agent_id:<22}  "
                f"${agent.cash:>9.2f}  "
                f"{agent.holdings:>10d}  "
                f"${wealth:>9.2f}  "
                f"{ret_str}"
            )

        # Performance metrics
        print(f"\n  Performance Metrics:")
        print(f"  {'Agent':<22}  {'Sharpe':>8}  {'MaxDD':>8}  {'Volatility':>10}  {'WinRate':>8}")
        print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")

        for agent in ranked:
            m = self._compute_agent_metrics(agent.agent_id)
            sharpe_color = Colors.GREEN if m['sharpe'] > 0 else (Colors.RED if m['sharpe'] < 0 else "")
            dd_color = Colors.RED if m['max_drawdown'] > 0.1 else ""
            sharpe_str = colorize(f"{m['sharpe']:8.2f}", sharpe_color)
            dd_str = colorize(f"{m['max_drawdown']*100:7.2f}%", dd_color)
            print(
                f"  {agent.agent_id:<22}  "
                f"{sharpe_str}  "
                f"{dd_str}  "
                f"{m['volatility']*100:9.2f}%  "
                f"{m['win_rate']*100:7.1f}%"
            )

        print(f"\n{'='*60}\n")

        if generate_charts:
            self._generate_charts(chart_output_dir)

    def _generate_charts(self, output_dir: str = "charts") -> None:
        """Generate visualization charts after simulation completes."""
        from .visualization import generate_all_charts

        # Collect event history from the unified sandbox.
        event_history = None
        if self.env.event_manager:
            event_history = getattr(self.env.event_manager, "event_history", None)

        print(f"\n  Generating charts in '{output_dir}/' ...")
        try:
            generate_all_charts(
                price_history=self.env.price_history,
                state_history=self.state_history,
                event_history=event_history,
                initial_price=self.env.config.initial_price,
                output_dir=output_dir,
                initial_wealths=self._initial_wealths or None,
            )
        except Exception as e:
            print(f"  (Chart generation skipped: {e})")

    def export_csv(self, filepath: str) -> None:
        """Export trade history to a CSV file."""
        import csv

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "agent_id", "action", "quantity", "price", "cash_change"])
            for t in self.env.trade_history:
                writer.writerow([
                    t.step, t.agent_id, t.action,
                    t.quantity, f"{t.price:.2f}", f"{t.cash_change:.2f}",
                ])
        print(f"Trade history exported to: {filepath}")

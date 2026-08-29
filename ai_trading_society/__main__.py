"""
Entry point: python -m ai_trading_society

Runs simulations with AI agents that trade on personality and live market events.

The CLI reads the same configuration (user_config.json) that the web
homepage saves, so a simulation configured in the browser runs
identically from the terminal. Optional flags override individual fields.
"""

from typing import Optional

from .agents.roster import DEFAULT_AI_MODELS, build_agent_roster, resolve_social_map
from .config import MarketConfig, StockSpec
from .config_store import load_config
from .market_env import MarketEnv
from .simulator import Simulator

# Re-export AI_MODELS for backwards compatibility
AI_MODELS = DEFAULT_AI_MODELS


def run_society_mode(
    interactive: bool = False,
    seed: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    steps: Optional[int] = None,
):
    """Run a simulation: AI traders act on their personality plus market events."""
    print("\n" + "=" * 60)
    print("  AI TRADING SANDBOX")
    print("  AI traders act on personality and live market events")
    print("=" * 60)

    cfg = load_config()
    provider = provider or cfg.get("provider") or "openai"
    model = model or cfg.get("model") or "gpt-4o"
    trader_configs = cfg.get("traders") or None
    steps = steps or cfg.get("steps") or 5
    deep_persona = bool(cfg.get("deep_persona", False))

    # Parse multi-stock configuration (falls back to a single default stock).
    # Config entries use {"name", "price", "hold"} (see config_store); accept
    # "symbol" as an alias like the web UI does.
    raw_stocks = cfg.get("stocks") or []
    stock_specs: list = []
    if raw_stocks:
        for s in raw_stocks:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name") or s.get("symbol") or "").strip()
            if not name:
                continue
            stock_specs.append(StockSpec(
                name=name,
                initial_price=float(
                    s.get("price", s.get("initial_price", cfg.get("price", 100.0)))
                ),
                initial_holdings=int(s.get("hold", s.get("initial_holdings", cfg.get("hold", 0)))),
                sector=str(s.get("sector") or ""),
                blurb=str(s.get("blurb") or s.get("description") or ""),
            ))
    if not stock_specs:
        stock_specs = [StockSpec(
            name="Stock 1",
            initial_price=float(cfg.get("price", 100.0)),
            initial_holdings=int(cfg.get("hold", 0)),
        )]

    config = MarketConfig(
        initial_price=float(cfg.get("price", 100.0)),
        price_sensitivity=0.02,
        max_price_change_ratio=0.10,
        event_probability_multiplier=1.5,
        deep_persona=deep_persona,
        seed=seed,
        fee_rate=float(cfg.get("fee", 0.001)),
        slippage_rate=float(cfg.get("slip", 0.001)),
        social_influence=float(cfg.get("social_influence", 0.0) or 0.0),
        stocks=stock_specs,
    )

    agents, player_agent = build_agent_roster(
        provider=provider,
        model=model,
        api_key=api_key,
        trader_configs=trader_configs,
        cash=float(cfg.get("cash", 10000.0)),
        holdings=int(cfg.get("hold", 20)),
        stocks=stock_specs,
        include_player=cfg.get("player_participates", True) is not False,
        deep_persona=deep_persona,
        mood_max_step=config.mood_max_step,
        mood_intensity=config.mood_intensity,
    )
    env = MarketEnv(config, agents, seed=seed)
    # Resolve social relationships (idol/friends/enemies) so herding works in
    # CLI mode too — same setup as the web dashboard.
    env.social_map = resolve_social_map(list(env.agents.values()))
    env._social_influence = config.social_influence
    # Wire the player agent to the environment so its buffered orders are
    # read during order collection (same mechanism as the web dashboard).
    if player_agent is not None:
        player_agent._env = env
    sim = Simulator(env)

    sim.run(
        steps=steps,
        verbose=True,
        round_by_round=True,
        interactive=interactive,
        seed=seed,
        player_agent=player_agent,
    )
    sim.report(generate_charts=True, chart_output_dir="charts")


def main(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    steps: Optional[int] = None,
):
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  AI TRADING SANDBOX")
    print("  Multi-Agent Stock Market Sandbox")
    print("=" * 60)

    print("\nThe web dashboard lets the always-present player observe or intervene.")
    print()
    print("Config is shared with the homepage (user_config.json). Pass --provider, ")
    print("--model, or --api-key to override what was saved in the browser.")
    print()
    print("In interactive mode you can also trade as the player, inject market")
    print("events, tweak market parameters (God Mode), and inspect social ties.")
    print()

    # Ask whether to run in interactive (step-by-step) mode.
    interactive = False
    try:
        ans = input(
            "Interactive mode? Pause after each round? (y/n) [default: n]: "
        ).strip().lower()
        if ans in ("y", "yes"):
            interactive = True
    except (EOFError, KeyboardInterrupt):
        pass

    run_society_mode(
        interactive=interactive,
        provider=provider,
        model=model,
        api_key=api_key,
        steps=steps,
    )


if __name__ == "__main__":
    main()

<div align="center">

<img src="logo.png" alt="AI Trading Sandbox Logo" width="160">

# AI Trading Sandbox

A multi-agent stock market sandbox where LLM traders with distinct personalities
trade multiple stocks while market events fire.

<img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
<img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue">
<img alt="Version" src="https://img.shields.io/badge/version-0.2.0-blue">
<img alt="Language" src="https://img.shields.io/badge/language-Python-3776AB">

</div>

> [!WARNING]
> Simulation for studying AI behavior — **not a trading system, not financial advice**.

## Features

- **LLM traders** — OpenAI, Anthropic, Google Gemini, or any OpenAI-compatible endpoint.
- **Multiple stocks** — configure any number of stocks, each with its own price and order book.
- **Personality traits** — panic, greed, FOMO, stubbornness, loss aversion,
  overconfidence, regret avoidance; individually or as presets.
- **Human player** — join as a trader or spectate; trade from the web dashboard.
- **Market events** — 41 templates across 10 categories (global events hit every
  stock; company-specific ones hit one).
- **Agent memory** — short-term decision summaries plus long-term memory of key events.
- **Social influence** — idol/friends/enemies relationships drive herding and fading.
- **God Mode** — inject events and tune market parameters live.
- **Replay & reports** — per-round snapshots, event timeline replay, agent decision
  logs, one-click self-contained HTML report with shareable read-only link.

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

Open http://localhost:5000, click **Configure API connections** to add at least
one trader API key, optionally configure stocks, then launch.

### CLI

```bash
python run.py --cli                                  # interactive round-by-round run
python run.py --cli --provider groq --model groq/compound-mini --api-key sk-...
python -m ai_trading_society                         # equivalent to --cli
```

The CLI reads the same `user_config.json` saved by the web homepage. Each round
pauses with a menu: continue, player trade, inject events, tune parameters,
show social relations, help, stop.

### Keyboard shortcuts (web)

`Space` next round · `A` auto-play · `S` end · `C` chat · `N` social graph ·
`E` export report · `?` shortcuts · `Esc` close

## Configuration

Saved to `user_config.json`, shared by the web UI and CLI:

| Key | Meaning |
| --- | --- |
| `traders` | Per-trader provider / model / API key / personality. |
| `stocks` | `{name, price, hold}` list — multiple stocks supported. |
| `cash`, `hold`, `fee`, `slip` | Starting capital and transaction costs. |
| `social_influence` | Strength of herding behavior (0–1). |
| `player_participates` | Whether you join the market as a trader (default true). |

## Personality Presets

`balanced` · `aggressive` · `conservative` · `panicky` · `greedy` ·
`fomo_driven` · `stubborn` · `emotional` — assigned per trader in the web UI,
or via `create_personality_agent(base_agent, personality=...)`.

## As a Library

```python
from ai_trading_society import (
    MarketConfig, StockSpec, MarketEnv, Simulator,
    build_agent_roster,
)

config = MarketConfig(
    stocks=[StockSpec(name="AAPL", initial_price=150.0),
            StockSpec(name="TSLA", initial_price=250.0)],
    fee_rate=0.001, slippage_rate=0.001, seed=42,
)
agents, player = build_agent_roster(provider="openai", model="gpt-4o",
                                    api_key="sk-...", stocks=config.stocks)
env = MarketEnv(config, agents)
sim = Simulator(env)
sim.run(steps=30)
sim.report()                      # rankings, Sharpe, drawdown
sim.export_csv("trades.csv")
```

Every run saves a snapshot (metadata, config, state history) to `runs/<run_id>/`;
reload one with `load_run_snapshot()`.

## API Reference

| Symbol | Description |
| --- | --- |
| `MarketConfig(...)` | Market parameters: fees, slippage, events, social influence, seed. |
| `StockSpec(name, initial_price, initial_holdings)` | One stock. |
| `MarketEnv(config, agents, seed=None)` | Engine: observations, matching, pricing, player buffer. |
| `Simulator(env)` / `.run(steps, interactive=False)` | Run loop; returns state history. |
| `.report()` / `.export_csv(filepath)` | Final report / trade CSV. |
| `build_agent_roster(..., include_player=True)` | Build `(agents, player_agent)`. |
| `ExternalAIAgent(agent_id, api_provider, model, api_key)` | LLM-backed trader with short/long-term memory. |
| `TraitAgent(base_agent, panic=0, greed=0, ...)` | Personality wrapper. |
| `create_personality_agent(base, personality="balanced")` | Named preset. |
| `PlayerAgent(agent_id, cash, holdings)` | Human trader (absent when spectating). |
| `EventManager` / `EVENT_TEMPLATES` | 41 event templates across 10 categories. |
| `load_run_snapshot(run_id)` | Reload a saved run. |

## Project Structure

```
run.py                    # Web dashboard / CLI entry point
ai_trading_society/
├── market_env.py         # Engine: observations, matching, pricing
├── market_events.py      # Event system (41 templates)
├── simulator.py          # Run loop, reporting, CSV export
├── web/app.py            # Flask dashboard API
├── agents/
│   ├── external_ai_agent.py   # LLM trader (multi-provider, memory)
│   ├── traits.py              # Personality wrapper
│   ├── player_agent.py        # Human player
│   └── roster.py              # Roster factory + social map
└── ...
templates/                # Web UI pages
docs/user_facing_text.md  # All user-visible copy (editable)
```

## Contributing

Fork → feature branch → pull request. Code style is enforced by `ruff`; run
`pytest` before submitting.

## License

MIT License. See [LICENSE](./LICENSE).


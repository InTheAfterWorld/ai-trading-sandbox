# AI Trading Society

A multi-agent stock market sandbox. Several AI traders buy and sell one stock
while market events fire, personalities distort their decisions, and a human
player can step in and trade alongside them.

This is a simulation for studying AI behavior in market-like environments, not
a trading system.

## How it works

Every run is one unified sandbox:

- **ExternalAIAgent** — a real LLM trader (OpenAI, Anthropic, Gemini, or any
  OpenAI-compatible API). Each step the market sends it an observation and it
  replies with a JSON trading decision plus reasoning.
- **TraitAgent** — wraps an AI trader with a personality that bends its
  behavior: panic selling, greed, FOMO, stubbornness, loss aversion, and more.
- **PlayerAgent** — the human player, always present. Trade through the web
  dashboard or simply watch.
- **Market events** — 28 templates across 7 categories (earnings, analyst,
  macro, social, regulatory, black swans, technical) that shift price and
  sentiment.
- **Order matching** — buy and sell orders are matched proportionally; price
  moves with net buying pressure, with mean reversion, fees, and slippage.

## Quick start

### Web dashboard (recommended)

```bash
pip install -r requirements.txt
python run.py
```

Open http://localhost:5000, configure your traders on the homepage, then run
the simulation step by step. The dashboard includes God Mode (inject events,
tune market parameters) and a chat panel for talking to each agent in
character.

### CLI

```bash
python run.py --cli                     # interactive round-by-round terminal run
python run.py --cli --provider groq --model groq/compound-mini
python run.py --cli --api-key sk-...
python -m ai_trading_society
```

### In code

```python
from ai_trading_society import MarketConfig, MarketEnv, Simulator
from ai_trading_society.agents import build_agent_roster

config = MarketConfig(
    initial_price=100.0,
    fee_rate=0.001,
    slippage_rate=0.001,
    seed=42,
)
agents, player = build_agent_roster(
    provider="openai", model="gpt-4o", api_key="sk-...",
    cash=10000, holdings=20,
)

env = MarketEnv(config, agents, seed=config.seed)
sim = Simulator(env)
sim.run(steps=25)
sim.report(generate_charts=True)
```

## Configuration

`MarketConfig` (in `ai_trading_society/config.py`) controls the market:

| Field | Default | Meaning |
| --- | --- | --- |
| `initial_price` | 100.0 | Starting stock price |
| `price_sensitivity` | 0.02 | How strongly net buying pressure moves price |
| `max_price_change_ratio` | 0.10 | Max single-step price move |
| `fee_rate` / `slippage_rate` | 0.0 | Per-trade costs |
| `price_history_length` | 20 | Prices each agent can see |
| `event_probability_multiplier` | 1.5 | Event frequency scale (0 disables events) |
| `seed` | None | Random seed for reproducible runs |

The web homepage and CLI share one config file (`user_config.json`), so a
simulation configured in the browser runs identically from the terminal.

## Using external AI APIs

API keys are **never** read from environment variables or a `.env` file.
Configure each trader on the homepage under **Configure Traders** (provider,
model, API key, optional base URL), or pass `api_key` explicitly in code:

```python
from ai_trading_society.agents import ExternalAIAgent

agent = ExternalAIAgent(
    agent_id="GPT_Trader",
    api_provider="openai",   # openai / anthropic / google / openrouter / groq / ...
    model="gpt-4o",
    api_key="sk-...",        # always passed explicitly
)
```

Supported providers: OpenAI, Anthropic, Google Gemini (native SDKs) plus any
OpenAI-compatible endpoint (OpenRouter, Groq, ChatAnywhere, ...) via
`base_url`.

## Personality traits

Wrap any agent with a personality, or use a preset:

```python
from ai_trading_society.agents import TraitAgent, create_personality_agent

agent = TraitAgent(
    base_agent,
    panic=0.7,          # panic-sell on big drawdowns
    greed=0.6,          # hold winners too long
    fomo=0.5,           # chase strong momentum
    stubbornness=0.4,   # repeat the previous action
    loss_aversion=0.3,  # cut losses fast
)

agent = create_personality_agent(base_agent, personality="greedy")
# presets: balanced, aggressive, conservative, panicky, greedy,
#          fomo_driven, stubborn, emotional
```

## Visualization (optional)

```python
pip install -e ".[viz]"
```

```python
from ai_trading_society import generate_all_charts

generate_all_charts(
    price_history=env.price_history,
    state_history=sim.state_history,
    event_history=env.event_manager.event_history,
    initial_price=env.config.initial_price,
    output_dir="./charts",
)
```

Or simply `sim.report(generate_charts=True)`.

## Run snapshots

Every `Simulator.run()` seeds randomness and saves a snapshot (metadata, config,
state history, trade history) to `runs/<run_id>/`, so results are reproducible
and reviewable after the fact.

## Project structure

```
ai_trading_society/
├── __init__.py          # Public API
├── __main__.py          # CLI entry point
├── config.py            # MarketConfig
├── config_store.py      # Shared user_config.json (web + CLI)
├── base_agent.py        # Agent interface
├── market_env.py        # Market engine: observations, matching, pricing
├── market_events.py     # Event system (28 templates)
├── simulator.py         # Run loop, reporting, CSV export
├── run_metadata.py      # Seed handling and run snapshots
├── visualization.py     # Optional charts
├── console_utils.py     # Terminal output helpers
├── web/
│   └── app.py           # Flask dashboard
└── agents/
    ├── external_ai_agent.py   # LLM trader (multi-provider)
    ├── traits.py              # Personality wrapper
    ├── player_agent.py        # Human player
    └── roster.py              # Roster factory

run.py                    # Web dashboard / CLI entry point
templates/                # Web UI pages
```

## Installation

```bash
# Core library (no dependencies)
pip install -e .

# Extras
pip install -e ".[viz]"   # matplotlib charts
pip install -e ".[ai]"    # OpenAI / Anthropic / Gemini SDKs
pip install -e ".[dev]"   # pytest, ruff, mypy

# Everything including the web dashboard
pip install -r requirements.txt
```

## License

MIT License.
<div align="center">

<img src="logo.png" alt="AI Trading Sandbox Logo" width="160">

# AI Trading Sandbox

A multi-agent stock market sandbox where autonomous LLM traders with distinct personalities, memories, and social relationships interact in a synthetic economy—creating a controlled environment for studying emergent behavior.

<img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
<img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue">
<img alt="Version" src="https://img.shields.io/badge/version-0.3.0-blue">
<img alt="Language" src="https://img.shields.io/badge/language-Python-3776AB">

</div>

<p align="center">
  <img src="assets/demo.png" alt="AI Trading Sandbox demo" width="900">
</p>

> [!WARNING]
> Simulation for studying AI behavior — **not a trading system, not financial advice**.

## Table of Contents

- [What is this?](#what-is-this)
- [Features](#features)
- [The Traders](#the-traders)
- [How a Simulation Works](#how-a-simulation-works)
- [Emergent Behavior](#emergent-behavior)
- [Quick Start](#quick-start)
  - [CLI](#cli)
  - [Keyboard shortcuts (web)](#keyboard-shortcuts-web)
- [Configuration](#configuration)
- [Personality Presets](#personality-presets)
- [Deep Personality Mode](#deep-personality-mode)
- [Tokens, Cost & Prompt Versions](#tokens-cost--prompt-versions)
- [As a Library](#as-a-library)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## What is this?

Imagine putting different AI traders into the same artificial stock market, 
and let them compete, adapt, and shape the market on their own.

Each trader has its own personality, memory, portfolio, and relationships with
other traders. They all observe and trade in the same market, but can make completely
different decisions.

As traders buy and sell, they change the market itself. Events affect the
traders, traders affect prices, and those new prices influence the next round
of decisions.

The result is a controllable environment for exploring how complex behavior
can emerge from interactions between autonomous AI agents.

## Features

- **LLM traders** — Multiple LLM-powered traders compete in this market.
  A hardened prompt plus JSON structured-output mode (with automatic fallback)
  keeps every decision a valid `{"decisions": [...], ...}` object.
- **Multiple stocks** — configure any number of stocks, each with its own price,
  order book, **sector tag, and company blurb** fed to the agents' prompts.
- **Synthetic history** — each stock gets a seed-reproducible random walk
  pre-history so agents analyze real trends from the very first round.
- **Personality** — each trader gets a written character (8 presets) in its
  prompt. Nothing overrides what the model decides, so an agent's action
  always matches the reasoning shown next to it.
- **Deep personality mode** — an optional switch (off by default) that gives
  each trader tunable sensitivity dials and a mood that shifts with results
  (confidence / stress / frustration), and adds more context to the prompt:
  how the trader ranks against the others, how the floor feels, what rivals
  said, and which moves it was and wasn't positioned for.
- **Human player** — join as a trader or spectate; trade from the web dashboard.
- **Market events** — 41 templates across 10 categories (global events hit every
  stock; company-specific ones hit one).
- **Agent memory** — short-term decision summaries, long-term memory of key
  events, and (in deep mode) one-line lessons a trader writes for itself. The
  last `memory_window` rounds (default 3) are replayed as the trader's own
  replies plus a one-line recap of each round, so request size stays flat
  across a long run instead of compounding.
- **Learning feedback** — each agent sees its last round's fills and their
  price moves since execution to learn from its own outcomes.
- **Grading** — every agent gets a blended 0-100 score + S/A/B/C/D grade from
  return, Sharpe, drawdown, and win rate.
- **Token & cost accounting** — every LLM call is counted per agent, tagged by
  round, and priced. The running spend sits in the dashboard's top bar, so a
  long run is not a surprise on a billing page later.
- **Prompt versioning** — each run records the prompt generation and a content
  hash of the prompt every agent actually ran on, so an old report can never
  be silently reinterpreted against a newer prompt.
- **Mood timeline** — in deep mode, an agent's decision log traces all three
  mood axes (confidence / stress / frustration) round by round, next to the
  trades they accompanied.
- **Agents who know themselves in chat** — every chat message carries a fresh
  briefing: the trader's real character text, its idol / friends / enemies by
  name, its mood, standing, recent decisions, lessons and the live market. Chat
  is read-only — nothing said to a trader reaches its trading memory.
- **Social relationships** — idol / friends / enemies decide which other
  traders an agent pays attention to: their recent moves, and in deep mode
  their stated reasoning.
- **Import / export config** — download the current setup as a JSON file
  from the homepage, or upload one to load it.
- **God Mode** — inject events and tune market parameters live.
- **Replay & reports** — per-round snapshots, event timeline replay, agent decision
  logs, one-click self-contained HTML report with shareable link.

## The Traders

Each trader has

- Personality
    - a written character (one of 8 presets, or your own text)
    - in deep mode: sensitivity dials and a mood that changes over time

- Memory

    - recent decisions
    - important past events
    - previous outcomes

- Portfolio

    - cash
    - holdings
    - realized/unrealized performance

- Social relationships

    - idols
    - friends
    - enemies

- Market perception

    - current prices
    - trends
    - events
    - available information
 
Two traders can receive the same market information and still make completely different decisions.

## How a Simulation Works

Each round follows the same cycle:

```text
Market State
     ↓
Agent Observations
     ↓
Personality + Memory + Social Relationships
     ↓
AI Decisions
     ↓
Orders & Trades
     ↓
Price / Portfolio Updates
     ↓
Feedback & Memory
     ↓
Next Round
```

The AI's decision is final. Personality and mood only shape the prompt the
model sees — they never rewrite the trade it chooses.

## Emergent Behavior

The sandbox does not just tell each agent what to do. Instead,
interesting behaviors can emerge from interactions between agents and the market.

Examples include:

- **Herding** — traders converge on the same decision.
- **Panic selling** — negative events trigger waves of selling.
- **FOMO** — rising prices attract increasingly aggressive buyers.
- **Contrarian behavior** — traders move against the crowd.
- **Social cascades** — one trader's behavior influences others.
- **Overreaction** — agents respond disproportionately to new information.
- **Behavioral persistence** — previous experiences affect future decisions.

None of these are scripted — they come from each trader's own judgment, not
from rules in the sandbox. These behaviors are to be explored.


## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

Open http://localhost:5000, click **Configure API connections** to add at least
one trader API key, optionally configure stocks, then launch.

> [!NOTE]
> The dashboard has no login and hands your saved API keys to anyone who can
> reach it. `run.py` listens on all network interfaces and reads a `PORT`
> variable, so keep it on your own machine or a trusted network.

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
| `traders` | Per-trader provider / model / API key / personality, plus optional `persona` text, `trait_notes`, and `dials` (used in deep mode). |
| `stocks` | `{name, price, hold, sector?, blurb?}` list — multiple stocks supported. |
| `cash`, `hold`, `fee`, `slip` | Starting capital and transaction costs. |
| `deep_persona` | Turn on deep personality mode: dials, mood, richer prompts (default false). |
| `social_influence` | Kept for older configs; no longer changes behavior. |
| `player_participates` | Whether you join the market as a trader (default true). |
| `parallel_agents` | Concurrent per-round agent actions (default true). |

In `MarketConfig`, deep mode has two tuning knobs: `mood_max_step` (how far a
mood can move in one round, default 3) and `mood_intensity` (how strongly the
mood events react, default 1). The synthetic pre-history length is
`history_backfill_steps` (default 30; set 0 to disable). Stock `sector`/`blurb` strings go straight into
each agent's observation and prompt. Set `parallel_agents=False` to collect
agent decisions strictly one-by-one.

## Personality Presets

`balanced` · `aggressive` · `conservative` · `panicky` · `greedy` ·
`fomo_driven` · `stubborn` · `emotional` — assigned per trader in the web UI,
or via `create_personality_agent(base_agent, personality=..., deep=False)`.
Pass `deep=True` for the mood + dials layer.

## Deep Personality Mode

By default each trader carries a one-line character in its prompt plus a lean
set of context. This is fast and cheap, and every trade still follows the
model's own stated reasoning.

Tick **Deep personality simulation** on the homepage (or pass
`deep_persona=True`) and each trader also gets:

- **Seven sensitivity dials** (0–10): `risk_appetite`, `loss_sensitivity`,
  `herd_pull`, `patience`, `resilience`, `envy`, `conviction`. Each preset
  ships a profile; adjust them per trader with the sliders.
- **A mood** — `confidence`, `stress`, `frustration` (0–10) — that starts from
  the preset and shifts each round on plain events (gains, losses, streaks,
  volatility, rank moves, rivals), then drifts back toward the baseline. The
  model still reports its own mood, but only as a small bounded adjustment
  around the rules.
- **Custom character text**: `trait_notes` is added to the preset, `persona`
  replaces it.
- **More context in the prompt**: where the trader ranks, how the rest of the
  floor behaved, what rivals said, and which moves it did and didn't catch.
- **Self-written notes**: the model may keep a short lesson and set stop-loss /
  target prices. These come back in later prompts as reminders — nothing acts
  on them automatically.

Mood is model-reported, so deep runs are not bit-for-bit reproducible, and the
longer prompts cost more tokens. Pair it with a fast model.

## Tokens, Cost & Prompt Versions

Every provider call an agent makes is recorded: the round it belonged to,
whether it was a decision or a repair re-ask, the tokens it moved, and what it
cost. The dashboard shows the running total in the top bar and the per-agent
figure in each trader's 📋 Timeline; `sim.report()` prints a per-agent table at
the end of a CLI run, and `GET /api/usage` returns the full breakdown including
a per-round cost curve.

Prices live in `ai_trading_society/model_prices.json`, in USD per 1M tokens.
**The table is user-maintained**: it ships only the rows this project could
verify against a first-party source, so most models start unpriced. An unpriced
model is still counted in full — its tokens are exact — but its cost is
reported as unknown rather than as zero, and any total that includes one is
marked as a lower bound (`$1.23+`). Add your own rows to that file, or point
`ATS_MODEL_PRICES` at your own:

```bash
ATS_MODEL_PRICES=/path/to/my_prices.json python run.py
```

A provider that returns no usage block at all falls back to a character-based
token estimate, flagged as an estimate wherever it appears.

Alongside the cost, each run records **which prompt produced it**:
`prompt_template_version` (bumped by hand when the shipped prompt changes in a
way that should invalidate comparison with older runs) and a fingerprint of the
text each agent actually sent. Both land in `runs/<run_id>/metadata.json` and in
the exported HTML report's header. A per-agent `source` distinguishes the
shipped template, a persona-prefixed one, and a fully custom prompt.

## As a Library

```python
from ai_trading_society import (
    MarketConfig, StockSpec, MarketEnv, Simulator,
    build_agent_roster,
)

config = MarketConfig(
    stocks=[
        StockSpec(name="TechTitan", initial_price=150.0, sector="AI chips",
                  blurb="High-growth chip designer"),
        StockSpec(name="MegaBank", initial_price=250.0, sector="Banking",
                  blurb="Defensive large-cap bank"),
    ],
    fee_rate=0.001, slippage_rate=0.001, history_backfill_steps=30, seed=42,
)
agents, player = build_agent_roster(provider="openai", model="gpt-4o",
                                    api_key="sk-...", stocks=config.stocks)
env = MarketEnv(config, agents)
sim = Simulator(env)
sim.run(steps=30)
sim.report()                      # rankings, Sharpe, drawdown
sim.export_csv("trades.csv")
```

Pass `deep_persona=True` to both `MarketConfig` and `build_agent_roster` for
the full mood + dials layer.

Every run saves a snapshot (metadata, config, state history) to `runs/<run_id>/`;
reload one with `load_run_snapshot()`.

## API Reference

| Symbol | Description |
| --- | --- |
| `MarketConfig(...)` | Market parameters: fees, slippage, events, `deep_persona` and mood knobs, seed. |
| `StockSpec(name, initial_price, initial_holdings, sector="", blurb="")` | One stock; optional sector tag & company blurb. |
| `MarketEnv(config, agents, seed=None)` | Engine: observations, matching, pricing, player buffer, last-round feedback. |
| `Simulator(env)` / `.run(steps, interactive=False)` | Run loop; returns state history. |
| `.report()` / `.export_csv(filepath)` | Final report / trade CSV. |
| `build_agent_roster(..., include_player=True)` | Build `(agents, player_agent)`. |
| `ExternalAIAgent(agent_id, api_provider, model, api_key)` | LLM-backed trader with short/long-term memory. |
| `TraitAgent(base_agent, personality_name="custom", deep=False, ...)` | Personality + mood wrapper; passes the base agent's decision through untouched. |
| `create_personality_agent(base, personality="balanced", deep=False)` | Named preset; `deep=True` adds mood + dials. |
| `PlayerAgent(agent_id, cash, holdings)` | Human trader (absent when spectating). |
| `EventManager` / `EVENT_TEMPLATES` | 41 event templates across 10 categories. |
| `load_run_snapshot(run_id)` | Reload a saved run. |
| `UsageTracker` / `collect_usage(agents)` | Per-agent token & cost accounting; aggregate a roster. |
| `model_price(model)` / `compute_cost(model, in, out)` | Price lookup; `None` means unpriced, never free. |
| `PROMPT_TEMPLATE_VERSION` / `describe_prompt(agent)` | Prompt generation and the fingerprint of the prompt in use. |
| `build_chat_system_prompt(env, agent_id)` | The read-only background briefing an agent carries into a chat. |
| `evaluate_wealth_curve(wealths)` | Sharpe / max-drawdown / volatility / win-rate from a wealth curve. |
| `grade_performance(ret, sharpe, drawdown, win_rate)` | Blend into a 0-100 score + S/A/B/C/D grade. |
| `grade_wealth_curve(wealths, initial_wealth)` | Metrics + score + grade in one call. |

## Project Structure

```
run.py                    # Web dashboard / CLI entry point
ai_trading_society/
├── market_env.py         # Engine: observations, matching, pricing
├── market_events.py      # Event system (41 templates)
├── simulator.py          # Run loop, reporting, performance grading, CSV export
├── usage.py              # Token & cost accounting per agent
├── model_prices.json     # USD per 1M tokens (user-maintained)
├── prompt_version.py     # Prompt generation + fingerprints
├── chat_context.py       # Background briefing sent when chatting
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

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for
the development setup, code quality gates (ruff / mypy / pytest), and commit
message conventions. By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md). Security vulnerabilities are handled
privately — see [SECURITY.md](SECURITY.md).

## License

MIT License. See [LICENSE](./LICENSE).


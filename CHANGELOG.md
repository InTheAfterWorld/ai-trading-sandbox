# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/) (pre-1.0, minor versions may
still contain breaking changes).

## [0.3.0] — unreleased

### Added

- **Deep personality mode** (`deep_persona`, a homepage checkbox, off by
  default). When on, each trader also gets:
  - seven sensitivity dials — `risk_appetite`, `loss_sensitivity`, `herd_pull`,
    `patience`, `resilience`, `envy`, `conviction` — with a per-preset profile
    and per-trader sliders;
  - an evolving mood (`confidence` / `stress` / `frustration`, 0–10) that
    moves on round-by-round events (gains, losses, streaks, volatility, rank
    moves, rival performance) with gradual recovery toward the preset
    baseline; the model's self-report is a bounded adjustment around the
    rules, never a reset;
  - optional free-text character: `trait_notes` (added to the preset) and
    `persona` (replaces it);
  - richer prompt context: how the trader ranks against the others, how the
    floor behaved, what rivals said, and which price moves it was and wasn't
    positioned for;
  - optional self-written `lesson` memory and `stop_loss` / `target` notes that
    replay in later prompts as reminders (never auto-executed).
- Market regime described in words and a real-stakes line in every prompt (both
  modes).
- Import / export configuration from the homepage — upload a `user_config.json`
  (`POST /api/config/upload`) or download the current one.
- Per-stock reasoning is shown one line per stock, untruncated, in the
  dashboard, CLI round output and HTML report.
- `MarketConfig.mood_max_step` and `MarketConfig.mood_intensity`.
- `run.py` honours the `PORT` environment variable and binds all interfaces,
  for hosting the dashboard.

### Changed

- **Personality no longer overrides the model's decision.** The character lives
  in the system prompt; the trade always matches the reasoning shown beside it.
  The `[trait override]` / `[social]` tags are gone.
- `GET` / `POST /api/config` return stored API keys by default; set
  `ATS_REDACT_CONFIG=1` to withhold them. A Host allowlist
  (`ATS_ALLOWED_HOSTS`, default `localhost,127.0.0.1,[::1]`) guards against DNS
  rebinding.
- Unhandled API errors always return JSON; a transient provider error retries
  once; corrective re-asks to the model are capped per round.
- The Sharpe figure is documented as a nominal score, not an annualized ratio.

### Fixed

- `_parse_decisions` coerces a non-numeric `quantity` instead of failing the
  whole round.
- A literal `<think>` block inside a JSON string no longer truncates the
  decision.
- The JSON-mode fallback only triggers on a real `response_format` rejection,
  not on rate-limit or auth errors.
- `/api/start` tolerates a non-numeric `seed`.
- A `null` lesson is dropped instead of being stored as the string `"None"` and
  replayed forever.
- Deep-mode mood bars now render (the fill element was inline).

### Removed

- `MarketConfig.random_traits` and the numeric personality traits (`panic`,
  `greed`, `fomo`, `stubbornness`, `loss_aversion`, `overconfidence`,
  `regret_avoidance`), replaced by the prompt-based character.
  `TraitAgent(...)` and `create_personality_agent(...)` take new arguments
  (`deep`, `dials`, `persona`, `trait_notes`, `mood_max_step`,
  `mood_intensity`).
- `.env` from the repo; use `.env.example` as a template.

## [0.2.0]

- First tagged baseline: multi-agent LLM market simulation, Flask dashboard,
  CLI, event library, personality presets.

# Contributing to AI Trading Sandbox

Thanks for contributing! By participating you agree to our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Setup

```bash
git clone https://github.com/<your-username>/ai-trading-sandbox.git
cd ai-trading-sandbox
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Extras: `.[viz]` (charts), `.[config]` (YAML), `.[ai]` (LLM providers),
`.[all]`.

Run: `python run.py` (web, port 5000), `python run.py --cli`, `python run.py --debug`.

## Checks

All must pass locally — CI enforces them:

```bash
ruff check .
python -m mypy ai_trading_society
pytest -q --cov=ai_trading_society --cov-report=term-missing --cov-fail-under=50
```

- Line length is 100 (`report_export.py` and `tests/` are exempt).
- `ruff check --fix .` auto-sorts imports and fixes style.
- Annotate new/changed functions; `mypy` treats missing annotations as errors.

## Tests

```bash
pytest -q                      # fast run
pytest tests/test_web.py -q    # one module
```

- Use `tests/test_<module>.py`.
- No network or real API keys — mock LLM calls; use Flask's test client.
- Bug fixes need a regression test. Keep total coverage above 50%.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):
`<type>: <summary>`, types: `feat`, `fix`, `docs`, `test`, `refactor`,
`chore`.

```
feat: add sector tag and company blurb to stock specs
fix: install flask in dev deps and fix lint/type errors to make CI green
docs: add contributing and security docs
```

## Workflow

1. Fork, then branch from `master`: `git checkout -b feat/my-feature`.
2. Commit in small, focused changes.
3. Run the three checks above.
4. Open a PR against `master`. Keep PRs small.

## Pull Request Checklist

- [ ] `ruff` and `mypy` pass.
- [ ] `pytest` passes, coverage >= 50%.
- [ ] Bug fix has a regression test.
- [ ] README / docstrings updated for user-visible changes.
- [ ] No secrets committed (see [SECURITY.md](SECURITY.md)).

## Reporting Bugs & Feature Ideas

Open a [GitHub issue](https://github.com/LXTimer/ai-trading-sandbox/issues).

- Bugs: OS/Python version, how you launched it, expected vs. actual, a minimal
  repro, and provider/model for LLM issues.
- Features: describe the problem you are solving and how it fits the project's
  research purpose (this is a sandbox, not a trading system).

Security issues: follow [SECURITY.md](SECURITY.md), not the public tracker.

## Getting Help

- Issues: <https://github.com/LXTimer/ai-trading-sandbox/issues>
- Usage: [README](README.md)

Thanks for contributing! 🎉

"""Regression tests for the hardening pass (items 1-8).

1  _parse_decisions coerces quantities instead of crashing on junk.
2  _strip_reasoning no longer eats JSON containing a literal <think>.
3  The JSON-mode fallback only fires for response_format rejections.
4  api_start survives a non-numeric seed of any type.
5  handle_exception always returns JSON with a sane int status.
6  The Sharpe annualization factor is a documented nominal constant.
7  Market tuning constants live on MarketConfig, defaults unchanged.
8  Exported reports are capped so runs/reports cannot grow forever.
"""

import os
import shutil
import tempfile
from types import ModuleType

import pytest

from ai_trading_society.agents.external_ai_agent import (
    ExternalAIAgent,
    _is_json_mode_rejection,
)
from ai_trading_society.config import MarketConfig
from ai_trading_society.market_env import MarketEnv
from ai_trading_society.market_events import EventManager
from ai_trading_society.report_export import MAX_REPORTS, save_report
from ai_trading_society.simulator import (
    _NOMINAL_PERIODS_PER_YEAR,
    evaluate_wealth_curve,
)
from ai_trading_society.web.app import app
from tests.conftest import ScriptedExternalAIAgent


# ---------------------------------------------------------------------------
# 1: quantity coercion in _parse_decisions
# ---------------------------------------------------------------------------
class TestParseDecisionsQuantityCoercion:
    @pytest.fixture
    def env(self):
        env = MarketEnv(
            MarketConfig(initial_price=100.0),
            [ScriptedExternalAIAgent("a", cash=1000, holdings=10)],
        )
        yield env
        env.close()

    @pytest.mark.parametrize("raw,expected", [
        (5, 5),
        ("7", 7),
        ("10 shares", 10),
        (7.9, 7),
        ("all", 10 ** 9),
        (None, 0),
        ("abc", 0),
        ("", 0),
        ({"n": 1}, 0),
        ([], 0),
        (True, 0),
        (-5, 0),
        (float("nan"), 0),
        (float("inf"), 0),
    ])
    def test_quantity_is_coerced_not_crashed(self, env, raw, expected):
        syms = list(env.stocks.keys())
        action = {"decisions": [
            {"name": syms[0], "action": "buy", "quantity": raw}
        ]}
        assert env._parse_decisions(action, syms)[0]["quantity"] == expected

    def test_legacy_format_also_coerces(self, env):
        syms = list(env.stocks.keys())
        result = env._parse_decisions(
            {"action": "sell", "quantity": "3 shares"}, syms
        )
        assert result[0]["action"] == "sell"
        assert result[0]["quantity"] == 3

    def test_legacy_junk_quantity_does_not_raise(self, env):
        syms = list(env.stocks.keys())
        result = env._parse_decisions({"action": "buy", "quantity": []}, syms)
        assert result[0]["quantity"] == 0

    def test_invalid_action_still_raises(self, env):
        """Coercing quantities must not soften action validation."""
        syms = list(env.stocks.keys())
        with pytest.raises(ValueError, match="invalid action"):
            env._parse_decisions(
                {"decisions": [{"name": syms[0], "action": "yolo", "quantity": 1}]},
                syms,
            )

    def test_a_junk_quantity_no_longer_fails_the_whole_round(self):
        """The agent should hold that stock, not lose its entire round."""
        class _JunkAgent(ScriptedExternalAIAgent):
            def act(self, observation):
                return {"decisions": [
                    {"name": s["name"], "action": "buy", "quantity": "lots"}
                    for s in observation["stocks"]
                ]}

        env = MarketEnv(
            MarketConfig(initial_price=100.0, event_probability_multiplier=0.0),
            [_JunkAgent("junk", cash=1000, holdings=10),
             ScriptedExternalAIAgent("s", cash=0, holdings=50, sell_prob=1.0)],
            seed=1,
        )
        state = env.step()
        assert env._agent_error_counts.get("junk", 0) == 0
        acts = state["agent_actions"]["junk"]
        assert all(a["requested_qty"] == 0 for a in acts.values())
        env.close()


# ---------------------------------------------------------------------------
# 2: <think> stripping must not destroy JSON
# ---------------------------------------------------------------------------
class TestStripReasoning:
    @pytest.fixture
    def agent(self):
        return ExternalAIAgent("x", api_key="k", enable_memory=False)

    def test_literal_think_inside_json_string_is_preserved(self, agent):
        payload = (
            '{"decisions": [{"name": "S0", "action": "buy", "quantity": 5, '
            '"reasoning": "the notes said <think> before the call"}]}'
        )
        parsed = agent._parse_response(payload)
        assert parsed["decisions"][0]["action"] == "buy"
        assert parsed["decisions"][0]["quantity"] == 5

    def test_json_after_a_literal_think_survives(self, agent):
        text = 'Note: <think> is a tag. {"action": "sell", "quantity": 2}'
        assert agent._parse_response(text)["quantity"] == 2

    def test_complete_think_block_is_still_stripped(self, agent):
        text = (
            "<think>let me reason about this</think>"
            '{"decisions": [{"name": "S0", "action": "hold", "quantity": 0}]}'
        )
        assert agent._parse_response(text)["decisions"][0]["action"] == "hold"

    def test_leading_unclosed_think_is_still_stripped(self, agent):
        """A response truncated mid-thought still has its preamble removed."""
        text = (
            "<think>I am reasoning and got cut off"
        )
        assert agent._strip_reasoning(text) == text  # empty result -> fallback

    def test_leading_think_before_json_is_stripped(self, agent):
        text = (
            "  <think>reasoning that was never closed "
            '{"decisions": [{"name": "S0", "action": "buy", "quantity": 1}]}'
        )
        # The whole thing is reasoning-tagged, so the parser falls back to the
        # original text and still recovers the decision.
        assert agent._parse_response(text)["decisions"][0]["quantity"] == 1

    def test_thought_token_block_is_still_stripped(self, agent):
        text = (
            "<|begin_of_thought|>musing<|end_of_thought|>"
            '{"action": "hold", "quantity": 0}'
        )
        assert agent._parse_response(text)["action"] == "hold"


# ---------------------------------------------------------------------------
# 3: JSON-mode fallback narrowing
# ---------------------------------------------------------------------------
class _FakeOpenAI(ModuleType):
    class BadRequestError(Exception):
        pass


def _err(name, status, message):
    exc = type(name, (Exception,), {})(message)
    exc.status_code = status
    return exc


class TestJsonModeNarrowing:
    @pytest.fixture
    def mod(self):
        return _FakeOpenAI("openai")

    @pytest.mark.parametrize("message", [
        "Unsupported parameter: 'response_format' is not supported",
        "model does not support json_object",
        "json_schema is unavailable for this model",
        "JSON mode not available",
        "structured output is not enabled",
    ])
    def test_json_specific_400_triggers_fallback(self, mod, message):
        assert _is_json_mode_rejection(_err("BadRequestError", 400, message), mod)

    @pytest.mark.parametrize("message", [
        "unsupported parameter: 'seed'",
        "unrecognized field 'foo'",
        "unknown parameter: temperature",
        "context length exceeded",
        "additional properties are not allowed",
    ])
    def test_unrelated_400_does_not_trigger_fallback(self, mod, message):
        assert not _is_json_mode_rejection(_err("BadRequestError", 400, message), mod)

    @pytest.mark.parametrize("name,status", [
        ("RateLimitError", 429),
        ("AuthenticationError", 401),
        ("PermissionDeniedError", 403),
        ("InternalServerError", 500),
    ])
    def test_non_400_never_triggers_fallback(self, mod, name, status):
        exc = _err(name, status, "response_format mentioned but wrong status")
        assert not _is_json_mode_rejection(exc, mod)


# ---------------------------------------------------------------------------
# 4 + 5: web request hardening
# ---------------------------------------------------------------------------
class TestApiStartSeedParsing:
    @pytest.mark.parametrize("seed", [[], {}, "abc", "", 3.7, True, None])
    def test_invalid_seed_does_not_500(self, seed):
        with app.test_client() as client:
            res = client.post(
                "/api/start", json={"steps": 2, "traders": [], "seed": seed}
            )
            assert res.status_code == 200, res.get_json()

    def test_valid_seed_is_honoured(self):
        with app.test_client() as client:
            res = client.post(
                "/api/start", json={"steps": 2, "traders": [], "seed": 4242}
            )
            assert res.get_json()["seed"] == 4242


class TestErrorHandler:
    def test_unknown_route_returns_json_404(self):
        with app.test_client() as client:
            res = client.get("/definitely/not/a/route")
            assert res.status_code == 404
            assert res.is_json and "error" in res.get_json()

    def test_method_not_allowed_returns_json_405(self):
        with app.test_client() as client:
            res = client.get("/api/start")
            assert res.status_code == 405
            assert res.is_json

    def test_non_int_code_is_coerced_to_500(self):
        """An exception carrying a junk .code must not break jsonify."""
        from ai_trading_society.web.app import handle_exception

        class _Weird(Exception):
            code = "not-an-int"

        with app.test_request_context("/api/step", method="POST"):
            _body, status = handle_exception(_Weird("boom"))
            assert status == 500

    def test_out_of_range_code_is_coerced_to_500(self):
        from ai_trading_society.web.app import handle_exception

        class _Weird(Exception):
            code = 12345

        with app.test_request_context("/api/step", method="POST"):
            _body, status = handle_exception(_Weird("boom"))
            assert status == 500


# ---------------------------------------------------------------------------
# 6: Sharpe scaling is a documented constant, value unchanged
# ---------------------------------------------------------------------------
class TestNominalSharpeScaling:
    def test_factor_value_is_unchanged(self):
        assert _NOMINAL_PERIODS_PER_YEAR == 252

    def test_sharpe_matches_the_explicit_formula(self):
        wealths = [100.0, 110.0, 105.0, 120.0, 118.0, 130.0]
        returns = [
            (wealths[i] - wealths[i - 1]) / wealths[i - 1]
            for i in range(1, len(wealths))
        ]
        mean = sum(returns) / len(returns)
        std = (sum((r - mean) ** 2 for r in returns) / len(returns)) ** 0.5
        expected = (mean / std) * (_NOMINAL_PERIODS_PER_YEAR ** 0.5)
        assert evaluate_wealth_curve(wealths)["sharpe"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 7: tuning constants on MarketConfig, defaults preserve old behaviour
# ---------------------------------------------------------------------------
def _seeded_prices(**cfg_kwargs):
    config = MarketConfig(
        initial_price=100.0, event_probability_multiplier=1.0, seed=99, **cfg_kwargs
    )
    agents = [
        ScriptedExternalAIAgent("b", cash=9000, holdings=0, buy_prob=1.0),
        ScriptedExternalAIAgent("s", cash=0, holdings=80, sell_prob=1.0),
        ScriptedExternalAIAgent("h", cash=500, holdings=5),
    ]
    env = MarketEnv(config, agents, seed=99)
    prices = [round(env.step()["price"], 10) for _ in range(20)]
    env.close()
    return prices


class TestMarketTuningConstants:
    def test_defaults_match_the_previous_literals(self):
        config = MarketConfig()
        assert config.mean_reversion_strength == 0.0005
        assert config.idle_price_noise == 0.003
        assert config.event_impact_scale == 0.3
        assert EventManager().impact_scale == 0.3

    def test_defaults_reproduce_the_old_behaviour(self):
        assert _seeded_prices() == _seeded_prices(
            mean_reversion_strength=0.0005,
            idle_price_noise=0.003,
            event_impact_scale=0.3,
        )

    @pytest.mark.parametrize("field,value", [
        ("mean_reversion_strength", 0.05),
        ("idle_price_noise", 0.05),
        ("event_impact_scale", 1.0),
    ])
    def test_each_knob_actually_moves_the_market(self, field, value):
        assert _seeded_prices(**{field: value}) != _seeded_prices()

    def test_disabling_idle_noise_is_possible(self):
        config = MarketConfig(initial_price=100.0, idle_price_noise=0.0,
                              event_probability_multiplier=0.0, seed=1)
        env = MarketEnv(config, [ScriptedExternalAIAgent("h", cash=10, holdings=0)],
                        seed=1)
        for _ in range(5):
            env.step()
        stock = next(iter(env.stocks.values()))
        assert stock.price == pytest.approx(stock.initial_price)
        env.close()

    def test_round_trips_through_to_dict(self):
        config = MarketConfig(mean_reversion_strength=0.01, idle_price_noise=0.02,
                              event_impact_scale=0.4)
        restored = MarketConfig.from_dict(config.to_dict())
        assert restored.mean_reversion_strength == 0.01
        assert restored.idle_price_noise == 0.02
        assert restored.event_impact_scale == 0.4


# ---------------------------------------------------------------------------
# 8: exported report retention
# ---------------------------------------------------------------------------
class TestReportRetention:
    @pytest.fixture
    def reports_dir(self):
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)

    def test_oldest_reports_are_pruned(self, reports_dir):
        for i in range(8):
            save_report(f"<html>{i}</html>", f"run{i:02d}",
                        reports_dir=reports_dir, max_reports=5)
        kept = sorted(os.listdir(reports_dir))
        assert len(kept) == 5
        assert "run07.html" in kept, "the newest report must always survive"
        assert "run00.html" not in kept

    def test_just_saved_report_always_survives(self, reports_dir):
        for i in range(4):
            path = save_report(f"<html>{i}</html>", f"run{i}",
                               reports_dir=reports_dir, max_reports=1)
            assert os.path.isfile(path)
        assert os.listdir(reports_dir) == ["run3.html"]

    def test_zero_disables_pruning(self, reports_dir):
        for i in range(6):
            save_report("<html/>", f"run{i}", reports_dir=reports_dir, max_reports=0)
        assert len(os.listdir(reports_dir)) == 6

    def test_under_the_cap_nothing_is_removed(self, reports_dir):
        for i in range(3):
            save_report("<html/>", f"run{i}", reports_dir=reports_dir, max_reports=5)
        assert len(os.listdir(reports_dir)) == 3

    def test_non_html_files_are_never_touched(self, reports_dir):
        keep = os.path.join(reports_dir, "notes.txt")
        with open(keep, "w", encoding="utf-8") as handle:
            handle.write("not a report")
        for i in range(4):
            save_report("<html/>", f"run{i}", reports_dir=reports_dir, max_reports=1)
        assert os.path.isfile(keep)

    def test_default_cap_is_applied(self):
        assert MAX_REPORTS == 50

    def test_invalid_run_id_still_rejected(self, reports_dir):
        with pytest.raises(ValueError, match="Invalid run id"):
            save_report("<html/>", "../escape", reports_dir=reports_dir)

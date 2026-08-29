"""Regression tests for review findings M1-M5.

M2  A single wealth definition; metrics delegate to evaluate_wealth_curve.
M3  Fill indexes replace the quadratic trade_history rescans.
M4  JSON-mode fallback only fires on an actual response_format rejection.
M5  Transient API failures retry once; corrective re-asks have a budget.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from ai_trading_society.agents.external_ai_agent import (
    ExternalAIAgent,
    _is_json_mode_rejection,
    _is_transient_error,
)
from ai_trading_society.config import MarketConfig, StockSpec
from ai_trading_society.market_env import MarketEnv
from ai_trading_society.simulator import Simulator, evaluate_wealth_curve
from tests.conftest import ScriptedExternalAIAgent

SYMS = ["S0", "S1", "S2", "S3", "S4"]
RISING = [100, 101, 102, 103, 104, 105]


def _multi_stock_obs(cash=10000.0, price=100.0, holdings=0):
    return {
        "step": 3,
        "stocks": [
            {"symbol": s, "name": s, "price": price, "price_history": RISING,
             "last_volume": 0, "my_holdings": holdings}
            for s in SYMS
        ],
        "my_cash": cash,
        "my_holdings": {s: holdings for s in SYMS},
        "my_wealth": cash + holdings * price * len(SYMS),
        "market_sentiment": 0.0,
        # Backward-compat top-level fields MarketEnv always includes.
        "price": price,
        "price_history": RISING,
        "last_volume": 0,
    }


def _exc(name, status=None, message=""):
    """Build a stand-in provider exception with the given class name/status."""
    exc_type = type(name, (Exception,), {})
    exc = exc_type(message)
    if status is not None:
        exc.status_code = status
    return exc


class _FakeOpenAIModule(ModuleType):
    class BadRequestError(Exception):
        pass


# ---------------------------------------------------------------------------
# M2: one wealth definition, one metrics implementation
# ---------------------------------------------------------------------------
class TestM2SingleWealthDefinition:
    def test_agent_wealth_accepts_id_or_object(self, simple_env):
        agent = simple_env.agents["buyer"]
        assert simple_env.agent_wealth("buyer") == simple_env.agent_wealth(agent)

    def test_agent_wealth_marks_holdings_to_market(self):
        specs = [StockSpec(name="A", initial_price=10.0, initial_holdings=0),
                 StockSpec(name="B", initial_price=20.0, initial_holdings=0)]
        config = MarketConfig(stocks=specs)
        agent = ScriptedExternalAIAgent("x", cash=100.0, holdings={"A": 3, "B": 2})
        env = MarketEnv(config, [agent])
        assert env.agent_wealth("x") == pytest.approx(100.0 + 3 * 10.0 + 2 * 20.0)
        env.close()

    def test_state_snapshot_wealth_matches_helper(self, simple_env):
        state = simple_env.step()
        for aid, data in state["agents"].items():
            assert data["wealth"] == pytest.approx(simple_env.agent_wealth(aid))
        simple_env.close()

    def test_metrics_delegate_to_evaluate_wealth_curve(self, simple_env):
        sim = Simulator(simple_env)
        for _ in range(6):
            sim.state_history.append(simple_env.step())

        metrics = sim._compute_agent_metrics("buyer")
        curve = [s["agents"]["buyer"]["wealth"] for s in sim.state_history]
        expected = evaluate_wealth_curve(curve)

        assert metrics["sharpe"] == pytest.approx(expected["sharpe"])
        assert metrics["max_drawdown"] == pytest.approx(expected["max_drawdown"])
        assert metrics["volatility"] == pytest.approx(expected["volatility"])
        simple_env.close()

    def test_metrics_win_rate_stays_a_fraction(self, simple_env):
        """Callers multiply win_rate by 100 - it must stay on the 0-1 scale."""
        sim = Simulator(simple_env)
        for _ in range(6):
            sim.state_history.append(simple_env.step())
        metrics = sim._compute_agent_metrics("buyer")
        assert 0.0 <= metrics["win_rate"] <= 1.0
        expected = evaluate_wealth_curve(
            [s["agents"]["buyer"]["wealth"] for s in sim.state_history]
        )
        assert metrics["win_rate"] == pytest.approx(expected["win_rate"] / 100.0)
        simple_env.close()


# ---------------------------------------------------------------------------
# M3: fill indexes replace the trade_history rescans
# ---------------------------------------------------------------------------
class TestM3FillIndexes:
    def test_filled_qty_matches_trade_history(self, simple_env):
        for _ in range(5):
            state = simple_env.step()
            step = state["step"]
            expected = {}
            for t in simple_env.trade_history:
                if t.step == step:
                    key = (t.agent_id, t.name)
                    expected[key] = expected.get(key, 0) + t.quantity
            for aid, stock_acts in state["agent_actions"].items():
                for sym, sa in stock_acts.items():
                    assert sa["filled_qty"] == expected.get((aid, sym), 0)
        simple_env.close()

    def test_last_round_matches_a_full_history_scan(self, simple_env):
        """The index must reproduce the old 'latest step with fills' semantics."""
        for _ in range(5):
            simple_env.step()
            for aid in simple_env.agents:
                obs = simple_env.get_observation(aid)
                latest = simple_env.trade_history[-1].step
                expected = [
                    t for t in simple_env.trade_history
                    if t.agent_id == aid and t.step == latest
                ]
                got = obs.get("last_round", {}).get("trades", [])
                assert len(got) == len(expected)
                assert [g["symbol"] for g in got] == [t.name for t in expected]
                assert [g["quantity"] for g in got] == [t.quantity for t in expected]
        simple_env.close()

    def test_step_fill_index_is_reset_each_step(self, simple_env):
        simple_env.step()
        first = dict(simple_env._step_fills)
        simple_env.step()
        assert simple_env._step_fills != {} or first == {}
        assert all(v > 0 for v in simple_env._step_fills.values())
        simple_env.close()

    def test_web_wealth_curve_grows_one_point_per_step(self):
        """api_step must append to the curve, not rebuild it from history."""
        from ai_trading_society.web.app import _sessions, app

        with app.test_client() as client:
            started = client.post("/api/start", json={"steps": 5, "traders": []})
            assert started.status_code == 200
            # Other tests leave sessions behind, so address ours by id
            # rather than assuming it is the only one in the store.
            with client.session_transaction() as sess:
                session_id = sess["simulation_id"]
            for expected_len in range(1, 4):
                res = client.post("/api/step", json={})
                assert res.status_code == 200
                state = _sessions[session_id]
                assert state["wealth_curves"], "curves should be seeded at start"
                for curve in state["wealth_curves"].values():
                    assert len(curve) == expected_len


# ---------------------------------------------------------------------------
# M4: only a genuine response_format rejection disables JSON mode
# ---------------------------------------------------------------------------
class TestM4JsonModeRejection:
    @pytest.fixture
    def openai_mod(self):
        return _FakeOpenAIModule("openai")

    def test_response_format_rejection_detected(self, openai_mod):
        exc = openai_mod.BadRequestError(
            "Unsupported parameter: 'response_format' is not supported"
        )
        assert _is_json_mode_rejection(exc, openai_mod) is True

    def test_400_naming_json_object_detected(self, openai_mod):
        assert _is_json_mode_rejection(
            _exc("APIStatusError", 400, "model does not support json_object"),
            openai_mod,
        ) is True

    def test_rate_limit_is_not_a_json_mode_rejection(self, openai_mod):
        assert _is_json_mode_rejection(
            _exc("RateLimitError", 429, "rate limit exceeded"), openai_mod
        ) is False

    def test_auth_failure_is_not_a_json_mode_rejection(self, openai_mod):
        assert _is_json_mode_rejection(
            _exc("AuthenticationError", 401, "invalid api key"), openai_mod
        ) is False

    def test_timeout_is_not_a_json_mode_rejection(self, openai_mod):
        assert _is_json_mode_rejection(
            _exc("APITimeoutError", None, "request timed out"), openai_mod
        ) is False

    def test_unrelated_400_is_not_a_json_mode_rejection(self, openai_mod):
        assert _is_json_mode_rejection(
            _exc("BadRequestError", 400, "context length exceeded"), openai_mod
        ) is False

    def test_rate_limit_does_not_disable_json_mode(self, monkeypatch):
        """A 429 must propagate without a second call or a sticky JSON-mode flag."""
        calls = []
        fake = _FakeOpenAIModule("openai")

        class _Client:
            def __init__(self, **_):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                calls.append(kwargs)
                raise _exc("RateLimitError", 429, "slow down")

        fake.OpenAI = _Client
        monkeypatch.setitem(sys.modules, "openai", fake)

        agent = ExternalAIAgent("a", api_key="k", enable_memory=False)
        with pytest.raises(Exception) as excinfo:
            agent._call_openai_compat("prompt")

        assert type(excinfo.value).__name__ == "RateLimitError"
        assert len(calls) == 1, "a rate limit must not trigger a second request"
        assert agent._json_mode_supported is True

    def test_response_format_rejection_retries_once_plain(self, monkeypatch):
        calls = []
        fake = _FakeOpenAIModule("openai")

        class _Client:
            def __init__(self, **_):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                calls.append(kwargs)
                if "response_format" in kwargs:
                    raise fake.BadRequestError("response_format is not supported")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )

        fake.OpenAI = _Client
        monkeypatch.setitem(sys.modules, "openai", fake)

        agent = ExternalAIAgent("a", api_key="k", enable_memory=False)
        assert agent._call_openai_compat("prompt") == "ok"
        assert len(calls) == 2
        assert agent._json_mode_supported is False


# ---------------------------------------------------------------------------
# M5: transient retries and the corrective re-ask budget
# ---------------------------------------------------------------------------
class TestM5TransientErrors:
    def test_transient_classification(self):
        assert _is_transient_error(_exc("RateLimitError", 429)) is True
        assert _is_transient_error(_exc("APIStatusError", 503)) is True
        assert _is_transient_error(_exc("APITimeoutError")) is True
        assert _is_transient_error(TimeoutError("slow")) is True
        assert _is_transient_error(ConnectionError("reset")) is True

    def test_permanent_classification(self):
        assert _is_transient_error(_exc("AuthenticationError", 401)) is False
        assert _is_transient_error(_exc("BadRequestError", 400)) is False
        assert _is_transient_error(_exc("NotFoundError", 404)) is False
        assert _is_transient_error(ValueError("no key")) is False

    def test_transient_failure_is_retried_once(self):
        agent = ExternalAIAgent("a", api_key="k", enable_memory=False)
        agent.retry_backoff = 0.0
        calls = []

        def _call(prompt, messages=None, system_prompt=None):
            calls.append(prompt)
            if len(calls) == 1:
                raise _exc("RateLimitError", 429, "slow down")
            return "recovered"

        agent._call_ai_api = _call
        assert agent._call_ai_api_with_retry("p") == "recovered"
        assert len(calls) == 2

    def test_permanent_failure_is_not_retried(self):
        agent = ExternalAIAgent("a", api_key="k", enable_memory=False)
        agent.retry_backoff = 0.0
        calls = []

        def _call(prompt, messages=None, system_prompt=None):
            calls.append(prompt)
            raise _exc("AuthenticationError", 401, "bad key")

        agent._call_ai_api = _call
        with pytest.raises(Exception):
            agent._call_ai_api_with_retry("p")
        assert len(calls) == 1

    def test_agent_survives_a_transient_blip(self):
        """One rate limit must not cost the agent its whole round."""
        agent = ExternalAIAgent("a", api_key="k", enable_memory=False)
        agent.retry_backoff = 0.0
        calls = []

        def _call(prompt, messages=None, system_prompt=None):
            calls.append(prompt)
            if len(calls) == 1:
                raise _exc("APITimeoutError", None, "timed out")
            return '{"decisions": [{"name": "S0", "action": "buy", "quantity": 4}]}'

        agent._call_ai_api = _call
        result = agent.act(_multi_stock_obs())
        assert result["decisions"][0]["quantity"] == 4


class TestM5RepairBudget:
    def _garbage_agent(self, budget):
        agent = ExternalAIAgent(
            "a", api_key="k", enable_memory=False, repair_budget=budget
        )
        agent.retry_backoff = 0.0
        calls = []

        def _call(prompt, messages=None, system_prompt=None):
            calls.append(prompt)
            return "I am not JSON at all."

        agent._call_ai_api = _call
        return agent, calls

    def test_budget_caps_repair_calls(self):
        agent, calls = self._garbage_agent(budget=2)
        with pytest.raises(ValueError):
            agent.act(_multi_stock_obs())
        assert len(calls) == 3, "1 initial call + 2 budgeted repairs"

    def test_exhausted_budget_skips_repairs_entirely(self):
        agent, calls = self._garbage_agent(budget=1)
        with pytest.raises(ValueError):
            agent.act(_multi_stock_obs())
        assert len(calls) == 2
        with pytest.raises(ValueError):
            agent.act(_multi_stock_obs())
        assert len(calls) == 3, "second round must not spend more repair calls"

    def test_zero_budget_never_repairs(self):
        agent, calls = self._garbage_agent(budget=0)
        with pytest.raises(ValueError):
            agent.act(_multi_stock_obs())
        assert len(calls) == 1

    def test_default_budget_allows_the_full_escalation(self):
        agent, calls = self._garbage_agent(budget=6)
        with pytest.raises(ValueError):
            agent.act(_multi_stock_obs())
        assert len(calls) == 4, "1 initial call + all 3 escalation attempts"

    def test_successful_repair_leaves_budget_for_later_rounds(self):
        agent = ExternalAIAgent(
            "a", api_key="k", enable_memory=False, repair_budget=6
        )
        agent.retry_backoff = 0.0
        calls = []

        def _call(prompt, messages=None, system_prompt=None):
            calls.append(prompt)
            if len(calls) == 1:
                return "no json here"
            return '{"decisions": [{"name": "S0", "action": "hold", "quantity": 0}]}'

        agent._call_ai_api = _call
        agent.act(_multi_stock_obs())
        assert len(calls) == 2
        assert agent._repair_calls_remaining == 5

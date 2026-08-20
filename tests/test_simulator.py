"""Tests for Simulator: run loop, metrics, and reporting."""

import pytest

from ai_trading_society.config import MarketConfig
from ai_trading_society.market_env import MarketEnv
from ai_trading_society.simulator import Simulator
from tests.conftest import ScriptedExternalAIAgent


@pytest.fixture
def sim_with_history():
    """A simulator that has run 10 steps."""
    config = MarketConfig(initial_price=100.0)
    agents = [
        ScriptedExternalAIAgent("a1", cash=10000, holdings=50, buy_prob=0.4, sell_prob=0.3),
        ScriptedExternalAIAgent("a2", cash=10000, holdings=50, buy_prob=0.3, sell_prob=0.4),
    ]
    env = MarketEnv(config, agents)
    sim = Simulator(env)
    sim.run(steps=10, verbose=False, round_by_round=False)
    return sim


class TestRunLoop:
    """Test simulation run loop."""

    def test_run_produces_state_history(self, sim_with_history):
        """run() should populate state_history with one entry per step."""
        assert len(sim_with_history.state_history) == 10

    def test_state_history_contains_agent_data(self, sim_with_history):
        """Each state should contain agent wealth data."""
        for state in sim_with_history.state_history:
            assert "agents" in state
            assert "a1" in state["agents"]
            assert "a2" in state["agents"]

    def test_run_with_zero_steps(self, classic_config):
        """Running 0 steps should produce empty history."""
        agents = [ScriptedExternalAIAgent("a1", cash=10000, holdings=10)]
        env = MarketEnv(classic_config, agents)
        sim = Simulator(env)
        sim.run(steps=0, verbose=False)
        assert len(sim.state_history) == 0

    def test_report_runs_without_error(self, sim_with_history):
        """report() should execute without raising."""
        sim_with_history.report()


class TestPerformanceMetrics:
    """Test performance metrics computation."""

    def test_metrics_return_required_keys(self, sim_with_history):
        """_compute_agent_metrics should return all required keys."""
        metrics = sim_with_history._compute_agent_metrics("a1")
        required = {"sharpe", "max_drawdown", "volatility", "win_rate"}
        assert required.issubset(metrics.keys())

    def test_max_drawdown_non_negative(self, sim_with_history):
        """Max drawdown should always be >= 0."""
        for agent_id in ["a1", "a2"]:
            metrics = sim_with_history._compute_agent_metrics(agent_id)
            assert metrics["max_drawdown"] >= 0.0

    def test_win_rate_in_valid_range(self, sim_with_history):
        """Win rate should be between 0 and 1."""
        for agent_id in ["a1", "a2"]:
            metrics = sim_with_history._compute_agent_metrics(agent_id)
            assert 0.0 <= metrics["win_rate"] <= 1.0

    def test_volatility_non_negative(self, sim_with_history):
        """Volatility should always be >= 0."""
        for agent_id in ["a1", "a2"]:
            metrics = sim_with_history._compute_agent_metrics(agent_id)
            assert metrics["volatility"] >= 0.0

    def test_metrics_with_insufficient_data(self, classic_config):
        """Metrics with < 2 data points should return zeros."""
        agents = [ScriptedExternalAIAgent("a1", cash=10000, holdings=10)]
        env = MarketEnv(classic_config, agents)
        sim = Simulator(env)
        # No steps run yet
        metrics = sim._compute_agent_metrics("a1")
        assert metrics["sharpe"] == 0.0
        assert metrics["max_drawdown"] == 0.0


class TestRealisticMode:
    """Test simulator in realistic mode with events."""

    def test_realistic_mode_runs_without_error(self, realistic_config):
        """Realistic mode simulation should complete without errors."""
        agents = [
            ScriptedExternalAIAgent("a1", cash=10000, holdings=50),
            ScriptedExternalAIAgent("a2", cash=10000, holdings=50),
        ]
        env = MarketEnv(realistic_config, agents)
        sim = Simulator(env)
        sim.run(steps=5, verbose=False, round_by_round=False)
        assert len(sim.state_history) == 5

    def test_triggered_events_appear_in_state(self, realistic_config):
        """When events trigger, they should appear in state as a list."""
        from ai_trading_society.market_events import EventManager, MarketEvent, EventType

        # Force event triggering by using probability=1.0 templates
        templates = [
            MarketEvent(
                name="forced_event",
                description="Forced test event",
                event_type=EventType.EARNINGS,
                duration_steps=2,
                probability=1.0,
            )
        ]
        agents = [ScriptedExternalAIAgent("a1", cash=10000, holdings=50)]
        env = MarketEnv(realistic_config, agents)
        # Replace the event manager with our forced one
        env.event_manager = EventManager(
            templates=templates,
            event_probability_multiplier=1.0,
        )
        sim = Simulator(env)
        state = env.step()
        assert "triggered_events" in state, "State should contain triggered_events"
        assert isinstance(state["triggered_events"], list)
        assert len(state["triggered_events"]) >= 1


class TestProgressAndTracking:
    """Test progress bar and wealth tracking features."""

    def test_total_steps_stored_after_run(self, classic_config):
        """_total_steps should be set after run()."""
        agents = [
            ScriptedExternalAIAgent("a1", cash=10000, holdings=50),
            ScriptedExternalAIAgent("a2", cash=10000, holdings=50),
        ]
        env = MarketEnv(classic_config, agents)
        sim = Simulator(env)
        sim.run(steps=10, verbose=False, round_by_round=False)
        assert sim._total_steps == 10

    def test_initial_wealths_tracked(self, classic_config):
        """_initial_wealths should be populated after first round."""
        agents = [
            ScriptedExternalAIAgent("a1", cash=10000, holdings=50),
            ScriptedExternalAIAgent("a2", cash=20000, holdings=30),
        ]
        env = MarketEnv(classic_config, agents)
        sim = Simulator(env)
        sim.run(steps=5, verbose=False, round_by_round=True)
        assert "a1" in sim._initial_wealths
        assert "a2" in sim._initial_wealths
        # a1 initial wealth = 10000 + 50 * 100 = 15000
        assert sim._initial_wealths["a1"] > 0
        # a2 should have more initial wealth than a1
        assert sim._initial_wealths["a2"] > sim._initial_wealths["a1"]

    def test_prev_wealths_tracked_after_run(self, classic_config):
        """_prev_wealths should be populated after running."""
        agents = [
            ScriptedExternalAIAgent("a1", cash=10000, holdings=50),
            ScriptedExternalAIAgent("a2", cash=10000, holdings=50),
        ]
        env = MarketEnv(classic_config, agents)
        sim = Simulator(env)
        sim.run(steps=3, verbose=False, round_by_round=True)
        assert len(sim._prev_wealths) == 2

    def test_report_colorized_does_not_raise(self, sim_with_history):
        """report() with colorized output should not raise."""
        sim_with_history.report()


class TestInteractiveMode:
    """Test interactive (step-by-step) simulation mode."""

    def test_prompt_continue_returns_false_on_enter(self, monkeypatch):
        """Empty input (Enter) should return False (continue)."""
        monkeypatch.setattr("builtins.input", lambda *a: "")
        result = Simulator._prompt_continue()
        assert result is False

    def test_prompt_continue_returns_true_on_q(self, monkeypatch):
        """Input 'q' should return True (stop)."""
        monkeypatch.setattr("builtins.input", lambda *a: "q")
        result = Simulator._prompt_continue()
        assert result is True

    def test_prompt_continue_returns_true_on_stop(self, monkeypatch):
        """Input 'stop' should return True (stop)."""
        monkeypatch.setattr("builtins.input", lambda *a: "stop")
        result = Simulator._prompt_continue()
        assert result is True

    def test_prompt_continue_returns_true_on_eof(self, monkeypatch):
        """EOFError should return True (stop)."""
        def raise_eof(*a):
            raise EOFError()
        monkeypatch.setattr("builtins.input", raise_eof)
        result = Simulator._prompt_continue()
        assert result is True

    def test_interactive_stops_early(self, classic_config, monkeypatch):
        """When user types 'q', simulation should stop before all steps."""
        # First call to input returns "" (continue), second returns "q" (stop)
        answers = iter(["", "q"])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))

        agents = [
            ScriptedExternalAIAgent("a1", cash=10000, holdings=50),
            ScriptedExternalAIAgent("a2", cash=10000, holdings=50),
        ]
        env = MarketEnv(classic_config, agents)
        sim = Simulator(env)
        sim.run(steps=10, verbose=False, round_by_round=True, interactive=True)
        # Should have stopped after 2 rounds (1 continue + 1 stop)
        assert len(sim.state_history) == 2

    def test_interactive_completes_all_steps(self, classic_config, monkeypatch):
        """When user always presses Enter, all steps should run."""
        monkeypatch.setattr("builtins.input", lambda *a: "")

        agents = [
            ScriptedExternalAIAgent("a1", cash=10000, holdings=50),
            ScriptedExternalAIAgent("a2", cash=10000, holdings=50),
        ]
        env = MarketEnv(classic_config, agents)
        sim = Simulator(env)
        sim.run(steps=5, verbose=False, round_by_round=True, interactive=True)
        assert len(sim.state_history) == 5

    def test_non_interactive_ignores_input(self, classic_config, monkeypatch):
        """interactive=False should never call input()."""
        call_count = [0]
        def mock_input(*a):
            call_count[0] += 1
            return ""
        monkeypatch.setattr("builtins.input", mock_input)

        agents = [
            ScriptedExternalAIAgent("a1", cash=10000, holdings=50),
            ScriptedExternalAIAgent("a2", cash=10000, holdings=50),
        ]
        env = MarketEnv(classic_config, agents)
        sim = Simulator(env)
        sim.run(steps=5, verbose=False, round_by_round=True, interactive=False)
        assert call_count[0] == 0

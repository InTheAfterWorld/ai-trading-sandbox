"""Tests for Simulator: run loop, metrics, and reporting."""

import pytest

from ai_trading_society.agents.player_agent import PlayerAgent
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
        from ai_trading_society.market_events import EventManager, EventType, MarketEvent

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


class TestInteractiveMenu:
    """Test the interactive command menu (player trade / God Mode / social)."""

    def _make_sim(self, config):
        agents = [
            ScriptedExternalAIAgent("a1", cash=10000, holdings=50),
            ScriptedExternalAIAgent("a2", cash=10000, holdings=50),
        ]
        player = PlayerAgent(agent_id="Player (You)", cash=10000, holdings=20)
        agents.append(player)
        env = MarketEnv(config, agents)
        player._env = env
        return Simulator(env), player

    def test_menu_continue_on_blank(self, classic_config, monkeypatch):
        sim, _ = self._make_sim(classic_config)
        monkeypatch.setattr("builtins.input", lambda *a: "")
        assert sim._interactive_menu() == "continue"

    def test_menu_stop_on_q(self, classic_config, monkeypatch):
        sim, _ = self._make_sim(classic_config)
        monkeypatch.setattr("builtins.input", lambda *a: "q")
        assert sim._interactive_menu() == "stop"

    def test_menu_unknown_command_loops(self, classic_config, monkeypatch):
        """Unknown command should not end the round; menu loops to continue."""
        sim, _ = self._make_sim(classic_config)
        answers = iter(["bogus", ""])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))
        assert sim._interactive_menu() == "continue"

    def test_menu_buy_then_continue(self, classic_config, monkeypatch):
        """'b' + quantity buffers a player buy, then blank advances the round."""
        sim, player = self._make_sim(classic_config)
        answers = iter(["b", "5", ""])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))
        assert sim._interactive_menu(player) == "continue"
        assert sim.env.pop_player_action() == {"action": "buy", "quantity": 5}

    def test_cmd_player_trade_buffers_buy(self, classic_config, monkeypatch):
        sim, player = self._make_sim(classic_config)
        monkeypatch.setattr("builtins.input", lambda *a: "30")
        sim._cmd_player_trade(player, "buy")
        assert sim.env.pop_player_action() == {"action": "buy", "quantity": 30}

    def test_cmd_player_trade_buffers_sell(self, classic_config, monkeypatch):
        sim, player = self._make_sim(classic_config)
        monkeypatch.setattr("builtins.input", lambda *a: "10")
        sim._cmd_player_trade(player, "sell")
        assert sim.env.pop_player_action() == {"action": "sell", "quantity": 10}

    def test_cmd_player_trade_cancels_on_empty(self, classic_config, monkeypatch):
        sim, player = self._make_sim(classic_config)
        monkeypatch.setattr("builtins.input", lambda *a: "")
        sim._cmd_player_trade(player, "buy")
        assert sim.env.pop_player_action() is None

    def test_cmd_player_trade_rejects_bad_quantity(self, classic_config, monkeypatch):
        sim, player = self._make_sim(classic_config)
        monkeypatch.setattr("builtins.input", lambda *a: "abc")
        sim._cmd_player_trade(player, "buy")
        assert sim.env.pop_player_action() is None

    def test_cmd_player_trade_unavailable_without_player(self, classic_config, monkeypatch):
        sim, _ = self._make_sim(classic_config)
        monkeypatch.setattr("builtins.input", lambda *a: "10")
        sim._cmd_player_trade(None, "buy")
        assert sim.env.pop_player_action() is None

    def test_cmd_god_event_injects_by_number(self, classic_config, monkeypatch):
        sim, _ = self._make_sim(classic_config)
        monkeypatch.setattr("builtins.input", lambda *a: "1")
        sim._cmd_god_event()
        assert any(e.get("forced") for e in sim.env.event_manager.event_history)

    def test_cmd_god_event_injects_by_name(self, classic_config, monkeypatch):
        from ai_trading_society.market_events import EVENT_TEMPLATES

        sim, _ = self._make_sim(classic_config)
        name = EVENT_TEMPLATES[0].name
        monkeypatch.setattr("builtins.input", lambda *a: name)
        sim._cmd_god_event()
        assert sim.env.event_manager.event_history[-1]["name"] == name
        assert sim.env.event_manager.event_history[-1]["forced"] is True

    def test_cmd_god_config_adjusts_sensitivity(self, classic_config, monkeypatch):
        sim, _ = self._make_sim(classic_config)
        answers = iter(["1", "0.05"])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))
        sim._cmd_god_config()
        assert sim.env.config.price_sensitivity == 0.05

    def test_cmd_god_config_adjusts_event_multiplier(self, classic_config, monkeypatch):
        sim, _ = self._make_sim(classic_config)
        answers = iter(["3", "2.5"])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))
        sim._cmd_god_config()
        assert sim.env.config.event_probability_multiplier == 2.5
        assert sim.env.event_manager.multiplier == 2.5

    def test_cmd_god_config_adjusts_sentiment_drift(self, classic_config, monkeypatch):
        sim, _ = self._make_sim(classic_config)
        answers = iter(["4", "0.3"])
        monkeypatch.setattr("builtins.input", lambda *a: next(answers))
        sim._cmd_god_config()
        assert sim.env._sentiment_drift == 0.3

    def test_show_social_does_not_raise(self, classic_config, capsys):
        sim, _ = self._make_sim(classic_config)
        sim._show_social()
        out = capsys.readouterr().out
        assert "Social Relationships" in out

"""Additional Simulator tests: verbose output, interactive menus, CSV export."""

import builtins

import pytest

from ai_trading_society.agents.player_agent import PlayerAgent
from ai_trading_society.market_env import MarketEnv
from ai_trading_society.simulator import Simulator, _safe, evaluate_wealth_curve, grade_performance
from tests.conftest import ScriptedExternalAIAgent


@pytest.fixture
def sim2(classic_config):
    agents = [
        ScriptedExternalAIAgent("a1", cash=10000, holdings=50, buy_prob=0.4),
        ScriptedExternalAIAgent("a2", cash=10000, holdings=50, sell_prob=0.4),
    ]
    return Simulator(MarketEnv(classic_config, agents))


class TestEvaluateWealthCurve:
    """Test evaluate_wealth_curve and helpers directly."""

    def test_short_curve_returns_defaults(self):
        result = evaluate_wealth_curve([100.0])
        assert result == {"sharpe": 0.0, "max_drawdown": 0.0,
                          "volatility": 0.0, "win_rate": 0.0}
        assert evaluate_wealth_curve([])["sharpe"] == 0.0

    def test_non_finite_values_sanitized(self):
        result = evaluate_wealth_curve([float("nan"), 100.0, float("inf"), 110.0])
        assert result["max_drawdown"] >= 0.0

    def test_safe_coercion(self):
        assert _safe("abc") == 0.0
        assert _safe(None, 5.0) == 5.0
        assert _safe(float("nan")) == 0.0
        assert _safe(3.5) == 3.5

    def test_perfect_steady_growth(self):
        result = evaluate_wealth_curve([100.0, 110.0, 115.0])
        assert result["win_rate"] == 100.0
        assert result["max_drawdown"] == 0.0
        assert result["sharpe"] > 0

    def test_flat_curve_zero_sharpe(self):
        assert evaluate_wealth_curve([100.0, 100.0, 100.0])["sharpe"] == 0.0

    def test_zero_prev_wealth_returns_zero_return(self):
        # When prev wealth <= 0 the step return is 0.0, no crash.
        result = evaluate_wealth_curve([0.0, 100.0])
        assert result["win_rate"] == 0.0


class TestGradePerformance:
    """Test grade_performance blending."""

    def test_grade_keys(self):
        result = grade_performance(10.0, 1.0, 5.0, 60.0)
        assert "score" in result and "grade" in result

    def test_score_bounds(self):
        for args in [(100.0, 5.0, 0.0, 100.0), (-100.0, -5.0, 90.0, 0.0)]:
            score = grade_performance(*args)["score"]
            assert 0.0 <= score <= 100.0

class TestVerboseRun:
    """Exercise verbose printing paths."""

    def test_verbose_round_by_round(self, sim2, capsys):
        sim2.run(steps=2, verbose=True, round_by_round=True, save_snapshot=False)
        out = capsys.readouterr().out
        assert "Simulation Start" in out
        assert "Agent Roster" in out
        assert "Round 1" in out

    def test_verbose_summary_mode(self, sim2, capsys):
        sim2.run(steps=6, verbose=True, round_by_round=False,
                 log_interval=2, save_snapshot=False)
        assert "Step" in capsys.readouterr().out

    def test_snapshot_saved(self, sim2, capsys, tmp_path):
        sim2.run(steps=2, verbose=True, save_snapshot=True,
                 runs_dir=str(tmp_path))
        assert "Run snapshot saved to" in capsys.readouterr().out
        assert any(tmp_path.iterdir())

    def test_no_snapshot(self, sim2, tmp_path):
        sim2.run(steps=2, verbose=False, save_snapshot=False)
        assert list(tmp_path.iterdir()) == []


class TestInteractive:
    """Test interactive menu paths via monkeypatched input."""

    def test_stop_early(self, sim2, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *a: "q")
        states = sim2.run(steps=10, verbose=False, round_by_round=True,
                          interactive=True, save_snapshot=False)
        assert len(states) < 10

    def test_eof_stops(self, sim2, monkeypatch):
        def raise_eof(*a):
            raise EOFError
        monkeypatch.setattr(builtins, "input", raise_eof)
        states = sim2.run(steps=5, verbose=False, round_by_round=True,
                          interactive=True, save_snapshot=False)
        assert len(states) < 5

    def test_menu_paths(self, sim2, monkeypatch, capsys):
        answers = iter(["h", "bogus", "r", ""])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        assert sim2._interactive_menu(None) == "continue"
        out = capsys.readouterr().out
        assert "Interactive commands" in out
        assert "Unknown command" in out
        assert "Social Relationships" in out

    def test_player_trade_without_player(self, sim2, monkeypatch, capsys):
        answers = iter(["b", ""])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        assert sim2._interactive_menu(None) == "continue"
        assert "Player mode is not available" in capsys.readouterr().out

    def test_player_trade_queue(self, classic_config, monkeypatch, capsys):
        player = PlayerAgent("human", cash=10000)
        agents = [ScriptedExternalAIAgent("a1", cash=10000, holdings=50)]
        sim = Simulator(MarketEnv(classic_config, agents + [player]))
        answers = iter(["b", "10", ""])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        sim._interactive_menu(player)
        assert "Queued BUY 10" in capsys.readouterr().out

    def test_player_trade_cancel_and_invalid(self, classic_config,
                                             monkeypatch, capsys):
        player = PlayerAgent("human", cash=10000)
        agents = [ScriptedExternalAIAgent("a1", cash=10000, holdings=50)]
        sim = Simulator(MarketEnv(classic_config, agents + [player]))
        for seq in (["b", "c", ""], ["b", "abc", ""], ["b", "-5", ""], ["s", "0", ""],
                    ["b", "0", ""], ["s", "x", ""]):
            answers = iter(seq)
            monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
            sim._interactive_menu(player)
        out = capsys.readouterr().out
        assert "Cancelled" in out
        assert "Invalid quantity" in out
        assert "positive" in out

    def test_prompt_continue_eof(self, monkeypatch):
        def raise_eof(*a):
            raise EOFError
        monkeypatch.setattr(builtins, "input", raise_eof)
        assert Simulator._prompt_continue() is True

class TestGodMode:
    """Test God Mode event injection and parameter editing."""

    def test_god_event_cancelled(self, sim2, monkeypatch, capsys):
        monkeypatch.setattr(builtins, "input", lambda *a: "c")
        sim2._cmd_god_event()
        assert "Cancelled" in capsys.readouterr().out

    def test_god_event_by_number(self, sim2, monkeypatch, capsys):
        monkeypatch.setattr(builtins, "input", lambda *a: "1")
        sim2._cmd_god_event()
        assert "Injected event" in capsys.readouterr().out

    def test_god_event_invalid_number(self, sim2, monkeypatch, capsys):
        monkeypatch.setattr(builtins, "input", lambda *a: "9999")
        sim2._cmd_god_event()
        assert "Invalid event number" in capsys.readouterr().out

    def test_god_event_eof(self, sim2, monkeypatch, capsys):
        def raise_eof(*a):
            raise EOFError
        monkeypatch.setattr(builtins, "input", raise_eof)
        sim2._cmd_god_event()
        assert "Cancelled" in capsys.readouterr().out

    @pytest.mark.parametrize("choice,newval,attr", [
        ("1", "0.05", "price_sensitivity"),
        ("2", "0.08", "max_price_change_ratio"),
        ("3", "2.0", "event_probability_multiplier"),
    ])
    def test_god_config_params(self, sim2, monkeypatch, choice, newval, attr):
        answers = iter([choice, newval])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        sim2._cmd_god_config()
        assert getattr(sim2.env.config, attr) == float(newval)

    def test_god_config_sentiment(self, sim2, monkeypatch):
        answers = iter(["4", "0.5"])
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        sim2._cmd_god_config()
        assert sim2.env._sentiment_drift == 0.5

    def test_god_config_invalid(self, sim2, monkeypatch, capsys):
        for seq in (["c"], [""], ["9", ""], ["1", "-1"], ["2", "0"], ["1", "notanum"],
                    ["1", ""]):
            answers = iter(seq)
            monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
            sim2._cmd_god_config()
        out = capsys.readouterr().out
        assert "Invalid choice" in out
        assert "must be >= 0" in out
        assert "must be > 0" in out
        assert "Invalid number" in out
        assert "Cancelled" in out


class TestRankingAndReporting:
    """Test mini ranking and summary printing paths."""

    def test_mini_ranking_skipped_for_few_agents(self, sim2, capsys):
        state = {"agents": {f"a{i}": {"wealth": 100 + i} for i in range(3)}}
        sim2._print_mini_ranking(state)
        assert capsys.readouterr().out == ""

    def test_mini_ranking_printed(self, sim2, capsys):
        state = {"agents": {f"a{i}": {"wealth": 100 + i} for i in range(8)}}
        sim2._print_mini_ranking(state)
        out = capsys.readouterr().out
        assert "Mini Ranking" in out
        assert "others" in out

    def test_should_show_ranking(self, sim2):
        sim2._ranking_interval = 5
        assert sim2._should_show_mini_ranking(10)
        assert not sim2._should_show_mini_ranking(7)
        sim2._ranking_interval = 0
        assert not sim2._should_show_mini_ranking(10)

    def test_report_with_charts_skipped(self, sim2, capsys, tmp_path, monkeypatch):
        sim2.run(steps=3, verbose=False, round_by_round=False,
                 save_snapshot=False)
        monkeypatch.setattr(
            "ai_trading_society.visualization.generate_all_charts",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        sim2.report(generate_charts=True, chart_output_dir=str(tmp_path))
        assert "skipped" in capsys.readouterr().out

    def test_export_csv(self, sim2, tmp_path):
        sim2.run(steps=3, verbose=False, save_snapshot=False)
        path = tmp_path / "trades.csv"
        sim2.export_csv(str(path))
        assert path.exists()
        assert "step,agent_id,action" in path.read_text(encoding="utf-8")

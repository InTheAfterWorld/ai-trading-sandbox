"""Phase 2A/2E/2F: the factual context blocks in the prompt.

Each block renders only when its data is in the observation, so simple
mode gets the regime + stakes lines and nothing else, while deep mode adds
concentration, exposure, standing, floor mood, peer quotes, lessons and
committed exit levels.
"""

import pytest

from ai_trading_society.agents.external_ai_agent import ExternalAIAgent
from ai_trading_society.agents.traits import create_personality_agent
from tests.conftest import ScriptedExternalAIAgent


def _agent(**kw):
    return ExternalAIAgent("t", api_key="k", enable_memory=True, **kw)


def _base_obs():
    """A simple-mode observation: only the always-on fields."""
    return {
        "step": 4,
        "stocks": [
            {"symbol": "Alpha", "name": "Alpha", "price": 120.0,
             "price_history": [100, 104, 109, 114, 120], "last_volume": 10,
             "my_holdings": 60, "move_since_last_pct": 5.3},
            {"symbol": "Beta", "name": "Beta", "price": 50.0,
             "price_history": [60, 57, 54, 52, 50], "last_volume": 5,
             "my_holdings": 0, "move_since_last_pct": -3.8},
        ],
        "my_cash": 800.0,
        "my_holdings": {"Alpha": 60, "Beta": 0},
        "my_wealth": 8000.0,
        "initial_wealth": 10000.0,
        "market_sentiment": 0.0,
        "price": 120.0,
        "price_history": [100, 104, 109, 114, 120],
        "last_volume": 10,
    }


def _deep_obs():
    """A deep-mode observation: everything MarketEnv adds when deep."""
    obs = _base_obs()
    obs["persona"] = {
        "name": "Nova", "disposition": "You are a jittery trader.",
        "mood": {"confidence": 3.0, "stress": 8.0, "frustration": 6.0},
        "dials": {"envy": 9},
        "pressure": "Last round you were down 4.2%; you are 20.0% below your peak.",
        "scale_hint": "0 = none at all, 10 = as strong as it gets",
    }
    obs["standing"] = {
        "rank": 5, "of": 6, "my_return_pct": -20.0,
        "leader_name": "Vega", "leader_return_pct": 22.0,
        "gap_to_leader_pct": 42.0,
    }
    obs["floor_mood"] = {
        "mood": "panicked",
        "sentence": "The floor feels panicked -- most traders dumped last round.",
    }
    obs["held_at_round_start"] = {"Alpha": 60.0, "Beta": 0.0}
    obs["social_peers"] = [
        {"id": "Vega", "relation": "idol", "action": "buy", "quantity": 12,
         "reasoning": "Loading up while everyone else panics."},
    ]
    return obs


class TestAlwaysOnBlocks:
    def test_stakes_line_in_both_modes(self):
        for obs in (_base_obs(), _deep_obs()):
            prompt = _agent()._build_prompt(obs)
            assert "you started with $10,000" in prompt
            assert "down $2,000" in prompt

    def test_regime_line_in_both_modes(self):
        for obs in (_base_obs(), _deep_obs()):
            agent = _agent()
            agent._build_prompt(obs)          # seeds the market history
            prompt = agent._build_prompt(obs)
            assert "Regime: the market is" in prompt

    @pytest.mark.parametrize("series,expected", [
        ([100, 101, 102, 103, 104, 105, 106, 112], "uptrend"),
        ([100, 99, 98, 97, 96, 95, 94, 88], "downtrend"),
        ([100, 130, 80, 125, 85, 130, 78, 128], "volatile"),
        ([100, 100.1, 100, 100.1, 100, 100.1, 100, 100.05], "calm"),
    ])
    def test_regime_names_the_shape(self, series, expected):
        agent = _agent()
        agent._market_history = list(enumerate(series))
        assert expected in agent._build_market_summary()

    def test_no_stakes_line_without_initial_wealth(self):
        obs = _base_obs()
        del obs["initial_wealth"]
        assert "you started with" not in _agent()._build_prompt(obs)


class TestDeepOnlyBlocks:
    def test_simple_mode_omits_all_of_them(self):
        prompt = _agent()._build_prompt(_base_obs())
        for marker in ("WHO YOU ARE", "HOW YOU FEEL", "Concentration:",
                       "Standing:", "Since last round:", "The floor feels",
                       "WHAT OTHERS ARE SAYING"):
            assert marker not in prompt, marker

    def test_deep_mode_includes_all_of_them(self):
        prompt = _agent()._build_prompt(_deep_obs())
        for marker in ("WHO YOU ARE", "HOW YOU FEEL", "Concentration:",
                       "Standing:", "Since last round:", "The floor feels",
                       "WHAT OTHERS ARE SAYING"):
            assert marker in prompt, marker

    def test_persona_block_shows_mood_and_pressure(self):
        prompt = _agent()._build_prompt(_deep_obs())
        assert "Confidence 3/10" in prompt and "Stress 8/10" in prompt
        assert "20.0% below your peak" in prompt
        assert "You are a jittery trader." in prompt

    def test_concentration_names_the_stock_and_share(self):
        prompt = _agent()._build_prompt(_deep_obs())
        assert "90% of your wealth is in Alpha" in prompt

    def test_concentration_silent_when_spread_out(self):
        obs = _deep_obs()
        obs["stocks"][0]["my_holdings"] = 5      # 7.5% of wealth
        assert "Concentration:" not in _agent()._build_prompt(obs)

    def test_standing_names_the_leader_and_gap(self):
        prompt = _agent()._build_prompt(_deep_obs())
        assert "you are 5 of 6" in prompt
        assert "Vega is leading" in prompt and "42.0 points ahead" in prompt

    def test_exposure_flags_held_versus_missed(self):
        prompt = _agent()._build_prompt(_deep_obs())
        assert "Alpha +5.3% (you held it)" in prompt
        assert "Beta -3.8% (you weren't in it)" in prompt

    def test_exposure_skips_flat_stocks(self):
        obs = _deep_obs()
        for s in obs["stocks"]:
            s["move_since_last_pct"] = 0.2
        assert "Since last round:" not in _agent()._build_prompt(obs)

    def test_peer_quote_is_rendered(self):
        prompt = _agent()._build_prompt(_deep_obs())
        assert "Vega" in prompt
        assert "Loading up while everyone else panics." in prompt

    def test_peers_without_reasoning_render_no_block(self):
        obs = _deep_obs()
        obs["social_peers"] = [{"id": "Vega", "relation": "idol",
                                "action": "buy", "quantity": 3}]
        assert "WHAT OTHERS ARE SAYING" not in _agent()._build_prompt(obs)


class TestLessonsAndPlans:
    def test_lesson_is_parsed_and_replayed(self):
        agent = _agent()
        parsed = agent._parse_response(
            '{"decisions": [{"name": "Alpha", "action": "hold", "quantity": 0}],'
            ' "lesson": "Stop chasing green candles."}'
        )
        assert parsed["lesson"] == "Stop chasing green candles."
        agent._record_lesson(parsed)
        prompt = agent._build_prompt(_deep_obs())
        assert "LESSONS YOU'VE LEARNED" in prompt
        assert "Stop chasing green candles." in prompt

    def test_lessons_are_capped_and_deduplicated(self):
        agent = _agent()
        for i in range(20):
            agent._record_lesson({"lesson": f"lesson {i}"})
        agent._record_lesson({"lesson": "lesson 19"})
        assert len(agent._lessons) == agent._MAX_LESSONS
        assert len(set(agent._lessons)) == len(agent._lessons)

    def test_stop_and_target_are_parsed(self):
        parsed = _agent()._parse_response(
            '{"decisions": [{"name": "Alpha", "action": "buy", "quantity": 5,'
            ' "stop_loss": 110, "target": "$140"}]}'
        )
        d = parsed["decisions"][0]
        assert d["stop_loss"] == 110.0 and d["target"] == 140.0

    def test_malformed_stop_is_dropped_not_fatal(self):
        parsed = _agent()._parse_response(
            '{"decisions": [{"name": "Alpha", "action": "buy", "quantity": 5,'
            ' "stop_loss": "soon"}]}'
        )
        assert parsed["decisions"][0]["quantity"] == 5
        assert "stop_loss" not in parsed["decisions"][0]

    def test_plan_is_replayed_next_round(self):
        agent = _agent()
        obs = _deep_obs()
        agent._record_position_plans(
            obs,
            {"decisions": [{"name": "Alpha", "action": "buy", "quantity": 5,
                            "stop_loss": 110.0}]},
        )
        prompt = agent._build_prompt(obs)
        assert "PLANS YOU COMMITTED TO" in prompt
        assert "stop at $110.00" in prompt and "it is now $120.00" in prompt

    def test_plan_cleared_when_the_position_closes(self):
        agent = _agent()
        obs = _deep_obs()
        agent._record_position_plans(
            obs, {"decisions": [{"name": "Alpha", "action": "buy",
                                 "quantity": 5, "stop_loss": 110.0}]}
        )
        assert "Alpha" in agent._position_plans
        closed = _deep_obs()
        closed["my_holdings"] = {"Alpha": 0, "Beta": 0}
        agent._record_position_plans(closed, {"decisions": []})
        assert "Alpha" not in agent._position_plans

    def test_optional_fields_absent_by_default(self):
        parsed = _agent()._parse_response(
            '{"decisions": [{"name": "Alpha", "action": "hold", "quantity": 0}]}'
        )
        assert "mood" not in parsed and "lesson" not in parsed

    def test_mood_rides_alongside_decisions(self):
        parsed = _agent()._parse_response(
            '{"decisions": [{"name": "Alpha", "action": "hold", "quantity": 0}],'
            ' "mood": {"confidence": 4, "stress": 7, "frustration": 2}}'
        )
        assert parsed["mood"]["stress"] == 7


class TestSchemaStillValid:
    def test_deep_prompt_documents_the_optional_fields(self):
        prompt = ExternalAIAgent._default_system_prompt(deep=True)
        assert "Optional extras" in prompt
        assert '"mood"' in prompt and '"lesson"' in prompt

    def test_simple_prompt_does_not(self):
        assert "Optional extras" not in ExternalAIAgent._default_system_prompt()

    def test_decision_parsing_unaffected_by_the_new_fields(self):
        agent = create_personality_agent(
            ScriptedExternalAIAgent("s", cash=100, holdings=5), "balanced",
            deep=True,
        )
        obs = _deep_obs()
        assert agent.act(obs)["decisions"] == agent.base_agent.act(obs)["decisions"]

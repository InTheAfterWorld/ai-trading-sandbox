"""Phase 2B: the 3-axis mood engine.

Hybrid: the model reports its own mood, Python clamps it, and a
deterministic formula fills in when the model reports nothing usable.
Mood only ever shapes the prompt -- never the decision.
"""

import pytest

from ai_trading_society.agents.traits import (
    MOOD_AXES,
    TraitAgent,
    create_personality_agent,
    preset_mood,
)
from ai_trading_society.config import MarketConfig
from ai_trading_society.market_env import MarketEnv
from tests.conftest import ScriptedExternalAIAgent


def _obs(wealth=10000.0, cash=1000.0, holdings=50, price=100.0, standing=None):
    obs = {
        "step": 2,
        "stocks": [{"symbol": "A", "name": "A", "price": price,
                    "price_history": [100, 100, price], "last_volume": 0,
                    "my_holdings": holdings, "move_since_last_pct": 0.0}],
        "my_cash": cash,
        "my_holdings": {"A": holdings},
        "my_wealth": wealth,
        "initial_wealth": 10000.0,
        "market_sentiment": 0.0,
        "price": price,
        "price_history": [100, 100, price],
        "last_volume": 0,
    }
    if standing is not None:
        obs["standing"] = standing
    return obs


class _Reporter(ScriptedExternalAIAgent):
    """Base agent that reports a mood of its choosing."""

    reported = None

    def act(self, observation):
        out = super().act(observation)
        if self.reported is not None:
            out["mood"] = self.reported
        return out


def _agent(personality="balanced", deep=True, **kw):
    base = _Reporter("t", cash=1000, holdings=50)
    return create_personality_agent(base, personality, deep=deep, **kw)


class TestSeeding:
    @pytest.mark.parametrize("personality", [
        "balanced", "aggressive", "conservative", "panicky",
        "greedy", "fomo_driven", "stubborn", "emotional",
    ])
    def test_mood_seeds_from_the_preset(self, personality):
        agent = _agent(personality)
        assert agent.mood == preset_mood(personality)
        assert set(agent.mood) == set(MOOD_AXES)

    def test_presets_do_not_all_start_the_same(self):
        assert preset_mood("panicky") != preset_mood("aggressive")

    def test_all_axes_in_range(self):
        for personality in ("panicky", "aggressive", "emotional"):
            assert all(0 <= v <= 10 for v in preset_mood(personality).values())


class TestReportedMoodIsClamped:
    def test_reported_mood_is_adopted(self):
        agent = _agent("balanced")
        agent.base_agent.reported = {
            "confidence": 6.0, "stress": 4.0, "frustration": 3.0
        }
        agent.act(_obs())
        assert agent.mood["stress"] == 4.0

    def test_clamped_to_zero_ten(self):
        agent = _agent("balanced")
        agent.mood_max_step = 100.0     # isolate the 0-10 clamp
        agent.base_agent.reported = {
            "confidence": 99.0, "stress": -20.0, "frustration": 5.0
        }
        agent.act(_obs())
        assert agent.mood["confidence"] == 10.0
        assert agent.mood["stress"] == 0.0

    def test_clamped_to_max_step(self):
        agent = _agent("balanced")
        start = dict(agent.mood)
        agent.base_agent.reported = {
            "confidence": 10.0, "stress": 10.0, "frustration": 10.0
        }
        agent.act(_obs())
        for axis in MOOD_AXES:
            assert agent.mood[axis] <= start[axis] + agent.mood_max_step + 1e-9

    @pytest.mark.parametrize("bad", [
        {"confidence": 5.0},                                  # incomplete
        {"confidence": "x", "stress": 1, "frustration": 1},   # non-numeric
        {"confidence": float("nan"), "stress": 1, "frustration": 1},
        "not a dict",
        None,
    ])
    def test_unusable_mood_falls_back_to_the_formula(self, bad):
        agent = _agent("balanced")
        agent.base_agent.reported = bad
        agent.act(_obs(wealth=8000.0))
        assert set(agent.mood) == set(MOOD_AXES)
        assert all(0 <= v <= 10 for v in agent.mood.values())


class TestFormulaFallback:
    def test_drawdown_raises_stress(self):
        agent = _agent("balanced")
        agent.act(_obs(wealth=12000.0))          # sets the peak
        before = agent.mood["stress"]
        agent.act(_obs(wealth=8000.0))           # 33% off the peak
        assert agent.mood["stress"] > before

    def test_a_green_round_raises_confidence(self):
        agent = _agent("balanced")
        agent.act(_obs(wealth=10000.0))
        before = agent.mood["confidence"]
        agent.act(_obs(wealth=13000.0))
        assert agent.mood["confidence"] > before

    def test_flat_round_decays_toward_baseline(self):
        agent = _agent("balanced")
        agent.mood["stress"] = 9.0
        agent.act(_obs(wealth=10000.0))
        agent.act(_obs(wealth=10000.0))
        assert agent.mood["stress"] < 9.0

    def test_rival_gap_raises_frustration(self):
        agent = _agent("balanced")
        agent.act(_obs())
        before = agent.mood["frustration"]
        agent.act(_obs(standing={
            "rank": 5, "of": 6, "my_return_pct": -8.0,
            "leader_name": "Nova", "leader_return_pct": 22.0,
            "gap_to_leader_pct": 30.0,
        }))
        assert agent.mood["frustration"] > before

    def test_mood_intensity_scales_the_move(self):
        calm = _agent("balanced", mood_intensity=0.1)
        hot = _agent("balanced", mood_intensity=2.0)
        for agent in (calm, hot):
            agent.act(_obs(wealth=12000.0))
            agent.act(_obs(wealth=7000.0))
        assert hot.mood["stress"] > calm.mood["stress"]


class TestDecisionUntouched:
    def test_decisions_identical_with_mood_on(self):
        agent = _agent("panicky")
        obs = _obs(wealth=6000.0)
        expected = agent.base_agent.act(obs)["decisions"]
        assert agent.act(obs)["decisions"] == expected

    def test_mood_attached_to_the_result(self):
        agent = _agent("greedy")
        result = agent.act(_obs())
        assert set(result["mood"]) == set(MOOD_AXES)

    def test_simple_mode_has_no_mood_and_no_persona(self):
        agent = _agent("panicky", deep=False)
        obs = _obs()
        result = agent.act(obs)
        assert "mood" not in result
        assert "persona" not in obs

    def test_deep_mode_attaches_persona_to_the_observation(self):
        agent = _agent("panicky")
        obs = _obs()
        agent.act(obs)
        persona = obs["persona"]
        assert persona["disposition"] == agent.disposition
        assert set(persona["mood"]) == set(MOOD_AXES)
        assert "dials" in persona


class TestSnapshotAndPlainAgents:
    def test_state_snapshot_carries_mood_in_deep_mode(self):
        config = MarketConfig(initial_price=100.0, deep_persona=True,
                              event_probability_multiplier=0.0)
        agent = create_personality_agent(
            ScriptedExternalAIAgent("a", cash=1000, holdings=10),
            "aggressive", deep=True,
        )
        env = MarketEnv(config, [agent], seed=1)
        state = env.step()
        assert set(state["agents"]["a"]["mood"]) == set(MOOD_AXES)
        env.close()

    def test_snapshot_mood_is_none_in_simple_mode(self):
        config = MarketConfig(initial_price=100.0, deep_persona=False,
                              event_probability_multiplier=0.0)
        agent = create_personality_agent(
            ScriptedExternalAIAgent("a", cash=1000, holdings=10), "aggressive"
        )
        env = MarketEnv(config, [agent], seed=1)
        assert env.step()["agents"]["a"]["mood"] is None
        env.close()

    def test_plain_agent_without_mood_does_not_break_snapshots(self):
        config = MarketConfig(initial_price=100.0, deep_persona=True,
                              event_probability_multiplier=0.0)
        env = MarketEnv(
            config, [ScriptedExternalAIAgent("plain", cash=1000, holdings=10)],
            seed=1,
        )
        assert env.step()["agents"]["plain"]["mood"] is None
        env.close()

    def test_traitagent_built_directly_still_works(self):
        agent = TraitAgent(ScriptedExternalAIAgent("d", cash=100, holdings=1))
        assert agent.act(_obs())["decisions"]

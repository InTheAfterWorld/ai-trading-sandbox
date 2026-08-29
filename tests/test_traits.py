"""Tests for TraitAgent: state sync, pass-through decisions, dispositions."""

import pytest

from ai_trading_society.agents.external_ai_agent import (
    _REASONING_DETAIL_DEEP,
    _REASONING_DETAIL_SIMPLE,
    ExternalAIAgent,
)
from ai_trading_society.agents.traits import (
    _PERSONALITY_DISPOSITIONS,
    TraitAgent,
    build_disposition,
    create_personality_agent,
)
from tests.conftest import ScriptedExternalAIAgent

PRESETS = [
    "balanced", "aggressive", "conservative", "panicky",
    "greedy", "fomo_driven", "stubborn", "emotional",
]


def _obs(step, price, price_history, my_cash, my_holdings, my_wealth):
    """Build a multi-stock observation dict."""
    return {
        "step": step,
        "stocks": [{
            "symbol": "ATSX",
            "name": "Stock 1",
            "price": price,
            "price_history": price_history,
            "last_volume": 0,
            "my_holdings": my_holdings,
        }],
        "my_cash": my_cash,
        "my_holdings": {"ATSX": my_holdings},
        "my_wealth": my_wealth,
        "market_sentiment": 0.0,
    }


class TestStateSync:
    """Test that TraitAgent stays in sync with its base_agent."""

    def test_cash_delegation(self):
        """Setting TraitAgent.cash should update base_agent.cash."""
        base = ScriptedExternalAIAgent("base", cash=10000, holdings=50)
        trait = TraitAgent(base)
        trait.cash = 5000
        assert base.cash == 5000, "base_agent.cash should sync with TraitAgent.cash"

    def test_holdings_delegation(self):
        """Setting TraitAgent.holdings should update base_agent.holdings."""
        base = ScriptedExternalAIAgent("base", cash=10000, holdings=50)
        trait = TraitAgent(base)
        trait.holdings = 25
        assert base.holdings == 25, "base_agent.holdings should sync with TraitAgent.holdings"

    def test_env_update_propagates(self):
        """When MarketEnv updates TraitAgent.cash, base_agent should see it."""
        from ai_trading_society.config import MarketConfig
        from ai_trading_society.market_env import MarketEnv

        base = ScriptedExternalAIAgent(
            "trader", cash=10000, holdings=0, buy_prob=1.0, sell_prob=0.0
        )
        trait = TraitAgent(base)

        config = MarketConfig(initial_price=100.0)
        seller = ScriptedExternalAIAgent(
            "seller", cash=0, holdings=100, buy_prob=0.0, sell_prob=1.0
        )
        env = MarketEnv(config, [trait, seller])
        env.step()

        # After a buy, cash should decrease in both trait and base
        assert trait.cash == base.cash, "TraitAgent and base_agent cash should match"
        assert trait.holdings == base.holdings, "TraitAgent and base_agent holdings should match"
        env.close()


class TestDecisionPassThrough:
    """The personality reaches the model via the prompt; act() never edits it.

    This is the core guarantee: an agent's action can no longer contradict
    the reasoning printed beside it.
    """

    def test_decisions_are_returned_unchanged(self):
        """act() must hand back exactly what the base agent produced."""
        base = ScriptedExternalAIAgent(
            "t", cash=10000, holdings=100, buy_prob=0.0, sell_prob=1.0
        )
        trait = create_personality_agent(base, "panicky")

        obs = _obs(2, 80.0, [100, 95, 90, 85, 80], 0.0, 100, 8000.0)
        expected = base.act(obs)
        assert trait.act(obs) == expected

    @pytest.mark.parametrize("personality", PRESETS)
    def test_no_personality_alters_the_decision(self, personality):
        """Every preset is prompt-only -- none of them rewrite an action."""
        base = ScriptedExternalAIAgent(
            "t", cash=10000, holdings=100, buy_prob=1.0, sell_prob=0.0
        )
        trait = create_personality_agent(base, personality)

        # A rally after a drawdown: the shape that used to trip panic,
        # FOMO, loss aversion and overconfidence all at once.
        trait.act(_obs(1, 100.0, [100] * 10, 10000.0, 100, 20000.0))
        obs = _obs(2, 105.0, [100, 90, 95, 100, 105], 10000.0, 100, 15000.0)
        assert trait.act(obs) == base.act(obs)

    def test_reasoning_is_never_prefixed(self):
        """No more [trait override] / [social] tags on the model's words."""
        base = ScriptedExternalAIAgent("t", cash=10000, holdings=100, sell_prob=1.0)
        trait = create_personality_agent(base, "emotional")
        trait.act(_obs(1, 100.0, [100] * 10, 10000.0, 100, 20000.0))
        result = trait.act(_obs(2, 70.0, [100, 90, 80, 75, 70], 10000.0, 100, 12000.0))
        for decision in result["decisions"]:
            assert "[trait override]" not in decision["reasoning"]
            assert "[social]" not in decision["reasoning"]

    def test_wealth_tracking_still_runs(self):
        """Peak/initial wealth are tracked for later phases, harmlessly."""
        base = ScriptedExternalAIAgent("t", cash=0, holdings=100)
        trait = TraitAgent(base)
        trait.act(_obs(1, 100.0, [100] * 10, 0.0, 100, 10000.0))
        trait.act(_obs(2, 120.0, [100, 110, 120], 0.0, 100, 12000.0))
        trait.act(_obs(3, 80.0, [120, 100, 80], 0.0, 100, 8000.0))
        assert trait._initial_wealth == 10000.0
        assert trait._peak_wealth == 12000.0


class TestDisposition:
    """The persona text handed to the model."""

    def test_identity_lines_always_present(self):
        text = build_disposition("balanced")
        assert "not a calculator" in text
        assert "reasoning and your action must agree" in text

    def test_simple_uses_the_one_liner(self):
        text = build_disposition("panicky", deep=False)
        assert _PERSONALITY_DISPOSITIONS["panicky"]["short"] in text
        assert _PERSONALITY_DISPOSITIONS["panicky"]["full"] not in text

    def test_deep_uses_the_full_paragraph(self):
        text = build_disposition("panicky", deep=True)
        assert _PERSONALITY_DISPOSITIONS["panicky"]["full"] in text

    @pytest.mark.parametrize("personality", PRESETS)
    def test_every_preset_has_both_depths(self, personality):
        entry = _PERSONALITY_DISPOSITIONS[personality]
        assert entry["short"] and entry["full"]
        assert len(entry["full"]) > len(entry["short"])

    def test_unknown_personality_falls_back_to_balanced(self):
        assert build_disposition("nonsense") == build_disposition("balanced")

    def test_dispositions_differ_between_presets(self):
        shorts = {p: _PERSONALITY_DISPOSITIONS[p]["short"] for p in PRESETS}
        assert len(set(shorts.values())) == len(PRESETS)


class TestPersonalityAgentWiring:
    """create_personality_agent writes the persona into the system prompt."""

    def test_system_prompt_carries_persona_and_rules(self):
        base = ExternalAIAgent("r", api_key="k", enable_memory=False)
        agent = create_personality_agent(base, "greedy")
        prompt = base.system_prompt
        assert agent.disposition in prompt
        # The JSON contract must survive the persona being prepended.
        assert "OUTPUT FORMAT" in prompt
        assert '"decisions"' in prompt

    def test_simple_asks_for_short_reasoning(self):
        base = ExternalAIAgent("r", api_key="k", enable_memory=False)
        create_personality_agent(base, "greedy", deep=False)
        assert _REASONING_DETAIL_SIMPLE in base.system_prompt
        assert _REASONING_DETAIL_DEEP not in base.system_prompt

    def test_deep_asks_for_longer_in_character_reasoning(self):
        base = ExternalAIAgent("r", api_key="k", enable_memory=False)
        create_personality_agent(base, "greedy", deep=True)
        assert _REASONING_DETAIL_DEEP in base.system_prompt

    def test_deep_flag_is_recorded(self):
        base = ExternalAIAgent("r", api_key="k", enable_memory=False)
        assert create_personality_agent(base, "greedy", deep=True).deep is True
        base2 = ExternalAIAgent("r2", api_key="k", enable_memory=False)
        assert create_personality_agent(base2, "greedy").deep is False

    def test_base_without_prompt_api_is_left_alone(self):
        """A plain agent with no system_prompt must not crash the factory."""
        from ai_trading_society.base_agent import BaseAgent

        class _Plain(BaseAgent):
            def act(self, observation):
                return {"decisions": []}

        agent = create_personality_agent(_Plain("p", 100.0, {}), "aggressive")
        assert agent.personality_name == "aggressive"
        assert agent.disposition


class TestPersonalityPresets:
    """Test personality preset creation."""

    def test_create_balanced_agent(self):
        base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
        agent = create_personality_agent(base, "balanced")
        assert agent.personality_name == "balanced"
        assert _PERSONALITY_DISPOSITIONS["balanced"]["short"] in agent.disposition

    def test_create_aggressive_agent(self):
        base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
        agent = create_personality_agent(base, "aggressive")
        assert _PERSONALITY_DISPOSITIONS["aggressive"]["short"] in agent.disposition

    def test_unknown_personality_defaults_to_balanced(self):
        base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
        agent = create_personality_agent(base, "unknown_type")
        assert _PERSONALITY_DISPOSITIONS["balanced"]["short"] in agent.disposition


class TestPersonalityDisplay:
    """Test personality name storage and description."""

    def test_personality_name_stored(self):
        """create_personality_agent should store the personality name."""
        base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
        agent = create_personality_agent(base, "aggressive")
        assert agent.personality_name == "aggressive"

    def test_default_personality_name_is_custom(self):
        """TraitAgent without personality_name should default to 'custom'."""
        base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
        agent = TraitAgent(base)
        assert agent.personality_name == "custom"

    def test_personality_description_aggressive(self):
        """Aggressive personality should have a descriptive string."""
        base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
        agent = create_personality_agent(base, "aggressive")
        desc = agent.personality_description
        assert "Aggressive" in desc
        assert "overconfident" in desc

    def test_personality_description_all_presets(self):
        """Every preset should have a non-empty description."""
        for name in PRESETS:
            base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
            agent = create_personality_agent(base, name)
            assert len(agent.personality_description) > 0, \
                f"Personality '{name}' should have a description"

    def test_repr_names_the_personality(self):
        base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
        agent = create_personality_agent(base, "stubborn")
        assert "stubborn" in repr(agent)

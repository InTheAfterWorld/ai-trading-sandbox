"""Tests for TraitAgent: state sync, regret avoidance, and other traits."""


from ai_trading_society.agents.traits import TraitAgent, create_personality_agent
from tests.conftest import ScriptedExternalAIAgent


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


class TestRegretAvoidance:
    """Test that regret_avoidance uses actual initial wealth, not hardcoded 10000."""

    def test_uses_actual_initial_wealth(self):
        """Agent with 50000 initial should use 50000 as threshold, not 10000."""
        base = ScriptedExternalAIAgent(
            "rich", cash=50000, holdings=100, buy_prob=0.0, sell_prob=1.0
        )
        trait = TraitAgent(base, regret_avoidance=1.0)

        # First act to capture initial wealth
        obs1 = {
            "step": 1,
            "price": 100.0,
            "price_history": [100] * 10,
            "my_cash": 50000.0,
            "my_holdings": 100,
            "my_wealth": 60000.0,  # 50000 + 100*100
            "market_sentiment": 0.0,
        }
        trait.act(obs1)

        # Now simulate being below threshold: wealth must be strictly < 57000
        # price = 69: wealth = 50000 + 100*69 = 56900 < 57000
        obs2 = {
            "step": 2,
            "price": 69.0,  # 50000 + 100*69 = 56900, which is < 60000*0.95=57000
            "price_history": [100, 95, 90, 85, 80, 75, 69],
            "my_cash": 50000.0,
            "my_holdings": 100,
            "my_wealth": 56900.0,
            "market_sentiment": 0.0,
        }
        # base agent wants to sell, but regret_avoidance=1.0 should block it
        action = trait.act(obs2)
        assert action["action"] == "hold", \
            "Regret avoidance should block selling when below initial wealth threshold"

    def test_does_not_trigger_when_wealth_above_threshold(self):
        """Agent with high wealth should still be able to sell."""
        base = ScriptedExternalAIAgent(
            "rich", cash=50000, holdings=100, buy_prob=0.0, sell_prob=1.0
        )
        trait = TraitAgent(base, regret_avoidance=1.0)

        obs1 = {
            "step": 1,
            "price": 100.0,
            "price_history": [100] * 10,
            "my_cash": 50000.0,
            "my_holdings": 100,
            "my_wealth": 60000.0,
            "market_sentiment": 0.0,
        }
        trait.act(obs1)

        # Wealth went UP, should still allow selling
        obs2 = {
            "step": 2,
            "price": 120.0,  # 50000 + 100*120 = 62000, above 60000*0.95=57000
            "price_history": [100, 105, 110, 115, 120],
            "my_cash": 50000.0,
            "my_holdings": 100,
            "my_wealth": 62000.0,
            "market_sentiment": 0.0,
        }
        action = trait.act(obs2)
        assert action["action"] == "sell", \
            "Regret avoidance should NOT block selling when above threshold"


class TestPanic:
    """Test panic selling trait."""

    def test_panic_sells_on_drawdown(self):
        """Panic trait should trigger sell on significant drawdown."""
        base = ScriptedExternalAIAgent(
            "panicker", cash=0, holdings=100, buy_prob=0.0, sell_prob=0.0
        )
        trait = TraitAgent(base, panic=1.0)

        # First act: establish peak wealth
        obs1 = {
            "step": 1,
            "price": 100.0,
            "price_history": [100] * 10,
            "my_cash": 0.0,
            "my_holdings": 100,
            "my_wealth": 10000.0,
            "market_sentiment": 0.0,
        }
        trait.act(obs1)

        # Now: significant drawdown (price drops to 80, wealth = 8000, drawdown = 20%)
        obs2 = {
            "step": 2,
            "price": 80.0,
            "price_history": [100, 95, 90, 85, 80],
            "my_cash": 0.0,
            "my_holdings": 100,
            "my_wealth": 8000.0,
            "market_sentiment": 0.0,
        }
        action = trait.act(obs2)
        assert action["action"] == "sell", "Panic should trigger sell on >10% drawdown"
        assert action["quantity"] == 100, "Panic should sell all holdings"


class TestOverconfidence:
    """Test overconfidence trait."""

    def test_overconfidence_increases_size(self):
        """Overconfidence should increase trade quantity."""
        base = ScriptedExternalAIAgent("tf", cash=10000, holdings=0, buy_prob=1.0)
        trait = TraitAgent(base, overconfidence=1.0)

        obs = {
            "step": 1,
            "price": 105.0,
            "price_history": [100, 101, 102, 103, 104, 105],
            "my_cash": 10000.0,
            "my_holdings": 0,
            "my_wealth": 10000.0,
            "market_sentiment": 0.0,
        }

        # Run multiple times to catch the overconfidence trigger
        import random
        random.seed(42)
        actions = [trait.act(obs) for _ in range(20)]

        # At least one action should have a quantity larger than what
        # the base agent would produce
        base_action = base.act(obs)
        max_trait_qty = max(a["quantity"] for a in actions if a["quantity"] > 0)
        assert max_trait_qty > base_action["quantity"], \
            "Overconfidence should produce larger trade sizes at least sometimes"


class TestPersonalityPresets:
    """Test personality preset creation."""

    def test_create_balanced_agent(self):
        """Balanced personality should have no traits."""
        base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
        agent = create_personality_agent(base, "balanced")
        assert agent.panic == 0.0
        assert agent.greed == 0.0

    def test_create_aggressive_agent(self):
        """Aggressive personality should have overconfidence and greed."""
        base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
        agent = create_personality_agent(base, "aggressive")
        assert agent.overconfidence > 0
        assert agent.greed > 0

    def test_unknown_personality_defaults_to_balanced(self):
        """Unknown personality should create balanced agent."""
        base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
        agent = create_personality_agent(base, "unknown_type")
        assert agent.panic == 0.0


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
        presets = [
            "balanced", "aggressive", "conservative", "panicky",
            "greedy", "fomo_driven", "stubborn", "emotional",
        ]
        for name in presets:
            base = ScriptedExternalAIAgent("r", cash=10000, holdings=0)
            agent = create_personality_agent(base, name)
            assert len(agent.personality_description) > 0, \
                f"Personality '{name}' should have a description"

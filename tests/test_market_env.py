"""Tests for MarketEnv: price updates, order matching, transaction costs."""

import pytest

from ai_trading_society.config import MarketConfig
from ai_trading_society.market_env import MarketEnv
from tests.conftest import ScriptedExternalAIAgent


class TestPriceUpdates:
    """Test price update mechanism."""

    def test_price_increases_with_net_buying(self, classic_config):
        """Price should rise when buy demand exceeds sell supply."""
        agents = [
            ScriptedExternalAIAgent("buyer", cash=100000, holdings=0, buy_prob=1.0, sell_prob=0.0),
            ScriptedExternalAIAgent("seller", cash=0, holdings=10, buy_prob=0.0, sell_prob=1.0),
        ]
        env = MarketEnv(classic_config, agents)
        initial_price = env.price
        env.step()
        assert env.price > initial_price, "Price should increase with net buying"

    def test_price_decreases_with_net_selling(self, classic_config):
        """Price should fall when sell supply exceeds buy demand."""
        agents = [
            ScriptedExternalAIAgent("buyer", cash=100, holdings=0, buy_prob=1.0, sell_prob=0.0),
            ScriptedExternalAIAgent("seller", cash=0, holdings=1000, buy_prob=0.0, sell_prob=1.0),
        ]
        env = MarketEnv(classic_config, agents)
        initial_price = env.price
        env.step()
        assert env.price < initial_price, "Price should decrease with net selling"

    def test_price_clamped_to_max_ratio(self, classic_config):
        """Single-step price change should not exceed max_price_change_ratio."""
        config = MarketConfig(
            initial_price=100.0,
            price_sensitivity=1.0,  # Very high sensitivity
            max_price_change_ratio=0.06,
        )
        agents = [
            ScriptedExternalAIAgent("buyer", cash=1_000_000, holdings=0, buy_prob=1.0, sell_prob=0.0),
            ScriptedExternalAIAgent("seller", cash=0, holdings=1, buy_prob=0.0, sell_prob=1.0),
        ]
        env = MarketEnv(config, agents)
        initial_price = env.price
        env.step()
        change_ratio = abs(env.price - initial_price) / initial_price
        assert change_ratio <= 0.06 + 1e-9, "Price change should be clamped"

    def test_price_history_grows(self, simple_env):
        """Price history should grow by one entry per step."""
        initial_len = len(simple_env.price_history)
        simple_env.step()
        assert len(simple_env.price_history) == initial_len + 1


class TestOrderMatching:
    """Test order matching logic."""

    def test_buyer_cash_decreases_after_buy(self, simple_env):
        """Buyer's cash should decrease after a successful buy."""
        buyer = simple_env.agents["buyer"]
        initial_cash = buyer.cash
        simple_env.step()
        assert buyer.cash < initial_cash, "Buyer cash should decrease"

    def test_seller_holdings_decrease_after_sell(self, simple_env):
        """Seller's holdings should decrease after a successful sell."""
        seller = simple_env.agents["seller"]
        initial_holdings = seller.holdings
        simple_env.step()
        assert seller.holdings < initial_holdings, "Seller holdings should decrease"

    def test_trade_history_records_trades(self, simple_env):
        """Trade history should contain records after a step."""
        simple_env.step()
        assert len(simple_env.trade_history) > 0, "Trades should be recorded"

    def test_matched_volume_recorded(self, simple_env):
        """Volume history should record the matched volume."""
        simple_env.step()
        assert len(simple_env.volume_history) == 1
        assert simple_env.volume_history[0] > 0, "Volume should be positive"

    def test_buy_quantity_clipped_to_affordable(self, classic_config):
        """Buy orders should be clipped to what the agent can afford."""
        agents = [
            ScriptedExternalAIAgent("buyer", cash=50, holdings=0, buy_prob=1.0, sell_prob=0.0),
            ScriptedExternalAIAgent("seller", cash=0, holdings=1000, buy_prob=0.0, sell_prob=1.0),
        ]
        env = MarketEnv(classic_config, agents)
        env.step()
        buyer = env.agents["buyer"]
        # With $50 and price ~$100, can only buy 0 shares
        assert buyer.holdings == 0 or buyer.cash >= 0, "Buy should be clipped to affordable"


class TestTransactionCosts:
    """Test fee and slippage application."""

    def test_fee_reduces_buyer_cash_more(self, fee_config):
        """Buyer should pay trade value + fee."""
        agents = [
            ScriptedExternalAIAgent("buyer", cash=100000, holdings=0, buy_prob=1.0, sell_prob=0.0),
            ScriptedExternalAIAgent("seller", cash=0, holdings=100, buy_prob=0.0, sell_prob=1.0),
        ]
        env_with_fee = MarketEnv(fee_config, agents)

        # Same setup without fees
        config_no_fee = MarketConfig(
            initial_price=100.0,
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        agents2 = [
            ScriptedExternalAIAgent("buyer", cash=100000, holdings=0, buy_prob=1.0, sell_prob=0.0),
            ScriptedExternalAIAgent("seller", cash=0, holdings=100, buy_prob=0.0, sell_prob=1.0),
        ]
        env_no_fee = MarketEnv(config_no_fee, agents2)

        # Use fixed random seed for comparison
        import random
        random.seed(42)
        env_with_fee.step()
        random.seed(42)
        env_no_fee.step()

        buyer_fee = env_with_fee.agents["buyer"]
        buyer_nofee = env_no_fee.agents["buyer"]
        assert buyer_fee.cash < buyer_nofee.cash, "Buyer with fees should have less cash"

    def test_slippage_worsens_execution_price(self, fee_config):
        """Slippage should make buyers pay more per share."""
        agents = [
            ScriptedExternalAIAgent("buyer", cash=100000, holdings=0, buy_prob=1.0, sell_prob=0.0),
            ScriptedExternalAIAgent("seller", cash=0, holdings=100, buy_prob=0.0, sell_prob=1.0),
        ]
        env = MarketEnv(fee_config, agents)
        env.step()

        # Find the buyer's trade record
        buyer_trades = [t for t in env.trade_history if t.agent_id == "buyer"]
        assert len(buyer_trades) > 0
        # With slippage, buyer's trade price should be higher than market price
        # (the market price before the trade was 100.0)
        assert buyer_trades[0].price > 100.0, "Slippage should increase buy price"


class TestObservations:
    """Test observation generation."""

    def test_observation_contains_required_keys(self, simple_env):
        """Observation should contain all required keys."""
        obs = simple_env.get_observation("buyer")
        required_keys = {"step", "price", "price_history", "my_cash",
                         "my_holdings", "my_wealth", "last_volume", "market_sentiment"}
        assert required_keys.issubset(obs.keys()), f"Missing keys: {required_keys - obs.keys()}"

    def test_sandbox_observation_has_event_data(self, simple_env):
        """Unified sandbox observations always expose event data and sentiment."""
        obs = simple_env.get_observation("buyer")
        assert "active_events" in obs
        assert "market_sentiment" in obs


class TestStateSnapshot:
    """Test state snapshot generation."""

    def test_state_contains_agent_data(self, simple_env):
        """State should contain wealth data for each agent."""
        state = simple_env.step()
        assert "agents" in state
        assert "buyer" in state["agents"]
        assert "wealth" in state["agents"]["buyer"]

    def test_state_contains_price_and_volume(self, simple_env):
        """State should contain price and volume."""
        state = simple_env.step()
        assert "price" in state
        assert "matched_volume" in state
        assert "step" in state

    def test_state_contains_market_pressure(self, simple_env):
        """State should contain total_buy and total_sell."""
        state = simple_env.step()
        assert "total_buy" in state
        assert "total_sell" in state
        assert state["total_buy"] > 0
        assert state["total_sell"] > 0

    def test_agent_actions_contain_reasoning_key(self, simple_env):
        """agent_actions should contain a reasoning key (empty for non-AI agents)."""
        state = simple_env.step()
        actions = state.get("agent_actions", {})
        for agent_id, act_info in actions.items():
            assert "reasoning" in act_info

    def test_sandbox_has_active_events_detail(self, simple_env):
        """Unified sandbox state always includes active event details."""
        state = simple_env.step()
        assert "active_events_detail" in state
        assert isinstance(state["active_events_detail"], list)

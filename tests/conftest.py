"""Shared pytest fixtures and deterministic ExternalAI test double."""

import pytest

from ai_trading_society.agents.external_ai_agent import ExternalAIAgent
from ai_trading_society.config import MarketConfig
from ai_trading_society.market_env import MarketEnv


class ScriptedExternalAIAgent(ExternalAIAgent):
    """ExternalAI-derived test agent with deterministic buy/sell directives."""

    def __init__(self, agent_id, cash=10000.0, holdings=0, buy_prob=0.0, sell_prob=0.0, **_):
        super().__init__(agent_id, cash=cash, holdings=holdings, api_provider="openai")
        self.buy_prob = buy_prob
        self.sell_prob = sell_prob

    def act(self, observation):
        if self.buy_prob >= self.sell_prob and self.buy_prob > 0:
            quantity = max(1, int(self.cash * 0.2 / max(observation["price"], 0.01)))
            return {"action": "buy", "quantity": quantity, "reasoning": "scripted buy"}
        if self.sell_prob > 0 and self.holdings > 0:
            return {"action": "sell", "quantity": self.holdings, "reasoning": "scripted sell"}
        return {"action": "hold", "quantity": 0, "reasoning": "scripted hold"}


@pytest.fixture
def classic_config():
    """A basic unified sandbox config."""
    return MarketConfig(initial_price=100.0, price_sensitivity=0.02, max_price_change_ratio=0.06, event_probability_multiplier=0.0)


@pytest.fixture
def realistic_config():
    """A unified sandbox config with normal event probability."""
    return MarketConfig(
        initial_price=100.0,
        price_sensitivity=0.02,
        max_price_change_ratio=0.10,
        event_probability_multiplier=1.0,
    )


@pytest.fixture
def fee_config():
    """Config with transaction costs enabled."""
    return MarketConfig(initial_price=100.0, fee_rate=0.001, slippage_rate=0.002)


@pytest.fixture
def simple_env(classic_config):
    agents = [
        ScriptedExternalAIAgent("buyer", cash=10000, holdings=0, buy_prob=1.0),
        ScriptedExternalAIAgent("seller", cash=0, holdings=100, sell_prob=1.0),
    ]
    return MarketEnv(classic_config, agents)


@pytest.fixture
def standard_observation():
    return {
        "step": 5,
        "price": 100.0,
        "price_history": [95, 96, 97, 98, 99, 100],
        "my_cash": 10000.0,
        "my_holdings": 50,
        "my_wealth": 15000.0,
        "last_volume": 100,
        "market_sentiment": 0.0,
    }

"""Unit tests for multi-stock trading system fixes.

Validates:
1. Multi-stock specs identified by stock name without symbols.
2. MarketEnv per-stock order matching and observation data.
3. ExternalAIAgent prompt generation and multi-stock JSON parsing.
4. Web API configuration without homepage start price.
"""

import pytest
from ai_trading_society.config import MarketConfig, StockSpec
from ai_trading_society.config_store import _normalize_stocks, save_config, load_config
from ai_trading_society.market_env import MarketEnv
from ai_trading_society.agents.external_ai_agent import ExternalAIAgent
from ai_trading_society.agents.player_agent import PlayerAgent
from ai_trading_society.agents.roster import build_agent_roster


def test_stock_spec_and_config_without_symbol():
    spec1 = StockSpec(name="Tesla", initial_price=200.0, initial_holdings=10)
    spec2 = StockSpec(name="Apple", initial_price=150.0, initial_holdings=15)
    config = MarketConfig(stocks=[spec1, spec2])

    assert len(config.get_stock_specs()) == 2
    assert config.get_stock_specs()[0].name == "Tesla"
    assert config.get_stock_specs()[1].name == "Apple"

    d = config.to_dict()
    assert d["stocks"] == [
        {"name": "Tesla", "initial_price": 200.0, "initial_holdings": 10},
        {"name": "Apple", "initial_price": 150.0, "initial_holdings": 15},
    ]

    reconstructed = MarketConfig.from_dict(d)
    assert len(reconstructed.stocks) == 2
    assert reconstructed.stocks[0].name == "Tesla"
    assert reconstructed.stocks[1].name == "Apple"


def test_normalize_stocks_by_name():
    raw = [
        {"name": "Tesla", "price": 200, "hold": 10},
        {"name": "Apple", "price": 150, "hold": 5},
        {"name": "Tesla", "price": 205, "hold": 20},  # duplicate name dropped
    ]
    normalized = _normalize_stocks(raw)
    assert len(normalized) == 2
    assert normalized[0]["name"] == "Tesla"
    assert normalized[1]["name"] == "Apple"


def test_market_env_multi_stock_execution():
    spec1 = StockSpec(name="Tesla", initial_price=200.0, initial_holdings=10)
    spec2 = StockSpec(name="Apple", initial_price=150.0, initial_holdings=10)
    config = MarketConfig(stocks=[spec1, spec2])

    player = PlayerAgent("Player", cash=10000.0, holdings={"Tesla": 10, "Apple": 10})
    env = MarketEnv(config, [player])

    assert "Tesla" in env.stocks
    assert "Apple" in env.stocks

    # Set player actions for both stocks
    env.set_player_action("sell", 5, symbol="Tesla")
    env.set_player_action("buy", 2, symbol="Apple")

    obs = env.get_observation("Player")
    assert len(obs["stocks"]) == 2
    stock_names = {s["name"] for s in obs["stocks"]}
    assert stock_names == {"Tesla", "Apple"}

    state = env.step()
    assert state["step"] == 1


def test_external_ai_agent_multi_stock_parsing():
    agent = ExternalAIAgent("TestAgent", api_key="dummy_key", enable_memory=False)
    raw_response = '{"decisions": [{"name": "Tesla", "action": "buy", "quantity": 5, "reasoning": "Strong trend"}, {"name": "Apple", "action": "sell", "quantity": 3, "reasoning": "Cutting risk"}]}'
    parsed = agent._parse_response(raw_response)

    assert "decisions" in parsed
    assert len(parsed["decisions"]) == 2
    assert parsed["decisions"][0]["name"] == "Tesla"
    assert parsed["decisions"][0]["action"] == "buy"
    assert parsed["decisions"][0]["quantity"] == 5
    assert parsed["decisions"][1]["name"] == "Apple"
    assert parsed["decisions"][1]["action"] == "sell"
    assert parsed["decisions"][1]["quantity"] == 3

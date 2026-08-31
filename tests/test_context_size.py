"""What an agent re-sends each round.

The conversation history used to replay every past round's prompt verbatim,
so most of a late-run request was duplicated market data the new prompt
already summarized. Requests grew 5.5x over twenty rounds, and a slow
request is what trips the step timeout. These tests hold the replacement in
place: recaps in, prompts out, and a history shape providers still accept.
"""

import json

from ai_trading_society.agents.external_ai_agent import ExternalAIAgent
from ai_trading_society.agents.traits import create_personality_agent
from ai_trading_society.config import MarketConfig, StockSpec
from ai_trading_society.market_env import MarketEnv

_REPLY = json.dumps({
    "decisions": [
        {"name": "TechTitan", "action": "buy", "quantity": 5,
         "reasoning": "Momentum looks constructive and I am sizing up."},
        {"name": "MegaBank", "action": "hold", "quantity": 0,
         "reasoning": "Nothing to do here yet."},
    ]
})


def build_env(rounds=0, deep=True, **agent_kwargs):
    base = ExternalAIAgent(
        "Ada", cash=10000, model="claude-sonnet-5", api_key="k", **agent_kwargs
    )
    base._call_ai_api = lambda p, messages=None, system_prompt=None: _REPLY
    agent = create_personality_agent(base, personality="aggressive", deep=deep)
    env = MarketEnv(
        MarketConfig(
            stocks=[StockSpec(name="TechTitan", initial_price=150.0),
                    StockSpec(name="MegaBank", initial_price=250.0)],
            seed=5, deep_persona=deep,
        ),
        [agent],
    )
    for _ in range(rounds):
        env.step()
    return env, base


def request_chars(base, prompt=""):
    """Total characters a trading request would carry."""
    total = len(base.system_prompt) + len(prompt)
    return total + sum(len(m["content"]) for m in base._conversation_history)


class TestConversationHistory:
    def test_stores_a_recap_not_the_prompt(self):
        _, base = build_env(rounds=3)
        user_turns = [
            m for m in base._conversation_history if m["role"] == "user"
        ]
        assert user_turns
        for turn in user_turns:
            assert turn["content"].startswith("[Round ")
            # The giveaway headings of a real prompt must not be in there.
            assert "Market Data (Step" not in turn["content"]
            assert "=== WHO YOU ARE ===" not in turn["content"]

    def test_recap_keeps_the_facts_that_anchor_a_reply(self):
        _, base = build_env(rounds=2)
        recap = [m for m in base._conversation_history if m["role"] == "user"][-1]
        text = recap["content"]
        assert "Round" in text
        assert "TechTitan $" in text and "MegaBank $" in text
        assert "your wealth $" in text

    def test_recap_is_far_smaller_than_a_prompt(self):
        env, base = build_env(rounds=2)
        recap = [m for m in base._conversation_history if m["role"] == "user"][-1]
        full_prompt = base._build_prompt(env.get_observation("Ada"))
        assert len(recap["content"]) < len(full_prompt) / 5

    def test_assistant_replies_are_kept_verbatim(self):
        # The model's own words are what carry continuity; only the question
        # is compressed.
        _, base = build_env(rounds=2)
        replies = [
            m for m in base._conversation_history if m["role"] == "assistant"
        ]
        assert replies and all(m["content"] == _REPLY for m in replies)

    def test_history_alternates_and_opens_on_a_user_turn(self):
        # Some providers reject a history that starts on an assistant turn.
        _, base = build_env(rounds=6)
        roles = [m["role"] for m in base._conversation_history]
        assert roles[0] == "user"
        assert all(a != b for a, b in zip(roles, roles[1:]))

    def test_memory_disabled_stores_nothing(self):
        _, base = build_env(rounds=3, enable_memory=False)
        assert base._conversation_history == []


class TestMemoryWindow:
    def test_default_is_three_rounds(self):
        agent = ExternalAIAgent("A", model="claude-sonnet-5", api_key="k")
        assert agent.memory_window == 3

    def test_history_is_capped_to_the_window(self):
        _, base = build_env(rounds=10)
        assert len(base._conversation_history) == base.memory_window * 2

    def test_short_term_summaries_share_the_window(self):
        _, base = build_env(rounds=10)
        assert len(base._short_term_memory) == base.memory_window


class TestRequestGrowth:
    def test_request_plateaus_instead_of_compounding(self):
        _, base = build_env(rounds=1)
        first = request_chars(base)
        _, base20 = build_env(rounds=20)
        twentieth = request_chars(base20)
        # It used to grow 5.5x across a run; the history is now bounded by
        # recaps rather than by whole prompts.
        assert twentieth < first * 2.5

    def test_history_is_a_minority_of_the_request(self):
        env, base = build_env(rounds=20)
        prompt = base._build_prompt(env.get_observation("Ada"))
        history = sum(len(m["content"]) for m in base._conversation_history)
        assert history < request_chars(base, prompt) / 2

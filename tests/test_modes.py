"""The deep_persona flag: simple (default) vs deep prompt depth.

Off by default -- a lean prompt with a one-line disposition. On, the model
gets the full personality paragraph and is asked for longer, in-character
reasoning. Nothing else about the simulation changes.
"""

import pytest

import ai_trading_society.config_store as config_store
from ai_trading_society.agents.external_ai_agent import (
    _REASONING_DETAIL_DEEP,
    _REASONING_DETAIL_SIMPLE,
    ExternalAIAgent,
)
from ai_trading_society.agents.roster import build_agent_roster
from ai_trading_society.agents.traits import _PERSONALITY_DISPOSITIONS
from ai_trading_society.config import MarketConfig
from ai_trading_society.config_store import DEFAULT_CONFIG, load_config, save_config
from ai_trading_society.web.app import app


@pytest.fixture
def cfg_path(monkeypatch, tmp_path):
    path = tmp_path / "user_config.json"
    monkeypatch.setattr(config_store, "CONFIG_PATH", path)
    return path


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _roster_prompt(deep, personality="panicky"):
    """System prompt of the first AI trader built at the given depth."""
    agents, _ = build_agent_roster(
        trader_configs=[{"name": "T", "api_key": "k", "personality": personality}],
        include_player=False,
        deep_persona=deep,
    )
    return agents[0].base_agent.system_prompt


class TestSimpleIsTheDefault:
    def test_config_default_is_off(self):
        assert MarketConfig().deep_persona is False
        assert DEFAULT_CONFIG["deep_persona"] is False

    def test_fresh_config_file_is_off(self, cfg_path):
        assert load_config(cfg_path)["deep_persona"] is False

    def test_roster_defaults_to_simple(self):
        prompt = _roster_prompt(deep=False)
        assert _REASONING_DETAIL_SIMPLE in prompt
        assert _REASONING_DETAIL_DEEP not in prompt


class TestPromptDepth:
    def test_simple_uses_the_one_liner(self):
        prompt = _roster_prompt(deep=False)
        assert _PERSONALITY_DISPOSITIONS["panicky"]["short"] in prompt
        assert _PERSONALITY_DISPOSITIONS["panicky"]["full"] not in prompt

    def test_deep_uses_the_full_paragraph(self):
        prompt = _roster_prompt(deep=True)
        assert _PERSONALITY_DISPOSITIONS["panicky"]["full"] in prompt
        assert _REASONING_DETAIL_DEEP in prompt

    def test_identity_lines_in_both_depths(self):
        for deep in (False, True):
            assert "not a calculator" in _roster_prompt(deep=deep)

    def test_json_contract_survives_in_both_depths(self):
        """The persona is prepended, never a replacement for the rules."""
        for deep in (False, True):
            prompt = _roster_prompt(deep=deep)
            assert "OUTPUT FORMAT" in prompt
            assert '"decisions"' in prompt

    def test_unwrapped_agent_keeps_the_lean_default(self):
        """A bare ExternalAIAgent is unaffected by the flag."""
        agent = ExternalAIAgent("solo", api_key="k", enable_memory=False)
        assert _REASONING_DETAIL_SIMPLE in agent.system_prompt


class TestConfigRoundTrip:
    def test_checkbox_round_trips(self, cfg_path):
        saved = save_config({"deep_persona": True}, path=cfg_path)
        assert saved["deep_persona"] is True
        assert load_config(cfg_path)["deep_persona"] is True

    def test_non_bool_keeps_the_default(self, cfg_path):
        saved = save_config({"deep_persona": "yes"}, path=cfg_path)
        assert saved["deep_persona"] is False

    def test_round_trips_through_the_api(self, client, cfg_path):
        client.post("/api/config", json={"deep_persona": True})
        assert client.get("/api/config").get_json()["config"]["deep_persona"] is True

    def test_start_reads_the_saved_flag(self, client, cfg_path):
        """A launch that omits deep_persona respects the homepage toggle."""
        save_config({"deep_persona": True}, path=cfg_path)
        res = client.post("/api/start", json={"steps": 2, "traders": []})
        assert res.status_code == 200

    def test_market_config_serialises_the_flag(self):
        restored = MarketConfig.from_dict(MarketConfig(deep_persona=True).to_dict())
        assert restored.deep_persona is True

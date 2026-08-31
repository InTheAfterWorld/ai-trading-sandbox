"""Tests for Web UI Flask endpoints."""

import pytest

from ai_trading_society.web.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_and_sim_routes(client):
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert b"AI Trading" in res_index.data

    res_sim = client.get("/sim")
    assert res_sim.status_code == 200
    assert b"Simulation" in res_sim.data


def test_api_start_and_step(client):
    start_res = client.post("/api/start", json={"steps": 5, "traders": []})
    assert start_res.status_code == 200
    start_data = start_res.get_json()
    assert "roster" in start_data

    step_res = client.post("/api/step")
    assert step_res.status_code == 200
    step_data = step_res.get_json()
    assert step_data.get("step") == 1
    assert "price" in step_data
    assert "agents" in step_data


@pytest.fixture
def tmp_config_path(monkeypatch, tmp_path):
    cfg_path = tmp_path / "user_config.json"
    import ai_trading_society.config_store as config_store
    monkeypatch.setattr(config_store, "CONFIG_PATH", cfg_path)
    return cfg_path


def test_api_config_get_empty(client, tmp_config_path):
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.get_json()
    assert "config" in data
    assert data["config"]["steps"] == 30
    assert data["config"]["traders"] == []


def test_api_config_post_and_get(client, tmp_config_path):
    payload = {
        "steps": 12,
        "price": 250,
        "cash": 5000,
        "hold": 10,
        "traders": [
            {"name": "A", "provider": "groq", "model": "groq/compound-mini", "api_key": "k"}
        ],
    }
    post_res = client.post("/api/config", json=payload)
    assert post_res.status_code == 200
    saved = post_res.get_json()["config"]
    assert saved["steps"] == 12
    assert saved["price"] == 250
    # By default the homepage gets its keys back so it can repopulate on load
    # (the Host allowlist keeps this endpoint local-only). ATS_REDACT_CONFIG=1
    # opts out -- covered in test_secret_handling.py.
    assert saved["traders"][0]["api_key"] == "k"

    get_res = client.get("/api/config")
    data = get_res.get_json()["config"]
    assert data["steps"] == 12
    assert data["cash"] == 5000
    assert data["hold"] == 10
    assert data["traders"][0]["provider"] == "groq"
    assert data["traders"][0]["api_key"] == "k"


def test_api_config_persists_homepage_adjustments(client, tmp_config_path):
    """The homepage sends every adjustment through POST /api/config; the two
    fields that were historically dropped (social-influence slider and the
    player-participation toggle) must round-trip through user_config.json."""
    post_res = client.post(
        "/api/config",
        json={"social_influence": "0.4", "player_participates": False},
    )
    assert post_res.status_code == 200
    saved = post_res.get_json()["config"]
    assert saved["social_influence"] == 0.4
    assert saved["player_participates"] is False

    # A fresh GET (as a new browser session would do) returns them too.
    get_res = client.get("/api/config")
    data = get_res.get_json()["config"]
    assert data["social_influence"] == 0.4
    assert data["player_participates"] is False


def test_api_env_keys_removed(client):
    res = client.get("/api/env_keys")
    assert res.status_code == 404


# --- Chat briefing: the background an agent carries into a conversation -----
# `chat` is monkeypatched at class level so these run without an API key: the
# assertion is about what the endpoint SENDS, not what a model replies.

def _capture_chat(monkeypatch):
    """Patch ExternalAIAgent.chat and return the dict it records into."""
    from ai_trading_society.agents.external_ai_agent import ExternalAIAgent

    captured = {}

    def fake_chat(self, message, system_prompt=None, history=None):
        captured["prompt"] = system_prompt
        captured["history"] = list(history or [])
        return "ok"

    monkeypatch.setattr(ExternalAIAgent, "chat", fake_chat)
    return captured


def _start_two_traders(client):
    return client.post("/api/start", json={
        "steps": 5,
        "deep_persona": True,
        "traders": [
            {"name": "Alice", "provider": "openai", "model": "gpt-4o",
             "api_key": "k", "personality": "aggressive"},
            {"name": "Bob", "provider": "openai", "model": "gpt-4o",
             "api_key": "k", "personality": "greedy"},
        ],
    })


def test_api_chat_sends_the_briefing(client, monkeypatch):
    captured = _capture_chat(monkeypatch)
    _start_two_traders(client)

    res = client.post("/api/chat", json={"agent_id": "Alice", "message": "hi"})
    assert res.status_code == 200
    assert res.get_json()["reply"] == "ok"

    prompt = captured["prompt"]
    assert "=== WHO YOU ARE ===" in prompt
    assert "=== WHERE YOU STAND ===" in prompt
    assert "Alice" in prompt
    # Relations resolved against the custom trader names from the homepage,
    # not the default model ids resolve_social_map starts from.
    assert "Bob" in prompt
    # The decision prompt's closing rule has no business in a conversation.
    assert "never explain a hold and then trade" not in prompt


def test_api_chat_briefing_is_rebuilt_each_round(client, monkeypatch):
    captured = _capture_chat(monkeypatch)
    _start_two_traders(client)

    client.post("/api/chat", json={"agent_id": "Alice", "message": "hi"})
    first = captured["prompt"]
    client.post("/api/step")
    client.post("/api/chat", json={"agent_id": "Alice", "message": "and now?"})
    second = captured["prompt"]

    assert first != second, "briefing must reflect the round it was sent in"


def test_api_chat_history_still_accumulates(client, monkeypatch):
    captured = _capture_chat(monkeypatch)
    _start_two_traders(client)

    client.post("/api/chat", json={"agent_id": "Alice", "message": "first"})
    assert captured["history"] == []
    client.post("/api/chat", json={"agent_id": "Alice", "message": "second"})
    assert captured["history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
    ]


def test_api_chat_builder_failure_returns_json(client, monkeypatch):
    _capture_chat(monkeypatch)
    _start_two_traders(client)

    # `ai_trading_society.web` re-exports the Flask object as `app`, which
    # shadows the submodule on attribute access -- reach the module itself.
    import importlib

    web_app = importlib.import_module("ai_trading_society.web.app")

    def boom(*args, **kwargs):
        raise RuntimeError("briefing exploded")

    monkeypatch.setattr(web_app, "build_chat_system_prompt", boom)
    res = client.post("/api/chat", json={"agent_id": "Alice", "message": "hi"})
    assert res.status_code == 500
    assert "briefing exploded" in res.get_json()["error"]

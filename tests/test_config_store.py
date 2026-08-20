"""Tests for the shared config store (user_config.json)."""

from ai_trading_society.config_store import (
    DEFAULT_CONFIG,
    load_config,
    save_config,
)


def test_defaults_when_missing(tmp_path):
    cfg = load_config(tmp_path / "nope.json")
    assert cfg["steps"] == DEFAULT_CONFIG["steps"]
    assert cfg["price"] == 100.0
    assert cfg["traders"] == []


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "user_config.json"
    saved = save_config(
        {
            "steps": 10,
            "price": 250,
            "cash": 5000,
            "hold": 8,
            "fee": 0.02,
            "slip": 0.03,
            "provider": "groq",
            "model": "groq/compound-mini",
            "traders": [
                {
                    "name": "Alice",
                    "provider": "groq",
                    "model": "groq/compound-mini",
                    "api_key": "secret",
                    "base_url": "https://api.groq.com/openai/v1",
                }
            ],
        },
        path=path,
    )
    assert saved["steps"] == 10
    assert saved["traders"][0]["name"] == "Alice"

    loaded = load_config(path)
    assert loaded["steps"] == 10
    assert loaded["price"] == 250
    assert loaded["cash"] == 5000
    assert loaded["hold"] == 8
    assert loaded["fee"] == 0.02
    assert loaded["slip"] == 0.03
    assert loaded["traders"][0]["api_key"] == "secret"


def test_partial_save_keeps_defaults(tmp_path):
    path = tmp_path / "user_config.json"
    saved = save_config({"steps": 5}, path=path)
    assert saved["steps"] == 5
    assert saved["price"] == DEFAULT_CONFIG["price"]
    assert saved["cash"] == DEFAULT_CONFIG["cash"]
    assert saved["traders"] == []


def test_malformed_traders_are_normalized(tmp_path):
    path = tmp_path / "user_config.json"
    saved = save_config({"traders": ["bogus", None, {"name": "A"}]}, path=path)
    assert len(saved["traders"]) == 1
    assert saved["traders"][0]["name"] == "A"
    assert saved["traders"][0]["provider"] == "openai"


def test_load_ignores_bad_file(tmp_path):
    path = tmp_path / "user_config.json"
    path.write_text("{{{ not json", encoding="utf-8")
    cfg = load_config(path)
    assert cfg == DEFAULT_CONFIG
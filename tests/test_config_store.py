"""Tests for the shared config store (user_config.json)."""

import pytest

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


# ---------------------------------------------------------------------------
# Homepage-adjustable fields must persist to user_config.json
# (social_influence slider, player participation toggle)
# ---------------------------------------------------------------------------
def test_homepage_adjustments_persist(tmp_path):
    """Everything adjustable on the homepage must survive a save/load cycle,
    including the social-influence slider and player participation toggle."""
    path = tmp_path / "user_config.json"
    saved = save_config(
        {
            "steps": "45",
            "social_influence": "0.65",
            "player_participates": False,
        },
        path=path,
    )
    assert saved["steps"] == 45
    assert saved["social_influence"] == 0.65
    assert saved["player_participates"] is False

    loaded = load_config(path)
    assert loaded["steps"] == 45
    assert loaded["social_influence"] == 0.65
    assert loaded["player_participates"] is False


def test_numeric_strings_are_coerced(tmp_path):
    """The homepage sends DOM input values (text) for every scalar; all of
    them must land as typed numbers in user_config.json."""
    saved = save_config(
        {"steps": "60", "cash": "20000", "hold": "7",
         "fee": "0.002", "slip": "0.003"},
        path=tmp_path / "user_config.json",
    )
    assert saved["steps"] == 60
    assert isinstance(saved["steps"], int)
    assert saved["cash"] == 20000.0
    assert isinstance(saved["cash"], float)
    assert saved["hold"] == 7
    assert saved["fee"] == 0.002
    assert saved["slip"] == 0.003


def test_social_influence_is_clamped_to_unit_range(tmp_path):
    path = tmp_path / "user_config.json"
    saved = save_config({"social_influence": 7.5}, path=path)
    assert saved["social_influence"] == 1.0
    saved = save_config({"social_influence": -3}, path=path)
    assert saved["social_influence"] == 0.0


_NON_FINITE_STRINGS = ["nan", "NaN", "inf", "-inf", "infinity", float("nan"), float("inf")]


@pytest.mark.parametrize("bad", _NON_FINITE_STRINGS)
@pytest.mark.parametrize("key", ["steps", "hold", "price", "cash", "fee", "slip", "social_influence"])
def test_non_finite_values_fall_back_to_defaults(tmp_path, key, bad):
    """Regression (C1): ``float()`` accepts nan/inf strings and bare JSON
    NaN literals; they must never reach the int()/clamp step in
    save_config — previously ``{'steps': 'nan'}`` raised ValueError and
    made POST /api/config return HTTP 500."""
    path = tmp_path / "user_config.json"
    saved = save_config({key: bad}, path=path)
    assert saved[key] == DEFAULT_CONFIG[key]

    # Same payload must survive a load round-trip without raising.
    loaded = load_config(path)
    assert loaded[key] == DEFAULT_CONFIG[key]


def test_numeric_ranges_are_clamped(tmp_path):
    """Out-of-range and negative numeric inputs must be clamped to safe bounds."""
    path = tmp_path / "user_config.json"
    saved = save_config(
        {
            "steps": -10,
            "hold": -5,
            "price": -100,
            "cash": -500,
            "fee": 0.8,
            "slip": 0.9,
            "social_influence": 2.5,
        },
        path=path,
    )
    assert saved["steps"] == 1
    assert saved["hold"] == 0
    assert saved["price"] == 0.01
    assert saved["cash"] == 0.0
    assert saved["fee"] == 0.5
    assert saved["slip"] == 0.5
    assert saved["social_influence"] == 1.0

    loaded = load_config(path)
    assert loaded["steps"] == 1
    assert loaded["hold"] == 0
    assert loaded["price"] == 0.01
    assert loaded["cash"] == 0.0
    assert loaded["fee"] == 0.5
    assert loaded["slip"] == 0.5
    assert loaded["social_influence"] == 1.0


def test_normalize_stocks_handles_invalid_and_non_finite_numbers(tmp_path):
    """Stocks with invalid, negative or non-finite price/hold must normalize cleanly."""
    path = tmp_path / "user_config.json"
    saved = save_config(
        {
            "stocks": [
                {"name": "Good", "price": "150.5", "hold": "10"},
                {"name": "BadNumbers", "price": "nan", "hold": "inf"},
                {"name": "Negative", "price": -20, "hold": -5},
            ]
        },
        path=path,
    )
    stocks = saved["stocks"]
    assert len(stocks) == 3
    assert stocks[0]["name"] == "Good"
    assert stocks[0]["price"] == 150.5
    assert stocks[0]["hold"] == 10

    assert stocks[1]["name"] == "BadNumbers"
    assert stocks[1]["price"] == 100.0
    assert stocks[1]["hold"] == 0

    assert stocks[2]["name"] == "Negative"
    assert stocks[2]["price"] == 0.01
    assert stocks[2]["hold"] == 0


def test_atomic_save_and_backup_creation(tmp_path):
    """save_config creates the file atomically and leaves a .bak backup."""
    path = tmp_path / "user_config.json"
    save_config({"steps": 15}, path=path)
    assert path.exists()
    bak = tmp_path / "user_config.json.bak"
    assert not bak.exists()  # first write has no prior generation to backup

    save_config({"steps": 25}, path=path)
    assert bak.exists()
    bak_cfg = load_config(bak)
    assert bak_cfg["steps"] == 15
    curr_cfg = load_config(path)
    assert curr_cfg["steps"] == 25


def test_bare_json_nan_literal_is_rejected_on_load(tmp_path):
    """json.load parses a bare ``NaN`` token into float('nan') by default;
    load_config must fall back to the default instead of propagating it."""
    path = tmp_path / "user_config.json"
    path.write_text('{"steps": NaN, "fee": Infinity}', encoding="utf-8")
    cfg = load_config(path)
    assert cfg["steps"] == DEFAULT_CONFIG["steps"]
    assert cfg["fee"] == DEFAULT_CONFIG["fee"]


def test_bad_types_for_new_fields_keep_defaults(tmp_path):
    path = tmp_path / "user_config.json"
    saved = save_config(
        {"social_influence": ["high"], "player_participates": "yes"},
        path=path,
    )
    assert saved["social_influence"] == DEFAULT_CONFIG["social_influence"]
    assert saved["player_participates"] == DEFAULT_CONFIG["player_participates"]

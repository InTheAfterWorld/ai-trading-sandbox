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
    assert saved["traders"][0]["api_key"] == "k"

    get_res = client.get("/api/config")
    data = get_res.get_json()["config"]
    assert data["steps"] == 12
    assert data["cash"] == 5000
    assert data["hold"] == 10
    assert data["traders"][0]["provider"] == "groq"


def test_api_env_keys_removed(client):
    res = client.get("/api/env_keys")
    assert res.status_code == 404

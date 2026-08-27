"""Secret handling for the local dashboard.

Default posture (single-user local tool):
  - GET/POST /api/config return the real trader keys so the homepage
    repopulates on load.
  - block_dns_rebinding rejects any non-local Host, which is what keeps that
    endpoint out of a malicious page's reach.

Opt-in hardening:
  - ATS_REDACT_CONFIG=1 -> redact_config() strips keys, adds has_api_key.
  - save_config() then restores a blanked key from disk (by name) so an
    autosave of another field is not a silent credential wipe.
  - /api/start and /api/test_api backfill a blank key from user_config.json.

Launch:
  - __main__ binds loopback and gates the debugger behind ATS_DEBUG.
"""

import ast
import io
import pathlib
import sys

import pytest

import ai_trading_society.config_store as config_store
import ai_trading_society.web.app  # noqa: F401  (register the real submodule)
from ai_trading_society.config_store import (
    load_config,
    redact_config,
    save_config,
)
from ai_trading_society.web.app import app

# ai_trading_society/web/__init__.py re-exports the Flask instance as
# `ai_trading_society.web.app`, shadowing the submodule for attribute access.
# Reach the real module through sys.modules.
web_app_module = sys.modules["ai_trading_society.web.app"]

_REPO = pathlib.Path(__file__).resolve().parent.parent


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


def _trader(name, key="", **extra):
    base = {"name": name, "provider": "groq", "model": "groq/compound-mini",
            "base_url": "https://api.groq.com/openai/v1", "api_key": key}
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# redact_config
# ---------------------------------------------------------------------------
class TestRedactConfig:
    def test_strips_key_and_adds_flag(self):
        cfg = {"traders": [_trader("A", "sk-secret"), _trader("B", "")]}
        out = redact_config(cfg)
        assert out["traders"][0]["api_key"] == ""
        assert out["traders"][0]["has_api_key"] is True
        assert out["traders"][1]["api_key"] == ""
        assert out["traders"][1]["has_api_key"] is False

    def test_does_not_mutate_the_input(self):
        cfg = {"traders": [_trader("A", "sk-secret")]}
        redact_config(cfg)
        assert cfg["traders"][0]["api_key"] == "sk-secret"

    def test_other_fields_survive(self):
        cfg = {"steps": 9, "traders": [_trader("A", "k", personality="greedy")]}
        out = redact_config(cfg)
        assert out["steps"] == 9
        assert out["traders"][0]["provider"] == "groq"
        assert out["traders"][0]["personality"] == "greedy"

    def test_handles_missing_traders_key(self):
        assert redact_config({"steps": 3})["traders"] == []


# ---------------------------------------------------------------------------
# save_config key preservation
# ---------------------------------------------------------------------------
class TestSaveConfigKeyPreservation:
    def test_blank_incoming_key_is_restored_from_disk(self, cfg_path):
        save_config({"traders": [_trader("A", "sk-original")]}, path=cfg_path)
        # A later autosave posts the same trader with a blanked key.
        saved = save_config({"traders": [_trader("A", "")]}, path=cfg_path)
        assert saved["traders"][0]["api_key"] == "sk-original"
        assert load_config(cfg_path)["traders"][0]["api_key"] == "sk-original"

    def test_editing_another_field_does_not_wipe_the_key(self, cfg_path):
        save_config({"traders": [_trader("A", "sk-original")]}, path=cfg_path)
        saved = save_config(
            {"steps": 42, "traders": [_trader("A", "", model="new-model")]},
            path=cfg_path,
        )
        assert saved["steps"] == 42
        assert saved["traders"][0]["model"] == "new-model"
        assert saved["traders"][0]["api_key"] == "sk-original"

    def test_a_real_new_key_still_overwrites(self, cfg_path):
        save_config({"traders": [_trader("A", "sk-old")]}, path=cfg_path)
        saved = save_config({"traders": [_trader("A", "sk-rotated")]}, path=cfg_path)
        assert saved["traders"][0]["api_key"] == "sk-rotated"

    def test_match_is_by_name(self, cfg_path):
        save_config(
            {"traders": [_trader("A", "sk-a"), _trader("B", "sk-b")]},
            path=cfg_path,
        )
        saved = save_config(
            {"traders": [_trader("B", ""), _trader("A", "")]}, path=cfg_path
        )
        by_name = {t["name"]: t["api_key"] for t in saved["traders"]}
        assert by_name == {"A": "sk-a", "B": "sk-b"}

    def test_renamed_trader_gets_no_stale_key(self, cfg_path):
        save_config({"traders": [_trader("A", "sk-a")]}, path=cfg_path)
        saved = save_config({"traders": [_trader("Renamed", "")]}, path=cfg_path)
        assert saved["traders"][0]["api_key"] == ""

    def test_partial_save_without_traders_key_is_unaffected(self, cfg_path):
        saved = save_config({"steps": 5}, path=cfg_path)
        assert saved["steps"] == 5
        assert saved["traders"] == []


@pytest.fixture
def redact_on(monkeypatch):
    monkeypatch.setenv("ATS_REDACT_CONFIG", "1")


# ---------------------------------------------------------------------------
# Default posture: the homepage gets its keys back on every load.
# ---------------------------------------------------------------------------
class TestConfigEndpointsExposeKeysByDefault:
    def test_get_config_returns_the_real_key(self, client, cfg_path):
        save_config({"traders": [_trader("A", "sk-live")]}, path=cfg_path)
        body = client.get("/api/config").get_json()["config"]
        assert body["traders"][0]["api_key"] == "sk-live"
        assert "has_api_key" not in body["traders"][0]

    def test_post_config_echoes_the_key(self, client, cfg_path):
        res = client.post(
            "/api/config", json={"traders": [_trader("A", "sk-live")]}
        )
        assert res.get_json()["config"]["traders"][0]["api_key"] == "sk-live"

    def test_round_trip_through_the_browser_keeps_the_key(self, client, cfg_path):
        client.post("/api/config", json={"traders": [_trader("A", "sk-live")]})
        reloaded = client.get("/api/config").get_json()["config"]
        client.post("/api/config", json=reloaded)
        assert load_config(cfg_path)["traders"][0]["api_key"] == "sk-live"


# ---------------------------------------------------------------------------
# ATS_REDACT_CONFIG=1: keys withheld from the client.
# ---------------------------------------------------------------------------
class TestConfigRedactionOptIn:
    def test_get_config_is_redacted(self, client, cfg_path, redact_on):
        save_config({"traders": [_trader("A", "sk-should-not-appear")]},
                    path=cfg_path)
        body = client.get("/api/config").get_json()["config"]
        assert body["traders"][0]["api_key"] == ""
        assert body["traders"][0]["has_api_key"] is True
        assert "sk-should-not-appear" not in client.get("/api/config").get_data(
            as_text=True
        )

    def test_post_config_response_is_redacted(self, client, cfg_path, redact_on):
        res = client.post(
            "/api/config", json={"traders": [_trader("A", "sk-secret")]}
        )
        assert "sk-secret" not in res.get_data(as_text=True)
        assert res.get_json()["config"]["traders"][0]["has_api_key"] is True

    def test_post_then_get_preserves_the_key_on_disk(
        self, client, cfg_path, redact_on
    ):
        client.post("/api/config", json={"traders": [_trader("A", "sk-secret")]})
        # UI reloads redacted config and posts it straight back.
        redacted = client.get("/api/config").get_json()["config"]
        client.post("/api/config", json=redacted)
        assert load_config(cfg_path)["traders"][0]["api_key"] == "sk-secret"


# ---------------------------------------------------------------------------
# DNS-rebinding guard -- the reason exposing keys locally is acceptable.
# ---------------------------------------------------------------------------
class TestDnsRebindingGuard:
    def test_foreign_host_is_rejected(self, client, cfg_path):
        save_config({"traders": [_trader("A", "sk-live")]}, path=cfg_path)
        res = client.get("/api/config", headers={"Host": "evil.example.com"})
        assert res.status_code == 403
        assert "sk-live" not in res.get_data(as_text=True)

    def test_foreign_host_with_port_is_rejected(self, client):
        res = client.get("/api/config", headers={"Host": "attacker.test:8080"})
        assert res.status_code == 403

    def test_localhost_is_allowed(self, client, cfg_path):
        save_config({"traders": [_trader("A", "sk-live")]}, path=cfg_path)
        assert client.get(
            "/api/config", headers={"Host": "localhost"}
        ).status_code == 200
        assert client.get(
            "/api/config", headers={"Host": "127.0.0.1:5000"}
        ).status_code == 200

    def test_allowlist_is_configurable(self, client, monkeypatch):
        monkeypatch.setattr(web_app_module, "_ALLOWED_HOSTS", {"dash.internal"})
        assert client.get(
            "/api/config", headers={"Host": "dash.internal"}
        ).status_code == 200
        assert client.get(
            "/api/config", headers={"Host": "localhost"}
        ).status_code == 403

    def test_guard_covers_every_route(self, client):
        for path in ("/", "/sim", "/api/events/list"):
            res = client.get(path, headers={"Host": "evil.example.com"})
            assert res.status_code == 403, path


class TestApiStartBackfill:
    def test_start_uses_stored_key_when_request_omits_it(self, client, cfg_path):
        save_config(
            {"traders": [_trader("A", "sk-live", personality="balanced")]},
            path=cfg_path,
        )
        res = client.post("/api/start", json={
            "steps": 2,
            "traders": [_trader("A", "", personality="balanced")],
        })
        assert res.status_code == 200, res.get_json()
        roster_ids = {r["id"] for r in res.get_json()["roster"]}
        assert "A" in roster_ids

    def test_explicit_request_key_still_wins(self, client, cfg_path):
        save_config({"traders": [_trader("A", "sk-stored")]}, path=cfg_path)
        res = client.post("/api/start", json={
            "steps": 2,
            "traders": [_trader("A", "sk-inline")],
        })
        assert res.status_code == 200


class TestApiTestApiBackfill:
    def test_lookup_by_name_when_key_missing(self, client, cfg_path, monkeypatch):
        save_config({"traders": [_trader("A", "sk-live")]}, path=cfg_path)
        seen = {}

        def _fake_call(self, *a, **k):
            seen["key"] = self.api_key
            return "OK"

        monkeypatch.setattr(
            "ai_trading_society.agents.external_ai_agent.ExternalAIAgent._call_ai_api",
            _fake_call,
        )
        res = client.post("/api/test_api", json={"name": "A", "provider": "groq"})
        assert res.status_code == 200, res.get_json()
        assert seen["key"] == "sk-live"

    def test_still_400_when_no_key_anywhere(self, client, cfg_path):
        res = client.post("/api/test_api", json={"name": "Unknown"})
        assert res.status_code == 400
        assert res.get_json()["ok"] is False


# ---------------------------------------------------------------------------
# Phase 4: the launch is loopback + opt-in debug
# ---------------------------------------------------------------------------
class TestLaunchHardening:
    def _main_call(self, path):
        tree = ast.parse(io.open(path, encoding="utf-8-sig").read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run"):
                return {kw.arg: kw.value for kw in node.keywords}
        raise AssertionError(f"no app.run(...) call found in {path}")

    def test_app_module_binds_loopback(self):
        kw = self._main_call(_REPO / "ai_trading_society" / "web" / "app.py")
        assert isinstance(kw.get("host"), ast.Constant)
        assert kw["host"].value == "127.0.0.1"

    def test_app_module_debug_is_env_gated_not_hardcoded_true(self):
        kw = self._main_call(_REPO / "ai_trading_society" / "web" / "app.py")
        debug = kw.get("debug")
        assert not (isinstance(debug, ast.Constant) and debug.value is True), \
            "debug must not be hard-coded True"
        # It should reference the environment, not a literal.
        assert isinstance(debug, ast.Compare)

    def test_run_py_binds_loopback(self):
        kw = self._main_call(_REPO / "run.py")
        assert isinstance(kw.get("host"), ast.Constant)
        assert kw["host"].value == "127.0.0.1"

    def test_no_env_file_shipped(self):
        assert not (_REPO / ".env").exists(), \
            ".env must not exist in the working tree (secrets live in user_config.json)"
        assert (_REPO / ".env.example").exists()

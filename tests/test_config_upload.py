"""POST /api/config/upload -- replace user_config.json with an uploaded file."""

import io
import json

import pytest

import ai_trading_society.config_store as config_store
from ai_trading_society.config_store import load_config, save_config
from ai_trading_society.web.app import _MAX_CONFIG_UPLOAD_BYTES, app


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


def _upload_file(client, payload, filename="user_config.json"):
    body = payload if isinstance(payload, (str, bytes)) else json.dumps(payload)
    if isinstance(body, str):
        body = body.encode("utf-8")
    return client.post(
        "/api/config/upload",
        data={"file": (io.BytesIO(body), filename)},
        content_type="multipart/form-data",
    )


A_FULL_CONFIG = {
    "steps": 12,
    "price": 250,
    "cash": 5000,
    "hold": 8,
    "fee": 0.002,
    "slip": 0.003,
    "provider": "groq",
    "model": "groq/compound-mini",
    "social_influence": 0.4,
    "player_participates": False,
    "traders": [
        {"name": "Alice", "provider": "groq", "model": "groq/compound-mini",
         "api_key": "sk-alice", "base_url": "https://api.groq.com/openai/v1",
         "personality": "aggressive"},
        {"name": "Bob", "provider": "openai", "model": "gpt-4o",
         "api_key": "sk-bob", "base_url": "", "personality": "panicky"},
    ],
    "stocks": [
        {"name": "Acme", "price": 100, "hold": 10, "sector": "Software"},
        {"name": "Globex", "price": 40, "hold": 5, "sector": "Energy"},
    ],
}


class TestUploadHappyPath:
    def test_multipart_file_is_saved_and_summarised(self, client, cfg_path):
        res = _upload_file(client, A_FULL_CONFIG)
        assert res.status_code == 200, res.get_json()
        body = res.get_json()
        assert body["ok"] is True
        assert body["summary"] == {
            "steps": 12, "traders": 2, "traders_with_keys": 2,
            "stocks": 2, "player_participates": False,
        }
        on_disk = load_config(cfg_path)
        assert on_disk["steps"] == 12
        assert on_disk["cash"] == 5000
        assert [t["name"] for t in on_disk["traders"]] == ["Alice", "Bob"]
        assert on_disk["traders"][0]["api_key"] == "sk-alice"
        assert [s["name"] for s in on_disk["stocks"]] == ["Acme", "Globex"]

    def test_raw_json_body_also_accepted(self, client, cfg_path):
        res = client.post(
            "/api/config/upload",
            data=json.dumps({"steps": 7, "traders": []}),
            content_type="application/json",
        )
        assert res.status_code == 200
        assert load_config(cfg_path)["steps"] == 7

    def test_response_config_lets_the_homepage_repopulate(self, client, cfg_path):
        body = _upload_file(client, A_FULL_CONFIG).get_json()
        assert body["config"]["steps"] == 12
        assert body["config"]["traders"][0]["api_key"] == "sk-alice"

    def test_utf8_bom_is_tolerated(self, client, cfg_path):
        raw = ("﻿" + json.dumps({"steps": 9})).encode("utf-8")
        res = _upload_file(client, raw)
        assert res.status_code == 200
        assert load_config(cfg_path)["steps"] == 9

    def test_previous_config_is_backed_up(self, client, cfg_path):
        save_config({"steps": 3}, path=cfg_path)
        _upload_file(client, {"steps": 99})
        assert (cfg_path.parent / (cfg_path.name + ".bak")).exists()
        assert load_config(cfg_path)["steps"] == 99


class TestUploadNormalisation:
    def test_garbage_keys_are_dropped_not_fatal(self, client, cfg_path):
        res = _upload_file(client, {
            "steps": 5, "totally_unknown": {"x": 1}, "traders": ["bogus", None,
                                                                 {"name": "Z"}],
        })
        assert res.status_code == 200
        saved = load_config(cfg_path)
        assert "totally_unknown" not in saved
        assert [t["name"] for t in saved["traders"]] == ["Z"]

    def test_out_of_range_numbers_are_clamped(self, client, cfg_path):
        _upload_file(client, {"fee": 9.0, "slip": -1, "social_influence": 5})
        saved = load_config(cfg_path)
        assert 0.0 <= saved["fee"] <= 0.5
        assert 0.0 <= saved["slip"] <= 0.5
        assert 0.0 <= saved["social_influence"] <= 1.0

    def test_blank_key_in_upload_keeps_the_stored_key(self, client, cfg_path):
        save_config({"traders": [
            {"name": "Alice", "api_key": "sk-existing"},
        ]}, path=cfg_path)
        res = _upload_file(client, {"traders": [
            {"name": "Alice", "provider": "groq", "model": "m", "api_key": ""},
        ]})
        assert res.get_json()["summary"]["traders_with_keys"] == 1
        assert load_config(cfg_path)["traders"][0]["api_key"] == "sk-existing"


class TestUploadRejections:
    def test_malformed_json_reports_position(self, client, cfg_path):
        res = _upload_file(client, b'{"steps": 5,,}')
        assert res.status_code == 400
        err = res.get_json()["error"]
        assert "Not valid JSON" in err and "line" in err

    @pytest.mark.parametrize("payload", ["[1, 2, 3]", '"just a string"', "42"])
    def test_non_object_json_is_rejected(self, client, cfg_path, payload):
        res = _upload_file(client, payload.encode())
        assert res.status_code == 400
        assert "JSON object" in res.get_json()["error"]

    def test_empty_upload_is_rejected(self, client, cfg_path):
        res = _upload_file(client, b"   ")
        assert res.status_code == 400
        assert res.get_json()["ok"] is False

    def test_oversized_upload_is_rejected(self, client, cfg_path):
        blob = b'{"blurb": "' + b"x" * (_MAX_CONFIG_UPLOAD_BYTES + 10) + b'"}'
        res = _upload_file(client, blob)
        assert res.status_code == 413
        assert "too large" in res.get_json()["error"]

    def test_invalid_utf8_is_rejected(self, client, cfg_path):
        res = _upload_file(client, b"\xff\xfe\x00bad")
        assert res.status_code == 400

    def test_a_rejected_upload_leaves_the_config_untouched(self, client, cfg_path):
        save_config({"steps": 33}, path=cfg_path)
        _upload_file(client, b"{ not json")
        assert load_config(cfg_path)["steps"] == 33


class TestUploadSecurityGuardsStillApply:
    def test_foreign_host_is_blocked(self, client, cfg_path):
        res = client.post(
            "/api/config/upload",
            data={"file": (io.BytesIO(b'{"steps": 5}'), "user_config.json")},
            content_type="multipart/form-data",
            headers={"Host": "evil.example.com"},
        )
        assert res.status_code == 403

    def test_cross_origin_post_is_blocked(self, client, cfg_path):
        res = client.post(
            "/api/config/upload",
            data={"file": (io.BytesIO(b'{"steps": 5}'), "user_config.json")},
            content_type="multipart/form-data",
            headers={"Origin": "http://evil.example"},
        )
        assert res.status_code == 403

    def test_redaction_mode_strips_keys_from_the_response(
        self, client, cfg_path, monkeypatch
    ):
        monkeypatch.setenv("ATS_REDACT_CONFIG", "1")
        body = _upload_file(client, {
            "traders": [{"name": "A", "provider": "groq", "model": "m",
                         "api_key": "sk-secret"}],
        }).get_json()
        assert "sk-secret" not in json.dumps(body)
        assert body["config"]["traders"][0]["has_api_key"] is True
        # ...but it still landed on disk.
        assert load_config(cfg_path)["traders"][0]["api_key"] == "sk-secret"

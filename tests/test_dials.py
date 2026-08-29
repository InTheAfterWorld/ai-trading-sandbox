"""Phase 2C/2D: sensitivity dials and free-text character fields.

Dials scale the mood formula and add a sentence to the disposition; the
free-text fields append to or replace the preset description. Neither ever
touches a decision.
"""

import pytest

import ai_trading_society.config_store as config_store
from ai_trading_society.agents.roster import build_agent_roster
from ai_trading_society.agents.traits import (
    _PERSONALITY_DISPOSITIONS,
    DIAL_NAMES,
    build_disposition,
    create_personality_agent,
    preset_dials,
    resolve_dials,
)
from ai_trading_society.config_store import load_config, save_config
from ai_trading_society.web.app import app
from tests.conftest import ScriptedExternalAIAgent


@pytest.fixture
def cfg_path(monkeypatch, tmp_path):
    path = tmp_path / "user_config.json"
    monkeypatch.setattr(config_store, "CONFIG_PATH", path)
    return path


def _obs(wealth=10000.0, standing=None):
    obs = {
        "step": 2,
        "stocks": [{"symbol": "A", "name": "A", "price": 100.0,
                    "price_history": [100, 100, 100], "last_volume": 0,
                    "my_holdings": 50, "move_since_last_pct": 0.0}],
        "my_cash": 1000.0, "my_holdings": {"A": 50}, "my_wealth": wealth,
        "initial_wealth": 10000.0, "market_sentiment": 0.0,
        "price": 100.0, "price_history": [100, 100, 100], "last_volume": 0,
    }
    if standing is not None:
        obs["standing"] = standing
    return obs


def _agent(dials=None, personality="balanced", **kw):
    return create_personality_agent(
        ScriptedExternalAIAgent("t", cash=1000, holdings=50),
        personality, deep=True, dials=dials, **kw,
    )


class TestDialProfiles:
    @pytest.mark.parametrize("personality", [
        "balanced", "aggressive", "conservative", "panicky",
        "greedy", "fomo_driven", "stubborn", "emotional",
    ])
    def test_every_preset_defines_all_seven(self, personality):
        dials = preset_dials(personality)
        assert set(dials) == set(DIAL_NAMES)
        assert all(0 <= v <= 10 for v in dials.values())

    def test_presets_differ(self):
        assert preset_dials("aggressive") != preset_dials("panicky")

    def test_overrides_merge_over_the_preset(self):
        merged = resolve_dials("aggressive", {"herd_pull": 9})
        assert merged["herd_pull"] == 9
        assert merged["risk_appetite"] == preset_dials("aggressive")["risk_appetite"]

    def test_overrides_are_clamped(self):
        merged = resolve_dials("balanced", {"envy": 99, "patience": -5})
        assert merged["envy"] == 10
        assert merged["patience"] == 0

    def test_junk_override_is_ignored(self):
        merged = resolve_dials("balanced", {"envy": "loads", "unknown": 3})
        assert merged["envy"] == preset_dials("balanced")["envy"]
        assert "unknown" not in merged


class TestDialsShapeTheDisposition:
    def test_each_dial_contributes_a_sentence(self):
        text = build_disposition("balanced", deep=True)
        # One sentence per dial, plus identity + body + closing check.
        assert text.count(".") >= len(DIAL_NAMES)

    def test_high_and_low_read_differently(self):
        high = build_disposition("balanced", deep=True, dials={**preset_dials("balanced"), "herd_pull": 10})
        low = build_disposition("balanced", deep=True, dials={**preset_dials("balanced"), "herd_pull": 0})
        assert "strong pull to move with the crowd" in high
        assert "barely care what other traders are doing" in low

    def test_simple_mode_has_no_dial_sentences(self):
        text = build_disposition("balanced", deep=False, dials={"herd_pull": 10})
        assert "strong pull to move with the crowd" not in text


class TestDialsScaleTheMoodFormula:
    def test_loss_sensitivity_drives_stress(self):
        base = preset_dials("balanced")
        calm = _agent({**base, "loss_sensitivity": 0, "resilience": 5})
        raw = _agent({**base, "loss_sensitivity": 10, "resilience": 5})
        for agent in (calm, raw):
            agent.act(_obs(wealth=12000.0))
            agent.act(_obs(wealth=7000.0))
        assert raw.mood["stress"] > calm.mood["stress"]

    def test_resilience_speeds_recovery(self):
        base = preset_dials("balanced")
        brittle = _agent({**base, "resilience": 0})
        tough = _agent({**base, "resilience": 10})
        for agent in (brittle, tough):
            agent.mood["stress"] = 9.0
            agent.act(_obs())
            agent.act(_obs())
        assert tough.mood["stress"] < brittle.mood["stress"]

    def test_envy_scales_the_rival_gap(self):
        base = preset_dials("balanced")
        serene = _agent({**base, "envy": 0})
        jealous = _agent({**base, "envy": 10})
        standing = {"rank": 6, "of": 6, "my_return_pct": -10.0,
                    "leader_name": "Nova", "leader_return_pct": 30.0,
                    "gap_to_leader_pct": 40.0}
        for agent in (serene, jealous):
            agent.act(_obs())
            agent.act(_obs(standing=standing))
        assert jealous.mood["frustration"] > serene.mood["frustration"]

    def test_dials_never_change_the_decision(self):
        base = preset_dials("balanced")
        obs = _obs(wealth=6000.0)
        wild = _agent({**base, "risk_appetite": 10, "herd_pull": 10})
        assert wild.act(obs)["decisions"] == wild.base_agent.act(obs)["decisions"]


class TestFreeTextCharacter:
    def test_trait_notes_are_appended_verbatim(self):
        note = "You hold a grudge against tech stocks."
        agent = _agent(trait_notes=note)
        assert note in agent.disposition
        # The preset body survives alongside the note.
        assert _PERSONALITY_DISPOSITIONS["balanced"]["full"] in agent.disposition

    def test_persona_replaces_the_preset_body(self):
        agent = _agent(persona="You are a retired fisherman who trades on gut feel.")
        assert "retired fisherman" in agent.disposition
        assert _PERSONALITY_DISPOSITIONS["balanced"]["full"] not in agent.disposition

    def test_free_text_is_ignored_in_simple_mode(self):
        text = build_disposition(
            "balanced", deep=False, trait_notes="NOTE", persona="PERSONA"
        )
        assert "NOTE" not in text and "PERSONA" not in text

    def test_long_text_is_capped(self):
        # Use a marker that cannot occur in the boilerplate prose.
        agent = _agent(trait_notes="§" * 5000)
        assert agent.disposition.count("§") == 1024


class TestConfigRoundTrip:
    def test_normalize_keeps_persona_fields_and_dials(self, cfg_path):
        saved = save_config({"traders": [{
            "name": "A", "provider": "groq", "model": "m",
            "persona": "bespoke", "trait_notes": "quirk",
            "dials": {"envy": 9, "bogus": 1},
        }]}, path=cfg_path)
        t = saved["traders"][0]
        assert t["persona"] == "bespoke"
        assert t["trait_notes"] == "quirk"
        assert t["dials"] == {"envy": 9.0}

    def test_text_fields_are_length_capped_on_save(self, cfg_path):
        saved = save_config({"traders": [
            {"name": "A", "persona": "p" * 5000, "trait_notes": "n" * 5000}
        ]}, path=cfg_path)
        assert len(saved["traders"][0]["persona"]) <= 1024
        assert len(saved["traders"][0]["trait_notes"]) <= 1024

    def test_missing_fields_default_empty(self, cfg_path):
        saved = save_config({"traders": [{"name": "A"}]}, path=cfg_path)
        t = saved["traders"][0]
        assert t["persona"] == "" and t["trait_notes"] == "" and t["dials"] == {}

    def test_survives_config_upload(self, cfg_path):
        app.config["TESTING"] = True
        with app.test_client() as client:
            import io as _io
            import json as _json
            payload = _json.dumps({"traders": [{
                "name": "A", "persona": "bespoke", "trait_notes": "quirk",
                "dials": {"conviction": 8},
            }]}).encode()
            res = client.post(
                "/api/config/upload",
                data={"file": (_io.BytesIO(payload), "user_config.json")},
                content_type="multipart/form-data",
            )
            assert res.status_code == 200
        t = load_config(cfg_path)["traders"][0]
        assert t["persona"] == "bespoke"
        assert t["dials"] == {"conviction": 8.0}

    def test_roster_threads_the_fields_through(self, cfg_path):
        agents, _ = build_agent_roster(
            trader_configs=[{
                "name": "T", "api_key": "k", "personality": "balanced",
                "trait_notes": "you never trust a rally",
                "dials": {"herd_pull": 10},
            }],
            include_player=False, deep_persona=True,
        )
        agent = agents[0]
        assert agent.dials["herd_pull"] == 10
        assert "you never trust a rally" in agent.disposition

"""Regression tests for the security/data-integrity bug fixes (Bugs 1-11, minus 9)."""

import pytest

from ai_trading_society.agents.external_ai_agent import ExternalAIAgent
from ai_trading_society.agents.traits import create_personality_agent
from ai_trading_society.config import MarketConfig
from ai_trading_society.config_store import load_config, save_config
from ai_trading_society.market_env import MarketEnv
from ai_trading_society.run_metadata import RunMetadata
from ai_trading_society.web.app import app
from tests.conftest import ScriptedExternalAIAgent


# ---------------------------------------------------------------------------
# Bug 1: sell orders could drive cash negative (fee/slippage had no upper bound)
# ---------------------------------------------------------------------------
class TestBug1SellNegativeCash:
    def test_config_clamps_fee_and_slippage(self):
        config = MarketConfig(initial_price=100.0, fee_rate=2.0, slippage_rate=5.0)
        assert 0.0 <= config.fee_rate <= 0.5
        assert 0.0 <= config.slippage_rate <= 0.5

    def test_seller_cash_stays_non_negative_with_extreme_rates(self):
        config = MarketConfig(initial_price=100.0, fee_rate=2.0, slippage_rate=5.0)
        agents = [
            ScriptedExternalAIAgent("buyer", cash=100000, holdings=0, buy_prob=1.0),
            ScriptedExternalAIAgent("seller", cash=0, holdings=100, sell_prob=1.0),
        ]
        env = MarketEnv(config, agents, seed=1)
        env.step()
        assert env.agents["seller"].cash >= 0

    def test_api_start_rejects_fee_above_max(self):
        with app.test_client() as client:
            res = client.post("/api/start", json={"steps": 3, "fee": 2.0, "traders": []})
            assert res.status_code == 400

    def test_api_start_rejects_slippage_above_max(self):
        with app.test_client() as client:
            res = client.post("/api/start", json={"steps": 3, "slip": 5.0, "traders": []})
            assert res.status_code == 400


# ---------------------------------------------------------------------------
# Bug 2: NaN/Infinity quantity crashed the JSON parser
# ---------------------------------------------------------------------------
class TestBug2NonFiniteQuantity:
    def test_nan_quantity_rejected(self):
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        assert agent._coerce_quantity(float("nan")) is None

    def test_infinity_quantity_rejected(self):
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        assert agent._coerce_quantity(float("inf")) is None
        assert agent._coerce_quantity(float("-inf")) is None

    def test_finite_quantity_still_parses(self):
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        assert agent._coerce_quantity(42.0) == 42


# ---------------------------------------------------------------------------
# Bug 3: duplicate trader names silently dropped agents
# ---------------------------------------------------------------------------
class TestBug3DuplicateAgentId:
    def test_duplicate_agent_id_raises(self):
        config = MarketConfig(initial_price=100.0)
        agents = [
            ScriptedExternalAIAgent("dup", cash=1000, holdings=0),
            ScriptedExternalAIAgent("dup", cash=1000, holdings=0),
        ]
        with pytest.raises(ValueError, match="Duplicate agent_id"):
            MarketEnv(config, agents)

    def test_api_start_rejects_duplicate_names(self):
        with app.test_client() as client:
            res = client.post(
                "/api/start",
                json={
                    "steps": 3,
                    "traders": [
                        {"name": "Same", "provider": "openai", "model": "gpt-4o"},
                        {"name": "Same", "provider": "openai", "model": "gpt-4o"},
                    ],
                },
            )
            assert res.status_code == 400
            assert "Duplicate" in res.get_json()["error"]


# ---------------------------------------------------------------------------
# Bug 4: RunMetadata snapshot dropped provider/model/personality of TraitAgent
# ---------------------------------------------------------------------------
class TestBug4MetadataSnapshot:
    def test_trait_agent_snapshot_includes_model_and_personality(self):
        base = ExternalAIAgent(
            "Trader A", cash=1000, holdings=5,
            api_provider="groq", model="groq/compound-mini", api_key="k",
        )
        trait = create_personality_agent(base, personality="aggressive")
        config = MarketConfig(initial_price=100.0, seed=42)
        meta = RunMetadata.create(config=config, agents=[trait], seed=42)
        info = meta.agents[0]
        assert info["api_provider"] == "groq"
        assert info["model"] == "groq/compound-mini"
        assert info["personality"] == "aggressive"


# ---------------------------------------------------------------------------
# Bug 5: CLI-side user_config.json had no type validation (could crash)
# ---------------------------------------------------------------------------
class TestBug5ConfigTypeValidation:
    def test_load_config_ignores_bad_numeric_types(self, tmp_path):
        path = tmp_path / "user_config.json"
        path.write_text(
            '{"steps": "abc", "price": [1,2], "hold": true, "fee": "0.1", "slip": null}',
            encoding="utf-8",
        )
        cfg = load_config(path)
        assert cfg["steps"] == 30
        assert cfg["price"] == 100.0
        assert cfg["hold"] == 20
        assert cfg["fee"] == 0.001
        assert cfg["slip"] == 0.001

    def test_save_config_keeps_defaults_for_bad_types(self, tmp_path):
        path = tmp_path / "user_config.json"
        saved = save_config({"steps": "abc", "cash": "lots"}, path=path)
        assert saved["steps"] == 30
        assert saved["cash"] == 10000.0


# ---------------------------------------------------------------------------
# Bug 6: trader names / agent ids were not escaped in sim.html (XSS)
# ---------------------------------------------------------------------------
class TestBug6XssEscaping:
    def test_sim_page_escapes_agent_id(self):
        with app.test_client() as client:
            html = client.get("/sim").get_data(as_text=True)
        # The old unsafe inline-JS string interpolation must be gone.
        assert "toggleAgentRow('${a.id}')" not in html
        # The id is now escaped at every HTML injection point.
        assert "toggleAgentRow(this.dataset.agent)" in html
        assert "${escapeHtml(a.id)}" in html


# ---------------------------------------------------------------------------
# Bug 7: /api/god/config and /api/god/event used bare float() with no validation
# ---------------------------------------------------------------------------
class TestBug7GodModeValidation:
    def _start(self, client):
        return client.post("/api/start", json={"steps": 5, "traders": []})

    def test_god_config_rejects_non_numeric(self):
        with app.test_client() as client:
            self._start(client)
            res = client.post("/api/god/config", json={"price_sensitivity": "abc"})
            assert res.status_code == 400

    def test_god_config_rejects_extreme_values(self):
        with app.test_client() as client:
            self._start(client)
            res = client.post("/api/god/config", json={"price_sensitivity": 1e9})
            assert res.status_code == 400

    def test_god_event_rejects_non_numeric_price_impact(self):
        with app.test_client() as client:
            self._start(client)
            res = client.post("/api/god/event", json={"event_name": "x", "price_impact": "abc"})
            assert res.status_code == 400


# ---------------------------------------------------------------------------
# Bug 8: /api/start with traders: [] created 7 default AI agents
# ---------------------------------------------------------------------------
class TestBug8EmptyTraderList:
    def test_start_with_empty_traders_keeps_only_player(self):
        with app.test_client() as client:
            res = client.post("/api/start", json={"steps": 5, "traders": []})
            assert res.status_code == 200
            data = res.get_json()
            assert len(data["roster"]) == 1
            assert data["roster"][0]["id"] == "Player (You)"


# ---------------------------------------------------------------------------
# Bug 10: hard-coded secret_key + no CSRF protection
# ---------------------------------------------------------------------------
class TestBug10Security:
    def test_secret_key_is_not_hardcoded_default(self):
        assert app.secret_key != "ai-trading-society-local-secret"

    def test_cross_origin_post_rejected(self, monkeypatch, tmp_path):
        self._patch_config_path(monkeypatch, tmp_path)
        with app.test_client() as client:
            res = client.post(
                "/api/config",
                json={"steps": 5},
                headers={"Origin": "http://evil.example"},
            )
            assert res.status_code == 403

    def test_same_origin_post_allowed(self, monkeypatch, tmp_path):
        self._patch_config_path(monkeypatch, tmp_path)
        with app.test_client() as client:
            res = client.post(
                "/api/config",
                json={"steps": 5},
                headers={"Origin": "http://localhost"},
            )
            assert res.status_code == 200

    @staticmethod
    def _patch_config_path(monkeypatch, tmp_path):
        import ai_trading_society.config_store as config_store

        monkeypatch.setattr(config_store, "CONFIG_PATH", tmp_path / "user_config.json")


# ---------------------------------------------------------------------------
# Bug 11: _parse_float silently fell back to defaults for bool/null
# ---------------------------------------------------------------------------
class TestBug11ParseFloatStrict:
    def test_bool_rejected(self):
        with app.test_client() as client:
            res = client.post("/api/start", json={"steps": 3, "price": True, "traders": []})
            assert res.status_code == 400

    def test_null_rejected(self):
        with app.test_client() as client:
            res = client.post("/api/start", json={"steps": 3, "price": None, "traders": []})
            assert res.status_code == 400

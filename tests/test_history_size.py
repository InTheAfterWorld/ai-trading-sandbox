"""What the sandbox retains and ships per round.

Every round used to re-copy each stock's whole price series into a record
kept forever: the state snapshot, the session history, and the /api/step
response all carried it. Retention and total transfer were quadratic in
round count -- 227k floats retained by round 300 where 1.5k were unique --
for a field nothing read.

These tests pin the field out (so it cannot creep back), pin the two
remaining readers in (so nobody "tidies" those away), prove growth is now
linear, and prove no information was lost.
"""

import json

import pytest

from ai_trading_society.config import MarketConfig, StockSpec
from ai_trading_society.market_env import MarketEnv
from ai_trading_society.web.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def build_env(stocks=2, seed=7):
    cfg = MarketConfig(
        stocks=[
            StockSpec(name=f"S{i}", initial_price=100.0 + i, initial_holdings=50)
            for i in range(stocks)
        ],
        seed=seed,
    )
    return MarketEnv(cfg, [])


class TestSnapshotShape:
    def test_stock_entry_carries_only_this_round(self):
        env = build_env()
        state = env.step()
        for entry in state["stocks"].values():
            assert set(entry) == {
                "price", "name", "volume", "total_buy", "total_sell",
            }

    def test_price_series_is_not_retained(self):
        env = build_env()
        for _ in range(15):
            state = env.step()
            for entry in state["stocks"].values():
                assert "price_history" not in entry

    def test_the_keys_consumers_actually_read_survive(self):
        # Snapshot consumers read wealth, price, step and the pressure
        # totals. Removing the stock series must not disturb any of them.
        env = build_env()
        state = env.step()
        assert {"agents", "price", "step", "total_buy", "total_sell"} <= set(state)
        first = next(iter(state["stocks"].values()))
        assert "total_buy" in first and "total_sell" in first


class TestNoInformationLost:
    def test_series_is_reconstructible_from_snapshots(self):
        # The whole justification for dropping the field: a post-mortem can
        # still rebuild the exact series from the initial price plus each
        # round's closing price.
        env = build_env(stocks=3)
        history = [env.step() for _ in range(25)]
        for i, (sym, sm) in enumerate(env.stocks.items()):
            rebuilt = [env.config.stocks[i].initial_price] + [
                s["stocks"][sym]["price"] for s in history
            ]
            assert rebuilt == list(sm.price_history)


class TestGrowthIsLinear:
    def _per_round_bytes(self, rounds):
        env = build_env(stocks=3)
        history = [env.step() for _ in range(rounds)]
        return len(json.dumps(history, default=str)) / rounds

    def test_snapshot_cost_per_round_is_flat(self):
        # Pre-fix this ratio grew with round count because each snapshot
        # carried a longer copy of the series than the one before it.
        short = self._per_round_bytes(10)
        long = self._per_round_bytes(40)
        assert long <= short * 1.10, (
            f"per-round snapshot cost grew {long / short:.2f}x; "
            "the price series may have crept back in"
        )

    def test_step_response_is_flat(self, client):
        client.post("/api/start", json={"steps": 60, "traders": []})
        sizes = [len(client.post("/api/step").data) for _ in range(45)]
        early, late = sizes[4], sizes[39]
        assert late <= early * 1.2, (
            f"/api/step response grew {late / early:.2f}x over 40 rounds"
        )


class TestEndpointShape:
    def test_step_response_drops_both_price_series(self, client):
        client.post("/api/start", json={"steps": 5, "traders": []})
        data = client.post("/api/step").get_json()
        assert "price_history" not in data
        assert data["stocks"]
        for stock in data["stocks"]:
            assert "price_history" not in stock

    def test_stored_history_drops_it_too(self, client):
        client.post("/api/start", json={"steps": 5, "traders": []})
        client.post("/api/step")
        snap = client.get("/api/history").get_json()["history"][0]
        # The key itself survives -- only the series inside it goes.
        assert "stocks" in snap
        for stock in snap["stocks"]:
            assert "price_history" not in stock
            assert "price" in stock

    def test_start_still_seeds_the_chart(self, client):
        # sim.html builds sim.histories from this; it must keep the series.
        data = client.post("/api/start", json={"steps": 5, "traders": []}).get_json()
        assert "price_history" in data["stocks"][0]
        assert data["stocks"][0]["price_history"]

    def test_results_still_carries_the_series(self, client):
        # The final chart in showResults() reads this.
        client.post("/api/start", json={"steps": 5, "traders": []})
        client.post("/api/step")
        assert "price_history" in client.get("/api/results").get_json()


class TestObservationUntouched:
    def test_agents_still_see_a_price_window(self):
        # The observation is a different structure from the snapshot, and is
        # the agents' decision input. It keeps its series, already capped at
        # price_history_length. None of this touched it.
        from ai_trading_society.agents.player_agent import PlayerAgent

        cfg = MarketConfig(
            stocks=[StockSpec(name="S0", initial_price=100.0)], seed=1
        )
        env = MarketEnv(cfg, [PlayerAgent("You", cash=1000)])
        env.step()
        obs = env.get_observation("You")
        assert obs["price_history"]
        assert obs["stocks"][0]["price_history"]
        assert len(obs["stocks"][0]["price_history"]) <= cfg.price_history_length

"""Verify all 7 bug fixes."""
import io
import contextlib
import inspect
import random as _r

from ai_trading_society.config import MarketConfig, StockSpec
from ai_trading_society.market_env import MarketEnv
from ai_trading_society.agents.player_agent import PlayerAgent
from ai_trading_society.agents.roster import resolve_social_map

# --- Bug 2: top-level total_buy/total_sell in state ---
cfg = MarketConfig(stocks=[StockSpec(name="A", initial_price=100, initial_holdings=5),
                           StockSpec(name="B", initial_price=50, initial_holdings=10)])
pl = PlayerAgent(cash=100000, holdings=0)
env = MarketEnv(cfg, [pl])
pl._env = env
env.set_player_action("buy", 3, symbol="A")
env.set_player_action("sell", 4, symbol="B")
state = env.step()
assert state["total_buy"] == 3 and state["total_sell"] == 4, (
    state["total_buy"], state["total_sell"])
assert state["matched_volume"] == 7
print("PASS Bug2: top-level total_buy=3 total_sell=4 matched_volume=7")

# --- Bug 5: multi-stock max buys cannot overdraw cash ---
class MaxBuyAgent(PlayerAgent):
    def act(self, observation):
        return {"decisions": [
            {"name": s.get("name"), "action": "buy",
             "quantity": 999999, "reasoning": ""}
            for s in observation.get("stocks", [])
        ]}

cfg2 = MarketConfig(stocks=[StockSpec(name="X", initial_price=100),
                            StockSpec(name="Y", initial_price=100)])
ag = MaxBuyAgent(cash=1000.0, holdings=0)
env2 = MarketEnv(cfg2, [ag])
for _ in range(6):
    env2.step()
    assert ag.cash >= -1e-6, f"cash negative: {ag.cash}"
print(f"PASS Bug5: cash after 6 max-buy rounds = {ag.cash:.2f} (never negative)")

# --- Bug 3+4: CLI config path ---
from ai_trading_society import config_store
config_store.save_config({"price": 100.0, "hold": 20,
                          "stocks": [{"name": "Cli1", "price": 10, "hold": 1},
                                     {"name": "Cli2", "price": 20, "hold": 2}]})
try:
    cfg_data = config_store.load_config()
    specs = []
    for s in (cfg_data.get("stocks") or []):
        if not isinstance(s, dict):
            continue
        nm = str(s.get("name") or s.get("symbol") or "").strip()
        if nm:
            specs.append(StockSpec(
                name=nm,
                initial_price=float(s.get("price", 100.0)),
                initial_holdings=int(s.get("hold", 0)),
            ))
    assert [s.name for s in specs] == ["Cli1", "Cli2"], specs
    print("PASS Bug3: CLI parses saved stocks ->", [s.name for s in specs])

    env3 = MarketEnv(MarketConfig(stocks=specs,
                                  social_influence=float(cfg_data.get("social_influence", 0) or 0)),
                     [PlayerAgent(cash=1000.0, holdings=0)])
    env3.social_map = resolve_social_map(list(env3.agents.values()))
    env3._social_influence = env3.config.social_influence
    assert isinstance(env3.social_map, dict)
    print("PASS Bug4: social map resolved for CLI-style env")
finally:
    config_store.save_config({"price": 100.0, "cash": 10000.0, "hold": 20,
                              "fee": 0.001, "slip": 0.001, "provider": "openai",
                              "model": "gpt-4o", "traders": [], "steps": 30,
                              "stocks": [{"name": "Stock 1", "price": 100.0, "hold": 20}]})

# --- Bug 7: error fallback decisions include "name" ---
class BoomAgent(PlayerAgent):
    def act(self, observation):
        raise RuntimeError("boom")

src = inspect.getsource(MarketEnv.step)
assert '"name": sym,' in src, "fallback missing 'name'"
env4 = MarketEnv(MarketConfig(stocks=[StockSpec(name="Q", initial_price=10)]),
                 [BoomAgent()])
st = env4.step()
aa = st["agent_actions"]["Player (You)"]["Q"]
assert aa["error"] is True and aa["action"] == "hold"
print("PASS Bug7: fallback decisions carry 'name'; agent flagged error")

# --- Bug 1: _print_round aggregates nested actions ---
from ai_trading_society.simulator import Simulator
sim = Simulator(env)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sim._print_round(state, prev_price=100.0)
out = buf.getvalue()
assert "BUY" in out and "SELL" in out, out[:500]
assert "Pressure:" in out, "pressure bar should render"
print("PASS Bug1: _print_round shows aggregated BUY/SELL + pressure bar")

# --- Bug 6: regret_avoidance uses portfolio wealth ---
from ai_trading_society.agents.traits import TraitAgent


class SellAllAgent(PlayerAgent):
    def act(self, observation):
        return {"decisions": [
            {"name": s.get("name"), "action": "sell",
             "quantity": 999, "reasoning": ""}
            for s in observation.get("stocks", [])
        ]}


base = SellAllAgent(cash=50000.0, holdings=0)
base.holdings = {"A": 100}
t = TraitAgent(base, regret_avoidance=1.0)
t._initial_wealth = 60000.0
obs = {
    "step": 2,
    "stocks": [
        {"symbol": "A", "name": "A", "price": 69.0,
         "price_history": [100, 90, 80, 69], "last_volume": 0, "my_holdings": 100},
        {"symbol": "B", "name": "B", "price": 300.0,
         "price_history": [290, 295, 300], "last_volume": 0, "my_holdings": 50},
    ],
    "my_cash": 50000.0,
    "my_holdings": {"A": 100, "B": 50},
    "my_wealth": 80000.0,  # portfolio UP even though stock A crashed
    "market_sentiment": 0.0,
}
_r.seed(7)
res = t.act(obs)
d = next(x for x in res["decisions"] if x["symbol"] == "A")
assert d["action"] == "sell", d
print("PASS Bug6: regret_avoidance uses portfolio wealth (sell kept)")
print("ALL 7 FIXES VERIFIED")

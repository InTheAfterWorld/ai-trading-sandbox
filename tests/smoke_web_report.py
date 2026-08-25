"""Smoke test: player trading, history snapshots, report export, read-only link."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading_society.web.app import app

client = app.test_client()

# 1. Start a player-only run (empty trader list = no AI agents).
r = client.post("/api/start", json={"steps": 4, "traders": [], "price": 100,
                                    "cash": 10000, "hold": 20})
data = r.get_json()
assert r.status_code == 200, data
assert any(a.get("is_player") for a in data["roster"]), "player must be in roster"
print("start ok — roster:", [a["id"] for a in data["roster"]])

# 2. Queue a player buy order.
r = client.post("/api/player_action", json={"action": "buy", "quantity": 5})
assert r.get_json().get("ok"), r.get_json()
print("player_action ok")

# 3. Run two rounds.
for i in range(2):
    r = client.post("/api/step")
    d = r.get_json()
    assert "step" in d, d
    print(f"step {d['step']} ok — price {d['price']}")

# Player's order should have executed in round 1.
player = [a for a in d["agents"] if a.get("is_player")][0]
print("player after 2 rounds:", {k: player[k] for k in ("cash", "wealth", "holdings")})
assert player["cash"] < 10000, "player buy should have consumed cash"

# 4. History snapshots. 
r = client.get("/api/history")
h = r.get_json()
assert r.status_code == 200 and len(h["history"]) == 2, h
snap = h["history"][0]
for key in ("step", "stocks", "agents", "events", "price"):
    assert key in snap, f"snapshot missing {key}"
assert any(a.get("is_player") for a in snap["agents"])
print("history ok — snapshots:", [s["step"] for s in h["history"]])

# 5. Export report.
r = client.post("/api/report/export")
rep = r.get_json()
assert rep.get("ok"), rep
print("export ok —", rep["url"], "run_id:", rep["run_id"])

# 6. Serve the read-only report page.
r = client.get(rep["url"])
assert r.status_code == 200, r.status_code
body = r.get_data(as_text=True)
for marker in ("Simulation Report", "Event Timeline", "Decision Log", "Final Rankings"):
    assert marker in body, f"report missing section: {marker}"
assert "Player (You)" in body
print("report page ok — length:", len(body))

# 7. Bad report id is rejected.
r = client.get("/report/..%5Cevil")
assert r.status_code in (400, 404), r.status_code
print("report id sanitization ok")

# 8. Results endpoint still fine.
r = client.get("/api/results")
res = r.get_json()
assert r.status_code == 200 and res["rankings"], res
assert any(x.get("is_player") for x in res["rankings"])
print("results ok")

print("\nALL SMOKE TESTS PASSED")

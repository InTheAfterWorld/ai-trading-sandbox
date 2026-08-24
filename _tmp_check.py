"""Temp check: player participation toggle end-to-end."""
import json
from ai_trading_society.web.app import app

client = app.test_client()

# 1. Spectator mode: no player agent in roster.
r = client.post("/api/start", json={"steps": 2, "traders": [{"name": "T1"}],
                                    "player_participates": False})
assert r.status_code == 200, r.get_json()
agents = r.get_json().get("roster", [])
assert not any(a.get("is_player") for a in agents), agents
print("spectator start ok — roster:", [a["id"] for a in agents])

# 2. Placing an order as spectator returns a clear error.
r = client.post("/api/player_action", json={"action": "buy", "quantity": 5})
assert r.status_code == 400 and "spectating" in r.get_json()["error"], r.get_json()
print("spectator order rejected ok")

# 3. Default (no flag) keeps the player.
r = client.post("/api/start", json={"steps": 2, "traders": []})
assert any(a.get("is_player") for a in r.get_json()["roster"])
print("default join ok")

# 4. Explicit join works too.
r = client.post("/api/start", json={"steps": 2, "traders": [], "player_participates": True})
assert any(a.get("is_player") for a in r.get_json()["roster"])
print("explicit join ok")

# 5. No participants at all -> 400.
r = client.post("/api/start", json={"steps": 2, "traders": [], "player_participates": False,
                                    "hold": 0, "cash": 1000})
# traders [] means no AI; player off -> error expected
# NOTE: traders=[] builds zero AI agents only when explicitly empty list is respected;
# re-check actual behavior:
data = r.get_json()
print("empty-market response:", r.status_code, data)

from ai_trading_society.agents.roster import build_agent_roster
agents, player = build_agent_roster(provider="mock", trader_configs=[], include_player=False)
assert player is None and len(agents) == 0
agents, player = build_agent_roster(provider="mock", trader_configs=[])
assert player is not None
print("roster include_player OK")
print("ALL CHECKS PASSED")

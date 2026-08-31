"""The dashboard's view of tokens, cost and mood.

A run started with no traders has no LLM agents, so these tests mostly pin
the shape of the payload and the honesty of the empty case -- a run that
called nothing must report nothing, not zero dollars of confirmed spend.
"""

import pytest

from ai_trading_society.usage import UsageTracker
from ai_trading_society.web.app import _usage_payload, app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestUsageEndpoint:
    def test_requires_an_active_simulation(self, client):
        res = client.get("/api/usage")
        assert res.status_code == 400
        assert "error" in res.get_json()

    def test_payload_shape(self, client):
        client.post("/api/start", json={"steps": 3, "traders": []})
        client.post("/api/step")
        data = client.get("/api/usage").get_json()
        assert set(data) == {"total", "agents", "by_round", "prompt"}
        assert data["prompt"]["template_version"]
        assert set(data["prompt"]["fingerprints"]) == {"simple", "deep"}

    def test_by_round_has_one_entry_per_completed_round(self, client):
        client.post("/api/start", json={"steps": 3, "traders": []})
        client.post("/api/step")
        client.post("/api/step")
        data = client.get("/api/usage").get_json()
        assert [r["step"] for r in data["by_round"]] == [1, 2]

    def test_a_run_with_no_llm_calls_reports_none(self, client):
        client.post("/api/start", json={"steps": 2, "traders": []})
        client.post("/api/step")
        total = client.get("/api/usage").get_json()["total"]
        assert total["calls"] == 0
        assert total["total_tokens"] == 0

    def test_step_response_carries_mood_and_usage_keys(self, client):
        client.post("/api/start", json={"steps": 2, "traders": []})
        agents = client.post("/api/step").get_json()["agents"]
        assert agents, "expected at least the player in the roster"
        for a in agents:
            assert "mood" in a
            assert "usage" in a


class TestUsagePayload:
    def test_agent_without_a_tracker_yields_none(self):
        class Bare:
            pass

        assert _usage_payload(Bare()) is None

    def test_running_total_and_round_slice(self):
        class Agent:
            usage = UsageTracker("A", "anthropic", "claude-sonnet-5")

        agent = Agent()
        agent.usage.begin_step(1)
        agent.usage.record(prompt_tokens=100, completion_tokens=10)
        agent.usage.begin_step(2)
        agent.usage.record(prompt_tokens=50, completion_tokens=5)

        payload = _usage_payload(agent, step=2)
        assert payload["total"]["prompt_tokens"] == 150
        assert payload["round"]["prompt_tokens"] == 50
        assert payload["priced"] is True

    def test_unpriced_agent_is_marked_unpriced(self):
        class Agent:
            usage = UsageTracker("A", "custom", "unknown-model-xyz")

        agent = Agent()
        agent.usage.record(prompt_tokens=10, completion_tokens=1)
        assert _usage_payload(agent)["priced"] is False

    def test_step_slice_is_omitted_when_no_step_is_given(self):
        class Agent:
            usage = UsageTracker("A", "x", "claude-sonnet-5")

        assert "round" not in _usage_payload(Agent())

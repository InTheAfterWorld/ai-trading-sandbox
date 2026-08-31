"""Token and cost accounting.

The load-bearing property here is that an unknown price is reported as
unknown. A cost feature that silently reports $0.00 for an unpriced model is
worse than no cost feature, so most of these tests are about that boundary.
"""

import json

import pytest

from ai_trading_society.agents.external_ai_agent import ExternalAIAgent
from ai_trading_society.usage import (
    UsageTracker,
    agent_usage,
    collect_usage,
    compute_cost,
    estimate_tokens,
    extract_usage,
    format_cost,
    load_prices,
    model_price,
    normalize_model,
)


class TestModelPricing:
    def test_known_model_is_priced(self):
        price = model_price("claude-sonnet-5")
        assert price is not None
        assert price["input"] > 0 and price["output"] > 0

    def test_unknown_model_has_no_price(self):
        assert model_price("totally-made-up-model-9000") is None

    def test_unknown_model_costs_none_not_zero(self):
        # The whole point: an unpriced call must not read as a free call.
        assert compute_cost("totally-made-up-model-9000", 1000, 1000) is None

    def test_openrouter_style_id_normalizes(self):
        assert normalize_model("anthropic/claude-opus-5:free") == "claude-opus-5"
        assert model_price("anthropic/claude-opus-5") == model_price("claude-opus-5")

    def test_dated_snapshot_matches_by_prefix(self):
        assert model_price("claude-sonnet-5-20260101") == model_price("claude-sonnet-5")

    def test_empty_model_is_unpriced(self):
        assert model_price("") is None
        assert model_price(None) is None  # type: ignore[arg-type]

    def test_cost_math(self):
        price = model_price("claude-sonnet-5")
        cost = compute_cost("claude-sonnet-5", 1_000_000, 1_000_000)
        assert cost == pytest.approx(price["input"] + price["output"])

    def test_price_file_override(self, tmp_path, monkeypatch):
        custom = tmp_path / "prices.json"
        custom.write_text(
            json.dumps({"models": {"my-model": {"input": 1.0, "output": 2.0}}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("ATS_MODEL_PRICES", str(custom))
        load_prices(force=True)
        try:
            assert model_price("my-model") == {"input": 1.0, "output": 2.0}
            assert compute_cost("my-model", 1_000_000, 0) == pytest.approx(1.0)
        finally:
            monkeypatch.delenv("ATS_MODEL_PRICES")
            load_prices(force=True)

    def test_missing_price_file_degrades_to_unknown(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATS_MODEL_PRICES", str(tmp_path / "nope.json"))
        load_prices(force=True)
        try:
            assert model_price("claude-sonnet-5") is None
        finally:
            monkeypatch.delenv("ATS_MODEL_PRICES")
            load_prices(force=True)


class TestExtractUsage:
    def test_openai_shape(self):
        class U:
            prompt_tokens = 120
            completion_tokens = 45

        class R:
            usage = U()

        assert extract_usage(R()) == (120, 45)

    def test_anthropic_shape(self):
        class U:
            input_tokens = 300
            output_tokens = 90

        class R:
            usage = U()

        assert extract_usage(R()) == (300, 90)

    def test_google_shape(self):
        class M:
            prompt_token_count = 77
            candidates_token_count = 12

        class R:
            usage_metadata = M()

        assert extract_usage(R()) == (77, 12)

    def test_dict_payload(self):
        assert extract_usage(
            {"usage": {"prompt_tokens": 5, "completion_tokens": 6}}
        ) == (5, 6)

    def test_no_usage_block_returns_none(self):
        class R:
            pass

        assert extract_usage(R()) is None
        assert extract_usage(None) is None
        assert extract_usage("a plain string") is None

    def test_partial_usage_fills_the_missing_half(self):
        assert extract_usage({"usage": {"prompt_tokens": 9}}) == (9, 0)


class TestUsageTracker:
    def test_totals_accumulate(self):
        t = UsageTracker("A", "anthropic", "claude-sonnet-5")
        t.record(prompt_tokens=100, completion_tokens=50)
        t.record(prompt_tokens=200, completion_tokens=25)
        assert t.total.calls == 2
        assert t.total.prompt_tokens == 300
        assert t.total.completion_tokens == 75
        assert t.total.total_tokens == 375
        assert t.total.cost_usd > 0
        assert t.total.unpriced_calls == 0

    def test_unpriced_call_is_flagged_not_zeroed(self):
        t = UsageTracker("A", "custom", "unknown-model-xyz")
        t.record(prompt_tokens=1000, completion_tokens=1000)
        assert t.total.prompt_tokens == 1000  # tokens still counted
        assert t.total.cost_usd == 0.0
        assert t.total.unpriced_calls == 1
        assert t.total.to_dict()["cost_complete"] is False

    def test_mixed_priced_and_unpriced_marks_total_incomplete(self):
        t = UsageTracker("A", "x", "claude-sonnet-5")
        t.record(prompt_tokens=100, completion_tokens=100)
        t.record(prompt_tokens=100, completion_tokens=100, model="unknown-xyz")
        d = t.total.to_dict()
        assert d["cost_usd"] > 0
        assert d["cost_complete"] is False

    def test_per_step_tagging(self):
        t = UsageTracker("A", "x", "claude-sonnet-5")
        t.begin_step(1)
        t.record(prompt_tokens=10, completion_tokens=10)
        t.begin_step(2)
        t.record(prompt_tokens=20, completion_tokens=20)
        t.record(prompt_tokens=5, completion_tokens=5)
        assert t.step_totals(1).calls == 1
        assert t.step_totals(2).calls == 2
        assert t.step_totals(99).calls == 0  # never touched

    def test_kinds_are_separated(self):
        t = UsageTracker("A", "x", "claude-sonnet-5")
        t.record(prompt_tokens=10, completion_tokens=10, kind="decision")
        t.record(prompt_tokens=10, completion_tokens=10, kind="repair")
        t.record(prompt_tokens=10, completion_tokens=10, kind="chat")
        assert set(t.by_kind) == {"decision", "repair", "chat"}
        assert t.by_kind["repair"].calls == 1

    def test_unknown_kind_falls_back_to_decision(self):
        t = UsageTracker("A", "x", "claude-sonnet-5")
        t.record(prompt_tokens=1, completion_tokens=1, kind="nonsense")
        assert "decision" in t.by_kind

    def test_estimated_calls_are_flagged(self):
        t = UsageTracker("A", "x", "claude-sonnet-5")
        t.record_estimated("a" * 400, "b" * 40)
        assert t.total.estimated_calls == 1
        assert t.total.prompt_tokens == estimate_tokens("a" * 400)

    def test_records_are_capped(self):
        t = UsageTracker("A", "x", "claude-sonnet-5")
        for _ in range(5100):
            t.record(prompt_tokens=1, completion_tokens=1)
        assert len(t.records) == 5000  # tail bounded
        assert t.total.calls == 5100  # totals still exact

    def test_negative_counts_are_clamped(self):
        t = UsageTracker("A", "x", "claude-sonnet-5")
        t.record(prompt_tokens=-5, completion_tokens=-5)
        assert t.total.prompt_tokens == 0


class TestAgentIntegration:
    def _agent(self, **kw):
        return ExternalAIAgent(
            "T", api_provider="anthropic", model="claude-sonnet-5",
            api_key="test-key", **kw,
        )

    def test_agent_has_a_tracker(self):
        agent = self._agent()
        assert agent_usage(agent) is not None
        assert agent.usage.model == "claude-sonnet-5"

    def test_tracker_is_reachable_through_the_persona_wrapper(self):
        from ai_trading_society.agents.traits import create_personality_agent

        wrapped = create_personality_agent(self._agent(), personality="balanced")
        assert agent_usage(wrapped) is agent_usage(wrapped.base_agent)

    def test_record_call_uses_provider_counts(self):
        agent = self._agent()

        class R:
            class usage:
                prompt_tokens = 800
                completion_tokens = 120

        agent._record_call(R(), "prompt text", "response text")
        assert agent.usage.total.prompt_tokens == 800
        assert agent.usage.total.estimated_calls == 0

    def test_record_call_estimates_when_provider_is_silent(self):
        agent = self._agent()
        agent._record_call(object(), "x" * 800, "y" * 80)
        assert agent.usage.total.estimated_calls == 1
        assert agent.usage.total.prompt_tokens > 0

    def test_accounting_failure_never_breaks_a_round(self):
        agent = self._agent()

        class Exploding:
            @property
            def usage(self):
                raise RuntimeError("provider object is hostile")

        agent._record_call(Exploding(), "p", "r")  # must not raise
        assert agent.usage.total.calls == 0

    def test_repair_calls_are_tagged_as_repairs(self, monkeypatch):
        agent = self._agent()
        agent._repair_calls_remaining = 1

        def fake_call(prompt, messages=None, system_prompt=None):
            # The kind is agent state, so it is set by the time we land here.
            assert agent._call_kind == "repair"
            return '{"decisions": [{"name": "S", "action": "hold", "quantity": 0}]}'

        monkeypatch.setattr(agent, "_call_ai_api", fake_call)
        result, _ = agent._retry_with_escalation(
            {"stocks": [{"name": "S", "price": 10, "my_holdings": 0}]},
            "prompt", "garbage", ValueError("bad"),
        )
        assert result["decisions"][0]["action"] == "hold"
        assert agent._call_kind == "decision"  # restored


class TestCollectUsage:
    def test_aggregates_across_agents(self):
        a = ExternalAIAgent("A", model="claude-sonnet-5", api_key="k")
        b = ExternalAIAgent("B", model="claude-sonnet-5", api_key="k")
        a.usage.record(prompt_tokens=100, completion_tokens=10)
        b.usage.record(prompt_tokens=200, completion_tokens=20)
        summary = collect_usage([a, b])
        assert summary["total"]["calls"] == 2
        assert summary["total"]["prompt_tokens"] == 300
        assert len(summary["agents"]) == 2

    def test_agents_without_trackers_are_skipped(self):
        from ai_trading_society.agents.player_agent import PlayerAgent

        summary = collect_usage([PlayerAgent("You")])
        assert summary["total"]["calls"] == 0
        assert summary["agents"] == []

    def test_empty_roster(self):
        assert collect_usage([])["total"]["calls"] == 0
        assert collect_usage(None)["total"]["calls"] == 0


class TestFormatCost:
    def test_unknown_reads_as_unknown(self):
        assert format_cost(None) == "n/a"

    def test_sub_cent_keeps_precision(self):
        assert format_cost(0.0031) == "$0.0031"

    def test_incomplete_total_is_marked_as_a_floor(self):
        assert format_cost(1.5, complete=False).endswith("+")

    def test_zero_is_not_padded_to_four_places(self):
        assert format_cost(0.0) == "$0.00"

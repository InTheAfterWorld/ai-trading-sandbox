"""Prompt versioning.

A run's decision log only means something next to the prompt that produced
it. These tests pin the two guarantees that make an old report readable:
the fingerprint tracks the prompt actually sent, and the metadata records it.
"""

from ai_trading_society.agents.external_ai_agent import ExternalAIAgent
from ai_trading_society.agents.traits import create_personality_agent
from ai_trading_society.config import MarketConfig
from ai_trading_society.prompt_version import (
    PROMPT_TEMPLATE_VERSION,
    describe_prompt,
    prompt_fingerprint,
    shipped_fingerprints,
)
from ai_trading_society.run_metadata import RunMetadata, get_code_version


def _agent(**kw):
    return ExternalAIAgent("T", model="claude-sonnet-5", api_key="k", **kw)


class TestFingerprint:
    def test_is_stable_for_identical_text(self):
        assert prompt_fingerprint("hello") == prompt_fingerprint("hello")

    def test_changes_when_the_prompt_changes(self):
        assert prompt_fingerprint("hello") != prompt_fingerprint("hello ")

    def test_is_short_and_hex(self):
        fp = prompt_fingerprint("anything")
        assert len(fp) == 12
        int(fp, 16)  # raises if not hex

    def test_empty_prompt_still_hashes(self):
        assert len(prompt_fingerprint("")) == 12

    def test_simple_and_deep_prompts_differ(self):
        shipped = shipped_fingerprints()
        assert shipped["simple"] != shipped["deep"]


class TestDescribePrompt:
    def test_default_prompt_is_recognised(self):
        info = describe_prompt(_agent())
        assert info["source"] == "default"
        assert info["template_version"] == PROMPT_TEMPLATE_VERSION
        assert info["fingerprint"] == shipped_fingerprints()["simple"]

    def test_custom_prompt_is_recognised(self):
        info = describe_prompt(_agent(system_prompt="be a pirate"))
        assert info["source"] == "custom"
        assert info["chars"] == len("be a pirate")

    def test_persona_prefixed_prompt_keeps_the_template_identity(self):
        # The persona layer prepends a disposition but leaves this version's
        # rules intact, so the template version still describes the run.
        wrapped = create_personality_agent(
            _agent(), personality="aggressive", deep=True
        )
        info = describe_prompt(wrapped)
        assert info["source"] == "persona+default"
        assert info["deep"] is True

    def test_fingerprint_follows_a_later_prompt_edit(self):
        # The persona layer rewrites system_prompt after construction; a
        # fingerprint frozen at __init__ would name a prompt never sent.
        agent = _agent()
        before = agent.prompt_fingerprint
        agent.system_prompt = agent.system_prompt + "\nExtra rule."
        assert agent.prompt_fingerprint != before
        assert describe_prompt(agent)["fingerprint"] == agent.prompt_fingerprint

    def test_agent_without_a_prompt_yields_none(self):
        from ai_trading_society.agents.player_agent import PlayerAgent

        assert describe_prompt(PlayerAgent("You")) is None


class TestRunMetadata:
    def test_version_block_pins_the_prompt(self):
        version = get_code_version()
        assert version["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
        assert set(version["prompt_fingerprints"]) == {"simple", "deep"}

    def test_roster_records_each_agent_prompt(self):
        agents = [
            _agent(),
            create_personality_agent(
                ExternalAIAgent("P", model="claude-sonnet-5", api_key="k"),
                personality="panicky", deep=True,
            ),
        ]
        meta = RunMetadata.create(config=MarketConfig(), agents=agents, seed=7)
        prompts = [a.get("prompt") for a in meta.agents]
        assert all(p is not None for p in prompts)
        assert prompts[0]["source"] == "default"
        assert prompts[1]["source"] == "persona+default"
        # Different prompts must be distinguishable in the record.
        assert prompts[0]["fingerprint"] != prompts[1]["fingerprint"]

    def test_metadata_survives_a_roster_with_no_prompts(self):
        from ai_trading_society.agents.player_agent import PlayerAgent

        meta = RunMetadata.create(
            config=MarketConfig(), agents=[PlayerAgent("You")], seed=1
        )
        assert "prompt" not in meta.agents[0]

    def test_attach_usage_folds_cost_into_the_summary(self):
        agent = _agent()
        agent.usage.record(prompt_tokens=100, completion_tokens=10)
        meta = RunMetadata.create(config=MarketConfig(), agents=[agent], seed=1)
        usage = meta.attach_usage([agent])
        assert usage["total"]["calls"] == 1
        assert meta.summary["usage"]["total"]["prompt_tokens"] == 100
        assert meta.to_dict()["summary"]["usage"]["total"]["calls"] == 1

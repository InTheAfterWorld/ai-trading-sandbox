"""Provider clients are built once per agent, not once per call.

Each SDK client owns its own HTTP connection pool, so constructing one per
call meant every request re-did the TCP and TLS handshake instead of reusing
a warm connection -- paid again by every retry and every repair re-ask. These
tests hold the reuse in place, and hold the invalidation honest: a changed
key or endpoint must not keep talking to the old connection.
"""

import pytest

from ai_trading_society.agents.external_ai_agent import (
    _REQUEST_TIMEOUT,
    ExternalAIAgent,
)


@pytest.fixture
def agent():
    return ExternalAIAgent(
        "Ada", api_provider="openai", model="gpt-4o", api_key="k"
    )


def call(agent, times=1):
    """Drive the call path; the request itself fails with no network."""
    for _ in range(times):
        try:
            agent._call_openai_compat("hi")
        except Exception:
            pass


class TestOpenAIClientReuse:
    def test_built_once_across_many_calls(self, agent, monkeypatch):
        import openai

        built = []
        real = openai.OpenAI
        monkeypatch.setattr(
            openai, "OpenAI",
            lambda **kw: (built.append(1), real(**kw))[1],
        )
        call(agent, times=10)
        assert len(built) == 1

    def test_same_client_object_every_call(self, agent):
        call(agent)
        first = agent._client_cache["openai"][1]
        call(agent, times=5)
        assert agent._client_cache["openai"][1] is first

    def test_one_connection_pool(self, agent):
        call(agent, times=5)
        client = agent._client_cache["openai"][1]
        # A second call must not have replaced the pool underneath it.
        pool = client._client._transport._pool
        call(agent)
        assert agent._client_cache["openai"][1]._client._transport._pool is pool

    def test_timeout_is_applied(self, agent):
        call(agent)
        client = agent._client_cache["openai"][1]
        assert client.timeout == _REQUEST_TIMEOUT

    def test_changed_base_url_rebuilds(self, agent):
        call(agent)
        first = agent._client_cache["openai"][1]
        agent.base_url = "https://example.test/v1"
        call(agent)
        assert agent._client_cache["openai"][1] is not first
        assert agent._client_cache["openai"][0][1] == "https://example.test/v1"

    def test_changed_api_key_rebuilds(self, agent):
        call(agent)
        first = agent._client_cache["openai"][1]
        agent.api_key = "different-key"
        call(agent)
        assert agent._client_cache["openai"][1] is not first

    def test_agents_do_not_share_a_client(self):
        a = ExternalAIAgent("A", api_provider="openai", model="gpt-4o", api_key="k")
        b = ExternalAIAgent("B", api_provider="openai", model="gpt-4o", api_key="k")
        call(a)
        call(b)
        assert a._client_cache["openai"][1] is not b._client_cache["openai"][1]


class TestAnthropicClient:
    def test_built_once_and_bounded_by_the_timeout(self, monkeypatch):
        anthropic = pytest.importorskip("anthropic")
        agent = ExternalAIAgent(
            "Ada", api_provider="anthropic", model="claude-sonnet-5", api_key="k"
        )
        built = []
        real = anthropic.Anthropic
        monkeypatch.setattr(
            anthropic, "Anthropic",
            lambda **kw: (built.append(kw), real(**kw))[1],
        )
        for _ in range(5):
            try:
                agent._call_anthropic("hi")
            except Exception:
                pass
        assert len(built) == 1
        # Previously left at the SDK's 600s default, long past the point the
        # dashboard has abandoned the round.
        assert built[0]["timeout"] == _REQUEST_TIMEOUT


class TestCacheHelper:
    def test_rebuilds_only_when_the_key_changes(self, agent):
        calls = []

        def build():
            calls.append(1)
            return object()

        first = agent._cached_client("x", ("a",), build)
        assert agent._cached_client("x", ("a",), build) is first
        second = agent._cached_client("x", ("b",), build)
        assert second is not first
        assert len(calls) == 2

    def test_kinds_are_isolated(self, agent):
        one = agent._cached_client("openai", ("k",), object)
        two = agent._cached_client("anthropic", ("k",), object)
        assert one is not two

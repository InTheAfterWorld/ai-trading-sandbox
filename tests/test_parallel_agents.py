"""Tests for parallel agent-action collection in MarketEnv.step()."""

import threading
import time

from ai_trading_society.config import MarketConfig
from ai_trading_society.market_env import MarketEnv


class _ThreadedAgent:
    """Minimal agent that records the thread it acted on and takes a fixed
    amount of time. Deliberately NOT a BaseAgent subclass that hits LLMs —
    just a black box decision maker like BaseAgent.act implies."""

    def __init__(self, agent_id: str, delay: float = 0.15):
        self.agent_id = agent_id
        self.cash = 10000.0
        self.holdings = {}
        self.delay = delay
        self.threads_seen = set()

    def act(self, observation):
        self.threads_seen.add(threading.get_ident())
        time.sleep(self.delay)
        sym = observation["stocks"][0]["symbol"]
        return {"decisions": [{"symbol": sym, "action": "hold", "quantity": 0}]}


class _FailingAgent(_ThreadedAgent):
    def act(self, observation):
        raise RuntimeError("LLM unavailable")


def _make_env(agents, seed=42, parallel=True):
    config = MarketConfig(initial_price=100.0, seed=seed)
    config.parallel_agents = parallel
    return MarketEnv(config, agents, seed=seed)


def test_parallel_collection_runs_agents_concurrently():
    """Multiple agents' act() calls must overlap in distinct threads (synchronized via Barrier)."""
    barrier = threading.Barrier(4, timeout=5.0)

    class BarrierAgent(_ThreadedAgent):
        def act(self, observation):
            self.threads_seen.add(threading.get_ident())
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                raise AssertionError("agents did not overlap concurrently")
            time.sleep(self.delay)
            sym = observation["stocks"][0]["symbol"]
            return {"decisions": [{"symbol": sym, "action": "hold", "quantity": 0}]}

    agents = [BarrierAgent(f"a{i}", delay=0.01) for i in range(4)]
    env = _make_env(agents)

    start = time.perf_counter()
    env.step()
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"expected quick completion with overlap, took {elapsed:.2f}s"
    used_threads = {t for a in agents for t in a.threads_seen}
    assert len(used_threads) > 1, "agents should act on separate worker threads"
    env.close()


def test_sequential_mode_when_disabled():
    """parallel_agents=False keeps everything on the calling thread."""
    agents = [_ThreadedAgent("x1", delay=0.01), _ThreadedAgent("x2", delay=0.01)]
    env = _make_env(agents, parallel=False)
    main_thread = threading.get_ident()
    env.step()
    for agent in agents:
        assert agent.threads_seen == {main_thread}
    env.close()


def test_parallel_and_sequential_produce_identical_results():
    """With deterministic agents, parallel collection must not change any
    market outcome versus sequential collection."""
    def build(parallel):
        from tests.conftest import ScriptedExternalAIAgent
        agents = [
            ScriptedExternalAIAgent("buyer", cash=10000, holdings=0, buy_prob=1.0),
            ScriptedExternalAIAgent("seller", cash=0, holdings=100, sell_prob=1.0),
            ScriptedExternalAIAgent("holder", cash=5000, holdings=10),
        ]
        return _make_env(agents, seed=7, parallel=parallel)

    env_p = build(True)
    env_s = build(False)
    for _ in range(8):
        env_p.step()
        env_s.step()

    prices_p = [sm.price_history[:] for sm in env_p.stocks.values()]
    prices_s = [sm.price_history[:] for sm in env_s.stocks.values()]
    assert prices_p == prices_s
    assert [t.__repr__() for t in env_p.trade_history] == [
        t.__repr__() for t in env_s.trade_history
    ]
    env_p.close()
    env_s.close()


def test_seeded_parallel_run_is_reproducible():
    """Two seeded runs with per-agent RNG streams must match exactly even
    though act() completion order may vary between runs."""
    def run(seed_val):
        from ai_trading_society.agents.traits import create_personality_agent
        from tests.conftest import ScriptedExternalAIAgent
        agents = []
        for i in range(3):
            base = ScriptedExternalAIAgent(
                f"t{i}", cash=8000 + i, holdings=i * 5,
                buy_prob=0.5, sell_prob=0.5,
            )
            agents.append(create_personality_agent(base, personality="aggressive"))
        env = _make_env(agents, seed=seed_val)
        sim_prices = []
        for _ in range(12):
            env.step()
            sm = next(iter(env.stocks.values()))
            sim_prices.append(sm.price)
        env.close()
        return sim_prices

    assert run(1234) == run(1234)


def test_failing_agent_records_error_in_parallel_mode():
    """Error handling parity: a raising agent yields hold decisions and an
    error count, same as the sequential path."""
    good = _ThreadedAgent("good", delay=0.01)
    bad = _FailingAgent("bad")
    env = _make_env([good, bad])
    env.step()

    assert env._agent_error_counts.get("bad") == 1
    assert env._agent_error_counts.get("good", 0) == 0
    env.close()


def test_single_agent_skips_executor():
    """With one agent there is nothing to overlap; take the serial path."""
    agent = _ThreadedAgent("only", delay=0.01)
    env = _make_env([agent])
    main_thread = threading.get_ident()
    env.step()
    assert agent.threads_seen == {main_thread}
    assert env._pool is None
    env.close()


def test_thread_pool_reused_across_steps_and_closed():
    """MarketEnv should reuse the ThreadPoolExecutor across steps and cleanly shut down on close()."""
    agents = [_ThreadedAgent(f"a{i}", delay=0.01) for i in range(3)]
    env = _make_env(agents)

    env.step()
    pool1 = env._pool
    assert pool1 is not None

    env.step()
    pool2 = env._pool
    assert pool1 is pool2

    env.close()
    assert env._pool is None


def test_thread_pool_workers_capped():
    """Max workers should be capped at 16 even with many agents."""
    agents = [_ThreadedAgent(f"a{i}", delay=0.001) for i in range(25)]
    env = _make_env(agents)
    pool = env._get_pool()
    assert getattr(pool, "_max_workers", 0) == 16
    env.close()

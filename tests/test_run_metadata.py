"""
Unit tests for run metadata, versioning, random seed reproducibility, and snapshot saving.
"""

import json
import os
import tempfile
import pytest

from ai_trading_society.config import MarketConfig
from ai_trading_society.market_env import MarketEnv
from ai_trading_society.simulator import Simulator
from tests.conftest import ScriptedExternalAIAgent
from ai_trading_society.run_metadata import (
    RunMetadata,
    get_code_version,
    load_run_snapshot,
    save_run_snapshot,
    set_seed,
)


def test_set_seed_reproducibility():
    """Test that set_seed ensures reproducible random numbers."""
    s1 = set_seed(42)
    val1 = [set_seed(42) and __import__("random").random() for _ in range(5)]

    s2 = set_seed(42)
    val2 = [set_seed(42) and __import__("random").random() for _ in range(5)]

    assert s1 == 42
    assert s2 == 42
    assert val1 == val2


def test_code_version_retrieval():
    """Test get_code_version returns valid version dictionary."""
    ver = get_code_version()
    assert isinstance(ver, dict)
    assert "package_version" in ver
    assert ver["package_version"] == "0.2.0"
    assert "git_commit" in ver
    assert "git_dirty" in ver


def test_run_metadata_creation():
    """Test creating RunMetadata object from config and agents."""
    config = MarketConfig(initial_price=100.0, seed=12345)
    agents = [
        ScriptedExternalAIAgent("Agent_A", cash=10000, holdings=20),
        ScriptedExternalAIAgent("Agent_B", cash=10000, holdings=20),
    ]

    meta = RunMetadata.create(config=config, agents=agents, seed=12345)

    assert meta.run_id.startswith("run_")
    assert meta.seed == 12345
    assert meta.config["initial_price"] == 100.0
    assert meta.config["seed"] == 12345
    assert len(meta.agents) == 2
    assert meta.agents[0]["agent_id"] == "Agent_A"
    assert meta.agents[1]["agent_id"] == "Agent_B"
    assert "python_version" in meta.environment


def test_simulation_reproducibility_with_seed():
    """Test that two simulation runs with the same seed yield identical trajectories."""
    def run_sim(seed_val):
        config = MarketConfig(
            initial_price=100.0,
            seed=seed_val,
        )
        agents = [
            ScriptedExternalAIAgent("Random_1", cash=10000, holdings=20),
            ScriptedExternalAIAgent("MM_1", cash=50000, holdings=500),
        ]
        env = MarketEnv(config, agents, seed=seed_val)
        sim = Simulator(env)
        return sim.run(steps=10, verbose=False, seed=seed_val, save_snapshot=False)

    history1 = run_sim(9999)
    history2 = run_sim(9999)

    prices1 = [s["price"] for s in history1]
    prices2 = [s["price"] for s in history2]

    assert prices1 == prices2


def test_save_and_load_run_snapshot():
    """Test saving and loading run snapshots for post-mortem analysis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MarketConfig(initial_price=150.0, seed=777)
        agents = [ScriptedExternalAIAgent("R1", cash=10000, holdings=10)]
        env = MarketEnv(config, agents, seed=777)
        sim = Simulator(env)
        state_history = sim.run(steps=5, verbose=False, seed=777, save_snapshot=True, runs_dir=tmpdir)

        run_id = sim.metadata.run_id
        run_folder = os.path.join(tmpdir, run_id)

        assert os.path.exists(os.path.join(run_folder, "metadata.json"))
        assert os.path.exists(os.path.join(run_folder, "config.json"))
        assert os.path.exists(os.path.join(run_folder, "state_history.json"))
        assert os.path.exists(os.path.join(run_folder, "trade_history.json"))

        # Load snapshot back
        snapshot = load_run_snapshot(run_id, base_dir=tmpdir)
        assert snapshot["metadata"]["run_id"] == run_id
        assert snapshot["metadata"]["seed"] == 777
        assert len(snapshot["state_history"]) == 5
        assert snapshot["metadata"]["config"]["initial_price"] == 150.0

"""
Run metadata and configuration snapshot management module.

Provides utilities for:
1. Setting reproducible random seeds across random and numpy.
2. Capturing code version (package version and git commit hash/status).
3. Generating unique run_id and metadata snapshots for every simulation.
4. Saving and loading run snapshots for exact post-mortem analysis ("precise review").
"""

import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import MarketConfig


def set_seed(seed: Optional[int] = None) -> int:
    """
    Set random seed across standard library `random` and `numpy` (if installed).

    Parameters
    ----------
    seed : int, optional
        Seed value. If None, a random seed between 1 and 1,000,000,000 is generated.

    Returns
    -------
    int
        The effective seed value used.
    """
    if seed is None:
        seed = random.randint(1, 1_000_000_000)

    random.seed(seed)

    # If numpy is available, seed it as well
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass

    return seed


def get_code_version() -> Dict[str, Any]:
    """
    Get current framework version and git commit information.

    Returns
    -------
    dict
        Dictionary containing:
        - package_version: str
        - git_commit: str or None
        - git_dirty: bool or None
    """
    from . import __version__

    git_commit = None
    git_dirty = None

    try:
        # Get short git commit hash
        commit_res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_res.returncode == 0:
            git_commit = commit_res.stdout.strip()

            # Check if there are uncommitted changes
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            )
            if status_res.returncode == 0:
                git_dirty = len(status_res.stdout.strip()) > 0
    except Exception:
        pass

    return {
        "package_version": __version__,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def generate_run_id(timestamp: Optional[datetime] = None) -> str:
    """
    Generate a unique run ID based on timestamp and random token.

    Example: run_20260811_071530_a1b2c3
    """
    if timestamp is None:
        timestamp = datetime.now()
    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
    rand_hex = f"{random.randint(0, 0xFFFFFF):06x}"
    return f"run_{time_str}_{rand_hex}"


@dataclass
class RunMetadata:
    """
    Comprehensive snapshot of a simulation run.
    """

    run_id: str
    timestamp: str
    seed: int
    version: Dict[str, Any]
    config: Dict[str, Any]
    agents: List[Dict[str, Any]]
    environment: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        config: MarketConfig,
        agents: List[Any],
        seed: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> "RunMetadata":
        """
        Create a new RunMetadata instance for a simulation run.
        """
        eff_seed = set_seed(seed)
        now = datetime.now()
        r_id = run_id or generate_run_id(now)
        version_info = get_code_version()

        # Update config seed if not set
        if config.seed is None:
            config.seed = eff_seed

        # Agent roster snapshot
        agent_roster = []
        for a in agents:
            # TraitAgent wraps the real agent: provider/model live on the
            # wrapped base_agent, while the personality name lives on the
            # wrapper itself.
            base = getattr(a, "base_agent", a)
            agent_info = {
                "agent_id": getattr(a, "agent_id", str(a)),
                "class_name": a.__class__.__name__,
                "initial_cash": getattr(a, "cash", 0.0),
                "initial_holdings": getattr(a, "holdings", 0),
            }
            if hasattr(base, "api_provider"):
                agent_info["api_provider"] = getattr(base, "api_provider")
            if hasattr(base, "model"):
                agent_info["model"] = getattr(base, "model")
            if hasattr(a, "personality_name"):
                agent_info["personality"] = getattr(a, "personality_name")
            # Persona state, so a run can be read back with the character
            # that produced it. Deep-mode only.
            if getattr(a, "deep", False):
                if getattr(a, "disposition", ""):
                    agent_info["disposition"] = getattr(a, "disposition")
                if getattr(a, "dials", None):
                    agent_info["dials"] = dict(getattr(a, "dials"))
                if getattr(a, "mood", None):
                    agent_info["mood"] = dict(getattr(a, "mood"))
            if hasattr(a, "agent_type"):
                agent_info["agent_type"] = getattr(a, "agent_type")
            agent_roster.append(agent_info)

        env_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }

        return cls(
            run_id=r_id,
            timestamp=now.isoformat(),
            seed=eff_seed,
            version=version_info,
            config=config.to_dict(),
            agents=agent_roster,
            environment=env_info,
            summary={},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert RunMetadata to dictionary."""
        return asdict(self)

    def save(self, output_dir: str = "runs") -> str:
        """
        Save metadata snapshot to json file inside `output_dir/<run_id>/metadata.json`.

        Returns
        -------
        str
            Filepath of saved metadata.json.
        """
        run_dir = os.path.join(output_dir, self.run_id)
        os.makedirs(run_dir, exist_ok=True)

        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        # Save standalone config snapshot as well
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        return metadata_path


def save_run_snapshot(
    metadata: RunMetadata,
    state_history: List[Dict[str, Any]],
    trade_history: List[Any],
    output_dir: str = "runs",
) -> str:
    """
    Save complete run artifacts (metadata, state history, trade history) for post-mortem analysis.

    Parameters
    ----------
    metadata : RunMetadata
        Run metadata object.
    state_history : list of dict
        Simulation step states.
    trade_history : list of TradeRecord
        Executed trades.
    output_dir : str
        Base directory to save run folder.

    Returns
    -------
    str
        Path to the run folder.
    """
    run_dir = os.path.join(output_dir, metadata.run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save metadata & config
    metadata.save(output_dir)

    # Save state history
    state_path = os.path.join(run_dir, "state_history.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state_history, f, indent=2, ensure_ascii=False, default=str)

    # Save trade history
    trades_data = []
    for t in trade_history:
        if hasattr(t, "__dict__"):
            trades_data.append(t.__dict__)
        elif isinstance(t, dict):
            trades_data.append(t)
        else:
            trades_data.append(str(t))

    trades_path = os.path.join(run_dir, "trade_history.json")
    with open(trades_path, "w", encoding="utf-8") as f:
        json.dump(trades_data, f, indent=2, ensure_ascii=False, default=str)

    return run_dir


def load_run_snapshot(run_id_or_dir: str, base_dir: str = "runs") -> Dict[str, Any]:
    """
    Load saved run snapshot files for post-mortem review / replay.

    Parameters
    ----------
    run_id_or_dir : str
        Run ID (e.g. "run_20260811_071530_a1b2c3") or path to run directory.
    base_dir : str
        Base runs directory if only run_id is supplied.

    Returns
    -------
    dict
        Dictionary containing metadata, config, state_history, trade_history.
    """
    if os.path.isdir(run_id_or_dir):
        run_dir = run_id_or_dir
    else:
        run_dir = os.path.join(base_dir, run_id_or_dir)

    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    meta_path = os.path.join(run_dir, "metadata.json")
    state_path = os.path.join(run_dir, "state_history.json")
    trades_path = os.path.join(run_dir, "trade_history.json")

    metadata = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    state_history = []
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state_history = json.load(f)

    trade_history = []
    if os.path.exists(trades_path):
        with open(trades_path, "r", encoding="utf-8") as f:
            trade_history = json.load(f)

    return {
        "run_dir": run_dir,
        "metadata": metadata,
        "state_history": state_history,
        "trade_history": trade_history,
    }

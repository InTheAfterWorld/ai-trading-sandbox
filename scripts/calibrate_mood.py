"""Calibration harness for the event-based mood engine.

Runs seeded sandbox simulations with scripted (deterministic) traders for
all 8 personality presets and reports distribution, correlation and preset-
separation statistics of the mood axes. Each preset follows a different
round-robin buy/sell schedule, so every trader alternates winning and losing
stretches (like real LLM runs -- unlike a constant bias, which produces
perpetual winners whose mood pins at 0/10), while the different buy
fractions and phases keep wealth paths apart so the rank / rival events
actually fire.

Reading the report:

- The per-axis band (p10-p90 within [2, 8]) is the gate to enforce. It is
  what the tuning constants in mood.py are chosen against.
- The correlation target is aspirational: every performance event moves
  confidence against stress/frustration at once, so the axes are strongly
  anti-correlated (~0.8) in any performance-driven market. They are not
  redundant copies of each other, but |r| < 0.6 is not reachable with
  simple, interpretable rules.
- Preset separation inherits the geometry of the frozen preset baselines
  (the closest baselines are 0-1.0 apart), so the minimum pairwise distance
  of the mean moods tops out around 1; the report prints the baseline
  minimum for comparison.

``--fit`` additionally grid-searches ``_BASE_POINT`` and a global ``_NORMAL``
scale so each axis's p10-p90 lands closer to [2, 8], and writes suggestions
to scripts/mood_params.suggested.json. Nothing is auto-applied: mood.py
stays the single source of truth -- copy values in by hand.

Usage (from the project root):
    python scripts/calibrate_mood.py
    python scripts/calibrate_mood.py --fit
"""

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import ai_trading_society.agents.mood as mood_mod  # noqa: E402
from ai_trading_society.agents.traits import (  # noqa: E402
    create_personality_agent,
    preset_mood,
)
from ai_trading_society.config import MarketConfig  # noqa: E402
from ai_trading_society.market_env import MarketEnv  # noqa: E402
from tests.conftest import ScriptedExternalAIAgent  # noqa: E402

PRESETS = [
    "balanced", "aggressive", "conservative", "panicky",
    "greedy", "fomo_driven", "stubborn", "emotional",
]
# Per-preset deterministic schedule: (pattern, phase). Pattern letters give
# the round's directive, shifted by ``phase`` rounds: "B" buys, "S" sells
# everything, "H" holds (wealth still moves with the market, so hold rounds
# produce small gains/losses like real runs). Presets come in two phase
# groups (phases 0-3 and 6-9, half a cycle apart) so one group's buying
# meets the other's selling: the market oscillates instead of drifting,
# every trader alternates winning and losing stretches, and streaks break
# before mood pins at an extreme.
SCHEDULES = {
    "balanced": ("BBBBSSSSHHHH", 0),
    "aggressive": ("BBBBBBSSHHHH", 1),
    "conservative": ("BSSHHHHHHHHH", 2),
    "panicky": ("BSBSHHHHHHHH", 3),
    "greedy": ("BBBBSSSSHHHH", 7),
    "fomo_driven": ("BBBBBBSSHHHH", 8),
    "stubborn": ("BBBBHHHHHHHH", 9),
    "emotional": ("BSBSBSBSBSSH", 6),
}
# Starting stock per preset: a spread of initial exposures gives the wealth
# paths room to diverge (real runs spread out a lot more than identical
# starting books), so the rank-change / rival-gap events actually fire.
HOLDINGS = {
    "balanced": 20, "aggressive": 60, "conservative": 10, "panicky": 15,
    "greedy": 45, "fomo_driven": 25, "stubborn": 30, "emotional": 20,
}
SEEDS = [11, 22, 33]
N_ROUNDS = 30

# Calibration targets (advisory gates printed in the report).
TARGET_LO, TARGET_HI = 2.0, 8.0
MAX_CORR = 0.60
MIN_PRESET_DIST = 1.5

_ORIGINAL_BASE_POINT = mood_mod._BASE_POINT
_ORIGINAL_NORMAL = dict(mood_mod._NORMAL)


def _apply_params(base_point: float, normal_scale: float) -> None:
    mood_mod._BASE_POINT = base_point
    mood_mod._NORMAL = {
        key: (value * normal_scale if key != "unit" else value)
        for key, value in _ORIGINAL_NORMAL.items()
    }


def _restore_params() -> None:
    mood_mod._BASE_POINT = _ORIGINAL_BASE_POINT
    mood_mod._NORMAL = dict(_ORIGINAL_NORMAL)


def collect_moods() -> dict:
    """One calibration pass; returns {preset: [[conf, stress, fr], ...]}."""
    samples: dict = {preset: [] for preset in PRESETS}
    for seed in SEEDS:
        config = MarketConfig(
            initial_price=100.0,
            deep_persona=True,
            parallel_agents=False,
            seed=seed,
        )
        agents = []
        bases: dict = {}
        for preset in PRESETS:
            base = ScriptedExternalAIAgent(
                preset, cash=10000.0, holdings=HOLDINGS[preset])
            bases[preset] = base
            agents.append(create_personality_agent(
                base, personality=preset, deep=True,
                use_reported_mood=False,
            ))
        env = MarketEnv(config, agents)
        try:
            for round_idx in range(N_ROUNDS):
                for preset in PRESETS:
                    pattern, phase = SCHEDULES[preset]
                    directive = pattern[(round_idx + phase) % len(pattern)]
                    bases[preset].buy_prob = 0.9 if directive == "B" else 0.0
                    bases[preset].sell_prob = 0.9 if directive == "S" else 0.0
                state = env.step()
                for preset in PRESETS:
                    mood = state["agents"][preset]["mood"]
                    if mood:
                        samples[preset].append(
                            [mood[a] for a in mood_mod.MOOD_AXES]
                        )
        finally:
            env.close()
    return samples


def _percentile(values: list, q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _pearson(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def _axis_columns(samples: dict) -> dict:
    cols = {axis: [] for axis in mood_mod.MOOD_AXES}
    for vectors in samples.values():
        for vector in vectors:
            for axis, value in zip(mood_mod.MOOD_AXES, vector):
                cols[axis].append(value)
    return cols


def _preset_means(samples: dict) -> dict:
    means = {}
    for preset, vectors in samples.items():
        means[preset] = [
            sum(v[i] for v in vectors) / len(vectors)
            if vectors else float("nan")
            for i in range(3)
        ]
    return means


def _min_pairwise_distance(means: dict):
    best = None
    presets = sorted(means)
    for i, a in enumerate(presets):
        for b in presets[i + 1:]:
            dist = math.dist(means[a], means[b])
            if best is None or dist < best[0]:
                best = (dist, a, b)
    return best


def _axis_gate(values: list) -> bool:
    p10 = _percentile(values, 0.10)
    p90 = _percentile(values, 0.90)
    return p10 >= TARGET_LO and p90 <= TARGET_HI


def print_report(samples: dict) -> None:
    total = sum(len(v) for v in samples.values())
    print(f"=== Mood calibration report: {len(PRESETS)} presets x "
          f"{len(SEEDS)} seeds x {N_ROUNDS} rounds ({total} mood vectors) ===")
    print()

    cols = _axis_columns(samples)
    print(f"Per-axis distribution "
          f"(target: p10-p90 within [{TARGET_LO:g}, {TARGET_HI:g}], "
          f"not pinned at 0/10):")
    for axis, values in cols.items():
        print(f"  {axis:<12} min {_percentile(values, 0.0):5.2f}  "
              f"p10 {_percentile(values, 0.10):5.2f}  "
              f"p50 {_percentile(values, 0.50):5.2f}  "
              f"p90 {_percentile(values, 0.90):5.2f}  "
              f"max {_percentile(values, 1.0):5.2f}  "
              f"-> {'PASS' if _axis_gate(values) else 'FAIL'}")
    print()

    axes = list(mood_mod.MOOD_AXES)
    print(f"Axis-pair Pearson |r| (target < {MAX_CORR:g}):")
    for i, a in enumerate(axes):
        for b in axes[i + 1:]:
            r = abs(_pearson(cols[a], cols[b]))
            print(f"  {a + '/' + b:<26} {r:5.2f}  "
                  f"-> {'PASS' if r < MAX_CORR else 'FAIL'}")
    print()

    means = _preset_means(samples)
    print(f"Preset mean mood (target: min pairwise distance > "
          f"{MIN_PRESET_DIST:g}):")
    for preset in PRESETS:
        m = means[preset]
        print(f"  {preset:<12} conf {m[0]:5.2f}  stress {m[1]:5.2f}  "
              f"frustration {m[2]:5.2f}")
    dist, a, b = _min_pairwise_distance(means)
    baselines = [preset_mood(p) for p in PRESETS]
    base_dist, ba, bb = _min_pairwise_distance(
        {p: list(v.values()) for p, v in zip(PRESETS, baselines)})
    print(f"  min pairwise distance {dist:.2f} ({a} vs {b})  "
          f"-> {'PASS' if dist > MIN_PRESET_DIST else 'FAIL'}")
    print(f"  (the preset baselines themselves come as close as "
          f"{base_dist:.2f}: {ba} vs {bb})")


def _fit_score(samples: dict) -> float:
    score = 0.0
    for values in _axis_columns(samples).values():
        p10 = _percentile(values, 0.10)
        p90 = _percentile(values, 0.90)
        score += max(0.0, TARGET_LO - p10) + max(0.0, p90 - TARGET_HI)
    return score


def fit() -> dict:
    """Grid-search _BASE_POINT and a global _NORMAL scale; write suggestions."""
    # The shipped constants are always a candidate, so a search that finds
    # nothing better recommends leaving mood.py alone instead of drifting
    # to the nearest grid point.
    grid = [(_ORIGINAL_BASE_POINT, 1.0)] + [
        (bp, ns)
        for bp in (0.5, 0.75, 1.0, 1.5, 2.0)
        for ns in (0.5, 1.0, 2.0)
        if (bp, ns) != (_ORIGINAL_BASE_POINT, 1.0)
    ]
    best = None
    for base_point, normal_scale in grid:
        _apply_params(base_point, normal_scale)
        try:
            samples = collect_moods()
        finally:
            _restore_params()
        score = _fit_score(samples)
        drift = abs(base_point - _ORIGINAL_BASE_POINT) + abs(normal_scale - 1.0)
        print(f"  base_point={base_point:<4} normal_scale={normal_scale:<4} "
              f"score={score:6.2f}")
        if best is None or (score, drift) < (best[0], best[1]):
            best = (score, drift, base_point, normal_scale, samples)
    score, _, base_point, normal_scale, samples = best
    return {
        "score": score, "base_point": base_point,
        "normal_scale": normal_scale, "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true",
                        help="grid-search tuning constants and write "
                             "scripts/mood_params.suggested.json")
    args = parser.parse_args()

    default_samples = collect_moods()
    print_report(default_samples)

    if not args.fit:
        return

    print()
    print("=== Fit: grid search over _BASE_POINT and a global _NORMAL scale ===")
    best = fit()
    print()
    print(f"Best: _BASE_POINT={best['base_point']} "
          f"normal_scale={best['normal_scale']} "
          f"(score {best['score']:.2f})")
    print("Per-axis before/after (default -> suggested):")
    for axis, before in _axis_columns(default_samples).items():
        after = _axis_columns(best["samples"])[axis]
        print(f"  {axis:<12} p10 {_percentile(before, 0.10):5.2f} -> "
              f"{_percentile(after, 0.10):5.2f}   "
              f"p90 {_percentile(before, 0.90):5.2f} -> "
              f"{_percentile(after, 0.90):5.2f}")

    out = {
        "base_point": best["base_point"],
        "normal_scale": best["normal_scale"],
        "score": best["score"],
        "per_axis": {
            axis: {
                "p10": round(_percentile(values, 0.10), 3),
                "p90": round(_percentile(values, 0.90), 3),
            }
            for axis, values in _axis_columns(best["samples"]).items()
        },
        "note": ("Advisory only: copy these into "
                 "ai_trading_society/agents/mood.py by hand; mood.py stays "
                 "the single source of truth."),
    }
    out_path = Path(__file__).resolve().parent / "mood_params.suggested.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSuggestions written to {out_path}")


if __name__ == "__main__":
    main()

"""Deterministic, event-based mood dynamics for deep-mode traders.

Mood is three axes (confidence / stress / frustration, 0-10). Each round a
handful of plain events nudge specific axes; stress and frustration then
recover toward the personality's baseline; every axis is capped to a small
move per round. Mood is prompt flavour only -- nothing here can change a
trade. See scripts/calibrate_mood.py to sanity-check the constants.

Both paths are capped by ``mood_max_step``: the rules alone cannot move an
axis further in one round, and the model's reported adjustment is clamped to
the rule value +/- ``_REPORT_ADJUST`` and then to the previous value +/-
``mood_max_step`` -- when the rules already moved an axis to the cap, a
report can only pull it back toward where it was, never push it further.
"""

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

# The three mood axes, 0-10.
MOOD_AXES = ("confidence", "stress", "frustration")

# --- tuning: mood points (0-10) or plain ratios; re-check with
# scripts/calibrate_mood.py ------------------------------------------------
_BASE_POINT = 0.4            # points a "one unit" event moves an axis
_NORMAL = {                  # what counts as one unit of each driver
    "gain_pct": 2.0,        # a 2% round               == 1 unit
    "drawdown_pct": 10.0,   # 10% off the peak         == 1 unit
    "vol_pct": 3.0,         # 3% average abs move     == 1 unit
    "rival_gap_pct": 10.0,  # 10 pts behind the leader == 1 unit
    "unit": 1.0,            # streak lengths and rank moves are in units
}
_SENS = (0.4, 0.12)         # dial -> sensitivity: 0.4 + 0.12*dial (5 -> 1.0)
_RECOVERY = (0.30, 0.04)    # resilience -> pull to baseline: 0.30 + 0.04*dial
_CONF_REVERT = 0.6          # confidence reverts to baseline faster than stress
_REPORT_ADJUST = 1.5        # max points the LLM may shift mood off the rules


class _EventRule(NamedTuple):
    """One mood event.

    ``driver`` returns the raw magnitude of what happened (percent points,
    streak lengths, rank moves); the event fires whenever it is positive.
    ``normal`` is the ``_NORMAL`` key counting as one unit: the driver is
    divided by it and clipped to ``[lo, hi]`` to get the event size. Each
    effect is ``(axis, coefficient, sensitivity dial or None)``.
    """

    name: str
    driver: Callable[[Dict[str, Any]], float]
    normal: str
    lo: float
    hi: float
    effects: Tuple[Tuple[str, float, Optional[str]], ...]


_EVENT_RULES: Tuple[_EventRule, ...] = (
    _EventRule(
        "gain_round", lambda m: float(m["round_return_pct"]),
        "gain_pct", 0.0, 3.0,
        (("confidence", 1.0, "risk_appetite"),),
    ),
    _EventRule(
        "loss_round", lambda m: -float(m["round_return_pct"]),
        "gain_pct", 0.0, 3.0,
        (("confidence", -1.0, None), ("stress", 1.0, "loss_sensitivity")),
    ),
    _EventRule(
        "drawdown", lambda m: float(m["drawdown_pct"]),
        "drawdown_pct", 0.0, 3.0,
        (("confidence", -0.5, None), ("stress", 1.0, "loss_sensitivity")),
    ),
    _EventRule(
        "losing_streak", lambda m: float(m["loss_streak"]) - 1.0,
        "unit", 0.0, 4.0,
        (("confidence", -0.5, None), ("stress", 0.5, "loss_sensitivity"),
         ("frustration", 1.0, "loss_sensitivity")),
    ),
    _EventRule(
        "winning_streak", lambda m: float(m["win_streak"]) - 1.0,
        "unit", 0.0, 4.0,
        (("confidence", 1.0, None), ("stress", -0.5, None),
         ("frustration", -0.5, None)),
    ),
    _EventRule(
        "volatile_market", lambda m: float(m["vol_pct"]),
        "vol_pct", 0.0, 2.0,
        (("stress", 0.5, "loss_sensitivity"),),
    ),
    _EventRule(
        "rank_up", lambda m: float(m["rank_delta"]),
        "unit", 0.0, 3.0,
        (("confidence", 0.5, None), ("frustration", -1.0, "envy")),
    ),
    _EventRule(
        "rank_down", lambda m: -float(m["rank_delta"]),
        "unit", 0.0, 3.0,
        (("confidence", -0.5, None), ("frustration", 1.0, "envy")),
    ),
    _EventRule(
        "rival_ahead", lambda m: float(m["rival_gap_pct"]),
        "rival_gap_pct", 0.0, 3.0,
        (("frustration", 1.0, "envy"),),
    ),
)


def _clip(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))


def _sensitivity(dials: Dict[str, float], dial: Optional[str]) -> float:
    """Map a 0-10 dial to a 0.4-1.6 sensitivity multiplier."""
    if dial is None:
        return 1.0
    return _SENS[0] + _SENS[1] * float(dials.get(dial, 5.0))


def _volatility_pct(observation: Dict[str, Any]) -> float:
    """Mean absolute percent move over the last <=5 visible prices."""
    history = observation.get("price_history")
    if not isinstance(history, (list, tuple)):
        return 0.0
    points: List[float] = []
    for raw in history[-5:]:
        try:
            points.append(float(raw))
        except (TypeError, ValueError):
            continue
    moves = [
        abs(points[i] / points[i - 1] - 1.0) * 100.0
        for i in range(1, len(points))
        if points[i - 1] > 0
    ]
    if not moves:
        return 0.0
    return sum(moves) / len(moves)


def objective_pressure(
    observation: Dict[str, Any],
    *,
    peak_wealth: Optional[float],
    prev_wealth: Optional[float],
    prev_rank: Optional[int],
    loss_streak: int,
    win_streak: int,
) -> Tuple[str, Dict[str, Any]]:
    """Describe what just happened to this agent, in plain language.

    Returns the sentence handed to the model plus the metrics dict the mood
    rules need. All percent metrics are in percentage points (e.g. -41.7).
    ``rank`` is the agent's current standing (None when unknown) and
    ``rank_delta`` is ``prev_rank - rank`` -- positive means the agent
    climbed; it is 0 on the first ranked round.
    """
    wealth = float(observation.get("my_wealth", 0.0) or 0.0)
    peak = peak_wealth if peak_wealth else wealth
    drawdown_pct = (peak - wealth) / peak * 100.0 if peak > 0 else 0.0
    round_return_pct = 0.0
    if prev_wealth and prev_wealth > 0:
        round_return_pct = (wealth - prev_wealth) / prev_wealth * 100.0

    standing = observation.get("standing") or {}
    rival_gap_pct = 0.0
    rank: Optional[int] = None
    if isinstance(standing, dict):
        gap = standing.get("gap_to_leader_pct")
        if gap is not None:
            try:
                rival_gap_pct = max(0.0, float(gap))
            except (TypeError, ValueError):
                rival_gap_pct = 0.0
        raw_rank = standing.get("rank")
        if raw_rank is not None:
            try:
                rank = int(float(raw_rank))
            except (TypeError, ValueError):
                rank = None
    rank_delta = 0.0
    if rank is not None and prev_rank is not None:
        rank_delta = float(prev_rank - rank)

    metrics: Dict[str, Any] = {
        "round_return_pct": round_return_pct,
        "drawdown_pct": drawdown_pct,
        "vol_pct": _volatility_pct(observation),
        "rival_gap_pct": rival_gap_pct,
        "loss_streak": float(loss_streak),
        "win_streak": float(win_streak),
        "rank": rank,
        "rank_delta": rank_delta,
    }

    bits = []
    if prev_wealth:
        direction = "up" if round_return_pct >= 0 else "down"
        bits.append(f"Last round you were {direction} {abs(round_return_pct):.1f}%")
    if drawdown_pct > 1.0:
        bits.append(f"you are {drawdown_pct:.1f}% below your peak")
    if loss_streak >= 2:
        bits.append(f"that is {loss_streak} losing rounds in a row")
    elif win_streak >= 2:
        bits.append(f"that is {win_streak} winning rounds in a row")
    if rank_delta >= 1:
        places = "place" if rank_delta == 1 else "places"
        bits.append(f"you climbed {int(rank_delta)} {places}")
    elif rank_delta <= -1:
        places = "place" if rank_delta == -1 else "places"
        bits.append(f"you slipped {int(-rank_delta)} {places}")
    pressure = ("; ".join(bits) + ".") if bits else ""
    return pressure, metrics


def event_mood(
    mood: Dict[str, float],
    baseline: Dict[str, float],
    dials: Dict[str, float],
    metrics: Dict[str, Any],
    *,
    mood_intensity: float,
    mood_max_step: float,
) -> Dict[str, float]:
    """One deterministic rule-engine step of the mood.

    Each firing event nudges specific axes by ``coefficient * size *
    _BASE_POINT * sensitivity``; the event deltas are scaled by
    ``mood_intensity`` and then stress / frustration / confidence are pulled
    back toward the baseline (confidence at half the rate). The total move
    per axis is capped at ``mood_max_step`` and the result at 0-10.
    """
    delta = {axis: 0.0 for axis in MOOD_AXES}
    for rule in _EVENT_RULES:
        raw = rule.driver(metrics)
        if raw <= 0.0:
            continue
        size = _clip(raw / _NORMAL[rule.normal], rule.lo, rule.hi)
        for axis, coef, dial in rule.effects:
            delta[axis] += coef * size * _BASE_POINT * _sensitivity(dials, dial)

    for axis in MOOD_AXES:
        delta[axis] *= mood_intensity

    r = _RECOVERY[0] + _RECOVERY[1] * float(dials.get("resilience", 5.0))
    delta["stress"] -= r * (mood["stress"] - baseline["stress"])
    delta["frustration"] -= r * (mood["frustration"] - baseline["frustration"])
    delta["confidence"] -= (
        r * _CONF_REVERT * (mood["confidence"] - baseline["confidence"])
    )

    out: Dict[str, float] = {}
    for axis in MOOD_AXES:
        step = _clip(delta[axis], -mood_max_step, mood_max_step)
        out[axis] = _clip(mood[axis] + step)
    return out


def _usable_reported(reported: Any) -> Optional[Dict[str, float]]:
    """Return the three finite axes if ``reported`` is usable, else None."""
    if not isinstance(reported, dict):
        return None
    values: Dict[str, float] = {}
    for axis in MOOD_AXES:
        raw = reported.get(axis)
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if value != value or value in (float("inf"), float("-inf")):
            return None
        values[axis] = value
    return values


def settle_mood(
    mood: Dict[str, float],
    baseline: Dict[str, float],
    dials: Dict[str, float],
    reported: Any,
    metrics: Dict[str, Any],
    *,
    use_reported_mood: bool,
    mood_intensity: float,
    mood_max_step: float,
) -> Dict[str, float]:
    """Controlled hybrid of the rule engine and the model's self-report.

    The rules always run and always produce a valid mood. When the model
    reports a usable mood, each axis is clamped to the rule value +/-
    ``_REPORT_ADJUST`` and then to the previous value +/- ``mood_max_step``,
    so one round can never move an axis further than ``mood_max_step`` no
    matter which path produced it.
    """
    rule_mood = event_mood(
        mood, baseline, dials, metrics,
        mood_intensity=mood_intensity, mood_max_step=mood_max_step,
    )
    values = _usable_reported(reported) if use_reported_mood else None
    if values is None:
        return rule_mood
    out: Dict[str, float] = {}
    for axis in MOOD_AXES:
        v = _clip(
            values[axis],
            rule_mood[axis] - _REPORT_ADJUST,
            rule_mood[axis] + _REPORT_ADJUST,
        )
        v = _clip(v, mood[axis] - mood_max_step, mood[axis] + mood_max_step)
        out[axis] = _clip(v)
    return out

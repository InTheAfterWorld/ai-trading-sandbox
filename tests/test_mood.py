"""Pure-function tests for the event-based mood engine (agents/mood.py).

Covers each event's direction, recovery toward the baseline, the per-round
cap, the controlled-hybrid clamps and the objective_pressure metrics.

Expected values are derived from the tuning constants in mood.py, so
recalibrating those constants does not silently break these tests: they pin
the rule-table semantics, not the current magnitudes.
"""

import itertools

import pytest

from ai_trading_society.agents.mood import (
    _BASE_POINT,
    _CONF_REVERT,
    _NORMAL,
    _RECOVERY,
    _REPORT_ADJUST,
    _SENS,
    MOOD_AXES,
    event_mood,
    objective_pressure,
    settle_mood,
)

BASELINE = {"confidence": 5.0, "stress": 3.0, "frustration": 2.0}
DIALS = {
    "risk_appetite": 5.0, "loss_sensitivity": 5.0, "herd_pull": 5.0,
    "patience": 5.0, "resilience": 5.0, "envy": 4.0, "conviction": 5.0,
}
# Dial sensitivities and the recovery rate implied by the constants above.
S_LOSS = _SENS[0] + _SENS[1] * DIALS["loss_sensitivity"]
S_ENVY = _SENS[0] + _SENS[1] * DIALS["envy"]
S_CONF = _SENS[0] + _SENS[1] * DIALS["risk_appetite"]
R = _RECOVERY[0] + _RECOVERY[1] * DIALS["resilience"]
B = _BASE_POINT


def _metrics(**over):
    m = {
        "round_return_pct": 0.0, "drawdown_pct": 0.0, "vol_pct": 0.0,
        "rival_gap_pct": 0.0, "loss_streak": 0.0, "win_streak": 0.0,
        "rank": None, "rank_delta": 0.0,
    }
    m.update(over)
    return m


def _event(mood=None, baseline=None, dials=None, **over):
    return event_mood(
        dict(mood or BASELINE), dict(baseline or BASELINE),
        dict(dials or DIALS), _metrics(**over),
        mood_intensity=1.0, mood_max_step=3.0,
    )


def _settle(mood=None, reported=None, **over):
    return settle_mood(
        dict(mood or BASELINE), dict(BASELINE), dict(DIALS),
        reported, _metrics(**over),
        use_reported_mood=True, mood_intensity=1.0, mood_max_step=3.0,
    )


def _obs(wealth=10000.0, standing=None, price_history=None):
    obs = {
        "my_wealth": wealth,
        "price": 100.0,
        "price_history": price_history if price_history is not None else [100, 100, 100],
    }
    if standing is not None:
        obs["standing"] = standing
    return obs


class TestEventDirections:
    def test_gain_round_raises_confidence(self):
        out = _event(round_return_pct=2.0)          # 2% round == one unit
        assert out["confidence"] == pytest.approx(5.0 + B * S_CONF)
        assert out["stress"] == pytest.approx(3.0)

    def test_gain_size_is_clipped(self):
        out = _event(round_return_pct=10.0)
        assert out["confidence"] == pytest.approx(5.0 + 3.0 * B * S_CONF)

    def test_loss_round_lowers_confidence_raises_stress(self):
        out = _event(round_return_pct=-2.0)
        assert out["confidence"] == pytest.approx(5.0 - B)
        assert out["stress"] == pytest.approx(3.0 + B * S_LOSS)

    def test_drawdown_bites_confidence_and_stress(self):
        out = _event(drawdown_pct=10.0)             # 10% off peak == one unit
        assert out["confidence"] == pytest.approx(5.0 - 0.5 * B)
        assert out["stress"] == pytest.approx(3.0 + B * S_LOSS)

    def test_losing_streak_hits_all_three_axes(self):
        out = _event(loss_streak=3.0)
        size = 3.0 - 1.0
        assert out["confidence"] == pytest.approx(5.0 - 0.5 * B * size)
        assert out["stress"] == pytest.approx(3.0 + 0.5 * B * size * S_LOSS)
        assert out["frustration"] == pytest.approx(2.0 + B * size * S_LOSS)

    def test_winning_streak_lifts_all_three_axes(self):
        out = _event(win_streak=3.0)
        size = 3.0 - 1.0
        assert out["confidence"] == pytest.approx(5.0 + B * size)
        assert out["stress"] == pytest.approx(3.0 - 0.5 * B * size)
        assert out["frustration"] == pytest.approx(2.0 - 0.5 * B * size)

    def test_volatile_market_raises_stress_only(self):
        out = _event(vol_pct=3.0)                   # 3% mean move == one unit
        assert out["stress"] == pytest.approx(3.0 + 0.5 * B * S_LOSS)
        assert out["confidence"] == pytest.approx(5.0)
        assert out["frustration"] == pytest.approx(2.0)

    def test_rank_up_raises_confidence_eases_frustration(self):
        out = _event(rank_delta=2.0)
        assert out["confidence"] == pytest.approx(5.0 + 0.5 * B * 2.0)
        assert out["frustration"] == pytest.approx(2.0 - B * 2.0 * S_ENVY)

    def test_rank_down_lowers_confidence_raises_frustration(self):
        out = _event(rank_delta=-2.0)
        assert out["confidence"] == pytest.approx(5.0 - 0.5 * B * 2.0)
        assert out["frustration"] == pytest.approx(2.0 + B * 2.0 * S_ENVY)

    def test_rival_ahead_raises_frustration(self):
        out = _event(rival_gap_pct=15.0)
        size = 15.0 / _NORMAL["rival_gap_pct"]
        assert out["frustration"] == pytest.approx(2.0 + B * size * S_ENVY)
        assert out["stress"] == pytest.approx(3.0)

    def test_flat_calm_round_changes_nothing(self):
        assert _event() == BASELINE


class TestRecovery:
    def test_stress_recovers_toward_baseline(self):
        mood = {**BASELINE, "stress": 7.0}         # 4 points above baseline
        out = _event(mood=mood)
        assert out["stress"] == pytest.approx(7.0 - R * 4.0)

    def test_frustration_recovers_toward_baseline(self):
        mood = {**BASELINE, "frustration": 6.0}    # 4 points above baseline
        out = _event(mood=mood)
        assert out["frustration"] == pytest.approx(6.0 - R * 4.0)

    def test_confidence_reverts_at_the_conf_revert_rate(self):
        mood = {**BASELINE, "confidence": 8.0}     # 3 points above baseline
        out = _event(mood=mood)
        assert out["confidence"] == pytest.approx(8.0 - R * _CONF_REVERT * 3.0)

    def test_below_baseline_pulls_up(self):
        mood = {**BASELINE, "stress": 0.0}         # 3 points below baseline
        out = _event(mood=mood)
        assert out["stress"] == pytest.approx(R * 3.0)

    def test_resilience_speeds_recovery(self):
        brittle = _event(mood={**BASELINE, "stress": 9.0},
                         dials={**DIALS, "resilience": 0.0})
        tough = _event(mood={**BASELINE, "stress": 9.0},
                       dials={**DIALS, "resilience": 10.0})
        assert tough["stress"] < brittle["stress"]


class TestPerRoundCapAndScaling:
    def test_rule_path_is_capped_per_round(self):
        # Every axis's raw delta far exceeds mood_max_step=3, so each lands
        # exactly one cap away from the previous mood.
        out = _event(round_return_pct=-50.0, drawdown_pct=50.0,
                    loss_streak=5.0, vol_pct=10.0, rank_delta=-3.0,
                    rival_gap_pct=30.0)
        assert out["stress"] == pytest.approx(3.0 + 3.0)
        assert out["confidence"] == pytest.approx(5.0 - 3.0)
        assert out["frustration"] == pytest.approx(2.0 + 3.0)

    def test_mood_intensity_scales_event_deltas_only(self):
        flat = _event(round_return_pct=-2.0)
        hot = event_mood(dict(BASELINE), dict(BASELINE), dict(DIALS),
                         _metrics(round_return_pct=-2.0),
                         mood_intensity=2.0, mood_max_step=3.0)
        assert hot["stress"] == pytest.approx(3.0 + 2 * (flat["stress"] - 3.0))
        # recovery is unaffected by intensity (mood == baseline here)
        calm = _event(mood={**BASELINE, "stress": 9.0})
        warm = event_mood({**BASELINE, "stress": 9.0}, dict(BASELINE),
                          dict(DIALS), _metrics(),
                          mood_intensity=2.0, mood_max_step=3.0)
        assert warm["stress"] == pytest.approx(calm["stress"])


class TestSettleHybrid:
    def test_reported_within_window_is_adopted(self):
        assert _settle(reported={"confidence": 6.0, "stress": 4.0,
                                "frustration": 3.0}) == {
            "confidence": 6.0, "stress": 4.0, "frustration": 3.0,
        }

    def test_reported_is_clamped_to_rule_window(self):
        out = _settle(reported={"confidence": 99.0, "stress": -20.0,
                                "frustration": 5.0})
        assert out["confidence"] == pytest.approx(BASELINE["confidence"] + _REPORT_ADJUST)
        assert out["stress"] == pytest.approx(BASELINE["stress"] - _REPORT_ADJUST)
        assert out["frustration"] == pytest.approx(BASELINE["frustration"] + _REPORT_ADJUST)
        assert all(0.0 <= v <= 10.0 for v in out.values())

    def test_extreme_low_report_stops_at_rule_minus_adjust(self):
        out = _settle(reported={"confidence": 0.0, "stress": 0.0,
                               "frustration": 0.0})
        assert out["confidence"] == pytest.approx(5.0 - _REPORT_ADJUST)
        assert out["stress"] == pytest.approx(3.0 - _REPORT_ADJUST)
        assert out["frustration"] == pytest.approx(2.0 - _REPORT_ADJUST)

    def test_report_cannot_leave_prev_plus_minus_max_step(self):
        # The rules already moved stress to the cap (baseline + mood_max_step)
        # on a brutal round; a report of 10 can only pull an axis back toward
        # the previous mood, never push it past the cap.
        out = _settle(reported={"confidence": 0.0, "stress": 10.0,
                                "frustration": 10.0},
                      round_return_pct=-50.0, drawdown_pct=50.0,
                      loss_streak=5.0, vol_pct=10.0, rank_delta=-3.0,
                      rival_gap_pct=30.0)
        assert out["stress"] == pytest.approx(3.0 + 3.0)
        assert out["confidence"] == pytest.approx(5.0 - 3.0)
        assert out["frustration"] == pytest.approx(2.0 + 3.0)

    def test_use_reported_mood_false_ignores_the_report(self):
        out = settle_mood(dict(BASELINE), dict(BASELINE), dict(DIALS),
                         {"confidence": 10.0, "stress": 10.0, "frustration": 10.0},
                         _metrics(),
                         use_reported_mood=False, mood_intensity=1.0,
                         mood_max_step=3.0)
        assert out == BASELINE

    @pytest.mark.parametrize("bad", [
        None, "not a dict", {},
        {"confidence": 5.0},                                            # incomplete
        {"confidence": "x", "stress": 1.0, "frustration": 1.0},         # non-numeric
        {"confidence": float("nan"), "stress": 1.0, "frustration": 1.0},
    ])
    def test_unusable_report_falls_back_to_the_rules(self, bad):
        assert _settle(reported=bad) == BASELINE

    def test_deterministic(self):
        kwargs = dict(reported={"confidence": 8.0, "stress": 0.0,
                                "frustration": 9.0},
                      round_return_pct=-7.0, drawdown_pct=20.0)
        assert _settle(**kwargs) == _settle(**kwargs)


class TestBoundsProperty:
    def test_mood_stays_in_range_and_within_the_step_cap(self):
        moods = [
            dict(BASELINE),
            {axis: 0.0 for axis in MOOD_AXES},
            {axis: 10.0 for axis in MOOD_AXES},
            {"confidence": 10.0, "stress": 0.0, "frustration": 10.0},
        ]
        combos = itertools.product(
            [0.0, -50.0, 30.0],          # round_return_pct
            [0.0, 50.0],                 # drawdown_pct
            [0.0, 30.0],                 # vol_pct
            [0.0, 5.0],                  # loss_streak
            [0.0, 5.0],                  # win_streak
            [-3.0, 0.0, 3.0],            # rank_delta
            [0.0, 60.0],                 # rival_gap_pct
        )
        for mood in moods:
            for combo in combos:
                keys = ("round_return_pct", "drawdown_pct", "vol_pct",
                        "loss_streak", "win_streak", "rank_delta",
                        "rival_gap_pct")
                out = _event(mood=mood, **dict(zip(keys, combo)))
                for axis in MOOD_AXES:
                    assert 0.0 <= out[axis] <= 10.0
                    assert abs(out[axis] - mood[axis]) <= 3.0 + 1e-9


class TestObjectivePressure:
    def test_basic_metrics(self):
        pressure, metrics = objective_pressure(
            _obs(wealth=8000.0), peak_wealth=10000.0, prev_wealth=10000.0,
            prev_rank=None, loss_streak=0, win_streak=0,
        )
        assert metrics["drawdown_pct"] == pytest.approx(20.0)
        assert metrics["round_return_pct"] == pytest.approx(-20.0)
        assert metrics["rank"] is None
        assert metrics["rank_delta"] == 0.0
        assert pressure == ("Last round you were down 20.0%; "
                            "you are 20.0% below your peak.")

    def test_volatility_is_the_mean_abs_move(self):
        _, metrics = objective_pressure(
            _obs(price_history=[100, 110, 99]), peak_wealth=None,
            prev_wealth=None, prev_rank=None, loss_streak=0, win_streak=0,
        )
        assert metrics["vol_pct"] == pytest.approx(10.0)

    def test_volatility_uses_only_the_last_five_points(self):
        _, metrics = objective_pressure(
            _obs(price_history=[110, 99, 100, 100, 100, 100, 100]),
            peak_wealth=None, prev_wealth=None, prev_rank=None,
            loss_streak=0, win_streak=0,
        )
        assert metrics["vol_pct"] == pytest.approx(0.0)

    def test_rank_and_rank_delta(self):
        standing = {"rank": 2, "gap_to_leader_pct": 30.0}
        _, metrics = objective_pressure(
            _obs(standing=standing), peak_wealth=None, prev_wealth=None,
            prev_rank=4, loss_streak=0, win_streak=0,
        )
        assert metrics["rank"] == 2
        assert metrics["rank_delta"] == 2.0
        assert metrics["rival_gap_pct"] == pytest.approx(30.0)

    def test_rank_climb_and_slip_sentences(self):
        standing = {"rank": 2, "gap_to_leader_pct": 0.0}
        pressure, _ = objective_pressure(
            _obs(standing=standing), peak_wealth=None, prev_wealth=None,
            prev_rank=4, loss_streak=0, win_streak=0,
        )
        assert "you climbed 2 places" in pressure

        pressure, _ = objective_pressure(
            _obs(standing=standing), peak_wealth=None, prev_wealth=None,
            prev_rank=1, loss_streak=0, win_streak=0,
        )
        assert "you slipped 1 place" in pressure

    def test_first_ranked_round_has_no_rank_clause(self):
        standing = {"rank": 5, "gap_to_leader_pct": 12.0}
        pressure, _ = objective_pressure(
            _obs(standing=standing), peak_wealth=None, prev_wealth=None,
            prev_rank=None, loss_streak=0, win_streak=0,
        )
        assert "climbed" not in pressure and "slipped" not in pressure

    def test_streak_sentence(self):
        pressure, _ = objective_pressure(
            _obs(), peak_wealth=None, prev_wealth=10000.0, prev_rank=None,
            loss_streak=3, win_streak=0,
        )
        assert "that is 3 losing rounds in a row" in pressure

    def test_junk_standing_does_not_crash(self):
        _, metrics = objective_pressure(
            _obs(standing="junk"), peak_wealth=None, prev_wealth=None,
            prev_rank=3, loss_streak=0, win_streak=0,
        )
        assert metrics["rank"] is None
        assert metrics["rank_delta"] == 0.0
        assert metrics["rival_gap_pct"] == 0.0

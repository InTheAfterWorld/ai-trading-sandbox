"""The mood timeline: three axes traced round by round.

Mood is bounded 0-10, so the chart pins its axis rather than fitting it to
the data -- a half-point drift must not render as a collapse. That, and the
fact that a run without deep mode says so instead of drawing an empty chart,
are what these tests hold in place.
"""

import re

from ai_trading_society.report_export import (
    MOOD_COLORS,
    _line_chart_svg,
    _mood_section,
    generate_report_html,
)


def _mood_points(*triples):
    return [
        {"step": i + 1, "confidence": c, "stress": s, "frustration": f}
        for i, (c, s, f) in enumerate(triples)
    ]


class TestMoodSection:
    def test_all_three_axes_are_drawn(self):
        html = _mood_section({"Alice": _mood_points((5, 3, 2), (7, 4, 1))})
        for color in MOOD_COLORS.values():
            assert color in html
        for axis in ("Confidence", "Stress", "Frustration"):
            assert axis in html

    def test_one_chart_per_agent(self):
        html = _mood_section({
            "Alice": _mood_points((5, 3, 2), (6, 3, 2)),
            "Bob": _mood_points((4, 6, 5), (3, 7, 6)),
        })
        assert html.count("<svg") == 2
        assert "Alice" in html and "Bob" in html

    def test_absent_mood_says_deep_mode_was_off(self):
        html = _mood_section({})
        assert "deep personality mode" in html.lower()
        assert "<svg" not in html

    def test_a_single_round_still_renders(self):
        html = _mood_section({"Alice": _mood_points((5, 3, 2))})
        assert "<svg" in html

    def test_agent_with_no_points_is_skipped(self):
        html = _mood_section({"Alice": []})
        assert "<svg" not in html

    def test_missing_axis_defaults_to_zero_rather_than_failing(self):
        html = _mood_section({"Alice": [{"confidence": 5}, {"confidence": 6}]})
        assert "<svg" in html


class TestBoundedAxis:
    def _y_positions(self, svg):
        # The y label text values along the axis.
        return re.findall(r'text-anchor="end">([^<]+)</text>', svg)

    def test_mood_axis_is_pinned_to_zero_ten(self):
        svg = _line_chart_svg(
            series=[{"label": "confidence", "color": "#000", "points": [5.0, 5.4]}],
            y_fmt=lambda v: f"{v:.0f}",
            y_min=0,
            y_max=10,
        )
        labels = self._y_positions(svg)
        assert labels[0] == "0" and labels[-1] == "10"

    def test_without_pinning_a_flat_series_autoscales(self):
        # Contrast case: this is exactly the behaviour mood must avoid.
        svg = _line_chart_svg(
            series=[{"label": "x", "color": "#000", "points": [5.0, 5.4]}],
            y_fmt=lambda v: f"{v:.2f}",
        )
        labels = self._y_positions(svg)
        assert labels[0] != "0.00"

    def test_out_of_range_values_do_not_escape_the_plot(self):
        svg = _line_chart_svg(
            series=[{"label": "x", "color": "#000", "points": [0, 10]}],
            height=200, y_fmt=lambda v: f"{v:.0f}", y_min=0, y_max=10,
        )
        assert "<svg" in svg


class TestReportIntegration:
    def _report(self, **kw):
        base = dict(
            run_id="run_test", seed=1, total_steps=2, steps_completed=2,
            stocks=[{"symbol": "S", "name": "S", "price_history": [10.0, 11.0]}],
            rankings=[], wealth_history={"Alice": [100.0, 110.0]},
            event_history=[], agent_logs={}, trade_summary={},
        )
        base.update(kw)
        return generate_report_html(**base)

    def test_mood_timeline_section_is_present(self):
        html = self._report(
            mood_history={"Alice": _mood_points((5, 3, 2), (7, 5, 1))}
        )
        assert "Mood Timeline" in html
        assert MOOD_COLORS["confidence"] in html

    def test_report_without_mood_still_renders(self):
        html = self._report()
        assert "Mood Timeline" in html
        assert "deep personality mode" in html.lower()

    def test_prompt_version_appears_in_the_header(self):
        html = self._report(prompt_info={
            "template_version": "1.0",
            "fingerprints": {"simple": "aaaaaaaaaaaa", "deep": "bbbbbbbbbbbb"},
        })
        assert "Prompt <b>v1.0</b>" in html
        assert "aaaaaaaaaaaa" in html

    def test_usage_section_shows_totals(self):
        html = self._report(usage={
            "total": {
                "calls": 4, "prompt_tokens": 1000, "completion_tokens": 200,
                "cost_usd": 0.05, "cost_complete": True, "total_tokens": 1200,
                "unpriced_calls": 0, "estimated_calls": 0,
            },
            "agents": [{
                "agent_id": "Alice", "provider": "anthropic",
                "model": "claude-sonnet-5", "priced": True,
                "total": {
                    "calls": 4, "prompt_tokens": 1000, "completion_tokens": 200,
                    "cost_usd": 0.05,
                },
            }],
        })
        assert "Tokens &amp; Cost" in html
        assert "claude-sonnet-5" in html
        assert "1,000" in html

    def test_unpriced_run_is_labelled_a_lower_bound(self):
        html = self._report(usage={
            "total": {
                "calls": 2, "prompt_tokens": 10, "completion_tokens": 5,
                "cost_usd": 0.0, "cost_complete": False, "total_tokens": 15,
                "unpriced_calls": 2, "estimated_calls": 0,
            },
            "agents": [{
                "agent_id": "Bob", "provider": "groq", "model": "mystery",
                "priced": False,
                "total": {"calls": 2, "prompt_tokens": 10, "completion_tokens": 5,
                          "cost_usd": 0.0},
            }],
        })
        assert "unpriced" in html
        assert "lower bound" in html

    def test_report_without_usage_says_so(self):
        html = self._report()
        assert "No API usage was recorded" in html

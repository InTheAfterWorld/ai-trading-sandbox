"""Unit tests for the self-contained HTML report builder (report_export)."""
from ai_trading_society.report_export import (
    _esc,
    _fmt_money,
    _fmt_pct,
    _line_chart_svg,
    generate_report_html,
)


class TestFormatters:
    def test_esc_escapes_html_and_quotes(self):
        assert _esc("<b>&</b>") == "&lt;b&gt;&amp;&lt;/b&gt;"
        assert _esc('say "hi"') == "say &quot;hi&quot;"

    def test_fmt_money(self):
        assert _fmt_money(1234.5) == "$1,234.50"
        assert _fmt_money(-9876.543) == "$-9,876.54"
        assert _fmt_money(0) == "$0.00"

    def test_fmt_pct(self):
        assert _fmt_pct(2.5) == "+2.50%"
        assert _fmt_pct(-3.1) == "-3.10%"
        assert _fmt_pct(0) == "+0.00%"
        assert _fmt_pct(2.5, signed=False) == "2.50%"


class TestLineChartSvg:
    def test_empty_series_renders_placeholder(self):
        assert '<div class="empty">No data.</div>' in _line_chart_svg([])

    def test_flat_series_does_not_divide_by_zero(self):
        svg = _line_chart_svg(
            [{"label": "A", "color": "#fff", "points": [5, 5, 5]}]
        )
        assert "<polyline" in svg

    def test_renders_polyline_and_event_marker(self):
        svg = _line_chart_svg(
            [{"label": "A", "color": "#fff", "points": [1, 2, 3]}],
            markers=[{"index": 1, "tip": "crash"}],
        )
        assert "<svg" in svg
        assert "<polyline" in svg
        assert "crash" in svg  # marker tip escaped into a <title>

    def test_marker_index_is_clamped(self):
        svg = _line_chart_svg(
            [{"label": "A", "color": "#fff", "points": [1, 2, 3]}],
            markers=[{"index": 99, "tip": "late"}],
        )
        assert "late" in svg


def _report_kwargs():
    """Minimal valid inputs for generate_report_html."""
    return dict(
        total_steps=10,
        steps_completed=3,
        stocks=[{"symbol": "Alpha", "name": "Alpha",
                 "price_history": [100.0, 104.0, 109.0]}],
        rankings=[{"rank": 1, "id": "Nova", "type": "AI", "cash": 500.0,
                   "wealth": 10500.0, "return_pct": 5.0, "sharpe": 1.2,
                   "max_drawdown": 3.0, "volatility": 0.8, "win_rate": 60.0,
                   "is_player": False}],
        wealth_history={"Nova": [10000.0, 10250.0, 10500.0]},
        event_history=[{"step": 2, "name": "Rate Hike", "scope": "global",
                        "description": "Rates up.", "price_impact": -0.04}],
        agent_logs={"Nova": [{"round": 2, "action": "buy", "requested": 10,
                              "filled": 10, "reasoning": "Momentum.",
                              "wealth": 10500.0, "delta": 250.0,
                              "actions": [{"symbol": "Alpha", "action": "buy",
                                           "filled": 10,
                                           "reasoning": "Momentum."}]}]},
        trade_summary={"total": 4, "buys": 3, "sells": 1},
    )


class TestGenerateReportHtml:
    def test_renders_every_section(self):
        html = generate_report_html("run_x", 42, **_report_kwargs())
        for marker in ("Simulation Report", "Price Charts", "Agent Wealth Curves",
                       "Final Rankings", "Event Timeline", "Agent Decision Log",
                       "Rate Hike", "Alpha BUY", "run_x"):
            assert marker in html, marker

    def test_run_id_is_escaped(self):
        html = generate_report_html("<script>x</script>", None,
                                    **_report_kwargs())
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_empty_sections_render_placeholders(self):
        html = generate_report_html(
            "r", 1, total_steps=0, steps_completed=0, stocks=[], rankings=[],
            wealth_history={}, event_history=[], agent_logs={},
            trade_summary={},
        )
        for placeholder in ("No rankings.", "No market events were triggered.",
                            "No decisions recorded."):
            assert placeholder in html, placeholder

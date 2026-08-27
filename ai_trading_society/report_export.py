"""
Self-contained HTML report export for simulation runs.

Generates a single .html file (no external assets, no JavaScript) that can
be shared or archived: market overview, inline-SVG price & wealth charts,
final rankings, the full event timeline, and a per-agent decision log.
"""

import html
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# Palette used for agent lines / rows (mirrors the web UI).
PALETTE = [
    "#a78bfa", "#60a5fa", "#22c55e", "#eab308", "#ef4444",
    "#f97316", "#06b6d4", "#ec4899", "#8b5cf6", "#84cc16",
]

_CSS = """
:root{--bg:#0b0e14;--card:#161b2a;--border:#232b42;--text:#e2e8f0;--dim:#7c8aa5;
--green:#22c55e;--red:#ef4444;--yellow:#eab308;--purple:#a78bfa;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
padding:32px 20px 60px;line-height:1.5;}
.wrap{max-width:1060px;margin:0 auto;}
h1{font-size:24px;background:linear-gradient(135deg,var(--purple),#60a5fa);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.8px;color:var(--dim);
margin:34px 0 12px;border-bottom:1px solid var(--border);padding-bottom:8px;}
.meta{font-size:12px;color:var(--dim);margin-top:6px;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 14px;}
.card .lb{font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);}
.card .vl{font-size:17px;font-weight:700;margin-top:3px;}
.pos{color:var(--green);} .neg{color:var(--red);}
table{width:100%;border-collapse:collapse;font-size:12.5px;}
th{text-align:left;padding:7px 9px;color:var(--dim);font-weight:600;border-bottom:1px solid var(--border);
font-size:10.5px;text-transform:uppercase;letter-spacing:.5px;}
td{padding:7px 9px;border-bottom:1px solid var(--border);}
tr:last-child td{border-bottom:none;}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:0;}
.badge{font-size:10px;font-weight:700;padding:2px 8px;border-radius:5px;display:inline-block;letter-spacing:.4px;}
.badge.global{background:rgba(96,165,250,.15);color:#60a5fa;}
.badge.stock{background:rgba(234,179,8,.15);color:var(--yellow);}
.badge.buy{background:rgba(34,197,94,.12);color:var(--green);}
.badge.sell{background:rgba(239,68,68,.12);color:var(--red);}
.badge.hold{background:rgba(124,138,165,.12);color:var(--dim);}
.evt{background:var(--card);border:1px solid var(--border);border-left:3px solid var(--yellow);
border-radius:10px;padding:9px 13px;margin-bottom:8px;font-size:12.5px;}
.evt .nm{font-weight:700;}
.evt .ds{color:var(--dim);font-size:11.5px;margin-top:2px;}
.evt .rd{color:var(--dim);font-size:10.5px;margin-top:3px;}
details{background:var(--card);border:1px solid var(--border);border-radius:12px;
margin-bottom:10px;overflow:hidden;}
summary{cursor:pointer;padding:11px 15px;font-size:13px;font-weight:700;list-style:none;}
summary::-webkit-details-marker{display:none;}
summary::before{content:'▸ ';color:var(--purple);}
details[open] summary::before{content:'▾ ';}
.log{padding:4px 15px 12px;}
.reason{color:var(--dim);font-style:italic;font-size:11.5px;}
.empty{color:var(--dim);font-style:italic;font-size:12.5px;}
footer{margin-top:40px;font-size:11px;color:var(--dim);text-align:center;}
.chart{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:10px;margin-bottom:12px;}
.chart .ttl{font-size:12px;font-weight:700;color:var(--dim);margin:2px 4px 6px;}
.legend{font-size:11px;color:var(--dim);padding:2px 4px 6px;}
.legend span{margin-right:12px;white-space:nowrap;}
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _fmt_pct(value: float, signed: bool = True) -> str:
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value:.2f}%"


def _line_chart_svg(
    series: List[Dict[str, Any]],
    width: int = 1000,
    height: int = 260,
    y_fmt=_fmt_money,
    x_step: int = 1,
    markers: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Render one or more polylines into an inline SVG.

    series: [{"label", "color", "points": [float, ...]}]
    markers: [{"index", "color", "tip"}] drawn as diamonds above the curve.
    """
    all_pts = [p for s in series for p in s["points"]]
    if not all_pts:
        return '<div class="empty">No data.</div>'

    pad_l, pad_r, pad_t, pad_b = 64, 16, 14, 26
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    lo: float = min(all_pts)
    hi: float = max(all_pts)
    if hi == lo:
        hi = lo + 1
    span = hi - lo
    lo -= span * 0.06
    hi += span * 0.06

    max_len = max(len(s["points"]) for s in series)

    def X(i: int) -> float:
        return pad_l + (w * i / max(1, max_len - 1))

    def Y(v: float) -> float:
        return pad_t + h - (h * (v - lo) / (hi - lo))

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']

    # Horizontal grid + y labels
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        y = Y(v)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="rgba(42,50,69,.6)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 3.5:.1f}" fill="#64748b" font-size="10" '
            f'text-anchor="end">{_esc(y_fmt(v))}</text>'
        )

    # X labels every few rounds
    label_every = max(1, max_len // 10)
    for i in range(0, max_len, label_every):
        x = X(i)
        parts.append(
            f'<text x="{x:.1f}" y="{height - 8}" fill="#64748b" font-size="10" '
            f'text-anchor="middle">{i * x_step}</text>'
        )

    # Polylines
    for s in series:
        pts = s["points"]
        if len(pts) < 2:
            continue
        path = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(pts))
        parts.append(
            f'<polyline points="{path}" fill="none" stroke="{s["color"]}" '
            f'stroke-width="2"/>'
        )
        # End-point dot
        parts.append(
            f'<circle cx="{X(len(pts) - 1):.1f}" cy="{Y(pts[-1]):.1f}" r="3" '
            f'fill="{s["color"]}"/>'
        )

    # Event markers (diamonds pinned near the top axis)
    if markers:
        for m in markers:
            idx = max(0, min(m.get("index", 0), max_len - 1))
            x = X(idx)
            my = pad_t + 10
            tip = _esc(m.get("tip", ""))
            parts.append(
                f'<g><title>{tip}</title>'
                f'<path d="M {x:.1f} {my - 5:.1f} l 5 5 l -5 5 l -5 -5 z" '
                f'fill="{m.get("color", "#eab308")}" stroke="#0b0e14" stroke-width="1"/>'
                f'<text x="{x:.1f}" y="{my - 9:.1f}" fill="{m.get("color", "#eab308")}" '
                f'font-size="9" text-anchor="middle">⚡</text></g>'
            )

    parts.append("</svg>")
    return "".join(parts)


def _price_section(stocks: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> str:
    """One chart per stock, with ⚡ markers at event rounds."""
    out = []
    for st in stocks:
        hist = st.get("price_history") or []
        if len(hist) < 2:
            continue
        # Marker index: price_history[0] = round 0 (initial price).
        markers = []
        for ev in events:
            if ev.get("stock") not in (None, st.get("symbol")):
                continue
            step = int(ev.get("step", 0))
            color = "#60a5fa" if ev.get("scope") == "global" else "#eab308"
            tip = f"Round {step}: {ev.get('name')} ({_fmt_pct((ev.get('price_impact') or 0) * 100)})"
            markers.append({"index": step, "color": color, "tip": tip})

        initial = hist[0]
        final = hist[-1]
        chg = (final / initial - 1) * 100 if initial else 0
        cls = "pos" if chg >= 0 else "neg"
        svg = _line_chart_svg(
            series=[{"label": st.get("symbol", ""), "color": "#a78bfa", "points": hist}],
            markers=markers,
        )
        out.append(
            f'<div class="chart"><div class="ttl">{_esc(st.get("symbol", "Stock"))} '
            f'— {_fmt_money(initial)} → <span class="{cls}">{_fmt_money(final)} '
            f'({_fmt_pct(chg)})</span></div>{svg}'
            f'<div class="legend"><span>◆ <span style="color:#60a5fa">market-wide event</span></span>'
            f'<span>◆ <span style="color:#eab308">stock event</span></span></div></div>'
        )
    if not out:
        return '<div class="empty">No price data.</div>'
    return "".join(out)


def _wealth_section(wealth_history: Dict[str, List[float]]) -> str:
    if not wealth_history:
        return '<div class="empty">No wealth data.</div>'
    series = []
    for i, (aid, pts) in enumerate(wealth_history.items()):
        series.append({
            "label": aid,
            "color": PALETTE[i % len(PALETTE)],
            "points": pts or [0],
        })
    legend = " ".join(
        f'<span><span class="dot" style="background:{s["color"]}"></span>{_esc(s["label"])}</span>'
        for s in series
    )
    svg = _line_chart_svg(series=series, height=300)
    return f'<div class="chart">{svg}<div class="legend">{legend}</div></div>'


def _rankings_section(rankings: List[Dict[str, Any]]) -> str:
    if not rankings:
        return '<div class="empty">No rankings.</div>'
    rows = []
    for r in rankings:
        ret = r.get("return_pct", 0)
        cls = "pos" if ret >= 0 else "neg"
        player_badge = (
            ' <span class="badge" style="background:rgba(167,139,250,.15);color:#a78bfa">you</span>'
            if r.get("is_player") else ""
        )
        rows.append(
            f'<tr><td style="font-weight:700">{r.get("rank", "")}</td>'
            f'<td><span class="dot" style="background:{PALETTE[(r.get("rank", 1) - 1) % len(PALETTE)]}"></span>'
            f'{_esc(r.get("id", ""))}{player_badge}</td>'
            f'<td class="num">{_esc(r.get("type", ""))}</td>'
            f'<td class="num">{_fmt_money(r.get("cash", 0))}</td>'
            f'<td class="num">{_fmt_money(r.get("wealth", 0))}</td>'
            f'<td class="num {cls}">{_fmt_pct(ret)}</td>'
            f'<td class="num">{r.get("sharpe", 0):.2f}</td>'
            f'<td class="num neg">{r.get("max_drawdown", 0):.1f}%</td>'
            f'<td class="num">{r.get("volatility", 0):.2f}%</td>'
            f'<td class="num">{r.get("win_rate", 0):.0f}%</td></tr>'
        )
    return (
        '<table><thead><tr><th>#</th><th>Agent</th><th>Type</th>'
        '<th class="num">Cash</th><th class="num">Wealth</th><th class="num">Return</th>'
        '<th class="num">Sharpe</th><th class="num">MaxDD</th><th class="num">Vol</th>'
        '<th class="num">Win%</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>"
    )


def _events_section(events: List[Dict[str, Any]]) -> str:
    if not events:
        return '<div class="empty">No market events were triggered.</div>'
    items = []
    for ev in sorted(events, key=lambda e: e.get("step", 0)):
        impact = (ev.get("price_impact") or 0) * 100
        cls = "pos" if impact >= 0 else "neg"
        scope = ev.get("scope") or "global"
        target = (
            f" · hits <b>{_esc(ev.get('stock'))}</b>" if ev.get("stock")
            else " · market-wide"
        )
        forced = " · forced (God Mode)" if ev.get("forced") else ""
        items.append(
            f'<div class="evt"><span class="nm">⚡ {_esc(ev.get("name", ""))}</span> '
            f'<span class="badge {scope}">{scope}</span> '
            f'<b class="{cls}">{_fmt_pct(impact)}</b>'
            f'<div class="ds">{_esc(ev.get("description", ""))}</div>'
            f'<div class="rd">Round {ev.get("step", "?")}{target}{forced}</div></div>'
        )
    return "".join(items)


def _decision_log_section(agent_logs: Dict[str, List[Dict[str, Any]]]) -> str:
    if not agent_logs:
        return '<div class="empty">No decisions recorded.</div>'
    blocks = []
    for i, (aid, rounds) in enumerate(agent_logs.items()):
        color = PALETTE[i % len(PALETTE)]
        rows = []
        for rd in rounds:
            act = str(rd.get("action", "hold")).lower()
            badge = f'<span class="badge {act}">{act.upper()}</span>'
            qty = ""
            if act in ("buy", "sell"):
                qty = f'{rd.get("filled", 0)}'
                if rd.get("requested") and rd.get("filled") != rd.get("requested"):
                    qty += f' <span class="reason">(asked {rd.get("requested")})</span>'
            wealth = rd.get("wealth", 0)
            delta = rd.get("delta")
            delta_str = (
                f'<span class="{"pos" if delta >= 0 else "neg"}">{delta:+,.0f}</span>'
                if delta is not None else "—"
            )
            reasoning = _esc(rd.get("reasoning") or "")
            rows.append(
                f'<tr><td>{rd.get("round", "?")}</td><td>{badge}</td>'
                f'<td class="num">{qty or "—"}</td>'
                f'<td class="num">{_fmt_money(wealth)}</td>'
                f'<td class="num">{delta_str}</td>'
                f'<td class="reason">{reasoning}</td></tr>'
            )
        blocks.append(
            f'<details><summary><span class="dot" style="background:{color}"></span>'
            f'{_esc(aid)} <span class="reason">({len(rounds)} rounds)</span></summary>'
            f'<div class="log"><table><thead><tr><th>Round</th><th>Action</th>'
            f'<th class="num">Qty</th><th class="num">Wealth</th>'
            f'<th class="num">Δ</th><th>Reasoning</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></details>'
        )
    return "".join(blocks)


def generate_report_html(
    run_id: str,
    seed: Any,
    total_steps: int,
    steps_completed: int,
    stocks: List[Dict[str, Any]],
    rankings: List[Dict[str, Any]],
    wealth_history: Dict[str, List[float]],
    event_history: List[Dict[str, Any]],
    agent_logs: Dict[str, List[Dict[str, Any]]],
    trade_summary: Dict[str, Any],
    generated_at: Optional[str] = None,
) -> str:
    """Assemble the full self-contained report page."""
    gen_time = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    overview_cards = []
    for st in stocks:
        hist = st.get("price_history") or []
        if not hist:
            continue
        initial, final = hist[0], hist[-1]
        chg = (final / initial - 1) * 100 if initial else 0
        cls = "pos" if chg >= 0 else "neg"
        overview_cards.append(
            f'<div class="card"><div class="lb">{_esc(st.get("symbol", "Stock"))}</div>'
            f'<div class="vl">{_fmt_money(final)}</div>'
            f'<div class="vl {cls}" style="font-size:13px">{_fmt_pct(chg)}</div></div>'
        )
    ts = trade_summary or {}
    overview_cards.extend([
        f'<div class="card"><div class="lb">Rounds</div><div class="vl">{steps_completed} / {total_steps}</div></div>',
        f'<div class="card"><div class="lb">Trades</div><div class="vl">{ts.get("total", 0)}</div>'
        f'<div class="vl" style="font-size:12px"><span class="pos">{ts.get("buys", 0)} buys</span> · '
        f'<span class="neg">{ts.get("sells", 0)} sells</span></div></div>',
        f'<div class="card"><div class="lb">Events</div><div class="vl">{len(event_history)}</div></div>',
    ])

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Trading Sandbox — Report {_esc(run_id)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>📈 AI Trading Sandbox — Simulation Report</h1>
  <div class="meta">Run <b>{_esc(run_id)}</b> · Seed <b>{_esc(seed)}</b> ·
  Generated {gen_time} · read-only snapshot</div>

  <h2>Market Overview</h2>
  <div class="cards">{''.join(overview_cards)}</div>

  <h2>Price Charts</h2>
  {_price_section(stocks, event_history)}

  <h2>Agent Wealth Curves</h2>
  {_wealth_section(wealth_history)}

  <h2>Final Rankings</h2>
  {_rankings_section(rankings)}

  <h2>Event Timeline</h2>
  {_events_section(event_history)}

  <h2>Agent Decision Log</h2>
  {_decision_log_section(agent_logs)}

  <footer>Generated by AI Trading Sandbox · report is self-contained (no external assets)</footer>
</div>
</body>
</html>
"""
    return doc


# Exported reports are disposable snapshots and nothing ever removed them,
# so a long-lived server grew runs/reports without bound. Keep the most
# recent N; pass max_reports=0 from a caller that wants to keep everything.
MAX_REPORTS = 50


def _prune_reports(reports_dir: str, max_reports: int) -> None:
    """Delete the oldest exported reports beyond ``max_reports``.

    Best-effort and deliberately narrow: only ``.html`` files directly
    inside ``reports_dir`` (the directory this module owns) are considered,
    and any filesystem error is ignored rather than failing an export.
    """
    if max_reports <= 0:
        return
    try:
        entries = [
            (os.path.getmtime(os.path.join(reports_dir, name)), name)
            for name in os.listdir(reports_dir)
            if name.endswith(".html")
            and os.path.isfile(os.path.join(reports_dir, name))
        ]
    except OSError:
        return
    if len(entries) <= max_reports:
        return
    entries.sort(reverse=True)  # newest first
    for _, name in entries[max_reports:]:
        try:
            os.remove(os.path.join(reports_dir, name))
        except OSError:
            pass


def save_report(
    html_text: str,
    run_id: str,
    reports_dir: str = "runs/reports",
    max_reports: int = MAX_REPORTS,
) -> str:
    """Write the report file and return its path. run_id is sanitized.

    Older reports beyond ``max_reports`` are pruned after the write, so
    the just-saved report is always kept.
    """
    import re as _re

    if not _re.fullmatch(r"[A-Za-z0-9_\-]+", run_id or ""):
        raise ValueError("Invalid run id")
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, f"{run_id}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)
    _prune_reports(reports_dir, max_reports)
    return path

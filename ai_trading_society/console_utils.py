"""
Console output utilities for simulation display.

Provides ANSI color codes, trend arrows, pressure bars, and agent type
labels — all without external dependencies. Designed for terminals that
support ANSI escape codes (Windows 10+ Terminal, PowerShell, Linux/macOS).
"""


class Colors:
    """ANSI color code constants."""

    GREEN = "\033[32m"   # Price up / buy
    RED = "\033[31m"     # Price down / sell
    YELLOW = "\033[33m"  # Market events
    GRAY = "\033[90m"    # Hold / secondary info
    DIM = "\033[2m"      # Dimmed (reasoning, annotations)
    BOLD = "\033[1m"     # Emphasis
    RESET = "\033[0m"


def colorize(text: str, color: str) -> str:
    """Wrap text with ANSI color codes. Returns plain text if color is empty."""
    if not color:
        return text
    return f"{color}{text}{Colors.RESET}"


def trend_arrow(change_pct: float) -> str:
    """Return a trend arrow based on price change percentage."""
    if change_pct > 0.01:
        return "+"
    elif change_pct < -0.01:
        return "-"
    else:
        return "->"


# Map class names to short display labels.
_NAME_MAP = {
    "ExternalAIAgent": "AI",
    "TraitAgent": "Trait",
    "PlayerAgent": "Player",
}


def _short_name(class_name: str) -> str:
    """Return a short label for a class name."""
    return _NAME_MAP.get(class_name, class_name[:8])


def agent_type_label(agent) -> str:
    """
    Return a short human-readable type label for an agent.

    Handles TraitAgent wrappers by inspecting the wrapped base_agent.
    Appends a personality abbreviation when available.
    """
    if hasattr(agent, "base_agent"):
        base_name = agent.base_agent.__class__.__name__
        label = f"Trait+{_short_name(base_name)}"
    else:
        label = _short_name(agent.__class__.__name__)

    # Append personality abbreviation if the agent has one.
    personality = agent_personality(agent)
    if personality and personality not in ("balanced", "custom", ""):
        abbr = _PERSONALITY_ABBR.get(personality, personality[:5])
        label = f"{label}|{abbr}"

    return label


def agent_personality(agent) -> str:
    """Return the personality name for an agent, or empty string."""
    if hasattr(agent, "personality_name"):
        return str(agent.personality_name)
    return ""


def agent_personality_desc(agent) -> str:
    """Return the personality description for an agent, or empty string."""
    if hasattr(agent, "personality_description"):
        return str(agent.personality_description)
    return ""


# Short abbreviations for personality names used in compact display.
_PERSONALITY_ABBR = {
    "balanced": "bal",
    "aggressive": "aggr",
    "conservative": "cons",
    "panicky": "panic",
    "greedy": "greed",
    "fomo_driven": "fomo",
    "stubborn": "stub",
    "emotional": "emo",
    "custom": "cust",
}


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list, width: int = 12) -> str:
    """
    Render a tiny ASCII sparkline from a list of numeric values.

    Uses Unicode block characters to show relative magnitude. Returns
    an empty string when fewer than 2 values are provided.

    Examples
    --------
    >>> sparkline([1, 2, 3, 4, 5])
    '▁▂▄▆█'
    >>> sparkline([5, 5, 5])
    '▄▄▄'
    """
    if len(values) < 2:
        return ""
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    if hi == lo:
        return _SPARK_CHARS[3] * len(recent)
    scaled = [(v - lo) / (hi - lo) for v in recent]
    return "".join(_SPARK_CHARS[min(7, int(s * 7.999))] for s in scaled)


def pressure_bar(total_buy: int, total_sell: int, width: int = 20) -> str:
    """
    Render a visual buy/sell pressure bar.

    Uses solid blocks for buy pressure (green) and light shade for sell
    pressure (red). Returns a no-activity indicator when both are zero.

    Examples
    --------
    >>> pressure_bar(60, 40)
    '[████████████░░░░░░░░] 60% buy / 40% sell'
    """
    total = total_buy + total_sell
    if total == 0:
        return f"[{'.' * width}] no activity"

    buy_pct = total_buy / total
    buy_blocks = round(buy_pct * width)
    sell_blocks = width - buy_blocks

    bar = (
        colorize("#" * buy_blocks, Colors.GREEN)
        + colorize("-" * sell_blocks, Colors.RED)
    )
    return f"[{bar}] {buy_pct * 100:.0f}% buy / {(1 - buy_pct) * 100:.0f}% sell"

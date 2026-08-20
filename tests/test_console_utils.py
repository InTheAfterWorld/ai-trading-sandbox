"""Tests for console output utilities."""

from ai_trading_society.agents.external_ai_agent import ExternalAIAgent
from ai_trading_society.agents.traits import TraitAgent
from ai_trading_society.console_utils import (
    Colors,
    agent_personality,
    agent_personality_desc,
    agent_type_label,
    colorize,
    pressure_bar,
    sparkline,
    trend_arrow,
)


class TestColorize:
    """Test the colorize function."""

    def test_wraps_text_with_color(self):
        """Colorize should wrap text with ANSI codes."""
        result = colorize("hello", Colors.GREEN)
        assert Colors.GREEN in result
        assert Colors.RESET in result
        assert "hello" in result

    def test_empty_color_returns_plain(self):
        """Empty color string should return plain text."""
        assert colorize("hello", "") == "hello"


class TestTrendArrow:
    """Test the trend_arrow function."""

    def test_up_arrow_for_positive(self):
        """Positive change should return up arrow."""
        assert trend_arrow(1.5) == "+"

    def test_down_arrow_for_negative(self):
        """Negative change should return down arrow."""
        assert trend_arrow(-1.5) == "-"

    def test_flat_arrow_for_zero(self):
        """Zero change should return flat arrow."""
        assert trend_arrow(0.0) == "->"


class TestPressureBar:
    """Test the pressure_bar function."""

    def test_no_activity(self):
        """Zero buy and sell should show no activity."""
        bar = pressure_bar(0, 0)
        assert "no activity" in bar

    def test_all_buy(self):
        """All buy pressure should show 100% buy."""
        bar = pressure_bar(100, 0)
        assert "100% buy" in bar

    def test_all_sell(self):
        """All sell pressure should show 0% buy."""
        bar = pressure_bar(0, 100)
        assert "0% buy" in bar

    def test_mixed(self):
        """Mixed pressure should show both percentages."""
        bar = pressure_bar(60, 40)
        assert "60% buy" in bar
        assert "40% sell" in bar


class TestAgentTypeLabel:
    """Test the agent_type_label function."""

    def test_external_ai_agent(self):
        """ExternalAIAgent should be labeled 'AI'."""
        agent = ExternalAIAgent("ai", api_provider="mock")
        assert agent_type_label(agent) == "AI"

    def test_trait_agent_wraps_base(self):
        """TraitAgent should show 'Trait+' prefix with base type."""
        base = ExternalAIAgent("ai", api_provider="mock")
        trait = TraitAgent(base)
        label = agent_type_label(trait)
        assert label == "Trait+AI"

    def test_trait_agent_with_personality_shows_abbr(self):
        """TraitAgent with a personality should include abbreviation in label."""
        from ai_trading_society.agents.traits import create_personality_agent
        base = ExternalAIAgent("r")
        agent = create_personality_agent(base, "aggressive")
        label = agent_type_label(agent)
        assert "aggr" in label

    def test_trait_agent_balanced_no_abbr(self):
        """Balanced personality should not add abbreviation."""
        from ai_trading_society.agents.traits import create_personality_agent
        base = ExternalAIAgent("r")
        agent = create_personality_agent(base, "balanced")
        label = agent_type_label(agent)
        assert "|" not in label


class TestAgentPersonality:
    """Test personality extraction functions."""

    def test_personality_from_trait_agent(self):
        """agent_personality should return the personality name."""
        from ai_trading_society.agents.traits import create_personality_agent
        base = ExternalAIAgent("r")
        agent = create_personality_agent(base, "panicky")
        assert agent_personality(agent) == "panicky"

    def test_personality_empty_for_plain_agent(self):
        """agent_personality should return empty string for non-trait agents."""
        agent = ExternalAIAgent("r")
        assert agent_personality(agent) == ""

    def test_personality_desc_from_trait_agent(self):
        """agent_personality_desc should return a human-readable description."""
        from ai_trading_society.agents.traits import create_personality_agent
        base = ExternalAIAgent("r")
        agent = create_personality_agent(base, "greedy")
        desc = agent_personality_desc(agent)
        assert "Greedy" in desc

    def test_personality_desc_empty_for_plain_agent(self):
        """agent_personality_desc should return empty string for non-trait agents."""
        agent = ExternalAIAgent("r")
        assert agent_personality_desc(agent) == ""


class TestSparkline:
    """Test the sparkline function."""

    def test_empty_for_single_value(self):
        """Single value should return empty string."""
        assert sparkline([42]) == ""

    def test_empty_for_empty_list(self):
        """Empty list should return empty string."""
        assert sparkline([]) == ""

    def test_increasing_values(self):
        """Increasing values should produce ascending block chars."""
        result = sparkline([1, 2, 3, 4, 5])
        assert len(result) == 5
        # First char should be the lowest block, last should be highest
        assert result[0] == "\u2581"  # ▁
        assert result[-1] == "\u2588"  # █

    def test_flat_values(self):
        """All-equal values should produce identical mid-range chars."""
        result = sparkline([5, 5, 5])
        assert len(result) == 3
        assert result[0] == result[1] == result[2]

    def test_respects_width(self):
        """Width parameter should limit the number of characters."""
        result = sparkline(list(range(20)), width=5)
        assert len(result) == 5

    def test_decreasing_values(self):
        """Decreasing values should produce descending block chars."""
        result = sparkline([5, 4, 3, 2, 1])
        assert result[0] == "\u2588"  # █ (highest)
        assert result[-1] == "\u2581"  # ▁ (lowest)

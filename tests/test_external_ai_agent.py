"""Tests for ExternalAIAgent."""

from ai_trading_society.agents.external_ai_agent import ExternalAIAgent


class TestActWithoutApiKey:
    """Test that act() raises when no API key is configured."""

    def test_raises_without_api_key(self, monkeypatch):
        """act() should raise RuntimeError when no API key is set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = ExternalAIAgent("test", api_provider="openai")
        obs = {
            "step": 1,
            "price": 100.0,
            "price_history": [95, 100],
            "my_cash": 10000.0,
            "my_holdings": 50,
            "my_wealth": 15000.0,
            "last_volume": 100,
            "market_sentiment": 0.0,
        }
        try:
            agent.act(obs)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "no API key" in str(e)


class TestResponseParsing:
    """Test the JSON response parser."""

    def test_parses_pure_json(self):
        """Should parse a pure JSON response."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        response = '{"action": "buy", "quantity": 50, "reasoning": "bullish"}'
        result = agent._parse_response(response)
        assert result["action"] == "buy"
        assert result["quantity"] == 50

    def test_parses_json_in_text(self):
        """Should extract JSON embedded in text."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        response = 'I think we should buy.\n{"action": "buy", "quantity": 30}\nThat is my recommendation.'
        result = agent._parse_response(response)
        assert result["action"] == "buy"
        assert result["quantity"] == 30

    def test_parses_json_in_code_block(self):
        """Should extract JSON from markdown code blocks."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        response = '```json\n{"action": "sell", "quantity": 10}\n```'
        result = agent._parse_response(response)
        assert result["action"] == "sell"
        assert result["quantity"] == 10

    def test_invalid_json_raises(self):
        """Invalid JSON should be reported to the caller."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        import pytest
        with pytest.raises(ValueError, match="valid JSON action"):
            agent._parse_response("This is not JSON at all")

    def test_negative_quantity_raises(self):
        """Negative quantities should be reported as invalid."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        response = '{"action": "buy", "quantity": -5}'
        import pytest
        with pytest.raises(ValueError, match="invalid action or quantity"):
            agent._parse_response(response)

    def test_invalid_action_raises(self):
        """Unknown action values should be reported as invalid."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        response = '{"action": "trade", "quantity": 10}'
        import pytest
        with pytest.raises(ValueError, match="invalid action or quantity"):
            agent._parse_response(response)

    def test_parses_reasoning_field(self):
        """_parse_response should capture the reasoning field."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        response = '{"action": "buy", "quantity": 50, "reasoning": "bullish trend"}'
        result = agent._parse_response(response)
        assert result["reasoning"] == "bullish trend"

    def test_reasoning_empty_when_absent(self):
        """reasoning should be empty string when not in response."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        response = '{"action": "sell", "quantity": 10}'
        result = agent._parse_response(response)
        assert result["reasoning"] == ""

    def test_nested_json_reasoning_is_parsed(self):
        """Nested JSON in reasoning should not break action parsing."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        result = agent._parse_response(
            'Analysis: {"action":"hold","quantity":0,'
            '"reasoning":"context: {\\"signal\\": \\"bullish\\"}"}'
        )
        assert result["action"] == "hold"

    def test_skips_non_action_json_before_real_action(self):
        """A JSON dict that is not an action should be skipped, not fatal."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        response = (
            'Let me analyze {"signal": "bullish", "strength": 0.8} first.\n'
            'Final answer: {"action": "buy", "quantity": 20, "reasoning": "uptrend"}'
        )
        result = agent._parse_response(response)
        assert result["action"] == "buy"
        assert result["quantity"] == 20

    def test_quantity_numeric_string_is_coerced(self):
        """Quantity strings like "10.0" or "10 shares" should be coerced."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        result = agent._parse_response('{"action": "buy", "quantity": "10.0"}')
        assert result["quantity"] == 10
        result = agent._parse_response('{"action": "sell", "quantity": "10 shares"}')
        assert result["quantity"] == 10

    def test_quantity_null_is_treated_as_zero(self):
        """A null quantity should parse as 0 instead of failing."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        result = agent._parse_response('{"action": "hold", "quantity": null}')
        assert result["quantity"] == 0

    def test_quantity_all_maps_to_large_int(self):
        '"all" should map to a huge int that the market env clips.'
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        result = agent._parse_response('{"action": "sell", "quantity": "all"}')
        assert result["action"] == "sell"
        assert result["quantity"] >= 10**9


class TestHtmlResponseGuard:
    """HTML responses must surface as a clear configuration error."""

    def _agent(self):
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        agent.api_provider = "custom"
        agent.base_url = "https://example.com/misconfigured"
        return agent

    def test_html_string_raises_config_error(self):
        agent = self._agent()
        import pytest
        with pytest.raises(RuntimeError, match="HTML page"):
            agent._reject_html('<!DOCTYPE html><html class="nv-dark"></html>')

    def test_html_with_leading_whitespace_raises(self):
        agent = self._agent()
        import pytest
        with pytest.raises(RuntimeError, match="base_url"):
            agent._reject_html('  \n <html lang="en"><body></body></html>')

    def test_normal_text_passes_through(self):
        agent = self._agent()
        assert agent._reject_html('{"action": "hold", "quantity": 0}') == (
            '{"action": "hold", "quantity": 0}'
        )

    def test_error_message_names_the_base_url(self):
        agent = self._agent()
        import pytest
        with pytest.raises(RuntimeError, match="https://example.com/misconfigured"):
            agent._reject_html("<!DOCTYPE html>")


class TestAPIKeyLoading:
    """API keys must come from the user's configuration, never the environment."""

    def test_no_key_when_not_provided(self, monkeypatch):
        """api_key should be None when nothing is passed (no env fallback)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = ExternalAIAgent("test", api_provider="openai")
        assert agent.api_key is None

    def test_environment_key_is_ignored(self, monkeypatch):
        """Even with an env var set, api_key stays None unless explicitly passed."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key_123")
        agent = ExternalAIAgent("test", api_provider="openai")
        assert agent.api_key is None

    def test_explicit_key_is_used(self, monkeypatch):
        """Explicitly provided key is used regardless of the environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "env_key")
        agent = ExternalAIAgent("test", api_provider="openai", api_key="explicit_key")
        assert agent.api_key == "explicit_key"

    def test_env_ignored_for_all_providers(self, monkeypatch):
        """No provider should read its key from the environment."""
        for provider in ("anthropic", "google", "groq", "openrouter"):
            monkeypatch.setenv("ANTHROPIC_API_KEY", "anthro_key")
            monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
            monkeypatch.setenv("GROQ_API_KEY", "groq_key")
            monkeypatch.setenv("OPENROUTER_API_KEY", "or_key")
            agent = ExternalAIAgent("test", api_provider=provider)
            assert agent.api_key is None


class TestPromptBuilding:
    """Test prompt construction from observations."""

    def test_prompt_contains_price(self):
        """Prompt should include the current price."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        agent.system_prompt = "test"
        agent._market_history = []
        agent.enable_memory = False
        obs = {
            "step": 1,
            "price": 100.0,
            "price_history": [100],
            "my_cash": 10000.0,
            "my_holdings": 50,
            "my_wealth": 15000.0,
            "last_volume": 100,
            "market_sentiment": 0.0,
        }
        prompt = agent._build_prompt(obs)
        assert "$100.00" in prompt

    def test_prompt_includes_sentiment_when_nonzero(self):
        """Prompt should include sentiment when it's nonzero."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        agent.system_prompt = "test"
        agent._market_history = []
        agent.enable_memory = False
        obs = {
            "step": 1,
            "price": 100.0,
            "price_history": [100],
            "my_cash": 10000.0,
            "my_holdings": 50,
            "my_wealth": 15000.0,
            "last_volume": 100,
            "market_sentiment": 0.5,
        }
        prompt = agent._build_prompt(obs)
        assert "Sentiment" in prompt

    def test_prompt_excludes_sentiment_when_zero(self):
        """Prompt should not include sentiment line when it's 0.0."""
        agent = ExternalAIAgent.__new__(ExternalAIAgent)
        agent.system_prompt = "test"
        agent._market_history = []
        agent.enable_memory = False
        obs = {
            "step": 1,
            "price": 100.0,
            "price_history": [100],
            "my_cash": 10000.0,
            "my_holdings": 50,
            "my_wealth": 15000.0,
            "last_volume": 100,
            "market_sentiment": 0.0,
        }
        prompt = agent._build_prompt(obs)
        assert "Sentiment" not in prompt

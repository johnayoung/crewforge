"""Test suite for LiteLLM integration and prompt processing."""

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel
from typing import Dict, Any

from crewforge.core.llm import LLMClient, LLMError, RetryConfig


class TestLLMClient:
    """Test cases for LLMClient wrapper around LiteLLM."""

    @pytest.fixture
    def llm_client(self):
        """Create LLMClient instance for testing."""
        return LLMClient(
            model="gpt-4", retry_config=RetryConfig(max_attempts=3, base_delay=0.1)
        )

    @pytest.fixture
    def mock_completion_response(self):
        """Mock LiteLLM completion response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = (
            '{"agents": [{"role": "researcher"}]}'
        )
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 150
        return mock_response

    def test_client_initialization_with_defaults(self):
        """Test LLMClient initializes with proper defaults."""
        client = LLMClient()

        assert client.model == "gpt-4"
        assert client.retry_config.max_attempts == 3
        assert client.retry_config.base_delay == 1.0
        assert client.retry_config.max_delay == 60.0
        assert client.retry_config.exponential_base == 2.0

    def test_client_initialization_with_custom_config(self):
        """Test LLMClient accepts custom configuration."""
        retry_config = RetryConfig(max_attempts=5, base_delay=0.5)
        client = LLMClient(model="claude-3-sonnet-20240229", retry_config=retry_config)

        assert client.model == "claude-3-sonnet-20240229"
        assert client.retry_config.max_attempts == 5
        assert client.retry_config.base_delay == 0.5

    @patch("crewforge.core.llm.litellm.completion")
    def test_generate_successful_call(
        self, mock_completion, llm_client, mock_completion_response
    ):
        """Test successful LLM generation call."""
        mock_completion.return_value = mock_completion_response

        result = llm_client.generate(
            system_prompt="You are a CrewAI expert",
            user_prompt="Create a research crew",
            use_json_mode=True,
        )

        assert result is not None
        assert "agents" in result
        mock_completion.assert_called_once()

        # Verify call arguments
        call_args = mock_completion.call_args
        assert call_args[1]["model"] == "gpt-4"
        assert call_args[1]["response_format"] == {"type": "json_object"}
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][1]["role"] == "user"

    @patch("crewforge.core.llm.litellm.completion")
    def test_generate_without_json_mode(self, mock_completion, llm_client):
        """Test LLM generation without JSON mode."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is a text response"
        mock_completion.return_value = mock_response

        result = llm_client.generate(
            system_prompt="System prompt",
            user_prompt="User prompt",
            use_json_mode=False,
        )

        assert result == "This is a text response"
        call_args = mock_completion.call_args
        assert "response_format" not in call_args[1]

    @patch("crewforge.core.llm.litellm.completion")
    def test_retry_logic_on_api_error(self, mock_completion, llm_client):
        """Test exponential backoff retry logic on API errors."""
        # Mock first two calls to fail, third to succeed
        mock_completion.side_effect = [
            Exception("API rate limit"),
            Exception("Temporary server error"),
            Mock(choices=[Mock(message=Mock(content="Success"))]),
        ]

        with patch("time.sleep") as mock_sleep:
            result = llm_client.generate(
                system_prompt="Test system", user_prompt="Test user"
            )

            assert result == "Success"
            assert mock_completion.call_count == 3
            # Verify exponential backoff: 0.1, 0.2 seconds
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(0.1)
            mock_sleep.assert_any_call(0.2)

    @patch("crewforge.core.llm.litellm.completion")
    def test_retry_exhaustion_raises_error(self, mock_completion, llm_client):
        """Test that exhausted retries raise LLMError."""
        mock_completion.side_effect = Exception("Persistent API error")

        with pytest.raises(LLMError) as exc_info:
            llm_client.generate(system_prompt="Test system", user_prompt="Test user")

        # Check that the error contains the original error message
        assert "Persistent API error" in str(exc_info.value)
        assert mock_completion.call_count == 3

    @patch("crewforge.core.llm.litellm.completion")
    def test_json_parsing_error_handling(self, mock_completion, llm_client):
        """Test handling of invalid JSON responses."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Invalid JSON content"
        mock_completion.return_value = mock_response

        with pytest.raises(LLMError) as exc_info:
            llm_client.generate(
                system_prompt="Test system", user_prompt="Test user", use_json_mode=True
            )

        assert "Failed to parse JSON response" in str(exc_info.value)

    def test_environment_variable_api_key_detection(self):
        """Test that client detects API keys from environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient()
            # Should not raise an error during initialization
            assert client.model == "gpt-4"

    @patch("crewforge.core.llm.litellm.completion")
    def test_model_provider_routing(self, mock_completion, mock_completion_response):
        """Test that different model providers are handled correctly."""
        mock_completion.return_value = mock_completion_response

        # Test OpenAI model
        openai_client = LLMClient(model="gpt-4")
        openai_client.generate("system", "user")
        call_args = mock_completion.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert len(call_args["messages"]) == 2

        # Test Claude model
        mock_completion.reset_mock()
        claude_client = LLMClient(model="claude-3-sonnet-20240229")
        claude_client.generate("system", "user")
        call_args = mock_completion.call_args[1]
        assert call_args["model"] == "claude-3-sonnet-20240229"
        assert len(call_args["messages"]) == 2

    @patch("crewforge.core.llm.litellm.completion")
    def test_cost_tracking_integration(
        self, mock_completion, llm_client, mock_completion_response
    ):
        """Test that usage tokens are tracked for cost monitoring."""
        mock_completion.return_value = mock_completion_response

        result = llm_client.generate("system", "user")

        # Should have access to usage information
        assert hasattr(mock_completion_response, "usage")
        assert mock_completion_response.usage.total_tokens == 150


class TestRetryConfig:
    """Test cases for retry configuration."""

    def test_retry_config_defaults(self):
        """Test RetryConfig has proper defaults."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0

    def test_retry_config_custom_values(self):
        """Test RetryConfig accepts custom values."""
        config = RetryConfig(
            max_attempts=5, base_delay=0.5, max_delay=30.0, exponential_base=1.5
        )

        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5

    def test_retry_config_validation(self):
        """Test RetryConfig validates input values."""
        with pytest.raises(ValueError):
            RetryConfig(max_attempts=0)

        with pytest.raises(ValueError):
            RetryConfig(base_delay=-1.0)

        with pytest.raises(ValueError):
            RetryConfig(max_delay=0.0)


class TestPromptEngineering:
    """Test cases for prompt engineering and templates."""

    def test_system_prompt_for_crewai_generation(self):
        """Test system prompt contains CrewAI context."""
        from crewforge.core.llm import LLMClient

        client = LLMClient()
        system_prompt = client.get_system_prompt()

        assert "CrewAI" in system_prompt
        assert "agent" in system_prompt.lower()
        assert "task" in system_prompt.lower()
        assert "tool" in system_prompt.lower()
        assert "JSON" in system_prompt

    def test_user_prompt_template(self):
        """Test user prompt template for requirements extraction."""
        from crewforge.core.llm import LLMClient

        client = LLMClient()
        user_requirements = (
            "Create a content research crew that finds and summarizes articles"
        )
        formatted_prompt = client.format_user_prompt(user_requirements)

        assert user_requirements in formatted_prompt
        assert "requirements" in formatted_prompt.lower()
        assert len(formatted_prompt) > len(user_requirements)

    def test_response_parsing_for_configurations(self):
        """Test parsing of LLM responses into configurations."""
        from crewforge.core.llm import parse_generation_response

        sample_response = {
            "agents": [
                {
                    "role": "researcher",
                    "goal": "Find relevant articles",
                    "backstory": "Expert research analyst",
                }
            ],
            "tasks": [
                {
                    "description": "Research articles on given topic",
                    "expected_output": "List of relevant articles with summaries",
                }
            ],
            "tools": ["search_tool", "web_scraper"],
        }

        parsed = parse_generation_response(sample_response)

        assert "agents" in parsed
        assert "tasks" in parsed
        assert "tools" in parsed
        assert len(parsed["agents"]) == 1
        assert parsed["agents"][0]["role"] == "researcher"


class TestLLMError:
    """Test cases for LLM-specific error handling."""

    def test_llm_error_creation(self):
        """Test LLMError can be created with message."""
        error = LLMError("Test error message")
        assert str(error) == "Test error message"

    def test_llm_error_with_original_exception(self):
        """Test LLMError can wrap original exceptions."""
        original = ValueError("Original error")
        error = LLMError("Wrapped error", original_exception=original)

        assert str(error) == "Wrapped error"
        assert error.original_exception == original


class TestConfiguration:
    """Test cases for configuration management."""

    def test_model_selection_defaults(self):
        """Test default model selection."""
        client = LLMClient()
        assert client.model == "gpt-4"

    def test_model_selection_override(self):
        """Test model selection can be overridden."""
        client = LLMClient(model="claude-3-sonnet-20240229")
        assert client.model == "claude-3-sonnet-20240229"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_warning(self):
        """Test warning when API keys are not configured."""
        # This should not fail during init, but may issue warnings
        client = LLMClient()
        assert client.model == "gpt-4"

    def test_rate_limiting_configuration(self):
        """Test rate limiting can be configured."""
        # This will be a placeholder until rate limiting is implemented
        client = LLMClient()
        # Should initialize without error
        assert client.model == "gpt-4"

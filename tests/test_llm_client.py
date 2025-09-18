"""
Tests for the LLM client module.

Tests cover API client initialization, token management, rate limiting,
error handling, and multi-provider support using liteLLM.
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch
from crewforge.llm import LLMClient, LLMError, RateLimitError, TokenLimitError


class TestLLMClientInitialization:
    """Test LLM client initialization and configuration."""

    def test_init_with_default_provider(self):
        """Test client initializes with OpenAI as default provider."""
        client = LLMClient()
        assert client.provider == "openai"
        assert client.model == "gpt-3.5-turbo"
        assert client.max_tokens == 4096

    def test_init_with_custom_provider_and_model(self):
        """Test client initializes with custom provider and model."""
        client = LLMClient(
            provider="anthropic", model="claude-3-sonnet-20240229", max_tokens=8192
        )
        assert client.provider == "anthropic"
        assert client.model == "claude-3-sonnet-20240229"
        assert client.max_tokens == 8192

    def test_init_with_api_key(self):
        """Test client initializes with API key."""
        client = LLMClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_init_without_api_key_uses_env(self):
        """Test client uses environment variable when no API key provided."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env_key"}):
            client = LLMClient()
            assert client.api_key == "env_key"

    def test_supported_providers(self):
        """Test that client supports expected providers."""
        expected_providers = ["openai", "anthropic", "google", "groq", "sambanova"]
        for provider in expected_providers:
            client = LLMClient(provider=provider)
            assert client.provider == provider

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises LLMError."""
        with pytest.raises(LLMError, match="Unsupported provider"):
            LLMClient(provider="unsupported_provider")


class TestTokenManagement:
    """Test token counting and management functionality."""

    def setup_method(self):
        """Set up test client for each test."""
        self.client = LLMClient()

    def test_count_tokens_simple_text(self):
        """Test token counting for simple text."""
        text = "Hello world"
        with patch("tiktoken.encoding_for_model") as mock_encoding:
            mock_enc = Mock()
            mock_enc.encode.return_value = [1, 2, 3]  # 3 tokens
            mock_encoding.return_value = mock_enc

            token_count = self.client.count_tokens(text)
            assert token_count == 3

    def test_count_tokens_complex_text(self):
        """Test token counting for complex text with special characters."""
        text = "Create a content research team with specialized agents for web scraping, data analysis, and report generation."
        with patch("tiktoken.encoding_for_model") as mock_encoding:
            mock_enc = Mock()
            mock_enc.encode.return_value = [1] * 20  # 20 tokens
            mock_encoding.return_value = mock_enc

            token_count = self.client.count_tokens(text)
            assert token_count == 20

    def test_count_tokens_messages_format(self):
        """Test token counting for messages format."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Create a research team."},
        ]
        with patch("tiktoken.encoding_for_model") as mock_encoding:
            mock_enc = Mock()
            mock_enc.encode.return_value = [1] * 10  # 10 tokens per message
            mock_encoding.return_value = mock_enc

            token_count = self.client.count_tokens(messages)
            # Should include message formatting tokens
            assert token_count > 20

    def test_check_token_limit_within_limit(self):
        """Test token limit check when within limits."""
        self.client.max_tokens = 100
        with patch.object(self.client, "count_tokens", return_value=50):
            # Should not raise an exception
            self.client.check_token_limit("test prompt")

    def test_check_token_limit_exceeds_limit(self):
        """Test token limit check when exceeding limits."""
        self.client.max_tokens = 100
        with patch.object(self.client, "count_tokens", return_value=150):
            with pytest.raises(TokenLimitError, match="Token limit exceeded"):
                self.client.check_token_limit("test prompt")


class TestRateLimiting:
    """Test rate limiting functionality."""

    def setup_method(self):
        """Set up test client for each test."""
        self.client = LLMClient()

    def test_rate_limit_tracking_new_client(self):
        """Test rate limiting starts with empty tracking."""
        assert len(self.client._request_history) == 0

    def test_rate_limit_allows_request_within_limits(self):
        """Test rate limiting allows requests within limits."""
        # Mock time to control rate limiting
        with patch("time.time", return_value=1000):
            self.client.check_rate_limit()
            # Should not raise exception

    def test_rate_limit_blocks_excessive_requests(self):
        """Test rate limiting blocks requests when limits exceeded."""
        # Fill up request history to trigger rate limit
        with patch("time.time", return_value=1000):
            # Add many requests in a short time
            for _ in range(self.client.requests_per_minute + 1):
                self.client._request_history.append(1000)

            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                self.client.check_rate_limit()

    def test_rate_limit_cleanup_old_requests(self):
        """Test rate limiting cleans up old requests from history."""
        # Add old requests that should be cleaned up
        old_time = 900  # 100 seconds ago
        current_time = 1000

        with patch("time.time", return_value=current_time):
            # Add old requests
            for _ in range(5):
                self.client._request_history.append(old_time)

            # Add current request
            self.client.check_rate_limit()

            # Old requests should be removed
            assert all(t >= current_time - 60 for t in self.client._request_history)


class TestMultiProviderSupport:
    """Test multi-provider functionality."""

    @patch("litellm.completion")
    async def test_openai_provider_completion(self, mock_completion):
        """Test completion with OpenAI provider."""
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="Test response"))]
        )

        client = LLMClient(provider="openai")
        response = await client.complete("Test prompt")

        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args[1]["model"].startswith("gpt-")
        assert response == "Test response"

    @patch("litellm.completion")
    async def test_anthropic_provider_completion(self, mock_completion):
        """Test completion with Anthropic provider."""
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="Claude response"))]
        )

        client = LLMClient(provider="anthropic", model="claude-3-sonnet-20240229")
        response = await client.complete("Test prompt")

        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args[1]["model"].startswith("claude-")
        assert response == "Claude response"

    @patch("litellm.completion")
    async def test_google_provider_completion(self, mock_completion):
        """Test completion with Google provider."""
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="Gemini response"))]
        )

        client = LLMClient(provider="google", model="gemini-pro")
        response = await client.complete("Test prompt")

        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args[1]["model"] == "gemini-pro"
        assert response == "Gemini response"


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""

    def setup_method(self):
        """Set up test client for each test."""
        self.client = LLMClient()

    @patch("litellm.completion")
    async def test_api_error_handling(self, mock_completion):
        """Test handling of API errors."""
        mock_completion.side_effect = Exception("API request failed")

        with pytest.raises(LLMError, match="API request failed"):
            await self.client.complete("Test prompt")

    @patch("litellm.completion")
    async def test_rate_limit_error_handling(self, mock_completion):
        """Test handling of rate limit errors from providers."""
        mock_completion.side_effect = Exception("rate limit exceeded")

        with pytest.raises(RateLimitError, match="Provider rate limit exceeded"):
            await self.client.complete("Test prompt")

    @patch("litellm.completion")
    async def test_token_limit_error_handling(self, mock_completion):
        """Test handling of token limit errors."""
        mock_completion.side_effect = Exception("context length exceeded")

        with pytest.raises(TokenLimitError, match="Context length exceeded"):
            await self.client.complete("Test prompt")

    @patch("litellm.completion")
    async def test_network_error_with_retry(self, mock_completion):
        """Test network error handling with retry logic."""
        import asyncio

        # First call fails, second succeeds
        mock_completion.side_effect = [
            ConnectionError("Network error"),
            Mock(choices=[Mock(message=Mock(content="Success after retry"))]),
        ]

        client = LLMClient(max_retries=2)
        response = await client.complete("Test prompt")

        assert mock_completion.call_count == 2
        assert response == "Success after retry"

    @patch("litellm.completion")
    async def test_max_retries_exceeded(self, mock_completion):
        """Test behavior when max retries are exceeded."""
        mock_completion.side_effect = ConnectionError("Persistent network error")

        client = LLMClient(max_retries=1)
        with pytest.raises(LLMError, match="Max retries exceeded"):
            await client.complete("Test prompt")

        assert mock_completion.call_count == 2  # Initial + 1 retry


class TestStructuredOutput:
    """Test structured output parsing functionality."""

    def setup_method(self):
        """Set up test client for each test."""
        self.client = LLMClient()

    @patch("litellm.completion")
    async def test_parse_json_response(self, mock_completion):
        """Test parsing JSON response from LLM."""
        json_response = json.dumps(
            {
                "agents": ["Research Agent", "Writing Agent"],
                "tasks": ["Gather information", "Write report"],
            }
        )

        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content=json_response))]
        )

        response = await self.client.complete_structured("Create a research team")

        assert isinstance(response, dict)
        assert response["agents"] == ["Research Agent", "Writing Agent"]
        assert response["tasks"] == ["Gather information", "Write report"]

    @patch("litellm.completion")
    async def test_invalid_json_response_handling(self, mock_completion):
        """Test handling of invalid JSON responses."""
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="This is not JSON"))]
        )

        with pytest.raises(LLMError, match="Failed to parse JSON response"):
            await self.client.complete_structured("Create a research team")

    @patch("litellm.completion")
    async def test_structured_output_with_schema_validation(self, mock_completion):
        """Test structured output with schema validation."""
        json_response = json.dumps(
            {
                "agents": [
                    {"name": "Research Agent", "role": "researcher"},
                    {"name": "Writing Agent", "role": "writer"},
                ]
            }
        )

        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content=json_response))]
        )

        schema = {
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                        },
                        "required": ["name", "role"],
                    },
                }
            },
            "required": ["agents"],
        }

        response = await self.client.complete_structured("Create agents", schema=schema)

        assert len(response["agents"]) == 2
        assert response["agents"][0]["name"] == "Research Agent"
        assert response["agents"][0]["role"] == "researcher"


class TestConfiguration:
    """Test client configuration and customization."""

    def test_configure_custom_model_per_provider(self):
        """Test configuring custom models for different providers."""
        configs = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-opus-20240229"),
            ("google", "gemini-1.5-pro"),
        ]

        for provider, model in configs:
            client = LLMClient(provider=provider, model=model)
            assert client.model == model

    def test_configure_temperature_and_parameters(self):
        """Test configuring generation parameters."""
        client = LLMClient(temperature=0.7, top_p=0.9)
        assert client.temperature == 0.7
        assert client.top_p == 0.9

    def test_configure_timeout_settings(self):
        """Test configuring timeout settings."""
        client = LLMClient(timeout=30)
        assert client.timeout == 30

    def test_update_configuration(self):
        """Test updating client configuration after initialization."""
        client = LLMClient()

        client.update_config(
            provider="anthropic", model="claude-3-sonnet-20240229", temperature=0.8
        )

        assert client.provider == "anthropic"
        assert client.model == "claude-3-sonnet-20240229"
        assert client.temperature == 0.8

"""
Additional tests for enhanced rate limiting and API error handling.

Tests cover advanced scenarios and edge cases for the LLM client
rate limiting and error handling functionality.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
import time

from crewforge.llm import LLMClient, LLMError, RateLimitError, TokenLimitError


class TestAdvancedRateLimiting:
    """Test advanced rate limiting scenarios."""

    def setup_method(self):
        """Set up test client with custom rate limiting."""
        self.client = LLMClient(requests_per_minute=5)  # Low limit for testing

    def test_rate_limit_with_concurrent_requests(self):
        """Test rate limiting behavior with concurrent requests."""
        # Fill up the rate limit almost completely
        current_time = time.time()
        self.client._request_history = [current_time - 10] * 4  # 4 recent requests

        # This should succeed (5th request)
        self.client.check_rate_limit()

        # This should fail (6th request)
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            self.client.check_rate_limit()

    def test_rate_limit_reset_after_time_window(self):
        """Test that rate limit resets properly after time window."""
        # Fill up rate limit with old requests
        old_time = time.time() - 120  # 2 minutes ago
        self.client._request_history = [old_time] * 10  # Many old requests

        # Should allow new requests since old ones are outside window
        self.client.check_rate_limit()  # Should succeed
        assert len(self.client._request_history) == 1  # Only current request

    def test_rate_limit_partial_cleanup(self):
        """Test partial cleanup of request history."""
        current_time = time.time()
        # Mix of old and recent requests
        self.client._request_history = [
            current_time - 120,  # Old (should be removed)
            current_time - 30,  # Recent (should be kept)
            current_time - 10,  # Recent (should be kept)
        ]

        # This should clean up old requests and succeed
        self.client.check_rate_limit()

        # Should have 3 requests now (2 recent + current)
        assert len(self.client._request_history) == 3

    def test_custom_rate_limit_values(self):
        """Test client with different rate limit configurations."""
        # Very strict rate limiting
        strict_client = LLMClient(requests_per_minute=1)
        strict_client.check_rate_limit()  # First should succeed

        with pytest.raises(RateLimitError):
            strict_client.check_rate_limit()  # Second should fail

        # Very permissive rate limiting
        permissive_client = LLMClient(requests_per_minute=1000)
        for _ in range(10):
            permissive_client.check_rate_limit()  # All should succeed


class TestAdvancedErrorHandling:
    """Test advanced error handling scenarios."""

    def setup_method(self):
        """Set up test client for each test."""
        self.client = LLMClient(max_retries=2)

    @patch("litellm.completion")
    async def test_authentication_error_handling(self, mock_completion):
        """Test handling of authentication errors."""
        mock_completion.side_effect = Exception("invalid api key")

        with pytest.raises(LLMError, match="API request failed: invalid api key"):
            await self.client.complete("Test prompt")

    @patch("litellm.completion")
    async def test_quota_exceeded_error_handling(self, mock_completion):
        """Test handling of quota exceeded errors."""
        mock_completion.side_effect = Exception("quota exceeded")

        with pytest.raises(LLMError, match="API request failed: quota exceeded"):
            await self.client.complete("Test prompt")

    @patch("litellm.completion")
    async def test_model_not_found_error_handling(self, mock_completion):
        """Test handling of model not found errors."""
        mock_completion.side_effect = Exception("model not found")

        with pytest.raises(LLMError, match="API request failed: model not found"):
            await self.client.complete("Test prompt")

    @patch("litellm.completion")
    async def test_server_error_with_retry(self, mock_completion):
        """Test server error handling with retry logic."""
        # Server error that should trigger retry
        mock_completion.side_effect = [
            Exception("internal server error"),  # First attempt
            Exception("service unavailable"),  # Second attempt
            Mock(choices=[Mock(message=Mock(content="Success"))]),  # Third succeeds
        ]

        client = LLMClient(max_retries=3)
        response = await client.complete("Test prompt")

        assert mock_completion.call_count == 3
        assert response == "Success"

    @patch("litellm.completion")
    async def test_timeout_error_with_exponential_backoff(self, mock_completion):
        """Test timeout error handling with proper exponential backoff."""
        import time

        start_time = time.time()
        mock_completion.side_effect = [
            TimeoutError("Request timeout"),
            Mock(choices=[Mock(message=Mock(content="Success after retry"))]),
        ]

        response = await self.client.complete("Test prompt")
        elapsed_time = time.time() - start_time

        # Should have waited at least 2^0 = 1 second for exponential backoff
        assert elapsed_time >= 1.0
        assert response == "Success after retry"

    @patch("litellm.completion")
    async def test_mixed_error_types_handling(self, mock_completion):
        """Test handling mixed error types in sequence."""
        mock_completion.side_effect = [
            ConnectionError("Network error"),  # Should retry
            Exception("rate limit exceeded"),  # Should raise RateLimitError immediately
        ]

        with pytest.raises(RateLimitError, match="Provider rate limit exceeded"):
            await self.client.complete("Test prompt")

        # Should have attempted twice (initial + 1 retry for connection error)
        assert mock_completion.call_count == 2

    @patch("litellm.completion")
    async def test_error_handling_preserves_original_exception(self, mock_completion):
        """Test that error handling preserves original exception details."""
        original_error = Exception("Detailed error message with context")
        mock_completion.side_effect = original_error

        try:
            await self.client.complete("Test prompt")
        except LLMError as e:
            assert "Detailed error message with context" in str(e)

    @patch("litellm.completion")
    async def test_structured_output_error_handling(self, mock_completion):
        """Test error handling in structured output parsing."""
        # Return invalid JSON
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="Invalid JSON response"))]
        )

        with pytest.raises(LLMError, match="Failed to parse JSON response"):
            await self.client.complete_structured("Test prompt")

    @patch("litellm.completion")
    async def test_schema_validation_error_handling(self, mock_completion):
        """Test schema validation error handling."""
        # Return valid JSON that doesn't match schema
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content='{"wrong": "format"}'))]
        )

        schema = {
            "type": "object",
            "properties": {"agents": {"type": "array"}},
            "required": ["agents"],
        }

        with pytest.raises(LLMError, match="Response doesn't match schema"):
            await self.client.complete_structured("Test prompt", schema=schema)


class TestRateLimitingIntegration:
    """Test rate limiting integration with actual API calls."""

    def setup_method(self):
        """Set up test client with aggressive rate limiting."""
        self.client = LLMClient(requests_per_minute=2)

    @patch("litellm.completion")
    async def test_rate_limiting_in_complete_method(self, mock_completion):
        """Test that complete method respects rate limiting."""
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="Response"))]
        )

        # First two requests should succeed
        await self.client.complete("Test 1")
        await self.client.complete("Test 2")

        # Third request should fail due to rate limiting
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await self.client.complete("Test 3")

    @patch("litellm.completion")
    async def test_rate_limiting_in_structured_complete_method(self, mock_completion):
        """Test that complete_structured method respects rate limiting."""
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content='{"result": "success"}'))]
        )

        # First two requests should succeed
        await self.client.complete_structured("Test 1")
        await self.client.complete_structured("Test 2")

        # Third request should fail due to rate limiting
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await self.client.complete_structured("Test 3")


class TestErrorHandlingEdgeCases:
    """Test error handling edge cases and unusual scenarios."""

    def setup_method(self):
        """Set up test client."""
        self.client = LLMClient()

    @patch("litellm.completion")
    async def test_empty_response_handling(self, mock_completion):
        """Test handling of empty responses."""
        mock_completion.return_value = Mock(choices=[Mock(message=Mock(content=""))])

        response = await self.client.complete("Test prompt")
        assert response == ""

    @patch("litellm.completion")
    async def test_whitespace_only_response_handling(self, mock_completion):
        """Test handling of whitespace-only responses."""
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="   \n\t  "))]
        )

        response = await self.client.complete("Test prompt")
        assert response == "   \n\t  "

    @patch("litellm.completion")
    async def test_unicode_error_handling(self, mock_completion):
        """Test handling of unicode errors."""
        mock_completion.side_effect = Exception("Unicode error: ñíçé characters")

        with pytest.raises(LLMError, match="Unicode error: ñíçé characters"):
            await self.client.complete("Test prompt")

    @patch("litellm.completion")
    async def test_nested_exception_handling(self, mock_completion):
        """Test handling of nested exceptions."""
        inner_exception = ValueError("Inner error")
        outer_exception = Exception("Outer error")
        outer_exception.__cause__ = inner_exception
        mock_completion.side_effect = outer_exception

        with pytest.raises(LLMError, match="Outer error"):
            await self.client.complete("Test prompt")

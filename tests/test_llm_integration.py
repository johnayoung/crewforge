#!/usr/bin/env python3
"""
Integration tests for the LLM client functionality.

These tests validate the LLM client integration without making actual API calls,
demonstrating key features like initialization, token management, rate limiting,
and multi-provider support.
"""
import pytest
from crewforge.llm import LLMClient, LLMError, RateLimitError, TokenLimitError


class TestLLMClientIntegration:
    """Integration tests for LLM client functionality."""

    def test_complete_initialization_workflow(self):
        """Test the complete client initialization workflow."""
        # Test basic initialization
        client = LLMClient()
        assert client.provider == "openai"
        assert client.model == "gpt-3.5-turbo"
        assert client.max_tokens == 4096

        # Test multi-provider initialization
        providers = ["openai", "anthropic", "google", "groq", "sambanova"]
        for provider in providers:
            test_client = LLMClient(provider=provider)
            assert test_client.provider == provider
            assert test_client.model == LLMClient.DEFAULT_MODELS[provider]

    def test_token_management_workflow(self):
        """Test token counting and limit validation workflow."""
        client = LLMClient()

        # Test various text formats
        test_cases = [
            ("Simple text", "Hello world"),
            (
                "Complex prompt",
                "Create a content research team with specialized agents",
            ),
            (
                "Message format",
                [{"role": "user", "content": "Design a multi-agent system"}],
            ),
        ]

        for description, text in test_cases:
            token_count = client.count_tokens(text)
            assert isinstance(token_count, int)
            assert token_count > 0, f"Token count should be positive for {description}"

        # Test token limit validation
        client.check_token_limit("Short prompt")  # Should not raise

        # Test token limit enforcement
        small_client = LLMClient(max_tokens=10)
        with pytest.raises(TokenLimitError):
            small_client.check_token_limit(
                "This is a longer prompt that should exceed the tiny token limit for testing"
            )

    def test_rate_limiting_workflow(self):
        """Test rate limiting functionality workflow."""
        client = LLMClient()

        # Basic rate limit check should pass
        client.check_rate_limit()

        # Test rate limit enforcement
        rate_limited_client = LLMClient(requests_per_minute=1)

        # Simulate exceeding rate limit
        import time

        current_time = time.time()

        # Add enough requests to exceed the limit
        for _ in range(2):  # Exceeds limit of 1
            rate_limited_client._request_history.append(current_time)

        with pytest.raises(RateLimitError):
            rate_limited_client.check_rate_limit()

    def test_configuration_management_workflow(self):
        """Test configuration updates and management."""
        client = LLMClient()

        # Test configuration updates
        client.update_config(provider="anthropic", temperature=0.8, max_tokens=8192)

        assert client.provider == "anthropic"
        assert client.temperature == 0.8
        assert client.max_tokens == 8192
        assert client.model == LLMClient.DEFAULT_MODELS["anthropic"]

        # Test model override
        client.update_config(model="claude-3-opus-20240229")
        assert client.model == "claude-3-opus-20240229"

    def test_error_handling_workflow(self):
        """Test comprehensive error handling."""
        # Test unsupported provider
        with pytest.raises(LLMError, match="Unsupported provider"):
            LLMClient(provider="unsupported_provider")

        # Test invalid configuration updates
        client = LLMClient()
        with pytest.raises(LLMError):
            client.update_config(provider="invalid_provider")

    def test_multi_provider_model_mapping(self):
        """Test that each provider gets the correct default model."""
        expected_mappings = {
            "openai": "gpt-3.5-turbo",
            "anthropic": "claude-3-sonnet-20240229",
            "google": "gemini-pro",
            "groq": "mixtral-8x7b-32768",
            "sambanova": "Meta-Llama-3.1-8B-Instruct",
        }

        for provider, expected_model in expected_mappings.items():
            client = LLMClient(provider=provider)
            assert (
                client.model == expected_model
            ), f"Provider {provider} should default to {expected_model}"

    def test_environment_variable_integration(self):
        """Test API key handling from environment variables."""
        import os
        from unittest.mock import patch

        # Test with environment variable
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            client = LLMClient()
            assert client.api_key == "test_key"

        # Test explicit API key overrides environment
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env_key"}):
            client = LLMClient(api_key="explicit_key")
            assert client.api_key == "explicit_key"

    def test_client_state_consistency(self):
        """Test that client maintains consistent state through operations."""
        client = LLMClient(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens=4000,
        )

        # Perform operations that shouldn't change core config
        client.check_rate_limit()
        client.count_tokens("test prompt")
        client.check_token_limit("test prompt")

        # Verify state unchanged
        assert client.provider == "anthropic"
        assert client.model == "claude-3-sonnet-20240229"
        assert client.temperature == 0.7
        assert client.max_tokens == 4000

    def test_structured_output_preparation(self):
        """Test preparation for structured output functionality."""
        client = LLMClient()

        # Test that structured prompts are properly formatted
        prompt = "Create a research team"

        # Simulate the structured output prompt preparation
        messages = [
            {
                "role": "user",
                "content": f"{prompt}\n\nRespond with valid JSON only, no additional text.",
            }
        ]

        # Verify token counting works with formatted messages
        token_count = client.count_tokens(messages)
        assert isinstance(token_count, int)
        assert token_count > client.count_tokens(
            prompt
        )  # Should be more tokens due to formatting

    def test_integration_readiness_checklist(self):
        """Validate that the client is ready for CrewAI integration."""
        client = LLMClient()

        # ✅ Multi-provider support
        assert len(LLMClient.SUPPORTED_PROVIDERS) >= 5
        assert "openai" in LLMClient.SUPPORTED_PROVIDERS
        assert "anthropic" in LLMClient.SUPPORTED_PROVIDERS

        # ✅ Token management
        assert hasattr(client, "count_tokens")
        assert hasattr(client, "check_token_limit")
        assert callable(client.count_tokens)
        assert callable(client.check_token_limit)

        # ✅ Rate limiting
        assert hasattr(client, "check_rate_limit")
        assert hasattr(client, "_request_history")
        assert callable(client.check_rate_limit)

        # ✅ Error handling
        assert issubclass(RateLimitError, LLMError)
        assert issubclass(TokenLimitError, LLMError)

        # ✅ Configuration management
        assert hasattr(client, "update_config")
        assert callable(client.update_config)

        # ✅ Async support ready
        assert hasattr(client, "complete")
        assert hasattr(client, "complete_structured")

        print("✅ LLM Client Integration Validation Complete")
        print("Ready for next milestone: Natural Language Prompt Processing")


def test_demo_functionality():
    """Demonstration function showing LLM client capabilities."""
    print("\n=== CrewForge LLM Client Integration Demo ===")

    client = LLMClient()
    print(f"✓ Initialized with provider: {client.provider}, model: {client.model}")

    # Demonstrate token counting
    sample_prompt = (
        "Create a content research team with web scraping and analysis agents"
    )
    tokens = client.count_tokens(sample_prompt)
    print(f"✓ Token counting: '{sample_prompt[:30]}...' = {tokens} tokens")

    # Demonstrate multi-provider support
    print("✓ Multi-provider support:")
    for provider in ["openai", "anthropic", "google"]:
        test_client = LLMClient(provider=provider)
        print(f"  - {provider}: {test_client.model}")

    # Demonstrate rate limiting
    client.check_rate_limit()
    print("✓ Rate limiting check passed")

    # Demonstrate configuration
    client.update_config(temperature=0.8, max_tokens=8192)
    print(
        f"✓ Configuration updated: temp={client.temperature}, max_tokens={client.max_tokens}"
    )

    print("=== Integration Demo Complete ===\n")


if __name__ == "__main__":
    # Run the demo when executed directly
    test_demo_functionality()

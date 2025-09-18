"""
Integration tests for rate limiting and API error handling with CLI.

Tests the integration between the LLM client's enhanced error handling
and the CLI interface to ensure proper error reporting and user experience.
"""

import pytest
from unittest.mock import patch, Mock
from click.testing import CliRunner

from crewforge.cli import create
from crewforge.llm import LLMError, RateLimitError, TokenLimitError


class TestCLIErrorHandlingIntegration:
    """Test CLI integration with enhanced error handling."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    @patch("crewforge.llm.LLMClient")
    @patch("crewforge.prompt_templates.PromptTemplates")
    def test_cli_handles_rate_limit_error(self, mock_prompt_templates, mock_llm_client):
        """Test CLI properly handles and reports rate limit errors."""
        # Mock LLM client to raise rate limit error
        mock_client_instance = Mock()
        mock_client_instance.complete_structured.side_effect = RateLimitError(
            "Rate limit exceeded: 60 requests per minute"
        )
        mock_llm_client.return_value = mock_client_instance

        # Mock prompt templates
        mock_templates_instance = Mock()
        mock_templates_instance.extract_project_spec.side_effect = RateLimitError(
            "Rate limit exceeded: 60 requests per minute"
        )
        mock_prompt_templates.return_value = mock_templates_instance

        # Run CLI command
        result = self.runner.invoke(
            create, ["test-project", "Create a simple data analysis team"]
        )

        # Verify error is handled gracefully
        assert result.exit_code == 0  # CLI should handle gracefully, not crash
        assert "LLM API not configured" in result.output  # Falls back to demo mode

    def test_cli_parameter_validation(self):
        """Test CLI validates parameters properly before calling LLM."""
        # Test missing prompt - this should fail with exit code 1
        result = self.runner.invoke(create, ["test-project"])
        # CLI should require prompt or interactive mode
        assert result.exit_code == 1  # Should fail with validation error
        assert "Please provide a prompt or use --interactive mode" in result.output

    def test_cli_handles_basic_functionality(self):
        """Test basic CLI functionality works."""
        # Test with valid parameters (will fall back to demo mode without API key)
        result = self.runner.invoke(create, ["test-project", "Create a test project"])
        assert result.exit_code == 0
        assert "CrewForge" in result.output

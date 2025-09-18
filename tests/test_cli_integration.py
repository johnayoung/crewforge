"""
Integration tests for CLI with prompt templates.

Tests the integration between the CLI interface and the prompt template functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from click.testing import CliRunner

from crewforge.cli import create
from crewforge.llm import LLMClient
from crewforge.prompt_templates import PromptTemplates


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_project_spec():
    """Create a mock project specification response."""
    return {
        "project_name": "test-project",
        "project_description": "A test project for validation",
        "agents": [
            {
                "role": "Test Agent",
                "goal": "Test system functionality",
                "backstory": "A specialized agent for testing",
                "tools": ["test_tool"],
            }
        ],
        "tasks": [
            {
                "description": "Run validation tests",
                "expected_output": "Test results report",
                "agent": "Test Agent",
            }
        ],
        "dependencies": ["crewai", "pytest"],
    }


class TestCLIIntegration:
    """Test integration between CLI and prompt templates."""

    @patch("crewforge.cli.LLMClient")
    @patch("crewforge.cli.PromptTemplates")
    @patch("crewforge.cli.asyncio.run")
    def test_create_command_with_prompt_success(
        self,
        mock_asyncio_run,
        mock_prompt_templates_class,
        mock_llm_client_class,
        cli_runner,
        mock_project_spec,
    ):
        """Test successful project creation with prompt."""
        # Setup mocks
        mock_llm_client = Mock(spec=LLMClient)
        mock_llm_client_class.return_value = mock_llm_client

        mock_prompt_templates = Mock(spec=PromptTemplates)
        mock_prompt_templates.extract_project_spec = AsyncMock(
            return_value=mock_project_spec
        )
        mock_prompt_templates_class.return_value = mock_prompt_templates

        mock_asyncio_run.return_value = mock_project_spec

        # Run CLI command
        result = cli_runner.invoke(
            create, ["test-project", "Create a test project with validation agents"]
        )

        # Verify results
        assert result.exit_code == 0
        assert "test-project" in result.output
        assert "Test Agent" in result.output
        assert "Extracted 1 agents and 1 tasks" in result.output

        # Verify mocks were called correctly
        mock_llm_client_class.assert_called_once()
        mock_prompt_templates_class.assert_called_once_with(llm_client=mock_llm_client)
        mock_asyncio_run.assert_called_once()

    @patch("crewforge.cli.LLMClient")
    def test_create_command_llm_client_error(self, mock_llm_client_class, cli_runner):
        """Test handling of LLM client initialization errors."""
        from crewforge.llm import LLMError

        mock_llm_client_class.side_effect = LLMError("API key not found")

        result = cli_runner.invoke(create, ["test-project", "Create a test project"])

        # CLI should gracefully handle LLM errors and fall back to demo mode
        assert result.exit_code == 0
        assert "LLM API not configured" in result.output
        assert "Project Analysis (Simulated)" in result.output

    @patch("crewforge.cli.LLMClient")
    @patch("crewforge.cli.PromptTemplates")
    @patch("crewforge.cli.asyncio.run")
    def test_create_command_prompt_template_error(
        self,
        mock_asyncio_run,
        mock_prompt_templates_class,
        mock_llm_client_class,
        cli_runner,
    ):
        """Test handling of prompt template processing errors."""
        from crewforge.prompt_templates import PromptTemplateError

        # Setup mocks - LLM client works, but prompt template fails
        mock_llm_client = Mock(spec=LLMClient)
        mock_llm_client_class.return_value = mock_llm_client

        mock_prompt_templates = Mock(spec=PromptTemplates)
        mock_prompt_templates_class.return_value = mock_prompt_templates

        mock_asyncio_run.side_effect = PromptTemplateError("Invalid prompt structure")

        result = cli_runner.invoke(
            create, ["test-project", "Invalid prompt that cannot be parsed"]
        )

        # CLI should gracefully handle prompt template errors and fall back to demo mode
        assert result.exit_code == 0
        assert "LLM API not configured" in result.output
        assert "Project Analysis (Simulated)" in result.output

    def test_create_command_empty_prompt_non_interactive(self, cli_runner):
        """Test handling of missing prompt without interactive mode."""
        result = cli_runner.invoke(create, ["test-project"])

        assert result.exit_code == 1
        assert "Please provide a prompt or use --interactive mode" in result.output

    @patch("crewforge.cli.click.prompt")
    def test_create_command_interactive_mode(self, mock_prompt, cli_runner):
        """Test interactive mode prompt collection."""
        mock_prompt.return_value = "Create a content writing team"

        # Mock LLM components to avoid actual API calls
        with (
            patch("crewforge.cli.LLMClient"),
            patch("crewforge.cli.PromptTemplates"),
            patch("crewforge.cli.asyncio.run") as mock_run,
        ):

            mock_run.return_value = {
                "project_name": "test-project",
                "project_description": "Interactive test project",
                "agents": [
                    {
                        "role": "Writer",
                        "goal": "Write",
                        "backstory": "Writer",
                        "tools": ["editor"],
                    }
                ],
                "tasks": [
                    {
                        "description": "Write content",
                        "expected_output": "Article",
                        "agent": "Writer",
                    }
                ],
                "dependencies": ["crewai"],
            }

            result = cli_runner.invoke(create, ["test-project", "--interactive"])

            assert result.exit_code == 0
            mock_prompt.assert_called_once()

    def test_create_command_invalid_project_name(self, cli_runner):
        """Test handling of invalid project names."""
        result = cli_runner.invoke(create, ["", "Test prompt"])
        assert result.exit_code == 1
        assert "Project name cannot be empty" in result.output

        result = cli_runner.invoke(create, ["invalid name with spaces", "Test prompt"])
        assert result.exit_code == 1
        # The actual error message from the CLI
        assert "must contain at least one alphanumeric character" in result.output

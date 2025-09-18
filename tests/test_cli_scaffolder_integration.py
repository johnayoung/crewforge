"""
Tests for CLI integration with CrewAI scaffolder

Test suite for the CLI integration with the scaffolder module,
including success and failure scenarios.
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
from click.testing import CliRunner

from crewforge.cli import create
from crewforge.scaffolder import CrewAIError


class TestCLIScaffolderIntegration:
    """Test suite for CLI integration with CrewAI scaffolder."""

    def test_cli_with_scaffolder_not_available(self):
        """Test CLI when CrewAI scaffolder is not available."""
        runner = CliRunner()

        with patch("crewforge.cli.CrewAIScaffolder") as mock_scaffolder_class:
            mock_scaffolder = Mock()
            mock_scaffolder.check_crewai_available.return_value = False
            mock_scaffolder_class.return_value = mock_scaffolder

            result = runner.invoke(
                create,
                [
                    "test-project",
                    "Create a simple content team",
                    "--output-dir",
                    "/tmp",
                ],
            )

            assert result.exit_code == 0
            assert "CrewAI CLI not found" in result.output
            assert "Scaffolding System Ready" in result.output
            assert "install crewai" in result.output

    def test_cli_with_scaffolder_success(self):
        """Test CLI when CrewAI scaffolder succeeds."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_path = temp_path / "test-project"
            project_path.mkdir()

            mock_result = {
                "success": True,
                "project_path": project_path,
                "project_name": "test-project",
                "returncode": 0,
                "stdout": "Project created successfully",
                "stderr": "",
            }

            with patch("crewforge.cli.CrewAIScaffolder") as mock_scaffolder_class:
                mock_scaffolder = Mock()
                mock_scaffolder.check_crewai_available.return_value = True
                mock_scaffolder.create_crew.return_value = mock_result
                mock_scaffolder_class.return_value = mock_scaffolder

                result = runner.invoke(
                    create,
                    [
                        "test-project",
                        "Create a simple content team",
                        "--output-dir",
                        temp_dir,
                    ],
                )

                assert result.exit_code == 0
                assert "CrewAI project structure created successfully" in result.output
                assert str(project_path) in result.output
                mock_scaffolder.create_crew.assert_called_once()

    def test_cli_with_scaffolder_failure(self):
        """Test CLI when CrewAI scaffolder fails."""
        runner = CliRunner()

        with patch("crewforge.cli.CrewAIScaffolder") as mock_scaffolder_class:
            mock_scaffolder = Mock()
            mock_scaffolder.check_crewai_available.return_value = True
            mock_scaffolder.create_crew.side_effect = CrewAIError(
                "Project creation failed"
            )
            mock_scaffolder_class.return_value = mock_scaffolder

            result = runner.invoke(
                create,
                [
                    "test-project",
                    "Create a simple content team",
                    "--output-dir",
                    "/tmp",
                ],
            )

            assert result.exit_code == 0  # Falls back to demo mode
            assert "CrewAI scaffolding failed" in result.output
            assert "Demo Mode" in result.output

    def test_cli_with_scaffolder_unsuccessful_result(self):
        """Test CLI when CrewAI scaffolder returns unsuccessful result."""
        runner = CliRunner()

        mock_result = {
            "success": False,
            "project_path": Path("/tmp/test-project"),
            "project_name": "test-project",
            "returncode": 1,
            "stdout": "",
            "stderr": "Project creation failed",
        }

        with patch("crewforge.cli.CrewAIScaffolder") as mock_scaffolder_class:
            mock_scaffolder = Mock()
            mock_scaffolder.check_crewai_available.return_value = True
            mock_scaffolder.create_crew.return_value = mock_result
            mock_scaffolder_class.return_value = mock_scaffolder

            result = runner.invoke(
                create,
                [
                    "test-project",
                    "Create a simple content team",
                    "--output-dir",
                    "/tmp",
                ],
            )

            assert result.exit_code == 1  # Click.Abort() returns exit code 1
            assert "Failed to create CrewAI project structure" in result.output

    def test_cli_scaffolder_path_handling(self):
        """Test that CLI correctly passes Path objects to scaffolder."""
        runner = CliRunner()

        with patch("crewforge.cli.CrewAIScaffolder") as mock_scaffolder_class:
            mock_scaffolder = Mock()
            mock_scaffolder.check_crewai_available.return_value = True
            mock_scaffolder.create_crew.side_effect = CrewAIError("Test error")
            mock_scaffolder_class.return_value = mock_scaffolder

            result = runner.invoke(
                create,
                [
                    "test-project",
                    "Create a simple content team",
                    "--output-dir",
                    "/tmp/test",
                ],
            )

            # Verify the scaffolder was called with a Path object
            mock_scaffolder.create_crew.assert_called_once_with(
                "test-project", Path("/tmp/test")
            )

    @patch("crewforge.cli.LLMClient")
    @patch("crewforge.cli.PromptTemplates")
    def test_cli_integration_full_flow(
        self, mock_prompt_templates_class, mock_llm_client_class
    ):
        """Test full CLI integration flow with all components."""
        runner = CliRunner()

        # Mock LLM components
        mock_llm_client = Mock()
        mock_llm_client_class.return_value = mock_llm_client

        mock_prompt_templates = Mock()
        mock_project_spec = {
            "project_name": "test-project",
            "project_description": "A test project",
            "agents": [{"role": "Writer"}, {"role": "Editor"}],
            "tasks": [{"name": "task1"}, {"name": "task2"}],
            "dependencies": ["crewai", "langchain"],
        }

        # Make the mock return an async coroutine
        async def mock_extract():
            return mock_project_spec

        mock_prompt_templates.extract_project_spec = Mock(return_value=mock_extract())
        mock_prompt_templates_class.return_value = mock_prompt_templates

        # Mock scaffolder
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_path = temp_path / "test-project"
            project_path.mkdir()

            mock_result = {
                "success": True,
                "project_path": project_path,
                "project_name": "test-project",
                "returncode": 0,
                "stdout": "Project created successfully",
                "stderr": "",
            }

            with patch("crewforge.cli.CrewAIScaffolder") as mock_scaffolder_class:
                mock_scaffolder = Mock()
                mock_scaffolder.check_crewai_available.return_value = True
                mock_scaffolder.create_crew.return_value = mock_result
                mock_scaffolder_class.return_value = mock_scaffolder

                result = runner.invoke(
                    create,
                    [
                        "test-project",
                        "Create a simple content team",
                        "--output-dir",
                        temp_dir,
                    ],
                )

                assert result.exit_code == 0
                assert "2 agents and 2 tasks" in result.output
                assert "CrewAI project structure created successfully" in result.output

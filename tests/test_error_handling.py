"""Tests for error handling and user experience improvements."""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from click.testing import CliRunner

from crewforge.cli.main import (
    cli,
    validate_prompt,
    validate_project_name,
    check_directory_conflicts,
)
from crewforge.core.llm import (
    LLMClient,
    LLMError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMNetworkError,
    LLMResponseError,
    RetryConfig,
)
from crewforge.core.scaffolding import (
    ProjectScaffolder,
    ScaffoldingError,
    CrewAICommandError,
    FileSystemError,
    ProjectStructureError,
)


class TestLLMErrorHandling:
    """Test LLM error handling and retry logic."""

    def test_retry_config_validation(self):
        """Test retry configuration validation."""
        # Valid config
        config = RetryConfig(max_attempts=3, base_delay=1.0, max_delay=60.0)
        assert config.max_attempts == 3

        # Invalid configs should raise validation errors
        with pytest.raises(Exception):
            RetryConfig(max_attempts=0)  # Must be >= 1

        with pytest.raises(Exception):
            RetryConfig(base_delay=-1.0)  # Must be >= 0

        with pytest.raises(Exception):
            RetryConfig(max_delay=0.0)  # Must be > 0

    @patch("crewforge.core.llm.litellm.completion")
    def test_llm_authentication_error(self, mock_completion):
        """Test LLM authentication error handling."""
        mock_completion.side_effect = Exception(
            "Authentication failed: Invalid API key"
        )

        client = LLMClient(retry_config=RetryConfig(max_attempts=1))

        with pytest.raises(LLMAuthenticationError) as exc_info:
            client.generate("system", "user")

        assert "authentication" in str(exc_info.value).lower()
        assert exc_info.value.error_type == "authentication"

    @patch("crewforge.core.llm.litellm.completion")
    def test_llm_rate_limit_error(self, mock_completion):
        """Test LLM rate limit error handling with retry-after."""
        mock_completion.side_effect = Exception(
            "Rate limit exceeded, retry in 30 seconds"
        )

        client = LLMClient(retry_config=RetryConfig(max_attempts=1))

        with pytest.raises(LLMRateLimitError) as exc_info:
            client.generate("system", "user")

        assert exc_info.value.error_type == "rate_limit"
        assert exc_info.value.retry_after == 30

    @patch("crewforge.core.llm.litellm.completion")
    def test_llm_network_error(self, mock_completion):
        """Test LLM network error handling."""
        mock_completion.side_effect = Exception("Connection timeout")

        client = LLMClient(retry_config=RetryConfig(max_attempts=1))

        with pytest.raises(LLMNetworkError) as exc_info:
            client.generate("system", "user")

        assert exc_info.value.error_type == "network"

    @patch("crewforge.core.llm.litellm.completion")
    def test_llm_retry_exhaustion(self, mock_completion):
        """Test retry exhaustion tracking."""
        mock_completion.side_effect = Exception("Temporary error")

        client = LLMClient(retry_config=RetryConfig(max_attempts=2, base_delay=0.1))

        with pytest.raises(LLMError) as exc_info:
            client.generate("system", "user")

        assert exc_info.value.retry_exhausted is True

    @patch("crewforge.core.llm.litellm.completion")
    def test_llm_json_parsing_error(self, mock_completion):
        """Test JSON parsing error handling."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json content"
        mock_completion.return_value = mock_response

        client = LLMClient()

        with pytest.raises(LLMResponseError) as exc_info:
            client.generate("system", "user", use_json_mode=True)

        assert "parse JSON" in str(exc_info.value)
        assert exc_info.value.error_type == "response"

    @patch("crewforge.core.llm.litellm.completion")
    def test_llm_successful_retry(self, mock_completion):
        """Test successful retry after temporary failure."""
        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Success!"

        mock_completion.side_effect = [Exception("Temporary error"), mock_response]

        client = LLMClient(retry_config=RetryConfig(max_attempts=2, base_delay=0.1))
        result = client.generate("system", "user")

        assert result == "Success!"
        assert mock_completion.call_count == 2


class TestScaffoldingErrorHandling:
    """Test scaffolding error handling."""

    def test_crewai_command_not_found(self):
        """Test handling when CrewAI CLI is not installed."""
        scaffolder = ProjectScaffolder()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("crewai: command not found")

            with pytest.raises(CrewAICommandError) as exc_info:
                scaffolder.create_crewai_project("test", Path("/tmp"))

            assert "not found" in str(exc_info.value).lower()

    def test_crewai_command_timeout(self):
        """Test handling CrewAI command timeout."""
        scaffolder = ProjectScaffolder()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("crewai", 120)

            with pytest.raises(CrewAICommandError) as exc_info:
                scaffolder.create_crewai_project("test", Path("/tmp"))

            assert "timed out" in str(exc_info.value).lower()

    def test_filesystem_permission_error(self):
        """Test file system permission error handling."""
        scaffolder = ProjectScaffolder()

        with pytest.raises(FileSystemError) as exc_info:
            # Try to create project in read-only directory
            readonly_dir = Path("/proc")  # This should be read-only on most systems
            scaffolder.create_crewai_project("test", readonly_dir)

        assert exc_info.value.error_type == "filesystem"

    def test_project_structure_validation_error(self):
        """Test project structure validation errors."""
        scaffolder = ProjectScaffolder()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "invalid_project"
            project_path.mkdir()

            # Create invalid structure (missing src directory)
            with pytest.raises(ProjectStructureError) as exc_info:
                scaffolder._validate_project_structure(project_path)

            assert exc_info.value.error_type == "structure"

    def test_disk_space_check(self):
        """Test disk space checking."""
        scaffolder = ProjectScaffolder()

        # Test with unreasonably large space requirement
        with pytest.raises(FileSystemError) as exc_info:
            scaffolder._check_disk_space(Path("/tmp"), min_space_mb=999999999)

        assert "disk space" in str(exc_info.value).lower()

    def test_file_backup_and_restore(self):
        """Test file backup and restore on failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            src_dir = project_path / "src"
            module_dir = src_dir / "test_module"
            module_dir.mkdir(parents=True)

            # Create existing file
            test_file = module_dir / "agents.py"
            test_file.write_text("original content")

            scaffolder = ProjectScaffolder()

            # Mock template engine to fail
            with patch.object(
                scaffolder.template_engine, "populate_template"
            ) as mock_template:
                mock_template.side_effect = Exception("Template error")

                with pytest.raises(FileSystemError):
                    scaffolder.populate_project_files(
                        project_path, [], [], {"selected_tools": []}
                    )

                # Original file should be restored
                assert test_file.exists()
                assert test_file.read_text() == "original content"


class TestCLIErrorHandling:
    """Test CLI error handling and user experience."""

    def test_prompt_validation_empty(self):
        """Test empty prompt validation."""
        with pytest.raises(Exception):
            validate_prompt("")

        with pytest.raises(Exception):
            validate_prompt("   ")

    def test_prompt_validation_too_short(self):
        """Test short prompt validation."""
        with pytest.raises(Exception):
            validate_prompt("short")

    def test_prompt_validation_too_long(self):
        """Test long prompt validation."""
        long_prompt = "a" * 2001
        with pytest.raises(Exception):
            validate_prompt(long_prompt)

    def test_prompt_validation_gibberish(self):
        """Test gibberish prompt validation."""
        with pytest.raises(Exception):
            validate_prompt("qwerty123 zxcvbnm qwertyuiop asdfghjkl")

    def test_project_name_validation_empty(self):
        """Test empty project name validation."""
        with pytest.raises(Exception):
            validate_project_name("", "valid prompt for testing purposes")

    def test_project_name_validation_special_chars(self):
        """Test project name with special characters."""
        result = validate_project_name("My Project! @#$", "valid prompt for testing")
        assert result == "my-project"

    def test_directory_conflicts(self):
        """Test directory conflict detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir) / "existing"
            existing_dir.mkdir()

            os.chdir(temp_dir)

            with pytest.raises(Exception):
                check_directory_conflicts("existing")

    def test_cli_verbose_mode(self):
        """Test CLI verbose mode functionality."""
        runner = CliRunner()

        with patch("crewforge.core.scaffolding.ProjectScaffolder") as mock_scaffolder:
            mock_instance = Mock()
            mock_scaffolder.return_value = mock_instance
            mock_instance.generate_project.side_effect = LLMAuthenticationError(
                "Auth failed"
            )

            result = runner.invoke(
                cli, ["generate", "--verbose", "test prompt for ai crew"]
            )

            assert result.exit_code != 0
            assert "authentication" in result.output.lower()

    def test_cli_no_color_mode(self):
        """Test CLI no-color mode functionality."""
        runner = CliRunner()

        with patch("crewforge.core.scaffolding.ProjectScaffolder") as mock_scaffolder:
            mock_instance = Mock()
            mock_scaffolder.return_value = mock_instance
            mock_instance.generate_project.side_effect = LLMNetworkError(
                "Network failed"
            )

            result = runner.invoke(
                cli, ["generate", "--no-color", "test prompt for ai crew"]
            )

            assert result.exit_code != 0
            assert "[ERROR]" in result.output or "ERROR" in result.output

    def test_cli_keyboard_interrupt(self):
        """Test CLI handling of keyboard interrupt."""
        runner = CliRunner()

        with patch("crewforge.core.scaffolding.ProjectScaffolder") as mock_scaffolder:
            mock_instance = Mock()
            mock_scaffolder.return_value = mock_instance
            mock_instance.generate_project.side_effect = KeyboardInterrupt()

            result = runner.invoke(cli, ["generate", "test prompt for ai crew"])

            assert result.exit_code != 0
            assert (
                "cancelled" in result.output.lower()
                or "interrupted" in result.output.lower()
            )


class TestIntegrationErrorHandling:
    """Test end-to-end error handling scenarios."""

    def test_full_pipeline_llm_failure(self):
        """Test full pipeline with LLM failure."""
        runner = CliRunner()

        with patch("crewforge.core.llm.litellm.completion") as mock_completion:
            mock_completion.side_effect = Exception("API key invalid")

            result = runner.invoke(
                cli, ["generate", "create a test crew for validation"]
            )

            assert result.exit_code != 0
            assert "authentication" in result.output.lower()

    def test_full_pipeline_crewai_failure(self):
        """Test full pipeline with CrewAI CLI failure."""
        runner = CliRunner()

        with patch("subprocess.run") as mock_run:
            # Mock successful LLM calls
            with patch("crewforge.core.llm.litellm.completion") as mock_completion:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = '{"agents": [], "tasks": []}'
                mock_completion.return_value = mock_response

                # Mock CrewAI command failure
                mock_run.side_effect = FileNotFoundError("crewai not found")

                result = runner.invoke(
                    cli, ["generate", "create a test crew for validation"]
                )

                assert result.exit_code != 0
                assert "crewai" in result.output.lower()

    def test_graceful_degradation(self):
        """Test graceful degradation on partial failures."""
        # This test would verify that the system provides helpful
        # error messages and cleanup on partial failures
        pass


class TestResourceManagement:
    """Test resource management and monitoring."""

    def test_memory_usage_monitoring(self):
        """Test that memory usage stays within reasonable bounds."""
        # This would test memory usage during generation
        pass

    def test_timeout_handling(self):
        """Test that long-running operations timeout appropriately."""
        # This would test operation timeouts
        pass

    def test_cleanup_on_failure(self):
        """Test that resources are cleaned up on failure."""
        # This would test cleanup of temporary files, processes, etc.
        pass


if __name__ == "__main__":
    pytest.main([__file__])

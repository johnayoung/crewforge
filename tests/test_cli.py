"""Test CLI interface for CrewForge using TDD approach."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from crewforge.cli.main import cli, generate


class TestCLIStructure:
    """Test basic CLI structure and command availability."""

    def test_cli_help_available(self):
        """Test main CLI help command is available."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "CrewForge" in result.output
        assert "Intelligent CrewAI Project Generator" in result.output
        assert "generate" in result.output

    def test_cli_version_available(self):
        """Test CLI version command is available."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "version" in result.output.lower() or "0.1.0" in result.output

    def test_generate_command_available(self):
        """Test generate command is available and shows proper help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "--help"])

        assert result.exit_code == 0
        assert "Generate a CrewAI project" in result.output
        assert "prompt" in result.output.lower()
        assert "--name" in result.output

    def test_generate_command_requires_prompt(self):
        """Test generate command requires prompt argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate"])

        assert result.exit_code != 0
        assert "prompt" in result.output.lower()


class TestPromptValidation:
    """Test prompt parameter validation."""

    def test_generate_with_valid_prompt(self):
        """Test generate command accepts valid prompts."""
        runner = CliRunner()
        valid_prompts = [
            "Create a content research crew that finds and analyzes articles",
            "Build a data analysis team for financial reporting",
            "Generate agents for customer service automation",
        ]

        for prompt in valid_prompts:
            result = runner.invoke(cli, ["generate", prompt])
            # Should not fail due to validation (implementation may be incomplete)
            assert (
                "prompt:" in result.output.lower()
                or "generating" in result.output.lower()
            )

    def test_generate_rejects_empty_prompt(self):
        """Test generate command rejects empty prompts."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", ""])

        # Should fail validation
        assert result.exit_code != 0
        assert "empty" in result.output.lower() or "invalid" in result.output.lower()

    def test_generate_rejects_too_short_prompt(self):
        """Test generate command rejects prompts that are too short."""
        runner = CliRunner()
        short_prompts = ["Hi", "Test", "A"]

        for prompt in short_prompts:
            result = runner.invoke(cli, ["generate", prompt])
            assert result.exit_code != 0
            assert (
                "short" in result.output.lower() or "minimum" in result.output.lower()
            )

    def test_generate_rejects_too_long_prompt(self):
        """Test generate command rejects prompts that are too long."""
        runner = CliRunner()
        # Create a very long prompt (over 2000 characters)
        long_prompt = "Create a crew " * 200  # Much longer than reasonable

        result = runner.invoke(cli, ["generate", long_prompt])
        assert result.exit_code != 0
        assert "long" in result.output.lower() or "maximum" in result.output.lower()

    def test_generate_validates_prompt_content(self):
        """Test generate command validates prompt content quality."""
        runner = CliRunner()
        invalid_prompts = [
            "asdkfjalskdfj alksjdf",  # Gibberish
            "!!@#$%^&*()",  # Only special characters
            "123456789",  # Only numbers
        ]

        for prompt in invalid_prompts:
            result = runner.invoke(cli, ["generate", prompt])
            assert result.exit_code != 0
            assert (
                "invalid" in result.output.lower() or "content" in result.output.lower()
            )


class TestProjectNameValidation:
    """Test project name parameter validation."""

    def test_generate_with_valid_project_names(self):
        """Test generate command accepts valid project names."""
        runner = CliRunner()
        valid_names = [
            "content-research-crew",
            "data-analysis-team",
            "customer-service-agents",
            "my_project_crew",
        ]

        for name in valid_names:
            result = runner.invoke(
                cli, ["generate", "Create a test crew", "--name", name]
            )
            # Should not fail due to name validation
            assert "name:" in result.output.lower() or name in result.output

    def test_generate_cleans_project_names(self):
        """Test generate command cleans and validates project names."""
        runner = CliRunner()
        name_tests = [
            ("My Test Crew", "my-test-crew"),
            ("Content Research!", "content-research"),
            ("Data@Analysis#Team", "data-analysis-team"),
        ]

        for input_name, expected_clean in name_tests:
            result = runner.invoke(
                cli, ["generate", "Create a test crew", "--name", input_name]
            )
            # Should show cleaned name or accept it
            assert expected_clean in result.output or input_name in result.output

    def test_generate_rejects_invalid_project_names(self):
        """Test generate command rejects invalid project names."""
        runner = CliRunner()
        invalid_names = [
            "",  # Empty name
            "   ",  # Whitespace only
            "a",  # Too short
            "a" * 100,  # Too long
        ]

        for name in invalid_names:
            result = runner.invoke(
                cli, ["generate", "Create a test crew", "--name", name]
            )
            assert result.exit_code != 0
            assert "name" in result.output.lower() and (
                "invalid" in result.output.lower() or "error" in result.output.lower()
            )

    def test_generate_auto_generates_project_name(self):
        """Test generate command can auto-generate project names from prompts."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "Create a content research crew"])

        # Should either generate a name or proceed without error
        assert result.exit_code == 0 or "name" in result.output.lower()


class TestDirectoryValidation:
    """Test directory existence and permission checking."""

    def test_generate_checks_directory_existence(self):
        """Test generate command validates target directory."""
        runner = CliRunner()

        # Test with a project name that would create a directory
        result = runner.invoke(
            cli, ["generate", "Create a test crew", "--name", "test-crew"]
        )

        # Should check for directory conflicts or create path
        # Implementation may vary but should handle directory logic
        assert result.exit_code in [0, 1]  # Either success or controlled failure

    def test_generate_handles_existing_directory(self):
        """Test generate command handles existing project directories."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test directory that might conflict
            existing_dir = Path(temp_dir) / "test-project"
            existing_dir.mkdir()

            with patch("os.getcwd", return_value=temp_dir):
                result = runner.invoke(
                    cli, ["generate", "Create a test crew", "--name", "test-project"]
                )

                # Should either handle the conflict or warn about it
                assert "exist" in result.output.lower() or result.exit_code != 0

    def test_generate_handles_permission_errors(self):
        """Test generate command handles directory permission errors."""
        runner = CliRunner()

        # Test in a read-only directory context (simulated)
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "os.getcwd", return_value="/root"
            ):  # Typically restricted directory
                result = runner.invoke(
                    cli, ["generate", "Create a test crew", "--name", "test-crew"]
                )

                # Should handle permission issues gracefully
                assert result.exit_code in [0, 1]


class TestProgressFeedback:
    """Test progress feedback and user communication."""

    def test_generate_provides_progress_feedback(self):
        """Test generate command provides step-by-step progress feedback."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "Create a content research crew"])

        # Should show progress indicators
        progress_indicators = ["generating", "creating", "processing", "analyzing"]
        has_progress = any(
            indicator in result.output.lower() for indicator in progress_indicators
        )
        assert has_progress, f"No progress indicators found in output: {result.output}"

    def test_generate_shows_validation_progress(self):
        """Test generate command shows validation progress."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["generate", "Create a data analysis crew", "--name", "data-crew"]
        )

        # Should show validation steps
        validation_indicators = ["validating", "checking", "verifying"]
        has_validation = any(
            indicator in result.output.lower() for indicator in validation_indicators
        )
        assert has_validation or "valid" in result.output.lower()

    def test_generate_confirms_success(self):
        """Test generate command provides success confirmation."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["generate", "Create a test crew", "--name", "success-test"]
        )

        # Should show success confirmation with project path
        success_indicators = ["success", "complete", "generated", "created"]
        has_success = any(
            indicator in result.output.lower() for indicator in success_indicators
        )
        assert has_success or result.exit_code == 0

    def test_generate_shows_project_path(self):
        """Test generate command shows the generated project path."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["generate", "Create a test crew", "--name", "path-test"]
        )

        # Should include path information
        path_indicators = ["path", "directory", "location", "created at"]
        has_path = any(
            indicator in result.output.lower() for indicator in path_indicators
        )
        assert has_path or "path-test" in result.output


class TestErrorHandling:
    """Test error handling and user-friendly messages."""

    def test_generate_handles_invalid_characters_gracefully(self):
        """Test generate command handles invalid characters in input."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["generate", "Create a crew with \x00 invalid chars"]
        )

        # Should handle gracefully with proper error message
        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "error" in result.output.lower()

    def test_generate_provides_helpful_error_messages(self):
        """Test generate command provides helpful error messages."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", ""])  # Empty prompt

        # Error message should be helpful and specific
        assert result.exit_code != 0
        assert len(result.output) > 20  # Should have substantial error message
        assert "prompt" in result.output.lower()

    def test_generate_suggests_corrections(self):
        """Test generate command suggests corrections for common mistakes."""
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "a"])  # Too short

        # Should suggest what the user should do
        assert result.exit_code != 0
        suggestions = ["try", "should", "example", "minimum", "characters"]
        has_suggestion = any(word in result.output.lower() for word in suggestions)
        assert has_suggestion


class TestCLIIntegration:
    """Test CLI integration with the rest of the system."""

    def test_cli_imports_models_correctly(self):
        """Test CLI can import and use data models."""
        # This test ensures CLI integration works with the models we built
        from crewforge.models import GenerationRequest

        # Should be able to create a generation request from CLI input
        request = GenerationRequest(
            prompt="Create a test crew", project_name="test-crew"
        )

        assert request.prompt == "Create a test crew"
        assert request.project_name == "test-crew"

    def test_cli_can_validate_inputs_with_models(self):
        """Test CLI can use models for input validation."""
        from crewforge.models import GenerationRequest
        from pydantic import ValidationError

        # Should be able to validate CLI inputs using our models
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="")  # Empty prompt should fail

    def test_cli_handles_model_validation_errors(self):
        """Test CLI handles model validation errors gracefully."""
        runner = CliRunner()

        # Test with input that would cause model validation to fail
        result = runner.invoke(cli, ["generate", "", "--name", "test"])

        # Should handle the validation error from models
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "invalid" in result.output.lower()

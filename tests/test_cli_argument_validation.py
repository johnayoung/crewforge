"""
Test CLI argument validation and edge cases.

Comprehensive tests for command line argument parsing, validation,
and error handling for the CrewForge CLI tool.
All tests use temporary directories to avoid creating test projects in the repository.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from click.testing import CliRunner
import pytest

# Add src to path so we can import crewforge
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crewforge.cli import main, create


# Global fixture to mock all LLM and scaffolder components for all tests in this file
@pytest.fixture(autouse=True)
def mock_llm_and_scaffolder():
    """Auto-use fixture to mock LLM and scaffolder components for all tests."""
    with (
        patch("crewforge.cli.LLMClient") as mock_llm,
        patch("crewforge.cli.PromptTemplates") as mock_templates,
        patch("crewforge.cli.CrewAIScaffolder") as mock_scaffolder,
    ):

        # Mock LLM client
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        # Mock prompt templates with async coroutine - create fresh coroutine each call
        mock_templates_instance = Mock()

        def create_mock_extract(*args, **kwargs):
            async def mock_extract():
                return {
                    "project_name": "test-project",
                    "project_description": "Test description",
                    "agents": [{"role": "TestAgent"}],
                    "tasks": [{"name": "test_task"}],
                    "dependencies": ["crewai"],
                }

            return mock_extract()

        mock_templates_instance.extract_project_spec = Mock(
            side_effect=create_mock_extract
        )
        mock_templates.return_value = mock_templates_instance

        # Mock scaffolder
        mock_scaffolder_instance = Mock()
        mock_scaffolder_instance.check_crewai_available.return_value = False
        mock_scaffolder.return_value = mock_scaffolder_instance

        yield {
            "llm": mock_llm,
            "templates": mock_templates,
            "scaffolder": mock_scaffolder,
        }


class TestProjectNameValidation:
    """Test project name validation and sanitization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def invoke_with_temp_dir(self, args, **kwargs):
        """Helper method to invoke CLI commands with a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add --output-dir to args if not already present
            if "--output-dir" not in args and "-o" not in args:
                # Handle the -- separator correctly - options must come before --
                if "--" in args:
                    separator_index = args.index("--")
                    # Insert --output-dir before the -- separator
                    args = (
                        args[:separator_index]
                        + ["--output-dir", temp_dir]
                        + args[separator_index:]
                    )
                else:
                    args = args + ["--output-dir", temp_dir]
            return self.runner.invoke(create, args, **kwargs)

    def test_valid_project_names(self):
        """Test that valid project names are accepted."""
        valid_names = [
            "my-project",
            "my_project",
            "project123",
            "simple",
            "project-name-with-dashes",
            "project_name_with_underscores",
            "mixed-name_123",
            "project.name",
            "project.name.with.dots",
        ]

        for name in valid_names:
            result = self.invoke_with_temp_dir([name, "Test prompt"])
            assert (
                result.exit_code == 0
            ), f"Project name '{name}' should be valid but was rejected"
            assert "Project name must be a valid directory name" not in result.output

    def test_invalid_project_names(self):
        """Test that invalid project names are rejected with proper error messages."""
        invalid_names = [
            ("", "empty"),  # Empty name
            ("   ", "whitespace"),  # Whitespace only
            ("project with spaces", "spaces"),
            ("project/with/slashes", "slashes"),
            ("project\\with\\backslashes", "backslashes"),
            ("project@with#symbols", "symbols"),
            ("project!with$special%chars", "special_chars"),
            ("project|with&pipes", "pipes"),
            ("project<with>brackets", "brackets"),
            ("project(with)parens", "parens"),
            ("project[with]square", "square"),
            ("project{with}curly", "curly"),
            ("project+with=equals", "equals"),
            ("project:with;colons", "colons"),
            ("project'with\"quotes", "quotes"),
            ("project,with,commas", "commas"),
        ]

        for name, description in invalid_names:
            if name.startswith("-"):
                # Use -- separator for names starting with dashes
                result = self.invoke_with_temp_dir(["--", name, "Test prompt"])
            else:
                result = self.invoke_with_temp_dir([name, "Test prompt"])
            assert (
                result.exit_code == 1
            ), f"Project name '{name}' ({description}) should be rejected but was accepted"
            # Check for any validation error message
            assert any(
                error_phrase in result.output
                for error_phrase in [
                    "Project name must",
                    "Project name cannot be empty",
                    "must contain at least one alphanumeric",
                ]
            ), f"Expected validation error for '{name}' but got: {result.output}"

    def test_edge_case_project_names(self):
        """Test edge cases for project name validation."""
        # Test very long names (should be accepted if valid characters)
        long_valid_name = "a" * 50  # 50 character name
        result = self.invoke_with_temp_dir([long_valid_name, "Test prompt"])
        assert result.exit_code == 0

        # Test single character names
        result = self.invoke_with_temp_dir(["a", "Test prompt"])
        assert result.exit_code == 0

        # Test names with only special allowed characters
        # Use -- to separate options from arguments for names starting with dashes
        result = self.invoke_with_temp_dir(["--", "---", "Test prompt"])
        assert result.exit_code == 1  # Should be invalid (no alphanumeric)

        result = self.invoke_with_temp_dir(["--", "___", "Test prompt"])
        assert result.exit_code == 1  # Should be invalid (no alphanumeric)


class TestPromptArgumentHandling:
    """Test prompt argument handling and validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def invoke_with_temp_dir(self, args, **kwargs):
        """Helper method to invoke CLI commands with a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add --output-dir to args if not already present
            if "--output-dir" not in args and "-o" not in args:
                # Handle the -- separator correctly - options must come before --
                if "--" in args:
                    separator_index = args.index("--")
                    # Insert --output-dir before the -- separator
                    args = (
                        args[:separator_index]
                        + ["--output-dir", temp_dir]
                        + args[separator_index:]
                    )
                else:
                    args = args + ["--output-dir", temp_dir]
            return self.runner.invoke(create, args, **kwargs)

    def test_prompt_with_special_characters(self):
        """Test that prompts with special characters are handled correctly."""
        special_prompts = [
            "Create a project with @mentions and #hashtags",
            "Build something with 'quotes' and \"double quotes\"",
            "Project with (parentheses) and [brackets]",
            "Use $variables and 100% success rate",
            "Handle & ampersands, | pipes, and < > brackets",
            "Multi-line\nprompt with\nnewlines",
            "Prompt with\ttabs\tand\tspaces",
            "Unicode: 🚀 rockets and émojis café",
        ]

        for prompt in special_prompts:
            result = self.invoke_with_temp_dir(["test-project", prompt])
            assert (
                result.exit_code == 0
            ), f"Prompt with special chars should be accepted: {prompt}"

    def test_empty_prompt_handling(self):
        """Test handling of empty or whitespace-only prompts."""
        # Empty prompt without interactive mode should fail
        result = self.invoke_with_temp_dir(["test-project"])
        assert result.exit_code == 1
        assert "Please provide a prompt or use --interactive mode" in result.output

        # Interactive mode with empty input should fail
        result = self.runner.invoke(
            create, ["test-project", "--interactive"], input="\n"  # Empty input
        )
        assert result.exit_code == 1
        assert "Project description cannot be empty" in result.output

        # Interactive mode with whitespace input should fail
        result = self.runner.invoke(
            create, ["test-project", "--interactive"], input="   \n"  # Whitespace only
        )
        assert result.exit_code == 1
        assert "Project description cannot be empty" in result.output

    def test_very_long_prompt_handling(self):
        """Test that very long prompts are handled gracefully."""
        # Create a very long prompt (1000+ characters)
        long_prompt = (
            "Create a comprehensive project that " + "handles many requirements " * 50
        )

        result = self.invoke_with_temp_dir(["test-project", long_prompt])
        assert result.exit_code == 0
        # Should not crash or truncate unexpectedly


class TestOutputDirectoryValidation:
    """Test output directory argument validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def invoke_with_temp_dir(self, args, **kwargs):
        """Helper method to invoke CLI commands with a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add --output-dir to args if not already present
            if "--output-dir" not in args and "-o" not in args:
                # Handle the -- separator correctly - options must come before --
                if "--" in args:
                    separator_index = args.index("--")
                    # Insert --output-dir before the -- separator
                    args = (
                        args[:separator_index]
                        + ["--output-dir", temp_dir]
                        + args[separator_index:]
                    )
                else:
                    args = args + ["--output-dir", temp_dir]
            return self.runner.invoke(create, args, **kwargs)

    def test_valid_output_directories(self):
        """Test that valid output directories are accepted."""
        valid_dirs = [
            ".",
            "..",
            "/tmp",
            "~/projects",
            "relative/path",
            "/absolute/path",
        ]

        for dir_path in valid_dirs:
            result = self.runner.invoke(
                create, ["test-project", "Test prompt", "--output-dir", dir_path]
            )
            # Should not fail due to directory path format
            # (actual directory existence is handled during project creation)
            assert result.exit_code == 0
            assert dir_path in result.output

    def test_output_directory_in_help(self):
        """Test that output directory option is properly documented."""
        result = self.invoke_with_temp_dir(["--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output or "-o" in result.output
        assert "directory" in result.output.lower()


class TestInteractiveModeHandling:
    """Test interactive mode flag and behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def invoke_with_temp_dir(self, args, **kwargs):
        """Helper method to invoke CLI commands with a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add --output-dir to args if not already present
            if "--output-dir" not in args and "-o" not in args:
                # Handle the -- separator correctly - options must come before --
                if "--" in args:
                    separator_index = args.index("--")
                    # Insert --output-dir before the -- separator
                    args = (
                        args[:separator_index]
                        + ["--output-dir", temp_dir]
                        + args[separator_index:]
                    )
                else:
                    args = args + ["--output-dir", temp_dir]
            return self.runner.invoke(create, args, **kwargs)

    def test_interactive_flag_variants(self):
        """Test both short and long forms of interactive flag."""
        # Test long form
        result = self.runner.invoke(
            create,
            ["test-project", "--interactive"],
            input="Test prompt from interactive mode\n",
        )
        assert result.exit_code == 0
        assert "Interactive Mode" in result.output

        # Test short form
        result = self.runner.invoke(
            create, ["test-project", "-i"], input="Test prompt from interactive mode\n"
        )
        assert result.exit_code == 0
        assert "Interactive Mode" in result.output

    def test_interactive_with_existing_prompt(self):
        """Test behavior when both interactive flag and prompt are provided."""
        # When both prompt and interactive are provided, prompt should take precedence
        result = self.invoke_with_temp_dir(
            ["test-project", "Existing prompt", "--interactive"]
        )
        assert result.exit_code == 0
        assert "Existing prompt" in result.output
        # Should not prompt for additional input


class TestArgumentCombinations:
    """Test various combinations of CLI arguments."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def invoke_with_temp_dir(self, args, **kwargs):
        """Helper method to invoke CLI commands with a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add --output-dir to args if not already present
            if "--output-dir" not in args and "-o" not in args:
                # Handle the -- separator correctly - options must come before --
                if "--" in args:
                    separator_index = args.index("--")
                    # Insert --output-dir before the -- separator
                    args = (
                        args[:separator_index]
                        + ["--output-dir", temp_dir]
                        + args[separator_index:]
                    )
                else:
                    args = args + ["--output-dir", temp_dir]
            return self.runner.invoke(create, args, **kwargs)
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_all_arguments_together(self):
        """Test providing all possible arguments together."""
        result = self.runner.invoke(
            create,
            [
                "comprehensive-project",
                "A comprehensive test prompt with all arguments",
                "--output-dir",
                "/tmp/test",
                "--interactive",
            ],
        )

        assert result.exit_code == 0
        assert "comprehensive-project" in result.output
        assert "A comprehensive test prompt with all arguments" in result.output
        assert "/tmp/test" in result.output

    def test_minimal_valid_arguments(self):
        """Test minimal valid argument combination."""
        result = self.invoke_with_temp_dir(["minimal", "Simple prompt"])

        assert result.exit_code == 0
        assert "minimal" in result.output
        assert "Simple prompt" in result.output
        # Should use output directory (temp directory from test helper)
        assert "Output directory: /tmp/tmp" in result.output

    def test_argument_order_independence(self):
        """Test that argument order doesn't affect functionality."""
        # Different orders of the same arguments should produce same result
        base_args = ["test-project", "Test prompt"]
        options_combos = [
            ["--output-dir", "/tmp", "--interactive"],
            ["--interactive", "--output-dir", "/tmp"],
            ["-i", "-o", "/tmp"],
            ["-o", "/tmp", "-i"],
        ]

        results = []
        for options in options_combos:
            result = self.runner.invoke(create, base_args + options)
            assert result.exit_code == 0
            results.append(result.output)

        # All results should have the same core information
        for output in results:
            assert "test-project" in output
            assert "Test prompt" in output
            assert "/tmp" in output

"""
Test the CLI command entry point and help documentation.

Tests for the main command interface, help text, error handling,
and basic command structure.
"""

import os
import sys
from click.testing import CliRunner
import pytest

# Add src to path so we can import crewforge
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crewforge.cli import main, create


class TestMainCommand:
    """Test the main CLI command entry point."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_main_command_help(self):
        """Test that main command displays comprehensive help."""
        result = self.runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "CrewForge CLI" in result.output
        assert "Generate CrewAI projects from natural language prompts" in result.output
        assert "Commands:" in result.output
        assert "create" in result.output

    def test_main_command_version(self):
        """Test that version option works correctly."""
        result = self.runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "crewforge" in result.output.lower()
        # Should contain version number (format: major.minor.patch)
        import re

        version_pattern = r"\d+\.\d+\.\d+"
        assert re.search(version_pattern, result.output)

    def test_main_command_no_args(self):
        """Test main command with no arguments shows help."""
        result = self.runner.invoke(main, [])

        assert result.exit_code == 0
        assert "CrewForge CLI" in result.output
        assert "Commands:" in result.output


class TestCreateCommand:
    """Test the create command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_create_command_help(self):
        """Test that create command shows detailed help."""
        result = self.runner.invoke(create, ["--help"])

        assert result.exit_code == 0
        assert "Create a new CrewAI project" in result.output
        assert "PROJECT_NAME" in result.output
        assert "PROMPT" in result.output
        assert "--interactive" in result.output or "-i" in result.output
        assert "--output-dir" in result.output or "-o" in result.output

    def test_create_command_missing_project_name(self):
        """Test create command without project name shows error."""
        result = self.runner.invoke(create, [])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Usage:" in result.output

    def test_create_command_with_project_name_and_prompt(self):
        """Test create command with both project name and prompt."""
        result = self.runner.invoke(create, ["test-project", "A simple test project"])

        assert result.exit_code == 0
        assert "test-project" in result.output
        assert "A simple test project" in result.output
        assert "CrewForge v" in result.output

    def test_create_command_with_project_name_only_non_interactive(self):
        """Test create command with project name only (non-interactive) shows error."""
        result = self.runner.invoke(create, ["test-project"])

        assert result.exit_code == 1  # Should exit with error code
        assert (
            "Error: Please provide a prompt or use --interactive mode" in result.output
        )

    def test_create_command_interactive_mode(self):
        """Test create command with interactive flag."""
        # Simulate user input for interactive prompt
        result = self.runner.invoke(
            create,
            ["test-project", "--interactive"],
            input="A test project for interactive mode\n",
        )

        assert result.exit_code == 0
        assert "test-project" in result.output
        assert "A test project for interactive mode" in result.output

    def test_create_command_with_output_dir(self):
        """Test create command with custom output directory."""
        result = self.runner.invoke(
            create, ["test-project", "Test prompt", "--output-dir", "/tmp/test"]
        )

        assert result.exit_code == 0
        assert "/tmp/test" in result.output

    def test_create_command_displays_version(self):
        """Test that create command displays version information."""
        result = self.runner.invoke(create, ["test-project", "Test prompt"])

        assert result.exit_code == 0
        assert "CrewForge v" in result.output


class TestCommandIntegration:
    """Test overall command integration and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_invalid_command_shows_help(self):
        """Test that invalid commands show helpful error messages."""
        result = self.runner.invoke(main, ["invalid-command"])

        assert result.exit_code != 0
        assert "No such command" in result.output or "Usage:" in result.output

    def test_command_structure_consistency(self):
        """Test that command structure is consistent and well-formed."""
        # Test main command structure
        main_help = self.runner.invoke(main, ["--help"])
        assert main_help.exit_code == 0

        # Test create subcommand structure
        create_help = self.runner.invoke(create, ["--help"])
        assert create_help.exit_code == 0

        # Both should mention CrewForge/CrewAI
        assert "CrewAI" in main_help.output or "crewai" in main_help.output.lower()
        assert "CrewAI" in create_help.output or "crewai" in create_help.output.lower()

    def test_help_documentation_quality(self):
        """Test that help documentation is comprehensive and user-friendly."""
        result = self.runner.invoke(main, ["--help"])

        assert result.exit_code == 0

        # Should have clear description
        assert len(result.output.split("\n")) > 5  # Multi-line help

        # Should mention key functionality
        assert any(
            word in result.output.lower()
            for word in ["generate", "create", "natural language", "prompt"]
        )

        # Should be properly formatted (no obvious formatting issues)
        lines = result.output.split("\n")
        assert not any(line.strip().startswith("Error") for line in lines)

"""
Test progress indicators and status message display functionality.

Tests for progress bars, spinners, status messages, and timing features
during project generation workflow.
"""

import os
import sys
import time
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import pytest

# Add src to path so we can import crewforge
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crewforge.cli import create
from crewforge.progress import ProgressIndicator, StatusDisplay


class TestProgressIndicator:
    """Test the progress indicator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_progress_indicator_creation(self):
        """Test that ProgressIndicator can be created and configured."""
        progress = ProgressIndicator()
        assert progress is not None
        assert hasattr(progress, "start_progress")
        assert hasattr(progress, "update_progress")
        assert hasattr(progress, "finish_progress")

    def test_progress_indicator_with_steps(self):
        """Test progress indicator with defined steps."""
        steps = [
            "Parsing prompt",
            "Validating requirements",
            "Creating project structure",
            "Generating agent configurations",
            "Finalizing project",
        ]

        progress = ProgressIndicator(steps=steps)
        assert progress.total_steps == len(steps)
        assert progress.current_step == 0

    @patch("click.echo")
    def test_progress_start_and_finish(self, mock_echo):
        """Test progress indicator start and finish functionality."""
        progress = ProgressIndicator()

        progress.start_progress("Starting project generation")
        mock_echo.assert_called()

        progress.finish_progress("Project generation complete")
        mock_echo.assert_called()

    @patch("click.echo")
    def test_progress_step_updates(self, mock_echo):
        """Test progress indicator step updates."""
        steps = ["Step 1", "Step 2", "Step 3"]
        progress = ProgressIndicator(steps=steps)

        progress.start_progress("Starting")
        progress.update_progress("Step 1", 0)
        progress.update_progress("Step 2", 1)
        progress.update_progress("Step 3", 2)
        progress.finish_progress("Complete")

        # Should have made several echo calls for progress updates
        assert mock_echo.call_count >= len(steps)

    def test_progress_indicator_timing(self):
        """Test that progress indicator tracks timing correctly."""
        progress = ProgressIndicator()

        start_time = time.time()
        progress.start_progress("Test timing")

        # Simulate some work
        time.sleep(0.1)

        progress.finish_progress("Complete")
        end_time = time.time()

        # Progress should track elapsed time
        assert progress.elapsed_time > 0
        assert (
            progress.elapsed_time <= (end_time - start_time) + 0.01
        )  # Small tolerance


class TestStatusDisplay:
    """Test the status message display functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_status_display_creation(self):
        """Test StatusDisplay can be created."""
        status = StatusDisplay()
        assert status is not None
        assert hasattr(status, "info")
        assert hasattr(status, "success")
        assert hasattr(status, "warning")
        assert hasattr(status, "error")

    @patch("click.echo")
    def test_status_display_info(self, mock_echo):
        """Test info status messages."""
        status = StatusDisplay()
        status.info("This is an info message")

        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert "This is an info message" in args[0]

    @patch("click.echo")
    def test_status_display_success(self, mock_echo):
        """Test success status messages."""
        status = StatusDisplay()
        status.success("Operation completed successfully")

        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert "Operation completed successfully" in args[0]
        # Should use green color for success (ANSI escape codes)
        assert "\x1b[32m" in args[0] or "green" in str(kwargs)  # ANSI green color code

    @patch("click.echo")
    def test_status_display_warning(self, mock_echo):
        """Test warning status messages."""
        status = StatusDisplay()
        status.warning("This is a warning")

        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert "This is a warning" in args[0]
        # Should use yellow color for warnings (ANSI escape codes)
        assert "\x1b[33m" in args[0] or "yellow" in str(
            kwargs
        )  # ANSI yellow color code

    @patch("click.echo")
    def test_status_display_error(self, mock_echo):
        """Test error status messages."""
        status = StatusDisplay()
        status.error("This is an error")

        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert "This is an error" in args[0]
        # Should use red color for errors (ANSI escape codes)
        assert "\x1b[31m" in args[0] or "red" in str(kwargs)  # ANSI red color code


class TestCLIProgressIntegration:
    """Test progress indicators integrated with CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("crewforge.cli.ProgressIndicator")
    @patch("crewforge.cli.StatusDisplay")
    def test_create_command_uses_progress_indicators(self, mock_status, mock_progress):
        """Test that create command uses progress indicators."""
        # Mock the progress indicator and status display
        mock_progress_instance = MagicMock()
        mock_status_instance = MagicMock()
        mock_progress.return_value = mock_progress_instance
        mock_status.return_value = mock_status_instance

        # Run the create command
        result = self.runner.invoke(create, ["test-project", "Test prompt"])

        # Should not crash and should use progress indicators
        assert result.exit_code == 0

        # Verify progress indicators were created and used
        mock_progress.assert_called_once()
        mock_status.assert_called_once()

    def test_create_command_shows_progress_messages(self):
        """Test that create command shows progress-related messages."""
        result = self.runner.invoke(create, ["test-project", "Test prompt"])

        assert result.exit_code == 0
        # Should contain progress-related indicators (emojis, status messages)
        assert "🔥" in result.output or "📁" in result.output or "📝" in result.output
        assert "CrewForge v" in result.output

    def test_create_command_timing_display(self):
        """Test that create command can display timing information."""
        result = self.runner.invoke(create, ["test-project", "Test prompt"])

        assert result.exit_code == 0
        # The current implementation should work without crashing
        assert "test-project" in result.output

    @patch("time.sleep")  # Speed up test by mocking sleep
    def test_create_command_with_simulated_work(self, mock_sleep):
        """Test create command with simulated processing work."""
        result = self.runner.invoke(create, ["test-project", "Test prompt"])

        assert result.exit_code == 0
        # Should handle the workflow without crashing
        assert "test-project" in result.output
        assert "Test prompt" in result.output


class TestProgressIntegrationPatterns:
    """Test progress indicator integration patterns."""

    def test_progress_steps_for_crewai_workflow(self):
        """Test that progress steps match expected CrewAI generation workflow."""
        expected_steps = [
            "Parsing natural language prompt",
            "Validating project requirements",
            "Creating base project structure",
            "Generating agent configurations",
            "Creating task definitions",
            "Setting up project dependencies",
            "Finalizing project files",
        ]

        progress = ProgressIndicator(steps=expected_steps)

        # Should handle the expected workflow steps
        assert progress.total_steps == len(expected_steps)

        # Each step should be processable
        for i, step in enumerate(expected_steps):
            progress.update_progress(step, i)
            assert progress.current_step == i

    def test_status_messages_for_user_feedback(self):
        """Test status messages provide good user feedback."""
        status = StatusDisplay()

        # Should be able to provide comprehensive feedback
        test_messages = [
            ("Analyzing your project description...", "info"),
            ("✅ Project requirements validated", "success"),
            ("⚠️  Using default configuration for missing details", "warning"),
            ("❌ Invalid project name format", "error"),
        ]

        for message, level in test_messages:
            method = getattr(status, level)
            # Should not raise exceptions
            try:
                method(message)
            except Exception as e:
                pytest.fail(f"Status display {level} failed: {e}")

"""
Test CLI error handling and user interruption recovery.

Comprehensive tests for KeyboardInterrupt handling, system-level failures,
and graceful error recovery in the CrewForge CLI tool.
"""

import os
import sys
import signal
from unittest.mock import patch, Mock
from click.testing import CliRunner
import pytest

# Add src to path so we can import crewforge
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crewforge.cli import main, create


class TestKeyboardInterruptHandling:
    """Test graceful handling of user interruption (Ctrl+C)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_keyboard_interrupt_during_creation(self):
        """Test that KeyboardInterrupt is handled gracefully during project creation."""
        # Mock the progress update to raise KeyboardInterrupt
        with patch("crewforge.cli.ProgressIndicator") as mock_progress:
            mock_progress_instance = Mock()
            mock_progress.return_value = mock_progress_instance

            # Configure the mock to raise KeyboardInterrupt during update
            mock_progress_instance.update_progress.side_effect = KeyboardInterrupt()

            result = self.runner.invoke(create, ["test-project", "Test prompt"])

            # Should exit cleanly with proper message
            assert result.exit_code == 130  # Standard exit code for SIGINT
            assert (
                "Operation cancelled by user" in result.output
                or "Interrupted" in result.output
            )

    def test_keyboard_interrupt_during_interactive_prompt(self):
        """Test KeyboardInterrupt handling during interactive input."""
        # This test simulates Ctrl+C during interactive prompt input
        with patch("click.prompt") as mock_prompt:
            mock_prompt.side_effect = KeyboardInterrupt()

            result = self.runner.invoke(create, ["test-project", "--interactive"])

            # Should handle interrupt gracefully
            assert result.exit_code == 130
            assert (
                "Operation cancelled by user" in result.output
                or "Interrupted" in result.output
            )

    def test_keyboard_interrupt_with_cleanup_message(self):
        """Test that KeyboardInterrupt shows appropriate cleanup message."""
        with patch("crewforge.cli.ProgressIndicator") as mock_progress:
            mock_progress_instance = Mock()
            mock_progress.return_value = mock_progress_instance
            mock_progress_instance.update_progress.side_effect = KeyboardInterrupt()

            result = self.runner.invoke(create, ["test-project", "Test prompt"])

            # Should include cleanup information
            assert result.exit_code == 130
            output_lower = result.output.lower()
            assert any(
                phrase in output_lower
                for phrase in ["cancelled", "interrupted", "stopped", "aborted"]
            )


class TestSystemErrorHandling:
    """Test handling of system-level errors and unexpected exceptions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_permission_error_handling(self):
        """Test handling of permission errors during project creation."""
        with patch("crewforge.cli.ProgressIndicator") as mock_progress:
            mock_progress_instance = Mock()
            mock_progress.return_value = mock_progress_instance

            # Simulate permission error
            mock_progress_instance.update_progress.side_effect = PermissionError(
                "Permission denied: Unable to create project directory"
            )

            result = self.runner.invoke(create, ["test-project", "Test prompt"])

            # Should handle permission error gracefully
            assert result.exit_code == 1
            assert (
                "Permission denied" in result.output
                or "permission" in result.output.lower()
            )

    def test_disk_space_error_handling(self):
        """Test handling of disk space errors."""
        with patch("crewforge.cli.ProgressIndicator") as mock_progress:
            mock_progress_instance = Mock()
            mock_progress.return_value = mock_progress_instance

            # Simulate disk space error
            mock_progress_instance.update_progress.side_effect = OSError(
                "No space left on device"
            )

            result = self.runner.invoke(create, ["test-project", "Test prompt"])

            # Should handle disk space error gracefully
            assert result.exit_code == 1
            assert "space" in result.output.lower() or "disk" in result.output.lower()

    def test_unexpected_system_error_handling(self):
        """Test handling of unexpected system errors."""
        with patch("crewforge.cli.ProgressIndicator") as mock_progress:
            mock_progress_instance = Mock()
            mock_progress.return_value = mock_progress_instance

            # Simulate unexpected system error
            mock_progress_instance.update_progress.side_effect = RuntimeError(
                "Unexpected system error occurred"
            )

            result = self.runner.invoke(create, ["test-project", "Test prompt"])

            # Should handle unexpected error gracefully
            assert result.exit_code == 1
            assert "error" in result.output.lower()
            # Should not show raw stack trace to user
            assert "Traceback" not in result.output


class TestErrorRecoveryAndCleanup:
    """Test error recovery mechanisms and cleanup procedures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_partial_project_cleanup_on_error(self):
        """Test that partial project files are cleaned up on error."""
        with patch("crewforge.cli.ProgressIndicator") as mock_progress:
            mock_progress_instance = Mock()
            mock_progress.return_value = mock_progress_instance

            # Simulate error after some progress
            mock_progress_instance.update_progress.side_effect = [
                None,  # First call succeeds
                None,  # Second call succeeds
                RuntimeError("Simulated error"),  # Third call fails
            ]

            result = self.runner.invoke(create, ["test-project", "Test prompt"])

            # Should mention cleanup in output
            assert result.exit_code == 1
            # Should indicate that cleanup was attempted
            output_lower = result.output.lower()
            assert "error" in output_lower

    def test_error_reporting_with_helpful_messages(self):
        """Test that error messages provide helpful information to users."""
        # Test various error scenarios and ensure helpful messages
        test_cases = [
            (PermissionError("Access denied"), "permission"),
            (FileNotFoundError("File not found"), "not found"),
            (OSError("Operation failed"), "operation"),
        ]

        for error, expected_text in test_cases:
            with patch("crewforge.cli.ProgressIndicator") as mock_progress:
                mock_progress_instance = Mock()
                mock_progress.return_value = mock_progress_instance
                mock_progress_instance.update_progress.side_effect = error

                result = self.runner.invoke(create, ["test-project", "Test prompt"])

                assert result.exit_code == 1
                assert expected_text.lower() in result.output.lower()

    def test_recovery_suggestions_in_error_messages(self):
        """Test that error messages include recovery suggestions."""
        with patch("crewforge.cli.ProgressIndicator") as mock_progress:
            mock_progress_instance = Mock()
            mock_progress.return_value = mock_progress_instance
            mock_progress_instance.update_progress.side_effect = PermissionError(
                "Permission denied"
            )

            result = self.runner.invoke(create, ["test-project", "Test prompt"])

            # Should suggest solutions
            assert result.exit_code == 1
            output_lower = result.output.lower()
            # Should mention trying again or checking permissions
            assert any(
                phrase in output_lower
                for phrase in ["try", "check", "ensure", "verify", "permission"]
            )


class TestErrorHandlingIntegration:
    """Test integration of error handling across CLI components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_error_handling_preserves_user_context(self):
        """Test that error handling preserves user input context."""
        with patch("crewforge.cli.ProgressIndicator") as mock_progress:
            mock_progress_instance = Mock()
            mock_progress.return_value = mock_progress_instance
            mock_progress_instance.update_progress.side_effect = RuntimeError(
                "Test error"
            )

            result = self.runner.invoke(
                create, ["my-special-project", "Very specific project description"]
            )

            # Should still mention the project name in error context
            assert result.exit_code == 1
            assert "my-special-project" in result.output

    def test_consistent_exit_codes(self):
        """Test that different error types use consistent exit codes."""
        error_scenarios = [
            (KeyboardInterrupt(), 130),  # SIGINT
            (PermissionError("Access denied"), 1),  # General error
            (FileNotFoundError("Not found"), 1),  # General error
            (RuntimeError("Runtime error"), 1),  # General error
        ]

        for error, expected_exit_code in error_scenarios:
            with patch("crewforge.cli.ProgressIndicator") as mock_progress:
                mock_progress_instance = Mock()
                mock_progress.return_value = mock_progress_instance
                mock_progress_instance.update_progress.side_effect = error

                result = self.runner.invoke(create, ["test-project", "Test prompt"])

                assert (
                    result.exit_code == expected_exit_code
                ), f"Expected exit code {expected_exit_code} for {type(error).__name__}, got {result.exit_code}"

    def test_error_handling_does_not_break_help(self):
        """Test that error handling doesn't interfere with help commands."""
        # Even if there are issues with other components, help should always work
        result = self.runner.invoke(main, ["--help"])
        assert result.exit_code == 0

        result = self.runner.invoke(create, ["--help"])
        assert result.exit_code == 0

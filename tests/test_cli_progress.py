"""Tests for CLI progress tracking integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from crewforge.cli.main import (
    generate,
    create_progress_callback,
    create_streaming_callbacks,
)
from crewforge.core.progress import (
    ProgressTracker,
    ProgressStep,
    ProgressEvent,
    ProgressStatus,
    StreamingCallbacks,
)


class TestCLIProgressIntegration:
    """Test progress tracking integration in CLI commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_scaffolder(self):
        """Create mock ProjectScaffolder."""
        scaffolder = Mock()
        scaffolder.generate_project.return_value = "/test/path"
        return scaffolder

    def test_create_progress_callback(self):
        """Test progress callback creation and functionality."""
        callback = create_progress_callback()

        # Test in-progress event
        event = ProgressEvent(
            step_id="test_step",
            description="Test step description",
            status=ProgressStatus.IN_PROGRESS,
            progress_percentage=50.0,
        )

        # Should not raise exception
        callback(event)

    def test_create_streaming_callbacks(self):
        """Test streaming callback creation and functionality."""
        callbacks = create_streaming_callbacks()

        assert isinstance(callbacks, StreamingCallbacks)

        # Test token handling (should not raise exception)
        callbacks.handle_token("test token")

        # Test completion handling (should not raise exception)
        callbacks.handle_completion("test response")

    @patch("crewforge.cli.main.ProjectScaffolder")
    @patch("crewforge.cli.main.os.getenv")
    def test_generate_command_with_progress(
        self, mock_getenv, mock_scaffolder_class, cli_runner
    ):
        """Test generate command creates and uses progress tracker."""
        # Setup mocks
        mock_getenv.return_value = "test-api-key"
        mock_scaffolder = Mock()
        mock_scaffolder.generate_project.return_value = "/test/project/path"
        mock_scaffolder_class.return_value = mock_scaffolder

        # Run command
        result = cli_runner.invoke(
            generate, ["--name", "test-crew", "Create a test crew"]
        )

        # Check success
        assert result.exit_code == 0

        # Verify scaffolder was called with progress tracker
        mock_scaffolder.generate_project.assert_called_once()
        args, kwargs = mock_scaffolder.generate_project.call_args

        assert "progress_tracker" in kwargs
        assert "streaming_callbacks" in kwargs
        assert isinstance(kwargs["progress_tracker"], ProgressTracker)
        assert isinstance(kwargs["streaming_callbacks"], StreamingCallbacks)

    @patch("crewforge.cli.main.ProjectScaffolder")
    @patch("crewforge.cli.main.os.getenv")
    def test_generate_command_progress_output(
        self, mock_getenv, mock_scaffolder_class, cli_runner
    ):
        """Test that progress tracking produces expected output."""
        # Setup mocks
        mock_getenv.return_value = "test-api-key"
        mock_scaffolder = Mock()
        mock_scaffolder.generate_project.return_value = "/test/project/path"
        mock_scaffolder_class.return_value = mock_scaffolder

        # Run command
        result = cli_runner.invoke(
            generate, ["--name", "test-crew", "Create a test crew"]
        )

        # Check for progress-related output
        assert "Generation pipeline started" in result.output
        assert "real-time progress tracking" in result.output

    def test_progress_callback_handles_all_status_types(self):
        """Test that progress callback handles all status types correctly."""
        callback = create_progress_callback()

        # Test different event types
        events = [
            ProgressEvent("step1", "Step 1", ProgressStatus.IN_PROGRESS, 25.0),
            ProgressEvent("step1", "Step 1", ProgressStatus.COMPLETED, 50.0),
            ProgressEvent(
                "step2", "Step 2", ProgressStatus.FAILED, 50.0, "Error message"
            ),
        ]

        # Should handle all event types without exceptions
        for event in events:
            callback(event)

    def test_streaming_callbacks_token_handling(self):
        """Test streaming callbacks token handling."""
        callbacks = create_streaming_callbacks()

        # Test multiple tokens
        tokens = ["Hello", " ", "world", "!"]
        for token in tokens:
            callbacks.handle_token(token)

        # Test completion
        callbacks.handle_completion("Hello world!")

    @patch("crewforge.cli.main.click.echo")
    def test_progress_callback_output_format(self, mock_echo):
        """Test that progress callback outputs in expected format."""
        callback = create_progress_callback()

        # Test in-progress event
        event = ProgressEvent(
            step_id="analyze_prompt",
            description="Analyzing prompt for requirements",
            status=ProgressStatus.IN_PROGRESS,
            progress_percentage=16.7,
        )

        callback(event)

        # Check output format
        mock_echo.assert_called_once()
        output = mock_echo.call_args[0][0]
        assert "⏳ Step analyze_prompt:" in output
        assert "Analyzing prompt for requirements" in output
        assert "16.7% complete" in output

    @patch("crewforge.cli.main.click.echo")
    def test_progress_callback_completion_format(self, mock_echo):
        """Test progress callback completion event format."""
        callback = create_progress_callback()

        event = ProgressEvent(
            step_id="generate_agents",
            description="Generating agent configurations",
            status=ProgressStatus.COMPLETED,
            progress_percentage=33.3,
        )

        callback(event)

        # Check output format
        mock_echo.assert_called_once()
        output = mock_echo.call_args[0][0]
        assert "✅" in output
        assert "Generating agent configurations" in output
        assert "Complete" in output
        assert "33.3%" in output

    @patch("crewforge.cli.main.click.echo")
    def test_progress_callback_failure_format(self, mock_echo):
        """Test progress callback failure event format."""
        callback = create_progress_callback()

        event = ProgressEvent(
            step_id="create_scaffold",
            description="Creating CrewAI project structure",
            status=ProgressStatus.FAILED,
            progress_percentage=66.7,
            error_message="CrewAI command failed",
        )

        callback(event)

        # Check output format
        mock_echo.assert_called_once()
        output = mock_echo.call_args[0][0]
        assert "❌" in output
        assert "Creating CrewAI project structure" in output
        assert "Failed" in output
        assert "CrewAI command failed" in output

    def test_progress_tracker_step_definitions_in_cli(self):
        """Test that CLI creates progress tracker with correct step definitions."""
        from crewforge.cli.main import generate

        # The expected steps that should be created in the CLI
        expected_steps = [
            "analyze_prompt",
            "generate_agents",
            "generate_tasks",
            "select_tools",
            "create_scaffold",
            "populate_files",
        ]

        # This test verifies the step IDs match what the ProjectScaffolder expects
        # by checking the hardcoded step definitions in the CLI
        import inspect

        source = inspect.getsource(generate)

        for step_id in expected_steps:
            assert step_id in source, f"Expected step '{step_id}' not found in CLI"


class TestProgressEventHandling:
    """Test progress event handling and display."""

    def test_progress_event_creation(self):
        """Test ProgressEvent creation with all parameters."""
        event = ProgressEvent(
            step_id="test_step",
            description="Test description",
            status=ProgressStatus.IN_PROGRESS,
            progress_percentage=42.5,
            error_message="Test error",
        )

        assert event.step_id == "test_step"
        assert event.description == "Test description"
        assert event.status == ProgressStatus.IN_PROGRESS
        assert event.progress_percentage == 42.5
        assert event.error_message == "Test error"

    def test_progress_event_without_error_message(self):
        """Test ProgressEvent creation without error message."""
        event = ProgressEvent(
            step_id="test_step",
            description="Test description",
            status=ProgressStatus.COMPLETED,
            progress_percentage=100.0,
        )

        assert event.error_message is None


class TestStreamingIntegration:
    """Test streaming callback integration."""

    def test_streaming_callbacks_initialization(self):
        """Test StreamingCallbacks can be initialized with None values."""
        callbacks = StreamingCallbacks()

        assert callbacks.on_token is None
        assert callbacks.on_completion is None

    def test_streaming_callbacks_with_handlers(self):
        """Test StreamingCallbacks with actual handler functions."""
        tokens = []
        completions = []

        def token_handler(token: str):
            tokens.append(token)

        def completion_handler(response: str):
            completions.append(response)

        callbacks = StreamingCallbacks(
            on_token=token_handler, on_completion=completion_handler
        )

        callbacks.handle_token("test")
        callbacks.handle_completion("done")

        assert tokens == ["test"]
        assert completions == ["done"]

    def test_streaming_callbacks_error_resilience(self):
        """Test that streaming callbacks handle handler errors gracefully."""

        def failing_token_handler(token: str):
            raise Exception("Token handler error")

        def failing_completion_handler(response: str):
            raise Exception("Completion handler error")

        callbacks = StreamingCallbacks(
            on_token=failing_token_handler, on_completion=failing_completion_handler
        )

        # Should not raise exceptions even if handlers fail
        callbacks.handle_token("test")
        callbacks.handle_completion("test")

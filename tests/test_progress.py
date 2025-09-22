"""Tests for progress tracking functionality."""

import pytest
from dataclasses import dataclass
from typing import Callable, List, Optional
from unittest.mock import Mock, call
import time

from crewforge.core.progress import (
    ProgressTracker,
    ProgressStep,
    StreamingCallbacks,
    ProgressStatus,
    ProgressEvent,
)


class TestProgressStep:
    """Test suite for ProgressStep dataclass."""

    def test_create_step_with_required_fields(self):
        """Test creating a progress step with required fields."""
        step = ProgressStep(
            id="step1",
            description="Test step",
        )

        assert step.id == "step1"
        assert step.description == "Test step"
        assert step.estimated_duration is None
        assert step.status == ProgressStatus.NOT_STARTED
        assert step.start_time is None
        assert step.end_time is None

    def test_create_step_with_all_fields(self):
        """Test creating a progress step with all fields."""
        step = ProgressStep(
            id="step1",
            description="Test step",
            estimated_duration=30.0,
        )

        assert step.id == "step1"
        assert step.description == "Test step"
        assert step.estimated_duration == 30.0
        assert step.status == ProgressStatus.NOT_STARTED

    def test_step_equality(self):
        """Test that steps with same id are equal."""
        step1 = ProgressStep("step1", "Test step")
        step2 = ProgressStep("step1", "Test step")
        step3 = ProgressStep("step2", "Test step")

        assert step1 == step2
        assert step1 != step3


class TestProgressTracker:
    """Test suite for ProgressTracker class."""

    @pytest.fixture
    def sample_steps(self):
        """Create sample progress steps for testing."""
        return [
            ProgressStep("step1", "Analyze prompt", 10.0),
            ProgressStep("step2", "Generate agents", 20.0),
            ProgressStep("step3", "Generate tasks", 15.0),
            ProgressStep("step4", "Create project", 5.0),
        ]

    @pytest.fixture
    def tracker(self, sample_steps):
        """Create a progress tracker with sample steps."""
        return ProgressTracker(sample_steps)

    @pytest.fixture
    def mock_callback(self):
        """Create a mock progress callback."""
        return Mock()

    def test_init_with_steps(self, sample_steps):
        """Test initialization with progress steps."""
        tracker = ProgressTracker(sample_steps)

        assert len(tracker.steps) == 4
        assert tracker.current_step_index == 0
        assert tracker.total_steps == 4
        assert tracker.completed_steps == 0
        assert tracker.progress_percentage == 0.0
        assert tracker.callbacks == []

    def test_init_empty_steps_raises_error(self):
        """Test initialization with empty steps raises error."""
        with pytest.raises(ValueError, match="Steps list cannot be empty"):
            ProgressTracker([])

    def test_add_callback(self, tracker, mock_callback):
        """Test adding progress callback."""
        tracker.add_callback(mock_callback)

        assert len(tracker.callbacks) == 1
        assert tracker.callbacks[0] == mock_callback

    def test_start_step_success(self, tracker, mock_callback):
        """Test starting the first step."""
        tracker.add_callback(mock_callback)

        tracker.start_step("step1")

        # Verify step status
        assert tracker.steps[0].status == ProgressStatus.IN_PROGRESS
        assert tracker.steps[0].start_time is not None
        assert tracker.current_step_index == 0
        assert tracker.progress_percentage == 0.0

        # Verify callback was called
        mock_callback.assert_called_once()
        event = mock_callback.call_args[0][0]
        assert isinstance(event, ProgressEvent)
        assert event.step_id == "step1"
        assert event.status == ProgressStatus.IN_PROGRESS

    def test_start_step_invalid_id_raises_error(self, tracker):
        """Test starting step with invalid id raises error."""
        with pytest.raises(ValueError, match="Step 'invalid' not found"):
            tracker.start_step("invalid")

    def test_start_step_already_in_progress_raises_error(self, tracker):
        """Test starting step that's already in progress raises error."""
        tracker.start_step("step1")

        with pytest.raises(ValueError, match="Step 'step1' is already in progress"):
            tracker.start_step("step1")

    def test_complete_step_success(self, tracker, mock_callback):
        """Test completing a step."""
        tracker.add_callback(mock_callback)
        tracker.start_step("step1")
        mock_callback.reset_mock()

        tracker.complete_step("step1")

        # Verify step status
        assert tracker.steps[0].status == ProgressStatus.COMPLETED
        assert tracker.steps[0].end_time is not None
        assert tracker.completed_steps == 1
        assert tracker.progress_percentage == 25.0  # 1/4 steps

        # Verify callback was called
        mock_callback.assert_called_once()
        event = mock_callback.call_args[0][0]
        assert event.step_id == "step1"
        assert event.status == ProgressStatus.COMPLETED

    def test_complete_step_not_in_progress_raises_error(self, tracker):
        """Test completing step that's not in progress raises error."""
        with pytest.raises(ValueError, match="Step 'step1' is not in progress"):
            tracker.complete_step("step1")

    def test_fail_step_success(self, tracker, mock_callback):
        """Test failing a step."""
        tracker.add_callback(mock_callback)
        tracker.start_step("step1")
        mock_callback.reset_mock()

        error_message = "Generation failed"
        tracker.fail_step("step1", error_message)

        # Verify step status
        assert tracker.steps[0].status == ProgressStatus.FAILED
        assert tracker.steps[0].end_time is not None
        assert tracker.steps[0].error_message == error_message

        # Verify callback was called
        mock_callback.assert_called_once()
        event = mock_callback.call_args[0][0]
        assert event.step_id == "step1"
        assert event.status == ProgressStatus.FAILED
        assert event.error_message == error_message

    def test_get_current_step(self, tracker):
        """Test getting current step."""
        current_step = tracker.get_current_step()

        assert current_step.id == "step1"
        assert current_step.description == "Analyze prompt"

    def test_get_step_by_id(self, tracker):
        """Test getting step by id."""
        step = tracker.get_step_by_id("step3")

        assert step.id == "step3"
        assert step.description == "Generate tasks"

    def test_get_step_by_id_not_found(self, tracker):
        """Test getting step by invalid id returns None."""
        step = tracker.get_step_by_id("invalid")

        assert step is None

    def test_is_complete(self, tracker):
        """Test checking if all steps are complete."""
        assert not tracker.is_complete()

        # Complete all steps
        for step in tracker.steps:
            tracker.start_step(step.id)
            tracker.complete_step(step.id)

        assert tracker.is_complete()

    def test_has_failures(self, tracker):
        """Test checking if any steps failed."""
        assert not tracker.has_failures()

        tracker.start_step("step1")
        tracker.fail_step("step1", "Test error")

        assert tracker.has_failures()

    def test_get_estimated_total_duration(self, tracker):
        """Test calculating total estimated duration."""
        total_duration = tracker.get_estimated_total_duration()

        # 10 + 20 + 15 + 5 = 50 seconds
        assert total_duration == 50.0

    def test_get_estimated_remaining_duration(self, tracker):
        """Test calculating estimated remaining duration."""
        # Initially all time remaining
        assert tracker.get_estimated_remaining_duration() == 50.0

        # Complete first step
        tracker.start_step("step1")
        tracker.complete_step("step1")

        # Should be 40 seconds remaining (20 + 15 + 5)
        assert tracker.get_estimated_remaining_duration() == 40.0


class TestStreamingCallbacks:
    """Test suite for StreamingCallbacks class."""

    @pytest.fixture
    def mock_token_callback(self):
        """Create a mock token callback."""
        return Mock()

    @pytest.fixture
    def mock_completion_callback(self):
        """Create a mock completion callback."""
        return Mock()

    @pytest.fixture
    def streaming_callbacks(self, mock_token_callback, mock_completion_callback):
        """Create streaming callbacks with mocks."""
        return StreamingCallbacks(
            on_token=mock_token_callback,
            on_completion=mock_completion_callback,
        )

    def test_init_with_callbacks(self, mock_token_callback, mock_completion_callback):
        """Test initialization with callbacks."""
        callbacks = StreamingCallbacks(
            on_token=mock_token_callback,
            on_completion=mock_completion_callback,
        )

        assert callbacks.on_token == mock_token_callback
        assert callbacks.on_completion == mock_completion_callback

    def test_init_without_callbacks(self):
        """Test initialization without callbacks."""
        callbacks = StreamingCallbacks()

        assert callbacks.on_token is None
        assert callbacks.on_completion is None

    def test_handle_token_with_callback(self, streaming_callbacks, mock_token_callback):
        """Test handling token with callback."""
        token = "test"
        streaming_callbacks.handle_token(token)

        mock_token_callback.assert_called_once_with(token)

    def test_handle_token_without_callback(self):
        """Test handling token without callback doesn't raise error."""
        callbacks = StreamingCallbacks()

        # Should not raise any exception
        callbacks.handle_token("test")

    def test_handle_completion_with_callback(
        self, streaming_callbacks, mock_completion_callback
    ):
        """Test handling completion with callback."""
        response = "Complete response"
        streaming_callbacks.handle_completion(response)

        mock_completion_callback.assert_called_once_with(response)

    def test_handle_completion_without_callback(self):
        """Test handling completion without callback doesn't raise error."""
        callbacks = StreamingCallbacks()

        # Should not raise any exception
        callbacks.handle_completion("test")


class TestProgressEvent:
    """Test suite for ProgressEvent dataclass."""

    def test_create_basic_event(self):
        """Test creating basic progress event."""
        event = ProgressEvent(
            step_id="step1",
            description="Test step",
            status=ProgressStatus.IN_PROGRESS,
            progress_percentage=25.0,
        )

        assert event.step_id == "step1"
        assert event.description == "Test step"
        assert event.status == ProgressStatus.IN_PROGRESS
        assert event.progress_percentage == 25.0
        assert event.error_message is None

    def test_create_error_event(self):
        """Test creating error progress event."""
        event = ProgressEvent(
            step_id="step1",
            description="Test step",
            status=ProgressStatus.FAILED,
            progress_percentage=0.0,
            error_message="Test error",
        )

        assert event.step_id == "step1"
        assert event.status == ProgressStatus.FAILED
        assert event.error_message == "Test error"


class TestProgressStatus:
    """Test suite for ProgressStatus enum."""

    def test_status_values(self):
        """Test all status enum values exist."""
        assert ProgressStatus.NOT_STARTED.value == "not_started"
        assert ProgressStatus.IN_PROGRESS.value == "in_progress"
        assert ProgressStatus.COMPLETED.value == "completed"
        assert ProgressStatus.FAILED.value == "failed"

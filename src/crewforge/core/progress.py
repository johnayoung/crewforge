"""Progress tracking system for long-running generation processes.

This module provides composable progress tracking with step visibility, percentage
completion, and LLM response streaming for the CrewForge generation pipeline.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ProgressStatus(Enum):
    """Status enumeration for progress steps."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProgressStep:
    """Represents a single step in the progress tracking system.

    Attributes:
        id: Unique identifier for the step
        description: Human-readable description of the step
        estimated_duration: Estimated duration in seconds (optional)
        status: Current status of the step
        start_time: Timestamp when step was started
        end_time: Timestamp when step was completed/failed
        error_message: Error message if step failed
    """

    id: str
    description: str
    estimated_duration: Optional[float] = None
    status: ProgressStatus = field(default=ProgressStatus.NOT_STARTED)
    start_time: Optional[float] = field(default=None)
    end_time: Optional[float] = field(default=None)
    error_message: Optional[str] = field(default=None)

    def __eq__(self, other):
        """Steps are equal if they have the same id."""
        if not isinstance(other, ProgressStep):
            return False
        return self.id == other.id


@dataclass
class ProgressEvent:
    """Event data structure for progress callbacks.

    Attributes:
        step_id: ID of the step that triggered the event
        description: Description of the step
        status: Current status of the step
        progress_percentage: Overall progress percentage (0-100)
        error_message: Error message if step failed
    """

    step_id: str
    description: str
    status: ProgressStatus
    progress_percentage: float
    error_message: Optional[str] = None


class StreamingCallbacks:
    """Callbacks for handling LLM response streaming.

    Provides hooks for token-by-token streaming and completion handling
    to show real-time LLM generation progress.
    """

    def __init__(
        self,
        on_token: Optional[Callable[[str], None]] = None,
        on_completion: Optional[Callable[[str], None]] = None,
    ):
        """Initialize streaming callbacks.

        Args:
            on_token: Callback function called for each token
            on_completion: Callback function called when response is complete
        """
        self.on_token = on_token
        self.on_completion = on_completion

    def handle_token(self, token: str) -> None:
        """Handle a single token from LLM streaming response.

        Args:
            token: Token string to handle
        """
        if self.on_token:
            self.on_token(token)

    def handle_completion(self, response: str) -> None:
        """Handle completion of LLM streaming response.

        Args:
            response: Complete response string
        """
        if self.on_completion:
            self.on_completion(response)


class ProgressTracker:
    """Tracks progress through multiple steps with callbacks and streaming support.

    Provides step tracking, percentage calculation, time estimation, and event
    callbacks for real-time progress updates during long-running operations.
    """

    def __init__(self, steps: List[ProgressStep]):
        """Initialize progress tracker with steps.

        Args:
            steps: List of progress steps to track

        Raises:
            ValueError: If steps list is empty
        """
        if not steps:
            raise ValueError("Steps list cannot be empty")

        self.steps = steps
        self.callbacks: List[Callable[[ProgressEvent], None]] = []
        self.current_step_index = 0

    @property
    def total_steps(self) -> int:
        """Get total number of steps."""
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        """Get number of completed steps."""
        return len([s for s in self.steps if s.status == ProgressStatus.COMPLETED])

    @property
    def progress_percentage(self) -> float:
        """Get overall progress percentage (0-100)."""
        if self.total_steps == 0:
            return 100.0
        return (self.completed_steps / self.total_steps) * 100.0

    def add_callback(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Add a progress callback function.

        Args:
            callback: Function to call on progress events
        """
        self.callbacks.append(callback)

    def _notify_callbacks(self, event: ProgressEvent) -> None:
        """Notify all callbacks of a progress event.

        Args:
            event: Progress event to send to callbacks
        """
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception:
                # Silently ignore callback errors to not interrupt progress
                pass

    def get_step_by_id(self, step_id: str) -> Optional[ProgressStep]:
        """Get step by ID.

        Args:
            step_id: ID of the step to find

        Returns:
            ProgressStep if found, None otherwise
        """
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_current_step(self) -> ProgressStep:
        """Get the current step being tracked.

        Returns:
            Current ProgressStep
        """
        return self.steps[self.current_step_index]

    def start_step(self, step_id: str) -> None:
        """Start a progress step.

        Args:
            step_id: ID of the step to start

        Raises:
            ValueError: If step not found or already in progress
        """
        step = self.get_step_by_id(step_id)
        if not step:
            raise ValueError(f"Step '{step_id}' not found")

        if step.status == ProgressStatus.IN_PROGRESS:
            raise ValueError(f"Step '{step_id}' is already in progress")

        step.status = ProgressStatus.IN_PROGRESS
        step.start_time = time.time()

        event = ProgressEvent(
            step_id=step.id,
            description=step.description,
            status=step.status,
            progress_percentage=self.progress_percentage,
        )
        self._notify_callbacks(event)

    def complete_step(self, step_id: str) -> None:
        """Complete a progress step.

        Args:
            step_id: ID of the step to complete

        Raises:
            ValueError: If step not found or not in progress
        """
        step = self.get_step_by_id(step_id)
        if not step:
            raise ValueError(f"Step '{step_id}' not found")

        if step.status != ProgressStatus.IN_PROGRESS:
            raise ValueError(f"Step '{step_id}' is not in progress")

        step.status = ProgressStatus.COMPLETED
        step.end_time = time.time()

        event = ProgressEvent(
            step_id=step.id,
            description=step.description,
            status=step.status,
            progress_percentage=self.progress_percentage,
        )
        self._notify_callbacks(event)

    def fail_step(self, step_id: str, error_message: str) -> None:
        """Fail a progress step.

        Args:
            step_id: ID of the step to fail
            error_message: Error message describing the failure

        Raises:
            ValueError: If step not found
        """
        step = self.get_step_by_id(step_id)
        if not step:
            raise ValueError(f"Step '{step_id}' not found")

        step.status = ProgressStatus.FAILED
        step.end_time = time.time()
        step.error_message = error_message

        event = ProgressEvent(
            step_id=step.id,
            description=step.description,
            status=step.status,
            progress_percentage=self.progress_percentage,
            error_message=error_message,
        )
        self._notify_callbacks(event)

    def is_complete(self) -> bool:
        """Check if all steps are completed.

        Returns:
            True if all steps are completed, False otherwise
        """
        return all(step.status == ProgressStatus.COMPLETED for step in self.steps)

    def has_failures(self) -> bool:
        """Check if any steps have failed.

        Returns:
            True if any step has failed, False otherwise
        """
        return any(step.status == ProgressStatus.FAILED for step in self.steps)

    def get_estimated_total_duration(self) -> float:
        """Get estimated total duration for all steps.

        Returns:
            Total estimated duration in seconds
        """
        return sum(step.estimated_duration or 0.0 for step in self.steps)

    def get_estimated_remaining_duration(self) -> float:
        """Get estimated remaining duration for incomplete steps.

        Returns:
            Estimated remaining duration in seconds
        """
        return sum(
            step.estimated_duration or 0.0
            for step in self.steps
            if step.status not in [ProgressStatus.COMPLETED, ProgressStatus.FAILED]
        )

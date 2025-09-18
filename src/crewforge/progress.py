"""
Progress indicators and status message display for CrewForge CLI.

Provides progress bars, spinners, status messages, and timing functionality
for long-running operations during project generation.
"""

import time
import click
from typing import List, Optional


class ProgressIndicator:
    """
    Progress indicator for multi-step operations.

    Provides step-by-step progress tracking with timing and visual feedback
    using Click's built-in progress utilities.
    """

    def __init__(self, steps: Optional[List[str]] = None):
        """
        Initialize progress indicator.

        Args:
            steps: List of step descriptions for the operation
        """
        self.steps = steps or []
        self.total_steps = len(self.steps)
        self.current_step = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._progress_bar = None

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since progress started."""
        if self.start_time is None:
            return 0.0

        end_time = self.end_time or time.time()
        return end_time - self.start_time

    def start_progress(self, message: str) -> None:
        """
        Start progress tracking with an initial message.

        Args:
            message: Initial status message to display
        """
        self.start_time = time.time()
        click.echo(f"🚀 {click.style(message, fg='blue', bold=True)}")

        if self.total_steps > 0:
            click.echo(f"📋 {self.total_steps} steps to complete")

    def update_progress(self, step_description: str, step_number: int) -> None:
        """
        Update progress with current step information.

        Args:
            step_description: Description of current step
            step_number: Current step number (0-based)
        """
        self.current_step = step_number

        if self.total_steps > 0:
            progress_pct = ((step_number + 1) / self.total_steps) * 100
            progress_bar = "█" * int(progress_pct // 10) + "░" * (
                10 - int(progress_pct // 10)
            )

            click.echo(
                f"⏳ [{progress_bar}] "
                f"{progress_pct:.0f}% - "
                f"{click.style(step_description, fg='cyan')}"
            )
        else:
            click.echo(f"⏳ {click.style(step_description, fg='cyan')}")

    def finish_progress(self, message: str) -> None:
        """
        Finish progress tracking with a completion message.

        Args:
            message: Completion message to display
        """
        self.end_time = time.time()
        elapsed = self.elapsed_time

        click.echo(f"✅ {click.style(message, fg='green', bold=True)}")

        if elapsed > 0:
            click.echo(f"🕐 Completed in {elapsed:.2f} seconds")


class StatusDisplay:
    """
    Status message display utility for user feedback.

    Provides consistent, colorized status messages using Click's styling.
    """

    def info(self, message: str) -> None:
        """
        Display an informational message.

        Args:
            message: Information message to display
        """
        click.echo(f"ℹ️  {click.style(message, fg='blue')}")

    def success(self, message: str) -> None:
        """
        Display a success message.

        Args:
            message: Success message to display
        """
        click.echo(f"✅ {click.style(message, fg='green', bold=True)}")

    def warning(self, message: str) -> None:
        """
        Display a warning message.

        Args:
            message: Warning message to display
        """
        click.echo(f"⚠️  {click.style(message, fg='yellow')}")

    def error(self, message: str) -> None:
        """
        Display an error message.

        Args:
            message: Error message to display
        """
        click.echo(f"❌ {click.style(message, fg='red', bold=True)}")


def simulate_work_with_progress(steps: List[str], work_duration: float = 0.5) -> None:
    """
    Simulate work with progress indicators for demonstration/testing.

    Args:
        steps: List of work steps to simulate
        work_duration: Duration to simulate for each step in seconds
    """
    progress = ProgressIndicator(steps=steps)
    status = StatusDisplay()

    progress.start_progress("Starting work simulation")

    for i, step in enumerate(steps):
        progress.update_progress(step, i)
        time.sleep(work_duration)  # Simulate work

    progress.finish_progress("Work simulation completed")
    status.success("All steps completed successfully!")

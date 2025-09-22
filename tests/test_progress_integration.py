"""Integration tests for progress tracking with ProjectScaffolder."""

import pytest
from unittest.mock import Mock, patch, call
from pathlib import Path

from crewforge.core.scaffolding import ProjectScaffolder
from crewforge.core.progress import (
    ProgressTracker,
    ProgressStep,
    ProgressStatus,
    ProgressEvent,
)
from crewforge.models import GenerationRequest, AgentConfig, TaskConfig


class TestProjectScaffolderProgressIntegration:
    """Integration tests for ProjectScaffolder with progress tracking."""

    @pytest.fixture
    def progress_steps(self):
        """Create standard progress steps for project generation."""
        return [
            ProgressStep("analyze_prompt", "Analyzing prompt requirements", 10.0),
            ProgressStep("generate_agents", "Generating agent configurations", 20.0),
            ProgressStep("generate_tasks", "Generating task definitions", 15.0),
            ProgressStep("select_tools", "Selecting appropriate tools", 5.0),
            ProgressStep("create_project", "Creating CrewAI project structure", 8.0),
            ProgressStep("populate_files", "Populating project files", 12.0),
        ]

    @pytest.fixture
    def progress_tracker(self, progress_steps):
        """Create a progress tracker with standard steps."""
        return ProgressTracker(progress_steps)

    @pytest.fixture
    def mock_progress_callback(self):
        """Create a mock progress callback for testing."""
        return Mock()

    @pytest.fixture
    def scaffolder_with_progress(self):
        """Create a ProjectScaffolder with mocked dependencies."""
        mock_generation_engine = Mock()
        mock_generation_engine.analyze_prompt.return_value = {"tools_needed": []}
        mock_generation_engine.generate_agents.return_value = [
            AgentConfig(role="Test Agent", goal="Test goal", backstory="Test backstory")
        ]
        mock_generation_engine.generate_tasks.return_value = [
            TaskConfig(
                description="Test task",
                expected_output="Test output",
                agent="Test Agent",
            )
        ]
        mock_generation_engine.select_tools.return_value = {
            "selected_tools": [],
            "unavailable_tools": [],
        }

        mock_template_engine = Mock()
        mock_template_engine.populate_template = Mock()

        return ProjectScaffolder(
            generation_engine=mock_generation_engine,
            template_engine=mock_template_engine,
        )

    @pytest.fixture
    def sample_request(self):
        """Sample generation request."""
        return GenerationRequest(
            prompt="Create a content research crew", project_name="test-crew"
        )

    def test_progress_tracker_integration_with_scaffolder(
        self,
        progress_tracker,
        mock_progress_callback,
        scaffolder_with_progress,
        sample_request,
    ):
        """Test that progress tracking integrates with scaffolding operations."""
        progress_tracker.add_callback(mock_progress_callback)

        # Simulate progress tracking during generation
        progress_tracker.start_step("analyze_prompt")
        progress_tracker.complete_step("analyze_prompt")

        progress_tracker.start_step("generate_agents")
        progress_tracker.complete_step("generate_agents")

        # Verify progress callbacks were called
        assert mock_progress_callback.call_count == 4  # 2 starts + 2 completions

        # Verify progress percentage calculation
        assert progress_tracker.progress_percentage == 33.33333333333333  # 2/6 steps

        # Verify proper event data
        events = [
            call_args[0][0] for call_args in mock_progress_callback.call_args_list
        ]

        # First event - start analyze_prompt
        assert events[0].step_id == "analyze_prompt"
        assert events[0].status == ProgressStatus.IN_PROGRESS

        # Second event - complete analyze_prompt
        assert events[1].step_id == "analyze_prompt"
        assert events[1].status == ProgressStatus.COMPLETED

        # Third event - start generate_agents
        assert events[2].step_id == "generate_agents"
        assert events[2].status == ProgressStatus.IN_PROGRESS

        # Fourth event - complete generate_agents
        assert events[3].step_id == "generate_agents"
        assert events[3].status == ProgressStatus.COMPLETED

    def test_error_handling_with_progress_tracking(
        self, progress_tracker, mock_progress_callback
    ):
        """Test error handling in progress tracking."""
        progress_tracker.add_callback(mock_progress_callback)

        progress_tracker.start_step("analyze_prompt")
        progress_tracker.fail_step("analyze_prompt", "Analysis failed")

        # Verify failure was tracked
        assert progress_tracker.has_failures()
        assert not progress_tracker.is_complete()

        # Verify callback received failure event
        failure_event = mock_progress_callback.call_args_list[-1][0][0]
        assert failure_event.status == ProgressStatus.FAILED
        assert failure_event.error_message == "Analysis failed"

    def test_time_estimation_during_progress(self, progress_tracker):
        """Test time estimation calculations during progress."""
        # Initially all time remaining
        total_duration = progress_tracker.get_estimated_total_duration()
        assert total_duration == 70.0  # Sum of all step durations

        remaining_duration = progress_tracker.get_estimated_remaining_duration()
        assert remaining_duration == 70.0

        # Complete first two steps
        progress_tracker.start_step("analyze_prompt")
        progress_tracker.complete_step("analyze_prompt")

        progress_tracker.start_step("generate_agents")
        progress_tracker.complete_step("generate_agents")

        # Remaining should be reduced by completed steps (10 + 20 = 30)
        remaining_duration = progress_tracker.get_estimated_remaining_duration()
        assert remaining_duration == 40.0  # 70 - 30 = 40

    @patch("crewforge.core.scaffolding.shutil.rmtree")
    def test_scaffolder_cleanup_with_progress_tracking(
        self,
        mock_rmtree,
        scaffolder_with_progress,
        sample_request,
        progress_tracker,
        mock_progress_callback,
    ):
        """Test that scaffolder cleanup doesn't interfere with progress tracking."""
        progress_tracker.add_callback(mock_progress_callback)

        # Setup scaffolder to fail after creating project
        with patch.object(
            scaffolder_with_progress, "create_crewai_project"
        ) as mock_create:
            mock_create.return_value = Path("/tmp/test-crew")

            with patch.object(
                scaffolder_with_progress, "populate_project_files"
            ) as mock_populate:
                mock_populate.side_effect = Exception("Population failed")

                # Start tracking before generation
                progress_tracker.start_step("create_project")

                # Generation should fail but progress tracking should still work
                with pytest.raises(Exception):
                    scaffolder_with_progress.generate_project(
                        sample_request, Path("/tmp")
                    )

                # Mark step as failed in progress tracker
                progress_tracker.fail_step("create_project", "Population failed")

                # Verify progress tracking still works after scaffolder failure
                assert progress_tracker.has_failures()
                failure_event = mock_progress_callback.call_args_list[-1][0][0]
                assert failure_event.status == ProgressStatus.FAILED

    def test_concurrent_step_tracking_validation(self, progress_tracker):
        """Test validation of concurrent step operations."""
        # Can't start same step twice
        progress_tracker.start_step("analyze_prompt")

        with pytest.raises(
            ValueError, match="Step 'analyze_prompt' is already in progress"
        ):
            progress_tracker.start_step("analyze_prompt")

        # Can't complete step that's not in progress
        with pytest.raises(
            ValueError, match="Step 'generate_agents' is not in progress"
        ):
            progress_tracker.complete_step("generate_agents")

        # Complete the in-progress step
        progress_tracker.complete_step("analyze_prompt")

        # Can't complete same step again
        with pytest.raises(
            ValueError, match="Step 'analyze_prompt' is not in progress"
        ):
            progress_tracker.complete_step("analyze_prompt")

    def test_progress_callback_error_handling(self, progress_tracker):
        """Test that callback errors don't interrupt progress tracking."""
        # Add a callback that raises an exception
        failing_callback = Mock(side_effect=Exception("Callback failed"))
        working_callback = Mock()

        progress_tracker.add_callback(failing_callback)
        progress_tracker.add_callback(working_callback)

        # Start a step - should not raise even though first callback fails
        progress_tracker.start_step("analyze_prompt")

        # Both callbacks should have been called
        failing_callback.assert_called_once()
        working_callback.assert_called_once()

        # Progress tracking should still work
        assert progress_tracker.steps[0].status == ProgressStatus.IN_PROGRESS

    def test_step_identification_and_retrieval(self, progress_tracker):
        """Test step identification and retrieval methods."""
        # Get current step
        current = progress_tracker.get_current_step()
        assert current.id == "analyze_prompt"

        # Get step by ID
        task_step = progress_tracker.get_step_by_id("generate_tasks")
        assert task_step.description == "Generating task definitions"

        # Get non-existent step
        missing_step = progress_tracker.get_step_by_id("nonexistent")
        assert missing_step is None

        # Test step equality
        step1 = ProgressStep("test", "Test step")
        step2 = ProgressStep("test", "Different description")
        step3 = ProgressStep("other", "Test step")

        assert step1 == step2  # Same ID
        assert step1 != step3  # Different ID

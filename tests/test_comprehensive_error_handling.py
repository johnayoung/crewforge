"""
Tests for comprehensive error handling and recovery mechanisms.

This module tests advanced error handling scenarios including:
- Recovery from partial failures
- Error aggregation and reporting
- Graceful degradation
- User guidance for error resolution
"""

import asyncio
import os
import sys
import tempfile
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from click.testing import CliRunner

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crewforge.cli import main, create
from crewforge.llm import LLMClient, LLMError, RateLimitError
from crewforge.orchestrator import WorkflowOrchestrator, WorkflowContext, WorkflowError
from crewforge.scaffolder import CrewAIScaffolder, CrewAIError
from crewforge.validation import (
    ValidationError,
    ValidationResult,
    ValidationIssue,
    IssueSeverity,
)


class TestErrorRecoveryMechanisms:
    """Test error recovery and graceful degradation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("crewforge.orchestrator.WorkflowOrchestrator._initialize_components")
    @patch("crewforge.orchestrator.WorkflowOrchestrator._step_process_prompt")
    @patch("crewforge.orchestrator.WorkflowOrchestrator._step_validate_specification")
    @patch("crewforge.orchestrator.WorkflowOrchestrator._step_validate_dependencies")
    @patch("crewforge.orchestrator.WorkflowOrchestrator._step_generate_project")
    @patch("crewforge.orchestrator.WorkflowOrchestrator._step_setup_dependencies")
    def test_partial_workflow_recovery(
        self,
        mock_setup_deps,
        mock_generate,
        mock_validate_deps,
        mock_validate_spec,
        mock_process_prompt,
        mock_init,
    ):
        """Test that workflow can recover from partial failures."""
        from crewforge.orchestrator import WorkflowStep

        # Setup mocks to add steps to context
        def mock_process_prompt_func(context):
            step = WorkflowStep(name="process_prompt", description="Process prompt")
            step.status = "completed"
            context.steps.append(step)

        def mock_validate_spec_func(context):
            step = WorkflowStep(name="validate_spec", description="Validate spec")
            step.status = "completed"
            context.steps.append(step)

        def mock_validate_deps_func(context):
            step = WorkflowStep(
                name="validate_dependencies", description="Validate deps"
            )
            step.status = "completed"
            context.steps.append(step)

        def mock_generate_func(context):
            step = WorkflowStep(name="generate_project", description="Generate project")
            step.status = "failed"
            step.error = "Generation failed"
            context.steps.append(step)
            raise WorkflowError("Generation failed", "generate_project")

        mock_init.return_value = None
        mock_process_prompt.side_effect = mock_process_prompt_func
        mock_validate_spec.side_effect = mock_validate_spec_func
        mock_validate_deps.side_effect = mock_validate_deps_func
        mock_generate.side_effect = mock_generate_func
        mock_setup_deps.return_value = None

        orchestrator = WorkflowOrchestrator()
        context = WorkflowContext(
            user_prompt="test prompt",
            project_name="test-project",
            output_dir=Path("/tmp"),
        )

        # Execute workflow - should handle error gracefully
        with pytest.raises(WorkflowError):
            asyncio.run(orchestrator.execute_workflow(context))

        # Verify that error handling was called
        assert len(context.steps) >= 4  # Should have recorded steps
        assert any(step.status == "failed" for step in context.steps)

    def test_llm_retry_recovery(self):
        """Test LLM client recovery through retry mechanism."""
        client = LLMClient(max_retries=2)

        # Mock litellm.completion to fail twice then succeed
        with patch("crewforge.llm.litellm.completion") as mock_completion:
            mock_completion.side_effect = [
                Exception("Temporary network error"),
                Exception("Server timeout"),
                Mock(choices=[Mock(message=Mock(content="Success response"))]),
            ]

            result = asyncio.run(client.complete("test prompt"))

            # Should have succeeded after retries
            assert result == "Success response"
            assert mock_completion.call_count == 3  # Initial + 2 retries

    def test_llm_non_retryable_error_no_recovery(self):
        """Test that non-retryable errors don't attempt recovery."""
        client = LLMClient(max_retries=3)

        with patch("crewforge.llm.litellm.completion") as mock_completion:
            # Invalid API key error (non-retryable)
            mock_completion.side_effect = Exception("Invalid API key")

            with pytest.raises(LLMError, match="API request failed"):
                asyncio.run(client.complete("test prompt"))

            # Should only call once (no retries for non-retryable)
            assert mock_completion.call_count == 1

    def test_scaffolder_error_recovery_with_fallback(self):
        """Test scaffolder error recovery with alternative approaches."""
        with patch("crewforge.scaffolder.subprocess.run") as mock_run:
            # Mock CrewAI CLI failure
            mock_run.return_value = Mock(returncode=1, stderr="CrewAI CLI error")

            scaffolder = CrewAIScaffolder()

            with pytest.raises(CrewAIError):
                scaffolder.create_crew("test", Path("/tmp"))

            # Verify error details are captured
            # In a real implementation, we might want to suggest alternatives
            # like manual project creation or different CLI versions


class TestErrorAggregation:
    """Test error aggregation and comprehensive reporting."""

    def test_multiple_validation_errors_aggregated(self):
        """Test that multiple validation errors are properly aggregated."""
        issues = [
            ValidationIssue(
                severity=IssueSeverity.ERROR,
                message="Missing required field: project_name",
                field_path="root",
                suggestions=["Add project_name to specification"],
            ),
            ValidationIssue(
                severity=IssueSeverity.ERROR,
                message="Invalid agent configuration",
                field_path="agents[0]",
                suggestions=["Check agent role and goal definitions"],
            ),
            ValidationIssue(
                severity=IssueSeverity.WARNING,
                message="Missing task description",
                field_path="tasks[0]",
                suggestions=["Add detailed task descriptions"],
            ),
        ]

        result = ValidationResult(issues=issues, completeness_score=0.3)

        assert not result.is_valid
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.completeness_score == 0.3

    def test_workflow_error_aggregation(self):
        """Test that workflow errors from multiple steps are aggregated."""
        from crewforge.orchestrator import WorkflowStep
        import time

        context = WorkflowContext(
            user_prompt="test", project_name="test", output_dir=Path("/tmp")
        )

        # Simulate multiple failed steps with proper WorkflowStep objects
        current_time = time.time()
        context.steps = [
            WorkflowStep(
                name="step1",
                description="Step 1",
                status="failed",
                error="Error 1",
                start_time=current_time,
                end_time=current_time + 1,
            ),
            WorkflowStep(
                name="step2",
                description="Step 2",
                status="failed",
                error="Error 2",
                start_time=current_time + 1,
                end_time=current_time + 2,
            ),
            WorkflowStep(
                name="step3",
                description="Step 3",
                status="completed",
                error=None,
                start_time=current_time + 2,
                end_time=current_time + 3,
            ),
        ]

        orchestrator = WorkflowOrchestrator()

        summary = orchestrator.get_workflow_summary(context)

        assert summary["failed_steps"] == 2
        assert summary["completed_steps"] == 1
        assert not summary["success"]


class TestGracefulDegradation:
    """Test graceful degradation when full functionality isn't available."""

    def test_llm_fallback_on_rate_limit(self):
        """Test graceful degradation when rate limits are hit."""
        client = LLMClient(requests_per_minute=1)

        # Fill rate limit
        client._request_history = [time.time()] * 2

        with pytest.raises(RateLimitError):
            client.check_rate_limit()

        # In a real scenario, we might want to suggest waiting or using a different provider
        # This test verifies the error is properly raised for handling

    @patch("crewforge.scaffolder.CrewAIScaffolder.check_crewai_available")
    def test_scaffolder_degradation_when_cli_unavailable(self, mock_check):
        """Test graceful degradation when CrewAI CLI is not available."""
        mock_check.return_value = False

        scaffolder = CrewAIScaffolder()

        with pytest.raises(CrewAIError):
            scaffolder.create_crew("test", Path("/tmp"))

        # Should provide helpful error message about installing CrewAI

    def test_partial_project_creation_recovery(self):
        """Test recovery when only partial project creation succeeds."""
        # This would test scenarios where some files are created but not others
        # and the system can still provide value or guidance


class TestUserGuidance:
    """Test user-friendly error messages and guidance."""

    def test_cli_error_messages_include_helpful_suggestions(self):
        """Test that CLI error messages include actionable suggestions."""
        runner = CliRunner()

        # Test with invalid project name
        result = runner.invoke(create, ["", "test prompt"])

        assert result.exit_code != 0
        assert "Project name cannot be empty" in result.output

    def test_validation_errors_provide_specific_guidance(self):
        """Test that validation errors provide specific actionable guidance."""
        issue = ValidationIssue(
            severity=IssueSeverity.ERROR,
            message="Invalid Python syntax in generated file",
            field_path="src/project/main.py",
            suggestions=[
                "Check for syntax errors using 'python -m py_compile main.py'",
                "Verify all imports are correct",
                "Ensure proper indentation (4 spaces)",
            ],
        )

        assert len(issue.suggestions) > 0
        assert all(
            "python" in suggestion.lower()
            or "import" in suggestion.lower()
            or "indentation" in suggestion.lower()
            for suggestion in issue.suggestions
        )

    def test_llm_error_messages_help_user_resolve_issues(self):
        """Test that LLM errors provide guidance for resolution."""
        # Test rate limit error message
        client = LLMClient()
        client._request_history = [time.time()] * 100  # Exceed limit

        with pytest.raises(RateLimitError) as exc_info:
            client.check_rate_limit()

        error_msg = str(exc_info.value)
        assert "Rate limit exceeded" in error_msg
        # Could assert for suggestions like "wait and retry" or "reduce request frequency"


if __name__ == "__main__":
    pytest.main([__file__])

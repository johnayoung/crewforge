"""
Tests for the WorkflowOrchestrator component.

This module provides comprehensive tests for the workflow orchestration system,
ensuring that all components work together correctly in the end-to-end pipeline.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from crewforge.orchestrator import (
    WorkflowOrchestrator,
    WorkflowContext,
    WorkflowStep,
    WorkflowError,
)
from crewforge.validation import ValidationResult, ValidationIssue, IssueSeverity
from crewforge.llm import LLMError
from crewforge.prompt_templates import PromptTemplateError
from crewforge.scaffolder import CrewAIError


class TestWorkflowOrchestrator:
    """Test suite for WorkflowOrchestrator."""

    @pytest.fixture
    def mock_progress(self):
        """Mock progress indicator."""
        return Mock()

    @pytest.fixture
    def orchestrator(self, mock_progress):
        """Create orchestrator instance with mocked progress."""
        return WorkflowOrchestrator(progress_tracker=mock_progress)

    @pytest.fixture
    def sample_context(self, tmp_path):
        """Create sample workflow context."""
        return WorkflowContext(
            user_prompt="Create a research team with writer and editor agents",
            project_name="research-team",
            output_dir=tmp_path / "output",
            llm_provider="openai",
            llm_model="gpt-3.5-turbo",
        )

    @pytest.fixture
    def sample_project_spec(self):
        """Sample project specification."""
        return {
            "project_name": "research-team",
            "project_description": "A research team with writer and editor agents",
            "agents": [
                {
                    "role": "researcher",
                    "goal": "Research topics thoroughly",
                    "backstory": "Expert researcher",
                    "tools": ["search", "analyze"],
                },
                {
                    "role": "writer",
                    "goal": "Write high-quality content",
                    "backstory": "Professional writer",
                    "tools": ["write", "edit"],
                },
            ],
            "tasks": [
                {
                    "description": "Research the given topic",
                    "expected_output": "Research report",
                    "agent": "researcher",
                }
            ],
            "dependencies": ["crewai", "openai"],
        }

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.progress is not None
        assert orchestrator.status is not None
        assert orchestrator.validation_cache is not None
        assert orchestrator.spec_validator is not None
        assert orchestrator.llm_client is None  # Not initialized yet
        assert orchestrator.prompt_templates is None  # Not initialized yet
        assert orchestrator.scaffolder is not None

    def test_initialize_components(self, orchestrator, sample_context):
        """Test component initialization."""
        orchestrator._initialize_components(sample_context)

        assert orchestrator.llm_client is not None
        assert orchestrator.prompt_templates is not None
        assert orchestrator.llm_client.provider == "openai"
        assert orchestrator.llm_client.model == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_step_process_prompt_success(
        self, orchestrator, sample_context, sample_project_spec
    ):
        """Test successful prompt processing step."""
        # Setup mocks
        orchestrator._initialize_components(sample_context)
        orchestrator.prompt_templates.extract_project_spec = AsyncMock(
            return_value=sample_project_spec
        )
        orchestrator.prompt_templates.validate_project_spec = Mock()

        # Execute step
        await orchestrator._step_process_prompt(sample_context)

        # Verify results
        assert len(sample_context.steps) == 1
        step = sample_context.steps[0]
        assert step.name == "process_prompt"
        assert step.status == "completed"
        assert sample_context.project_spec == sample_project_spec
        assert step.output == sample_project_spec

    @pytest.mark.asyncio
    async def test_step_process_prompt_failure(self, orchestrator, sample_context):
        """Test prompt processing step failure."""
        # Setup mocks
        orchestrator._initialize_components(sample_context)
        orchestrator.prompt_templates.extract_project_spec = AsyncMock(
            side_effect=LLMError("API Error")
        )

        # Execute step and expect error
        with pytest.raises(WorkflowError) as exc_info:
            await orchestrator._step_process_prompt(sample_context)

        # Verify error handling
        assert len(sample_context.steps) == 1
        step = sample_context.steps[0]
        assert step.status == "failed"
        assert "API Error" in step.error
        assert exc_info.value.step == "process_prompt"

    @pytest.mark.asyncio
    async def test_step_validate_specification_success(
        self, orchestrator, sample_context, sample_project_spec
    ):
        """Test successful specification validation."""
        # Setup context
        sample_context.project_spec = sample_project_spec

        # Mock validator
        mock_result = ValidationResult(issues=[], completeness_score=0.95)
        orchestrator.spec_validator.validate = Mock(return_value=mock_result)

        # Execute step
        await orchestrator._step_validate_specification(sample_context)

        # Verify results
        assert len(sample_context.steps) == 1
        step = sample_context.steps[0]
        assert step.name == "validate_spec"
        assert step.status == "completed"
        assert sample_context.validation_result == mock_result

    @pytest.mark.asyncio
    async def test_step_validate_specification_with_errors(
        self, orchestrator, sample_context, sample_project_spec
    ):
        """Test specification validation with errors."""
        # Setup context
        sample_context.project_spec = sample_project_spec

        # Mock validator with errors
        mock_result = ValidationResult(
            issues=[
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    message="Missing required field",
                    field_path="agents[0].goal",
                )
            ],
            completeness_score=0.5,
        )
        orchestrator.spec_validator.validate = Mock(return_value=mock_result)

        # Execute step and expect error
        with pytest.raises(WorkflowError) as exc_info:
            await orchestrator._step_validate_specification(sample_context)

        # Verify error handling
        assert "Critical validation errors" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_step_validate_dependencies_success(
        self, orchestrator, sample_context, sample_project_spec, tmp_path
    ):
        """Test successful dependency validation."""
        # Setup context
        sample_context.project_spec = sample_project_spec
        sample_context.output_dir = tmp_path

        # Mock scaffolder
        orchestrator.scaffolder.check_crewai_available = Mock(return_value=True)

        # Execute step
        await orchestrator._step_validate_dependencies(sample_context)

        # Verify results
        assert len(sample_context.steps) == 1
        step = sample_context.steps[0]
        assert step.name == "validate_dependencies"
        assert step.status == "completed"

    @pytest.mark.asyncio
    async def test_step_validate_dependencies_crewai_missing(
        self, orchestrator, sample_context
    ):
        """Test dependency validation when CrewAI is missing."""
        # Mock scaffolder
        orchestrator.scaffolder.check_crewai_available = Mock(return_value=False)

        # Execute step and expect error
        with pytest.raises(WorkflowError) as exc_info:
            await orchestrator._step_validate_dependencies(sample_context)

        # Verify error
        assert "CrewAI CLI not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_step_generate_project_success(
        self, orchestrator, sample_context, tmp_path
    ):
        """Test successful project generation."""
        # Setup context
        sample_context.output_dir = tmp_path

        # Mock scaffolder
        mock_result = {"success": True, "project_path": tmp_path / "test-project"}
        orchestrator.scaffolder.create_crew = Mock(return_value=mock_result)

        # Execute step
        await orchestrator._step_generate_project(sample_context)

        # Verify results
        assert len(sample_context.steps) == 1
        step = sample_context.steps[0]
        assert step.name == "generate_project"
        assert step.status == "completed"
        assert sample_context.scaffold_result == mock_result

    @pytest.mark.asyncio
    async def test_step_generate_project_failure(self, orchestrator, sample_context):
        """Test project generation failure."""
        # Mock scaffolder
        mock_result = {"success": False, "error": "Scaffolding failed"}
        orchestrator.scaffolder.create_crew = Mock(return_value=mock_result)

        # Execute step and expect error
        with pytest.raises(WorkflowError) as exc_info:
            await orchestrator._step_generate_project(sample_context)

        # Verify error
        assert "Project generation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_workflow_success(
        self, orchestrator, sample_context, sample_project_spec, tmp_path
    ):
        """Test complete successful workflow execution."""
        # Setup context
        sample_context.output_dir = tmp_path

        # Mock all components to avoid actual LLM calls
        with (
            patch.object(orchestrator, "_initialize_components"),
            patch.object(orchestrator, "_step_process_prompt") as mock_process,
            patch.object(orchestrator, "_step_validate_specification") as mock_validate,
            patch.object(orchestrator, "_step_validate_dependencies") as mock_deps,
            patch.object(orchestrator, "_step_generate_project") as mock_generate,
            patch.object(orchestrator, "_step_setup_dependencies") as mock_setup_deps,
        ):

            # Setup successful mocks
            mock_process.return_value = None
            mock_validate.return_value = None
            mock_deps.return_value = None
            mock_generate.return_value = None
            mock_setup_deps.return_value = None

            # Execute workflow
            result = await orchestrator.execute_workflow(sample_context)

            # Verify all steps were called
            mock_process.assert_called_once()
            mock_validate.assert_called_once()
            mock_deps.assert_called_once()
            mock_generate.assert_called_once()
            mock_setup_deps.assert_called_once()

            # Verify result
            assert result == sample_context

    @pytest.mark.asyncio
    async def test_execute_workflow_with_failure(self, orchestrator, sample_context):
        """Test workflow execution with step failure."""
        # Setup mocks to fail at prompt processing
        orchestrator._initialize_components(sample_context)
        orchestrator.prompt_templates.extract_project_spec = AsyncMock(
            side_effect=LLMError("API Error")
        )

        # Execute workflow and expect error
        with pytest.raises(WorkflowError):
            await orchestrator.execute_workflow(sample_context)

        # Verify error handling
        assert len(sample_context.steps) >= 1
        failed_step = sample_context.steps[0]
        assert failed_step.status == "failed"
        assert (
            "API Error" in failed_step.error or "Failed to extract" in failed_step.error
        )

    def test_get_workflow_summary_success(self, orchestrator, sample_context):
        """Test workflow summary generation for successful workflow."""
        # Setup completed steps
        import time

        start_time = time.time()
        end_time = start_time + 2.5

        sample_context.steps = [
            WorkflowStep("step1", "Step 1", "completed", start_time, end_time - 1),
            WorkflowStep("step2", "Step 2", "completed", end_time - 1, end_time),
        ]

        # Get summary
        summary = orchestrator.get_workflow_summary(sample_context)

        # Verify summary
        assert summary["total_steps"] == 2
        assert summary["completed_steps"] == 2
        assert summary["failed_steps"] == 0
        assert summary["success"] is True
        assert summary["total_time_seconds"] >= 1.0  # At least 1 second
        assert len(summary["steps"]) == 2

    def test_get_workflow_summary_with_failures(self, orchestrator, sample_context):
        """Test workflow summary generation with failed steps."""
        # Setup mixed steps
        sample_context.steps = [
            WorkflowStep("step1", "Step 1", "completed"),
            WorkflowStep("step2", "Step 2", "failed", error="Test error"),
            WorkflowStep("step3", "Step 3", "pending"),
        ]

        # Get summary
        summary = orchestrator.get_workflow_summary(sample_context)

        # Verify summary
        assert summary["total_steps"] == 3
        assert summary["completed_steps"] == 1
        assert summary["failed_steps"] == 1
        assert summary["success"] is False

    @pytest.mark.asyncio
    async def test_handle_workflow_error(self, orchestrator, sample_context):
        """Test workflow error handling."""
        # Setup running step
        sample_context.steps = [
            WorkflowStep("test_step", "Test Step", "running", start_time=0)
        ]

        # Create test error
        test_error = Exception("Test workflow error")

        # Handle error
        await orchestrator._handle_workflow_error(sample_context, test_error)

        # Verify error handling
        assert len(sample_context.steps) == 1
        step = sample_context.steps[0]
        assert step.status == "failed"
        assert step.error == "Test workflow error"
        assert step.end_time is not None

    def test_workflow_context_creation(self, tmp_path):
        """Test WorkflowContext creation and properties."""
        context = WorkflowContext(
            user_prompt="Test prompt",
            project_name="test-project",
            output_dir=tmp_path / "output",
            llm_provider="anthropic",
            llm_model="claude-3-sonnet-20240229",
            api_key="test-key",
        )

        assert context.user_prompt == "Test prompt"
        assert context.project_name == "test-project"
        assert context.output_dir == tmp_path / "output"
        assert context.llm_provider == "anthropic"
        assert context.llm_model == "claude-3-sonnet-20240229"
        assert context.api_key == "test-key"
        assert context.steps == []

    def test_workflow_step_creation(self):
        """Test WorkflowStep creation and properties."""
        import time

        start = time.time()
        step = WorkflowStep(
            name="test_step",
            description="Test step description",
            status="running",
            start_time=start,
        )

        assert step.name == "test_step"
        assert step.description == "Test step description"
        assert step.status == "running"
        assert step.start_time == start
        assert step.end_time is None
        assert step.error is None
        assert step.output is None


class TestWorkflowError:
    """Test suite for WorkflowError."""

    def test_workflow_error_creation(self):
        """Test WorkflowError creation."""
        error = WorkflowError("Test error message", "test_step")

        assert str(error) == "Test error message"
        assert error.step == "test_step"
        assert error.context is None

    def test_workflow_error_with_context(self, tmp_path):
        """Test WorkflowError with context."""
        context = WorkflowContext(
            user_prompt="Test prompt",
            project_name="test-project",
            output_dir=tmp_path / "output",
        )
        error = WorkflowError("Test error", "test_step", context)

        assert error.step == "test_step"
        assert error.context == context


class TestIntegrationWorkflow:
    """Integration tests for the complete workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_simulation(self, tmp_path):
        """Test end-to-end workflow with realistic simulation."""
        # This would be an integration test that exercises the full pipeline
        # For now, we'll create a comprehensive mock-based test

        context = WorkflowContext(
            user_prompt="Create a simple blog writing team",
            project_name="blog-team",
            output_dir=tmp_path / "projects",
            llm_provider="openai",
        )

        orchestrator = WorkflowOrchestrator()

        # Mock all external dependencies
        with (
            patch.object(orchestrator, "_initialize_components"),
            patch.object(orchestrator, "_step_process_prompt") as mock_process,
            patch.object(orchestrator, "_step_validate_specification") as mock_validate,
            patch.object(orchestrator, "_step_validate_dependencies") as mock_deps,
            patch.object(orchestrator, "_step_generate_project") as mock_generate,
            patch.object(orchestrator, "_step_setup_dependencies") as mock_setup_deps,
        ):

            # Setup successful mocks
            mock_process.return_value = None
            mock_validate.return_value = None
            mock_deps.return_value = None
            mock_generate.return_value = None
            mock_setup_deps.return_value = None

            # Execute workflow
            result = await orchestrator.execute_workflow(context)

            # Verify all steps were called
            mock_process.assert_called_once()
            mock_validate.assert_called_once()
            mock_deps.assert_called_once()
            mock_generate.assert_called_once()
            mock_setup_deps.assert_called_once()

            # Verify result
            assert result == context

    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, tmp_path):
        """Test workflow error recovery and cleanup."""
        context = WorkflowContext(
            user_prompt="Test prompt",
            project_name="test-project",
            output_dir=tmp_path / "output",
        )

        orchestrator = WorkflowOrchestrator()

        # Mock failure at validation step
        with (
            patch.object(orchestrator, "_initialize_components"),
            patch.object(orchestrator, "_step_process_prompt") as mock_process,
            patch.object(orchestrator, "_step_validate_specification") as mock_validate,
        ):

            mock_process.return_value = None
            mock_validate.side_effect = WorkflowError(
                "Validation failed", "validate_spec", context
            )

            # Execute workflow and expect error
            with pytest.raises(WorkflowError):
                await orchestrator.execute_workflow(context)

            # Verify error handling was called
            mock_process.assert_called_once()
            mock_validate.assert_called_once()

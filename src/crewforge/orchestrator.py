"""
Workflow orchestration system for CrewForge.

This module provides a comprehensive workflow orchestrator that connects all
CrewForge components into a cohesive end-to-end project generation pipeline.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from crewforge.llm import LLMClient, LLMError
from crewforge.prompt_templates import PromptTemplates, PromptTemplateError
from crewforge.validation import (
    ValidationCache,
    ValidationIssue,
    ValidationResult,
    SpecificationValidator,
    IssueSeverity,
    validate_generated_project,
)
from crewforge.scaffolder import CrewAIScaffolder, CrewAIError
from crewforge.progress import ProgressIndicator, StatusDisplay


@dataclass
class WorkflowStep:
    """Represents a single step in the workflow execution."""

    name: str
    description: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    output: Optional[Any] = None


@dataclass
class WorkflowContext:
    """Context object passed through workflow steps."""

    user_prompt: str
    project_name: str
    output_dir: Path
    llm_provider: str = "openai"
    llm_model: Optional[str] = None
    api_key: Optional[str] = None
    project_spec: Optional[Dict[str, Any]] = None
    validation_result: Optional[ValidationResult] = None
    scaffold_result: Optional[Dict[str, Any]] = None
    steps: List[WorkflowStep] = field(default_factory=list)


class WorkflowError(Exception):
    """Base exception for workflow orchestration errors."""

    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        context: Optional[WorkflowContext] = None,
    ):
        super().__init__(message)
        self.step = step
        self.context = context


class WorkflowOrchestrator:
    """
    Orchestrates the complete CrewForge workflow from prompt to project generation.

    This class manages the entire pipeline:
    1. Process natural language prompt into structured specification
    2. Validate specification completeness and requirements
    3. Generate CrewAI project structure
    4. Handle errors and provide recovery mechanisms
    """

    def __init__(self, progress_tracker: Optional[ProgressIndicator] = None):
        """
        Initialize the workflow orchestrator.

        Args:
            progress_tracker: Optional progress tracker for UI feedback
        """
        self.progress = progress_tracker or ProgressIndicator()
        self.status = StatusDisplay()
        self.validation_cache = ValidationCache()
        self.spec_validator = SpecificationValidator()

        # Initialize components
        self.llm_client = None
        self.prompt_templates = None
        self.scaffolder = CrewAIScaffolder()

    def _initialize_components(self, context: WorkflowContext) -> None:
        """Initialize workflow components with context configuration."""
        # Initialize LLM client
        self.llm_client = LLMClient(
            provider=context.llm_provider,
            model=context.llm_model,
            api_key=context.api_key,
        )

        # Initialize prompt templates
        self.prompt_templates = PromptTemplates(self.llm_client)

    async def execute_workflow(self, context: WorkflowContext) -> WorkflowContext:
        """
        Execute the complete workflow pipeline.

        Args:
            context: Workflow context with user inputs

        Returns:
            Updated context with results

        Raises:
            WorkflowError: If workflow execution fails
        """
        try:
            # Initialize components
            self._initialize_components(context)

            # Execute workflow steps
            await self._step_process_prompt(context)
            await self._step_validate_specification(context)
            await self._step_validate_dependencies(context)
            await self._step_generate_project(context)

            return context

        except Exception as e:
            # Handle workflow failures
            await self._handle_workflow_error(context, e)
            raise WorkflowError(
                f"Workflow execution failed: {str(e)}", context=context
            ) from e

    async def _step_process_prompt(self, context: WorkflowContext) -> None:
        """Step 1: Process natural language prompt into structured specification."""
        step = WorkflowStep(
            name="process_prompt",
            description="Extract project specifications from natural language prompt",
        )
        context.steps.append(step)

        try:
            step.status = "running"
            step.start_time = time.time()
            self.progress.update_progress("Processing prompt", 0)

            # Extract project specification from prompt
            if self.prompt_templates is None:
                raise WorkflowError(
                    "Prompt templates not initialized", "process_prompt", context
                )
            spec = await self.prompt_templates.extract_project_spec(context.user_prompt)

            # Store result
            context.project_spec = spec
            step.output = spec
            step.status = "completed"
            step.end_time = time.time()

            self.progress.update_progress("Prompt processed successfully", 1)

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.end_time = time.time()
            raise WorkflowError(
                f"Prompt processing failed: {str(e)}", "process_prompt", context
            ) from e

    async def _step_validate_specification(self, context: WorkflowContext) -> None:
        """Step 2: Validate project specification completeness."""
        step = WorkflowStep(
            name="validate_spec",
            description="Validate project specification completeness and requirements",
        )
        context.steps.append(step)

        try:
            step.status = "running"
            step.start_time = time.time()
            self.progress.update_progress("Validating specification", 2)

            if not context.project_spec:
                raise WorkflowError("No project specification available for validation")

            # Validate specification using SpecificationValidator
            validation_result = self.spec_validator.validate(context.project_spec)

            context.validation_result = validation_result
            step.output = validation_result

            if not validation_result.is_valid:
                # Collect critical errors
                critical_errors = validation_result.errors
                if critical_errors:
                    error_msg = f"Critical validation errors: {len(critical_errors)}"
                    raise WorkflowError(error_msg, "validate_spec", context)

            step.status = "completed"
            step.end_time = time.time()
            self.progress.update_progress("Specification validated", 3)

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.end_time = time.time()
            raise

    async def _step_validate_dependencies(self, context: WorkflowContext) -> None:
        """Step 3: Validate CrewAI dependencies and environment."""
        step = WorkflowStep(
            name="validate_dependencies",
            description="Validate CrewAI CLI and dependencies",
        )
        context.steps.append(step)

        try:
            step.status = "running"
            step.start_time = time.time()
            self.progress.update_progress("Validating dependencies", 4)

            # Check CrewAI CLI availability
            if not self.scaffolder.check_crewai_available():
                raise WorkflowError(
                    "CrewAI CLI not available. Install with: pip install crewai",
                    "validate_dependencies",
                    context,
                )

            # Validate project structure requirements
            if context.project_spec:
                # Use basic validation for project structure
                validation_result = ValidationResult(issues=[], completeness_score=1.0)

                # Check if output directory is valid
                if not context.output_dir.exists():
                    try:
                        context.output_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        # Create a new ValidationResult with the error
                        validation_result = ValidationResult(
                            issues=[
                                ValidationIssue(
                                    severity=IssueSeverity.ERROR,
                                    message=f"Cannot create output directory: {str(e)}",
                                    field_path="output_dir",
                                    suggestions=[
                                        "Choose a different output directory or check permissions"
                                    ],
                                )
                            ],
                            completeness_score=0.0,
                        )

                if not validation_result.is_valid:
                    raise WorkflowError(
                        f"Project structure validation failed: {len(validation_result.errors)} errors",
                        "validate_dependencies",
                        context,
                    )

            step.status = "completed"
            step.end_time = time.time()
            self.progress.update_progress("Dependencies validated", 5)

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.end_time = time.time()
            raise

    async def _step_generate_project(self, context: WorkflowContext) -> None:
        """Step 4: Generate CrewAI project structure."""
        step = WorkflowStep(
            name="generate_project",
            description="Generate CrewAI project structure and files",
        )
        context.steps.append(step)

        try:
            step.status = "running"
            step.start_time = time.time()
            self.progress.update_progress("Generating project", 6)

            # Create the project using CrewAI CLI
            result = self.scaffolder.create_crew(
                context.project_name, context.output_dir
            )

            if not result["success"]:
                raise WorkflowError(
                    f"Project generation failed: {result.get('error', 'Unknown error')}",
                    "generate_project",
                    context,
                )

            context.scaffold_result = result
            step.output = result
            step.status = "completed"
            step.end_time = time.time()

            self.progress.update_progress("Project generated successfully", 7)

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.end_time = time.time()
            raise

    async def _handle_workflow_error(
        self, context: WorkflowContext, error: Exception
    ) -> None:
        """Handle workflow execution errors with cleanup and recovery."""
        # Mark current step as failed
        if context.steps and context.steps[-1].status == "running":
            context.steps[-1].status = "failed"
            context.steps[-1].error = str(error)
            context.steps[-1].end_time = time.time()

        # Update progress with error
        self.progress.finish_progress(f"Workflow failed: {str(error)}")

        # Log error details for debugging
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "current_step": context.steps[-1].name if context.steps else None,
            "completed_steps": len(
                [s for s in context.steps if s.status == "completed"]
            ),
            "total_steps": len(context.steps),
        }

        # Could implement rollback logic here if needed
        # For now, just ensure clean state

    def get_workflow_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Get a summary of the workflow execution."""
        completed_steps = [s for s in context.steps if s.status == "completed"]
        failed_steps = [s for s in context.steps if s.status == "failed"]
        total_time = 0.0

        if context.steps:
            # Get steps with timing information
            timed_steps = [
                s
                for s in context.steps
                if s.start_time is not None and s.end_time is not None
            ]
            if timed_steps:
                start_time = min(s.start_time for s in timed_steps)
                end_time = max(s.end_time for s in timed_steps)
                total_time = end_time - start_time

        return {
            "total_steps": len(context.steps),
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "total_time_seconds": total_time,
            "success": len(failed_steps) == 0 and len(completed_steps) > 0,
            "steps": [
                {
                    "name": s.name,
                    "description": s.description,
                    "status": s.status,
                    "duration": (
                        s.end_time - s.start_time if s.start_time and s.end_time else 0
                    ),
                    "error": s.error,
                }
                for s in context.steps
            ],
        }

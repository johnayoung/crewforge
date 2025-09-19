"""Crew models for CrewForge using Pydantic v2 syntax."""

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from .agent import AgentConfig
from .task import TaskConfig


class CrewConfig(BaseModel):
    """Configuration model for complete CrewAI crews.

    Combines agents and tasks into a functional crew configuration
    with execution parameters and process settings.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Required fields for CrewAI crew definition
    name: str = Field(
        ...,
        description="Unique name for the crew (used for project directory)",
        min_length=1,
        max_length=100,
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of what the crew accomplishes",
        max_length=500,
    )
    agents: List[AgentConfig] = Field(
        ...,
        description="List of agents that belong to this crew",
        min_length=1,
    )
    tasks: List[TaskConfig] = Field(
        ...,
        description="List of tasks to be executed by the crew",
        min_length=1,
    )

    # Optional CrewAI crew parameters
    verbose: bool = Field(
        default=False,
        description="Whether the crew should provide verbose output during execution",
    )
    process: Literal["sequential", "hierarchical"] = Field(
        default="sequential",
        description="Execution process type for the crew",
    )
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum requests per minute for the crew",
        ge=1,
        le=1000,
    )
    language: Optional[str] = Field(
        default="en",
        description="Primary language for the crew operations",
        max_length=10,
    )
    full_output: bool = Field(
        default=False,
        description="Whether to return full output from all tasks",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and clean crew name for filesystem compatibility."""
        if not v or v.isspace():
            raise ValueError("Crew name cannot be empty or only whitespace")

        # Convert to lowercase and replace spaces/special chars with hyphens
        clean_name = v.strip().lower()
        clean_name = re.sub(r"[^a-z0-9\-_]", "-", clean_name)
        clean_name = re.sub(
            r"-+", "-", clean_name
        )  # Replace multiple hyphens with single
        clean_name = clean_name.strip("-")  # Remove leading/trailing hyphens

        if not clean_name:
            raise ValueError(
                "Crew name must contain at least one alphanumeric character"
            )

        return clean_name

    @field_validator("agents")
    @classmethod
    def validate_agents_not_empty(cls, v: List[AgentConfig]) -> List[AgentConfig]:
        """Validate agents list is not empty."""
        if not v:
            raise ValueError("Crew must have at least one agent")
        return v

    @field_validator("tasks")
    @classmethod
    def validate_tasks_not_empty(cls, v: List[TaskConfig]) -> List[TaskConfig]:
        """Validate tasks list is not empty."""
        if not v:
            raise ValueError("Crew must have at least one task")
        return v


class GenerationRequest(BaseModel):
    """Model for user generation requests and prompts.

    Captures user intent and requirements for generating CrewAI projects
    from natural language prompts.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Required fields
    prompt: str = Field(
        ...,
        description="Natural language description of the desired crew functionality",
        min_length=10,
        max_length=2000,
    )

    # Optional fields with defaults
    project_name: Optional[str] = Field(
        default=None,
        description="Desired name for the generated project",
        max_length=100,
    )
    description: Optional[str] = Field(
        default=None,
        description="Additional description or context for the crew",
        max_length=500,
    )
    requirements: List[str] = Field(
        default_factory=list,
        description="Specific tools or capabilities required",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Domain or industry context (e.g., 'marketing', 'research')",
        max_length=50,
    )
    complexity: Optional[Literal["simple", "moderate", "advanced"]] = Field(
        default="moderate",
        description="Desired complexity level of the generated crew",
    )
    output_format: Optional[str] = Field(
        default="markdown",
        description="Preferred output format for generated content",
        max_length=20,
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt is meaningful and not too short."""
        if not v or v.isspace():
            raise ValueError("Prompt cannot be empty or only whitespace")

        clean_prompt = v.strip()
        if len(clean_prompt) < 10:
            raise ValueError("Prompt must be at least 10 characters long")

        # Check for meaningful content (not just repeated characters)
        if len(set(clean_prompt.replace(" ", ""))) < 5:
            raise ValueError("Prompt must contain meaningful, varied content")

        return clean_prompt

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate and clean project name for filesystem compatibility."""
        if v is None:
            return None

        if not v or v.isspace():
            raise ValueError("Project name cannot be empty or only whitespace")

        # Convert to lowercase and replace spaces/special chars with hyphens
        clean_name = v.strip().lower()
        clean_name = re.sub(r"[^a-z0-9\-_]", "-", clean_name)
        clean_name = re.sub(
            r"-+", "-", clean_name
        )  # Replace multiple hyphens with single
        clean_name = clean_name.strip("-")  # Remove leading/trailing hyphens

        if not clean_name:
            raise ValueError(
                "Project name must contain at least one alphanumeric character"
            )

        return clean_name

    @field_validator("requirements")
    @classmethod
    def validate_requirements(cls, v: List[str]) -> List[str]:
        """Validate and clean requirements list."""
        if not v:
            return []

        cleaned_requirements = []
        for req in v:
            if req and not req.isspace():
                clean_req = req.strip().lower().replace(" ", "_")
                if clean_req not in cleaned_requirements:
                    cleaned_requirements.append(clean_req)

        return cleaned_requirements


class ValidationResult(BaseModel):
    """Model for project validation results.

    Contains comprehensive validation information about generated
    CrewAI projects including syntax, compliance, and functionality.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Required fields
    is_valid: bool = Field(
        ...,
        description="Overall validation status - True if project is valid",
    )
    project_path: str = Field(
        ...,
        description="Path to the validated project directory",
        min_length=1,
    )

    # Detailed validation results
    syntax_valid: Optional[bool] = Field(
        default=None,
        description="Whether all Python files have valid syntax",
    )
    crewai_compliant: Optional[bool] = Field(
        default=None,
        description="Whether the project follows CrewAI patterns and conventions",
    )
    functional: Optional[bool] = Field(
        default=None,
        description="Whether the crew can be instantiated and basic functionality works",
    )

    # Error and warning information
    errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors found",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warnings found",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="List of suggestions for improvement",
    )

    # Additional metadata
    validation_timestamp: Optional[str] = Field(
        default=None,
        description="ISO timestamp when validation was performed",
    )
    files_checked: List[str] = Field(
        default_factory=list,
        description="List of files that were validated",
    )

    @computed_field
    @property
    def summary(self) -> str:
        """Generate a human-readable validation summary."""
        if self.is_valid:
            return f"✅ Project validation passed successfully. All checks completed without critical errors."
        else:
            error_count = len(self.errors)
            warning_count = len(self.warnings)

            summary_parts = [
                f"❌ Project validation failed with {error_count} error(s)"
            ]

            if warning_count > 0:
                summary_parts.append(f"and {warning_count} warning(s)")

            if self.syntax_valid is False:
                summary_parts.append("Syntax errors detected.")
            if self.crewai_compliant is False:
                summary_parts.append("CrewAI compliance issues found.")
            if self.functional is False:
                summary_parts.append("Functionality issues detected.")

            return " ".join(summary_parts)

    @computed_field
    @property
    def error_count(self) -> int:
        """Get the total number of errors."""
        return len(self.errors)

    @computed_field
    @property
    def warning_count(self) -> int:
        """Get the total number of warnings."""
        return len(self.warnings)

    @field_validator("project_path")
    @classmethod
    def validate_project_path(cls, v: str) -> str:
        """Validate project path is not empty."""
        if not v or v.isspace():
            raise ValueError("Project path cannot be empty or only whitespace")
        return v.strip()


class CrewGenerationResult(BaseModel):
    """Model for complete crew generation results.

    Contains the generated crew configuration along with metadata
    about the generation process and validation results.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Core results
    crew_config: CrewConfig = Field(
        ...,
        description="The generated crew configuration",
    )
    generation_request: GenerationRequest = Field(
        ...,
        description="The original user request that generated this crew",
    )
    project_path: Optional[str] = Field(
        default=None,
        description="Path where the project was created (if applicable)",
    )

    # Generation metadata
    generation_timestamp: Optional[str] = Field(
        default=None,
        description="ISO timestamp when generation was completed",
    )
    generation_time_seconds: Optional[float] = Field(
        default=None,
        description="Time taken to generate the crew in seconds",
        ge=0,
    )
    llm_model_used: Optional[str] = Field(
        default=None,
        description="LLM model that was used for generation",
        max_length=100,
    )

    # Validation results
    validation_result: Optional[ValidationResult] = Field(
        default=None,
        description="Validation results if project was validated",
    )

    # Quality metrics
    agent_count: int = Field(
        default=0,
        description="Number of agents in the generated crew",
        ge=0,
    )
    task_count: int = Field(
        default=0,
        description="Number of tasks in the generated crew",
        ge=0,
    )

    def model_post_init(self, __context: Any) -> None:
        """Post-init to set computed fields."""
        # Set counts from crew config if not provided
        if self.agent_count == 0:
            self.agent_count = len(self.crew_config.agents)
        if self.task_count == 0:
            self.task_count = len(self.crew_config.tasks)

    @computed_field
    @property
    def is_valid(self) -> bool:
        """Whether the generated crew passed validation (if validated)."""
        return self.validation_result.is_valid if self.validation_result else True

    @computed_field
    @property
    def summary(self) -> str:
        """Generate a human-readable generation summary."""
        summary_parts = [
            f"Generated '{self.crew_config.name}' crew with {self.agent_count} agent(s) and {self.task_count} task(s)"
        ]

        if self.generation_time_seconds:
            summary_parts.append(f"in {self.generation_time_seconds:.1f} seconds")

        if self.validation_result:
            if self.validation_result.is_valid:
                summary_parts.append("✅ Validation passed")
            else:
                summary_parts.append(
                    f"❌ Validation failed ({self.validation_result.error_count} errors)"
                )

        return " ".join(summary_parts)

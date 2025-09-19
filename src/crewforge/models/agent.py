"""Agent models for CrewForge using Pydantic v2 syntax."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentConfig(BaseModel):
    """Configuration model for CrewAI agents.

    Defines the structure and validation for agent definitions including
    role, goal, backstory, and optional CrewAI-specific parameters.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Required fields for CrewAI agent definition
    role: str = Field(
        ...,
        description="The role of the agent (e.g., 'Content Researcher', 'Data Analyst')",
        min_length=1,
        max_length=100,
    )
    goal: str = Field(
        ...,
        description="The primary goal or objective of the agent",
        min_length=1,
        max_length=500,
    )
    backstory: str = Field(
        ...,
        description="The background story and expertise of the agent",
        min_length=1,
        max_length=1000,
    )

    # Optional CrewAI agent parameters
    verbose: bool = Field(
        default=False,
        description="Whether the agent should provide verbose output during execution",
    )
    allow_delegation: bool = Field(
        default=True,
        description="Whether the agent can delegate tasks to other agents",
    )
    max_iter: Optional[int] = Field(
        default=None,
        description="Maximum number of iterations for the agent",
        ge=1,
        le=100,
    )
    max_execution_time: Optional[int] = Field(
        default=None,
        description="Maximum execution time in seconds for the agent",
        ge=1,
        le=3600,  # 1 hour max
    )
    tools: Optional[List[str]] = Field(
        default=None,
        description="List of tool names available to this agent",
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate agent role for CrewAI compliance."""
        if not v or v.isspace():
            raise ValueError("Role cannot be empty or only whitespace")
        if len(v.strip()) == 0:
            raise ValueError("Role cannot be empty after stripping whitespace")
        return v.strip()

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, v: str) -> str:
        """Validate agent goal for CrewAI compliance."""
        if not v or v.isspace():
            raise ValueError("Goal cannot be empty or only whitespace")
        if len(v.strip()) == 0:
            raise ValueError("Goal cannot be empty after stripping whitespace")
        return v.strip()

    @field_validator("backstory")
    @classmethod
    def validate_backstory(cls, v: str) -> str:
        """Validate agent backstory for CrewAI compliance."""
        if not v or v.isspace():
            raise ValueError("Backstory cannot be empty or only whitespace")
        if len(v.strip()) == 0:
            raise ValueError("Backstory cannot be empty after stripping whitespace")
        return v.strip()


class AgentTemplate(BaseModel):
    """Template model for generating agent configurations.

    Provides Jinja2-compatible template patterns for generating
    agent definitions with variable substitution.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    name: str = Field(
        ...,
        description="Unique name for this agent template",
        min_length=1,
        max_length=50,
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of what this template generates",
        max_length=200,
    )

    # Jinja2 template patterns
    role_pattern: str = Field(
        ...,
        description="Jinja2 template pattern for agent role (e.g., '{{domain}} Researcher')",
        min_length=1,
        max_length=100,
    )
    goal_pattern: str = Field(
        ...,
        description="Jinja2 template pattern for agent goal",
        min_length=1,
        max_length=500,
    )
    backstory_pattern: str = Field(
        ...,
        description="Jinja2 template pattern for agent backstory",
        min_length=1,
        max_length=1000,
    )

    # Optional template metadata
    variables: Optional[List[str]] = Field(
        default=None,
        description="List of template variables that can be substituted",
    )
    category: Optional[str] = Field(
        default=None,
        description="Category of agent this template generates (e.g., 'research', 'analysis')",
        max_length=50,
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tags for organizing and searching templates",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate template name is not empty and properly formatted."""
        if not v or v.isspace():
            raise ValueError("Template name cannot be empty or only whitespace")
        # Convert to lowercase and replace spaces with underscores for consistency
        clean_name = v.strip().lower().replace(" ", "_")
        return clean_name

    def render_agent_config(self, variables: Dict[str, Any]) -> AgentConfig:
        """Render this template into an AgentConfig using provided variables.

        Args:
            variables: Dictionary of template variables to substitute

        Returns:
            AgentConfig with rendered values

        Raises:
            ValueError: If required template variables are missing
        """
        # This is a placeholder implementation - full Jinja2 rendering
        # will be implemented in the template engine module
        try:
            # Simple string substitution for now
            role = self.role_pattern
            goal = self.goal_pattern
            backstory = self.backstory_pattern

            # Replace variables in templates
            for key, value in variables.items():
                placeholder = "{{" + key + "}}"
                role = role.replace(placeholder, str(value))
                goal = goal.replace(placeholder, str(value))
                backstory = backstory.replace(placeholder, str(value))

            return AgentConfig(
                role=role,
                goal=goal,
                backstory=backstory,
            )
        except Exception as e:
            raise ValueError(f"Failed to render agent template: {e}")

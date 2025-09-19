"""Task models for CrewForge using Pydantic v2 syntax."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .agent import AgentConfig


class TaskConfig(BaseModel):
    """Configuration model for CrewAI tasks.

    Defines the structure and validation for task definitions including
    description, expected output, agent assignment, and optional parameters.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # Required fields for CrewAI task definition
    description: str = Field(
        ...,
        description="Clear description of what the task should accomplish",
        min_length=1,
        max_length=1000,
    )
    expected_output: str = Field(
        ...,
        description="Detailed description of the expected task output format and content",
        min_length=1,
        max_length=1000,
    )
    agent: str = Field(
        ...,
        description="ID or name of the agent responsible for executing this task",
        min_length=1,
        max_length=100,
    )

    # Optional CrewAI task parameters
    async_execution: bool = Field(
        default=False,
        description="Whether this task should be executed asynchronously",
    )
    context: Optional[List[str]] = Field(
        default=None,
        description="List of context items or previous task outputs to use",
    )
    tools: Optional[List[str]] = Field(
        default=None,
        description="List of tool names required for this task",
    )
    output_file: Optional[str] = Field(
        default=None,
        description="File path where the task output should be saved",
        max_length=255,
    )
    callback: Optional[str] = Field(
        default=None,
        description="Callback function to execute after task completion",
        max_length=100,
    )

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate task description for CrewAI compliance."""
        if not v or v.isspace():
            raise ValueError("Description cannot be empty or only whitespace")
        if len(v.strip()) == 0:
            raise ValueError("Description cannot be empty after stripping whitespace")
        return v.strip()

    @field_validator("expected_output")
    @classmethod
    def validate_expected_output(cls, v: str) -> str:
        """Validate expected output for CrewAI compliance."""
        if not v or v.isspace():
            raise ValueError("Expected output cannot be empty or only whitespace")
        if len(v.strip()) == 0:
            raise ValueError(
                "Expected output cannot be empty after stripping whitespace"
            )
        return v.strip()

    @field_validator("agent")
    @classmethod
    def validate_agent(cls, v: str) -> str:
        """Validate agent assignment for CrewAI compliance."""
        if not v or v.isspace():
            raise ValueError("Agent cannot be empty or only whitespace")
        if len(v.strip()) == 0:
            raise ValueError("Agent cannot be empty after stripping whitespace")
        return v.strip()


class TaskDependency(BaseModel):
    """Model for defining task dependencies and execution sequencing.

    Enables complex workflow orchestration by defining how tasks
    depend on each other and execution order constraints.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    task_id: str = Field(
        ...,
        description="Unique identifier of the task that has dependencies",
        min_length=1,
        max_length=100,
    )
    depends_on: List[str] = Field(
        ...,
        description="List of task IDs that must complete before this task can start",
        min_length=1,
    )
    dependency_type: Literal["sequential", "parallel", "conditional"] = Field(
        default="sequential",
        description="Type of dependency relationship",
    )
    condition: Optional[str] = Field(
        default=None,
        description="Condition expression for conditional dependencies",
        max_length=200,
    )

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        """Validate task ID format."""
        if not v or v.isspace():
            raise ValueError("Task ID cannot be empty or only whitespace")
        # Convert to lowercase and replace spaces/special chars for consistency
        clean_id = v.strip().lower().replace(" ", "_").replace("-", "_")
        return clean_id

    @field_validator("depends_on")
    @classmethod
    def validate_depends_on(cls, v: List[str]) -> List[str]:
        """Validate dependency list."""
        if not v:
            raise ValueError("Dependencies list cannot be empty")

        # Clean up dependency IDs for consistency
        clean_deps = []
        for dep_id in v:
            if not dep_id or dep_id.isspace():
                raise ValueError("Dependency ID cannot be empty or only whitespace")
            clean_deps.append(
                dep_id.strip().lower().replace(" ", "_").replace("-", "_")
            )

        # Check for duplicates
        if len(clean_deps) != len(set(clean_deps)):
            raise ValueError("Dependencies list cannot contain duplicates")

        return clean_deps


def validate_agent_task_compatibility(agent: AgentConfig, task: TaskConfig) -> bool:
    """Validate that an agent is compatible with a task.

    Checks if the agent has the necessary tools and capabilities
    to execute the given task successfully.

    Args:
        agent: AgentConfig instance to validate
        task: TaskConfig instance to validate against

    Returns:
        bool: True if agent and task are compatible, False otherwise
    """
    # If task doesn't specify tools, any agent can handle it
    if not task.tools:
        return True

    # If agent doesn't specify tools, it can't handle tool-specific tasks
    if not agent.tools:
        return False

    # Check if agent has all required tools
    agent_tools = set(agent.tools)
    required_tools = set(task.tools)

    # Agent must have all tools required by the task
    missing_tools = required_tools - agent_tools
    return len(missing_tools) == 0


def validate_task_dependency_graph(
    dependencies: List[TaskDependency],
) -> Dict[str, Any]:
    """Validate a task dependency graph for cycles and consistency.

    Args:
        dependencies: List of TaskDependency instances

    Returns:
        Dict with validation results and any detected issues
    """
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "task_count": 0,
        "dependency_count": len(dependencies),
    }

    if not dependencies:
        return validation_result

    # Build dependency graph
    task_deps = {}
    all_tasks = set()

    for dep in dependencies:
        task_deps[dep.task_id] = dep.depends_on
        all_tasks.add(dep.task_id)
        all_tasks.update(dep.depends_on)

    validation_result["task_count"] = len(all_tasks)

    # Check for circular dependencies using depth-first search
    def has_cycle(task_id: str, visited: set, recursion_stack: set) -> bool:
        """Check for cycles starting from a specific task."""
        if task_id in recursion_stack:
            return True
        if task_id in visited:
            return False

        visited.add(task_id)
        recursion_stack.add(task_id)

        # Check dependencies
        if task_id in task_deps:
            for dep_task in task_deps[task_id]:
                if has_cycle(dep_task, visited, recursion_stack):
                    return True

        recursion_stack.remove(task_id)
        return False

    # Check for cycles
    visited = set()
    for task_id in all_tasks:
        if task_id not in visited:
            if has_cycle(task_id, visited, set()):
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Circular dependency detected involving task: {task_id}"
                )

    # Check for orphaned tasks (tasks referenced as dependencies but not defined)
    defined_tasks = set(task_deps.keys())
    referenced_tasks = set()
    for deps in task_deps.values():
        referenced_tasks.update(deps)

    orphaned_tasks = referenced_tasks - defined_tasks
    if orphaned_tasks:
        validation_result["warnings"].append(
            f"Tasks referenced but not defined: {list(orphaned_tasks)}"
        )

    return validation_result

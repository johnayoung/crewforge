"""Data models for CrewForge agents, tasks, and crews.

This module provides Pydantic v2 models for defining and validating
CrewAI project configurations including agents, tasks, crews, and
generation requests.
"""

# Agent models
from .agent import AgentConfig, AgentTemplate

# Task models
from .task import (
    TaskConfig,
    TaskDependency,
    validate_agent_task_compatibility,
    validate_task_dependency_graph,
)

# Crew models
from .crew import (
    CrewConfig,
    CrewGenerationResult,
    GenerationRequest,
    ValidationResult,
)

# Export all public models and functions
__all__ = [
    # Agent models
    "AgentConfig",
    "AgentTemplate",
    # Task models
    "TaskConfig",
    "TaskDependency",
    "validate_agent_task_compatibility",
    "validate_task_dependency_graph",
    # Crew models
    "CrewConfig",
    "CrewGenerationResult",
    "GenerationRequest",
    "ValidationResult",
]

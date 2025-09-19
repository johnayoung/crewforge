"""
Specification validation and completeness checking system for CrewForge.

This module provides comprehensive validation for parsed project specifications,
ensuring they are complete, valid, and suitable for CrewAI project generation.
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


class IssueSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    def __lt__(self, other):
        """Enable comparison of severity levels."""
        if not isinstance(other, IssueSeverity):
            return NotImplemented
        order = {
            IssueSeverity.INFO: 0,
            IssueSeverity.WARNING: 1,
            IssueSeverity.ERROR: 2,
        }
        return order[self] < order[other]

    def __gt__(self, other):
        """Enable comparison of severity levels."""
        if not isinstance(other, IssueSeverity):
            return NotImplemented
        order = {
            IssueSeverity.INFO: 0,
            IssueSeverity.WARNING: 1,
            IssueSeverity.ERROR: 2,
        }
        return order[self] > order[other]


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a specification."""

    severity: IssueSeverity
    message: str
    field_path: str

    def __str__(self) -> str:
        """String representation of validation issue."""
        return f"{self.severity.value.upper()}: {self.message} (at {self.field_path})"


@dataclass
class ValidationResult:
    """Results of specification validation."""

    issues: List[ValidationIssue] = field(default_factory=list)
    completeness_score: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Check if specification is valid (no errors)."""
        return not any(issue.severity == IssueSeverity.ERROR for issue in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [issue for issue in self.issues if issue.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [
            issue for issue in self.issues if issue.severity == IssueSeverity.WARNING
        ]

    @property
    def info_messages(self) -> List[ValidationIssue]:
        """Get all info-level issues."""
        return [issue for issue in self.issues if issue.severity == IssueSeverity.INFO]

    def __str__(self) -> str:
        """String representation of validation result."""
        lines = []
        lines.append(f"Validation {'PASSED' if self.is_valid else 'FAILED'}")
        lines.append(f"Completeness Score: {self.completeness_score:.1%}")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  - {error}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        if self.info_messages:
            lines.append(f"\nInfo ({len(self.info_messages)}):")
            for info in self.info_messages:
                lines.append(f"  - {info}")

        return "\n".join(lines)


class ValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(
        self, message: str, validation_result: Optional[ValidationResult] = None
    ):
        super().__init__(message)
        self.validation_result = validation_result


class SpecificationValidator:
    """
    Validates CrewAI project specifications for completeness and correctness.

    Provides comprehensive validation including:
    - Required field validation
    - Data type validation
    - Business logic validation (agent-task relationships, etc.)
    - Ambiguity detection for better user guidance
    - Completeness scoring
    - Warning generation for best practices
    """

    # Required fields for each component
    REQUIRED_SPEC_FIELDS = {
        "project_name",
        "project_description",
        "agents",
        "tasks",
        "dependencies",
    }

    REQUIRED_AGENT_FIELDS = {"role", "goal", "backstory", "tools"}

    REQUIRED_TASK_FIELDS = {"description", "expected_output", "agent"}

    # Validation thresholds
    MAX_FIELD_LENGTH = 500
    MIN_DESCRIPTION_LENGTH = 10
    MAX_AGENTS = 10
    MAX_TASKS = 20

    # Valid project name pattern (kebab-case)
    PROJECT_NAME_PATTERN = re.compile(r"^[a-z]+(-[a-z0-9]+)*$")

    # Ambiguity detection patterns and rules
    VAGUE_ROLE_PATTERNS = {
        "agent",
        "assistant",
        "helper",
        "worker",
        "bot",
        "ai",
        "system",
        "generic",
        "general",
        "basic",
        "simple",
        "default",
        "main",
    }

    VAGUE_GOAL_KEYWORDS = {
        "help",
        "assist",
        "do",
        "work",
        "handle",
        "manage",
        "process",
        "deal",
        "take care",
        "support",
        "execute",
        "perform",
        "run",
    }

    VAGUE_TASK_KEYWORDS = {
        "task",
        "work",
        "job",
        "activity",
        "action",
        "operation",
        "step",
        "process",
        "handle",
        "deal",
        "manage",
        "execute",
        "perform",
        "run",
    }

    VAGUE_OUTPUT_KEYWORDS = {
        "output",
        "result",
        "outcome",
        "response",
        "answer",
        "data",
        "information",
        "content",
        "report",
        "summary",
        "analysis",
    }

    # Tool-role mapping for mismatch detection
    ROLE_TOOL_MAPPINGS = {
        "writer": [
            "document_writer",
            "text_editor",
            "grammar_checker",
            "content_generator",
        ],
        "content": [
            "document_writer",
            "text_editor",
            "grammar_checker",
            "content_generator",
        ],
        "research": ["web_search", "web_scraper", "research_tool", "data_collector"],
        "analyst": [
            "data_analyzer",
            "statistical_tool",
            "visualization_tool",
            "report_generator",
        ],
        "data": [
            "data_analyzer",
            "database_query",
            "statistical_tool",
            "visualization_tool",
        ],
        "social": ["social_media_tool", "engagement_tracker", "content_scheduler"],
        "marketing": [
            "social_media_tool",
            "analytics_tool",
            "campaign_manager",
            "seo_tool",
        ],
        "design": [
            "image_editor",
            "graphic_tool",
            "design_software",
            "visualization_tool",
        ],
        "developer": ["code_editor", "debugger", "compiler", "version_control"],
        "programmer": ["code_editor", "debugger", "compiler", "version_control"],
    }

    def __init__(self):
        """Initialize the specification validator."""
        self.issues: List[ValidationIssue] = []

    def validate(self, spec: Any) -> ValidationResult:
        """
        Validate a project specification.

        Args:
            spec: Project specification to validate

        Returns:
            ValidationResult with issues and completeness score
        """
        self.issues = []  # Reset issues for new validation

        # Basic type and structure validation
        if not self._validate_basic_structure(spec):
            return self._create_result()

        # Cast to dict for type safety
        spec_dict = spec

        # Validate required fields
        self._validate_required_fields(spec_dict)

        # Validate individual components
        self._validate_project_metadata(spec_dict)
        self._validate_agents(spec_dict.get("agents", []))
        self._validate_tasks(spec_dict.get("tasks", []), spec_dict.get("agents", []))
        self._validate_dependencies(spec_dict.get("dependencies", []))

        # Cross-component validation
        self._validate_agent_task_relationships(spec_dict)

        # Ambiguity detection
        self._detect_ambiguities(spec_dict)

        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(spec_dict)

        return self._create_result(completeness_score)

    def _validate_basic_structure(self, spec: Any) -> bool:
        """Validate basic structure and type of specification."""
        if spec is None:
            self._add_error("Specification cannot be null", "root")
            return False

        if not isinstance(spec, dict):
            self._add_error("Specification must be a dictionary", "root")
            return False

        return True

    def _validate_required_fields(self, spec: Dict[str, Any]) -> None:
        """Validate that all required top-level fields are present."""
        for field in self.REQUIRED_SPEC_FIELDS:
            if field not in spec:
                self._add_error(f"Required field '{field}' is missing", field)
            elif spec[field] is None:
                self._add_error(f"Field '{field}' cannot be null", field)

    def _validate_project_metadata(self, spec: Dict[str, Any]) -> None:
        """Validate project name and description."""
        # Validate project name
        project_name = spec.get("project_name")
        if project_name:
            if not isinstance(project_name, str):
                self._add_error("Project name must be a string", "project_name")
            elif not project_name.strip():
                self._add_error("Project name cannot be empty", "project_name")
            elif not self.PROJECT_NAME_PATTERN.match(project_name):
                self._add_error(
                    "Project name must be kebab-case (lowercase letters and hyphens only, "
                    "starting with a letter, no consecutive hyphens)",
                    "project_name",
                )
        elif project_name == "":
            # Handle empty string specifically
            self._add_error("Project name cannot be empty", "project_name")

        # Validate project description
        description = spec.get("project_description")
        if description is not None:
            if not isinstance(description, str):
                self._add_error(
                    "Project description must be a string", "project_description"
                )
            elif not description.strip():
                self._add_error(
                    "Project description cannot be empty", "project_description"
                )
            elif len(description.strip()) < self.MIN_DESCRIPTION_LENGTH:
                self._add_warning(
                    f"Project description is very short (less than {self.MIN_DESCRIPTION_LENGTH} characters)",
                    "project_description",
                )

    def _validate_agents(self, agents: Any) -> None:
        """Validate agents list and individual agent specifications."""
        if not isinstance(agents, list):
            self._add_error("Agents must be a list", "agents")
            return

        if len(agents) == 0:
            self._add_error("At least one agent is required", "agents")
            return

        if len(agents) > self.MAX_AGENTS:
            self._add_warning(
                f"Large number of agents ({len(agents)}), consider simplifying",
                "agents",
            )

        # Track agent roles for duplicate detection
        agent_roles: Set[str] = set()

        for i, agent in enumerate(agents):
            agent_path = f"agents[{i}]"

            if not isinstance(agent, dict):
                self._add_error(f"Agent must be a dictionary", agent_path)
                continue

            # Validate required fields
            for field in self.REQUIRED_AGENT_FIELDS:
                if field not in agent:
                    self._add_error(
                        f"Agent missing required field '{field}'",
                        f"{agent_path}.{field}",
                    )
                elif agent[field] is None:
                    self._add_error(
                        f"Agent field '{field}' cannot be null", f"{agent_path}.{field}"
                    )

            # Validate field types and content
            self._validate_agent_fields(agent, agent_path)

            # Check for duplicate roles
            role = agent.get("role")
            if role and isinstance(role, str):
                if role in agent_roles:
                    self._add_warning(
                        f"Duplicate agent role '{role}'", f"{agent_path}.role"
                    )
                agent_roles.add(role)

    def _validate_agent_fields(self, agent: Dict[str, Any], agent_path: str) -> None:
        """Validate individual agent field types and content."""
        # Role validation
        role = agent.get("role")
        if role is not None:
            if not isinstance(role, str):
                self._add_error("Agent role must be a string", f"{agent_path}.role")
            elif not role.strip():
                self._add_error("Agent role cannot be empty", f"{agent_path}.role")

        # Goal validation
        goal = agent.get("goal")
        if goal is not None:
            if not isinstance(goal, str):
                self._add_error("Agent goal must be a string", f"{agent_path}.goal")
            elif not goal.strip():
                self._add_error("Agent goal cannot be empty", f"{agent_path}.goal")

        # Backstory validation
        backstory = agent.get("backstory")
        if backstory is not None:
            if not isinstance(backstory, str):
                self._add_error(
                    "Agent backstory must be a string", f"{agent_path}.backstory"
                )
            elif not backstory.strip():
                self._add_error(
                    "Agent backstory cannot be empty", f"{agent_path}.backstory"
                )
            elif len(backstory) > self.MAX_FIELD_LENGTH:
                self._add_warning(
                    f"Agent backstory is very long ({len(backstory)} characters)",
                    f"{agent_path}.backstory",
                )

        # Tools validation
        tools = agent.get("tools")
        if tools is not None:
            if not isinstance(tools, list):
                self._add_error("Agent tools must be a list", f"{agent_path}.tools")
            elif len(tools) == 0:
                self._add_warning("Agent has no tools specified", f"{agent_path}.tools")
            else:
                for j, tool in enumerate(tools):
                    if not isinstance(tool, str):
                        self._add_error(
                            f"Tool must be a string", f"{agent_path}.tools[{j}]"
                        )
                    elif not tool.strip():
                        self._add_error(
                            f"Tool name cannot be empty", f"{agent_path}.tools[{j}]"
                        )

    def _detect_ambiguities(self, spec: Dict[str, Any]) -> None:
        """Detect various types of ambiguity in the specification."""
        self._detect_vague_project_description(spec)
        self._detect_vague_agent_roles(spec.get("agents", []))
        self._detect_vague_agent_goals(spec.get("agents", []))
        self._detect_vague_agent_backstories(spec.get("agents", []))
        self._detect_tool_role_mismatches(spec.get("agents", []))
        self._detect_vague_task_descriptions(spec.get("tasks", []))
        self._detect_vague_task_outputs(spec.get("tasks", []))

    def _detect_vague_project_description(self, spec: Dict[str, Any]) -> None:
        """Detect vague project descriptions."""
        description = spec.get("project_description", "")
        if not isinstance(description, str):
            return

        description_lower = description.lower().strip()

        # Check for very short descriptions
        if len(description_lower) < 15:
            self._add_warning(
                "Project description is very short and may be too vague",
                "project_description",
            )
            return

        # Check for vague patterns
        vague_patterns = [
            "a project",
            "some project",
            "project for",
            "simple project",
            "basic project",
            "general project",
            "new project",
            "project to",
        ]

        if any(pattern in description_lower for pattern in vague_patterns):
            self._add_warning(
                "Project description appears vague - consider adding more specific details about purpose and scope",
                "project_description",
            )

    def _detect_vague_agent_roles(self, agents: List[Dict[str, Any]]) -> None:
        """Detect vague agent roles."""
        if not isinstance(agents, list):
            return

        for i, agent in enumerate(agents):
            if not isinstance(agent, dict):
                continue

            role = agent.get("role", "")
            if not isinstance(role, str):
                continue

            role_lower = role.lower().strip()

            # Check for vague role names
            if role_lower in self.VAGUE_ROLE_PATTERNS:
                self._add_warning(
                    f"Agent role '{role}' is vague - consider a more specific role like 'Research Analyst' or 'Content Writer'",
                    f"agents[{i}].role",
                )

    def _detect_vague_agent_goals(self, agents: List[Dict[str, Any]]) -> None:
        """Detect vague agent goals."""
        if not isinstance(agents, list):
            return

        for i, agent in enumerate(agents):
            if not isinstance(agent, dict):
                continue

            goal = agent.get("goal", "")
            if not isinstance(goal, str):
                continue

            goal_lower = goal.lower().strip()

            # Check for very short goals
            if len(goal_lower) < 10:
                self._add_warning(
                    f"Agent goal is very short and may be too vague",
                    f"agents[{i}].goal",
                )
                continue

            # Check for vague goal patterns
            if any(keyword in goal_lower for keyword in self.VAGUE_GOAL_KEYWORDS):
                # Additional check - if it's just a single vague keyword with minimal context
                words = goal_lower.split()
                if len(words) <= 3 and any(
                    keyword in words for keyword in self.VAGUE_GOAL_KEYWORDS
                ):
                    self._add_warning(
                        f"Agent goal appears vague - consider adding specific details about what this agent should accomplish",
                        f"agents[{i}].goal",
                    )

    def _detect_vague_agent_backstories(self, agents: List[Dict[str, Any]]) -> None:
        """Detect vague agent backstories."""
        if not isinstance(agents, list):
            return

        for i, agent in enumerate(agents):
            if not isinstance(agent, dict):
                continue

            backstory = agent.get("backstory", "")
            if not isinstance(backstory, str):
                continue

            backstory_lower = backstory.lower().strip()

            # Check for very short backstories
            if len(backstory_lower) < 15:
                self._add_warning(
                    f"Agent backstory is very short and may lack sufficient detail",
                    f"agents[{i}].backstory",
                )

    def _detect_tool_role_mismatches(self, agents: List[Dict[str, Any]]) -> None:
        """Detect potential mismatches between agent roles and their assigned tools."""
        if not isinstance(agents, list):
            return

        for i, agent in enumerate(agents):
            if not isinstance(agent, dict):
                continue

            role = agent.get("role", "")
            tools = agent.get("tools", [])

            if not isinstance(role, str) or not isinstance(tools, list):
                continue

            role_lower = role.lower()

            # Find potential role keywords
            role_keywords = []
            for keyword in self.ROLE_TOOL_MAPPINGS.keys():
                if keyword in role_lower:
                    role_keywords.append(keyword)

            if not role_keywords:
                continue  # No recognizable role patterns

            # Check if any tools match the role
            expected_tools = set()
            for keyword in role_keywords:
                expected_tools.update(self.ROLE_TOOL_MAPPINGS[keyword])

            tool_names = [tool.lower() for tool in tools if isinstance(tool, str)]

            # Check for any overlap
            has_matching_tools = False
            for tool in tool_names:
                if any(expected_tool in tool for expected_tool in expected_tools):
                    has_matching_tools = True
                    break

            if not has_matching_tools and len(tools) > 0:
                self._add_warning(
                    f"Agent role '{role}' may not match assigned tools - consider tools more aligned with the role",
                    f"agents[{i}].tools",
                )

    def _detect_vague_task_descriptions(self, tasks: List[Dict[str, Any]]) -> None:
        """Detect vague task descriptions."""
        if not isinstance(tasks, list):
            return

        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue

            description = task.get("description", "")
            if not isinstance(description, str):
                continue

            description_lower = description.lower().strip()

            # Check for very short descriptions
            if len(description_lower) < 10:
                self._add_warning(
                    f"Task description is very short and may be too vague",
                    f"tasks[{i}].description",
                )
                continue

            # Check for vague task patterns
            words = description_lower.split()
            if len(words) <= 3 and any(
                keyword in words for keyword in self.VAGUE_TASK_KEYWORDS
            ):
                self._add_warning(
                    f"Task description appears vague - consider adding specific details about what should be accomplished",
                    f"tasks[{i}].description",
                )

    def _detect_vague_task_outputs(self, tasks: List[Dict[str, Any]]) -> None:
        """Detect vague task expected outputs."""
        if not isinstance(tasks, list):
            return

        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue

            expected_output = task.get("expected_output", "")
            if not isinstance(expected_output, str):
                continue

            output_lower = expected_output.lower().strip()

            # Check for very short outputs
            if len(output_lower) < 8:
                self._add_warning(
                    f"Task expected output is very short and may be too vague",
                    f"tasks[{i}].expected_output",
                )
                continue

            # Check for vague output patterns
            words = output_lower.split()
            if len(words) <= 2 and any(
                keyword in words for keyword in self.VAGUE_OUTPUT_KEYWORDS
            ):
                self._add_warning(
                    f"Task expected output appears vague - consider describing the specific format and content expected",
                    f"tasks[{i}].expected_output",
                )

    def _validate_tasks(self, tasks: Any, agents: List[Dict[str, Any]]) -> None:
        """Validate tasks list and individual task specifications."""
        if not isinstance(tasks, list):
            self._add_error("Tasks must be a list", "tasks")
            return

        if len(tasks) == 0:
            self._add_error("At least one task is required", "tasks")
            return

        if len(tasks) > self.MAX_TASKS:
            self._add_warning(
                f"Large number of tasks ({len(tasks)}), consider grouping", "tasks"
            )

        # Get agent roles for validation
        agent_roles = {
            agent.get("role")
            for agent in agents
            if isinstance(agent, dict) and agent.get("role") is not None
        }
        agent_roles = {
            role for role in agent_roles if isinstance(role, str)
        }  # Filter out None values

        for i, task in enumerate(tasks):
            task_path = f"tasks[{i}]"

            if not isinstance(task, dict):
                self._add_error("Task must be a dictionary", task_path)
                continue

            # Validate required fields
            for field in self.REQUIRED_TASK_FIELDS:
                if field not in task:
                    self._add_error(
                        f"Task missing required field '{field}'", f"{task_path}.{field}"
                    )
                elif task[field] is None:
                    self._add_error(
                        f"Task field '{field}' cannot be null", f"{task_path}.{field}"
                    )

            # Validate field types and content
            self._validate_task_fields(task, task_path, agent_roles)

    def _validate_task_fields(
        self, task: Dict[str, Any], task_path: str, agent_roles: Set[str]
    ) -> None:
        """Validate individual task field types and content."""
        # Description validation
        description = task.get("description")
        if description is not None:
            if not isinstance(description, str):
                self._add_error(
                    "Task description must be a string", f"{task_path}.description"
                )
            elif not description.strip():
                self._add_error(
                    "Task description cannot be empty", f"{task_path}.description"
                )

        # Expected output validation
        expected_output = task.get("expected_output")
        if expected_output is not None:
            if not isinstance(expected_output, str):
                self._add_error(
                    "Task expected_output must be a string",
                    f"{task_path}.expected_output",
                )
            elif not expected_output.strip():
                self._add_error(
                    "Task expected_output cannot be empty",
                    f"{task_path}.expected_output",
                )

        # Agent reference validation
        agent = task.get("agent")
        if agent is not None:
            if not isinstance(agent, str):
                self._add_error("Task agent must be a string", f"{task_path}.agent")
            elif not agent.strip():
                self._add_error("Task agent cannot be empty", f"{task_path}.agent")
            elif agent not in agent_roles:
                self._add_error(
                    f"Task references non-existent agent role '{agent}'. Available agents: {sorted(agent_roles)}",
                    f"{task_path}.agent",
                )

    def _validate_dependencies(self, dependencies: Any) -> None:
        """Validate dependencies list."""
        if not isinstance(dependencies, list):
            self._add_error("Dependencies must be a list", "dependencies")
            return

        # Check for crewai dependency
        if "crewai" not in dependencies:
            self._add_error(
                "Dependencies must include 'crewai' package", "dependencies"
            )

        # Validate individual dependencies
        for i, dep in enumerate(dependencies):
            if not isinstance(dep, str):
                self._add_error(f"Dependency must be a string", f"dependencies[{i}]")
            elif not dep.strip():
                self._add_error(f"Dependency cannot be empty", f"dependencies[{i}]")

    def _validate_agent_task_relationships(self, spec: Dict[str, Any]) -> None:
        """Validate relationships between agents and tasks."""
        agents = spec.get("agents", [])
        tasks = spec.get("tasks", [])

        if not isinstance(agents, list) or not isinstance(tasks, list):
            return  # Basic validation should have caught this

        # Track which agents are used in tasks
        used_agent_roles: Set[str] = set()
        for task in tasks:
            if isinstance(task, dict):
                agent_role = task.get("agent")
                if isinstance(agent_role, str):
                    used_agent_roles.add(agent_role)

        # Check for unused agents
        for i, agent in enumerate(agents):
            if isinstance(agent, dict):
                role = agent.get("role")
                if isinstance(role, str) and role not in used_agent_roles:
                    self._add_warning(
                        f"Agent '{role}' is not assigned to any tasks",
                        f"agents[{i}].role",
                    )

    def _calculate_completeness_score(self, spec: Dict[str, Any]) -> float:
        """Calculate completeness score based on specification quality."""
        score = 0.0
        max_score = 0.0

        # Required fields presence (40% of score)
        required_weight = 0.4
        required_present = sum(
            1 for field in self.REQUIRED_SPEC_FIELDS if field in spec and spec[field]
        )
        score += required_weight * (required_present / len(self.REQUIRED_SPEC_FIELDS))
        max_score += required_weight

        # Agent quality (25% of score)
        agents = spec.get("agents", [])
        if isinstance(agents, list) and agents:
            agent_weight = 0.25
            agent_score = 0.0
            for agent in agents:
                if isinstance(agent, dict):
                    # Check completeness of each agent
                    agent_fields_present = sum(
                        1
                        for field in self.REQUIRED_AGENT_FIELDS
                        if field in agent and agent[field]
                    )
                    agent_score += agent_fields_present / len(
                        self.REQUIRED_AGENT_FIELDS
                    )

                    # Bonus for having tools
                    tools = agent.get("tools", [])
                    if isinstance(tools, list) and tools:
                        agent_score += 0.2  # 20% bonus for having tools

            if agents:
                score += agent_weight * min(1.0, agent_score / len(agents))
            max_score += agent_weight

        # Task quality (25% of score)
        tasks = spec.get("tasks", [])
        if isinstance(tasks, list) and tasks:
            task_weight = 0.25
            task_score = 0.0
            for task in tasks:
                if isinstance(task, dict):
                    # Check completeness of each task
                    task_fields_present = sum(
                        1
                        for field in self.REQUIRED_TASK_FIELDS
                        if field in task and task[field]
                    )
                    task_score += task_fields_present / len(self.REQUIRED_TASK_FIELDS)

            if tasks:
                score += task_weight * (task_score / len(tasks))
            max_score += task_weight

        # Dependencies quality (10% of score)
        dependencies = spec.get("dependencies", [])
        if isinstance(dependencies, list):
            dep_weight = 0.1
            # Base score for having crewai
            if "crewai" in dependencies:
                score += dep_weight * 0.5
            # Bonus for having additional relevant dependencies
            if len(dependencies) > 1:
                score += dep_weight * 0.5
            max_score += dep_weight

        # Normalize score
        return score / max_score if max_score > 0 else 0.0

    def _add_error(self, message: str, field_path: str) -> None:
        """Add an error-level validation issue."""
        self.issues.append(ValidationIssue(IssueSeverity.ERROR, message, field_path))

    def _add_warning(self, message: str, field_path: str) -> None:
        """Add a warning-level validation issue."""
        self.issues.append(ValidationIssue(IssueSeverity.WARNING, message, field_path))

    def _add_info(self, message: str, field_path: str) -> None:
        """Add an info-level validation issue."""
        self.issues.append(ValidationIssue(IssueSeverity.INFO, message, field_path))

    def _create_result(self, completeness_score: float = 0.0) -> ValidationResult:
        """Create validation result from current issues."""
        return ValidationResult(
            issues=self.issues.copy(), completeness_score=completeness_score
        )


# Python File Validation Functions


def validate_python_syntax(file_path: Union[str, Path]) -> ValidationResult:
    """
    Validate Python file syntax using AST parsing.

    Args:
        file_path: Path to Python file to validate

    Returns:
        ValidationResult with syntax validation issues
    """
    import ast
    import sys
    from pathlib import Path

    file_path = Path(file_path)
    issues: List[ValidationIssue] = []

    # Check if file exists
    if not file_path.exists():
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR, f"File not found: {file_path}", str(file_path)
            )
        )
        return ValidationResult(issues=issues)

    # Check if path is actually a file
    if not file_path.is_file():
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR, f"Path is not a file: {file_path}", str(file_path)
            )
        )
        return ValidationResult(issues=issues)

    try:
        # Read file content
        content = file_path.read_text(encoding="utf-8")

        # Check for empty file
        if not content.strip():
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    f"Empty file: {file_path.name}",
                    str(file_path),
                )
            )
            return ValidationResult(issues=issues)

        # Parse with AST
        try:
            ast.parse(content, filename=str(file_path))
            issues.append(
                ValidationIssue(
                    IssueSeverity.INFO,
                    f"File has valid Python syntax: {file_path.name}",
                    str(file_path),
                )
            )
        except IndentationError as e:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Indentation error in {file_path.name} at line {e.lineno}: {e.msg}",
                    str(file_path),
                )
            )
        except SyntaxError as e:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Syntax error in {file_path.name} at line {e.lineno}: {e.msg}",
                    str(file_path),
                )
            )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Parse error in {file_path.name}: {str(e)}",
                    str(file_path),
                )
            )

    except UnicodeDecodeError as e:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Encoding error in {file_path.name}: Cannot decode file as UTF-8",
                str(file_path),
            )
        )
    except Exception as e:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Error reading {file_path.name}: {str(e)}",
                str(file_path),
            )
        )

    return ValidationResult(issues=issues)


def validate_python_imports(
    file_path: Union[str, Path], project_root: Optional[str] = None
) -> ValidationResult:
    """
    Validate Python imports in a file.

    Args:
        file_path: Path to Python file to validate
        project_root: Root directory of the project for relative import resolution

    Returns:
        ValidationResult with import validation issues
    """
    import ast
    import importlib.util
    import sys
    from pathlib import Path

    file_path = Path(file_path)
    issues: List[ValidationIssue] = []

    # First validate syntax
    syntax_result = validate_python_syntax(file_path)
    if not syntax_result.is_valid:
        # Return syntax errors - can't validate imports if syntax is broken
        issues.extend(syntax_result.errors)
        return ValidationResult(issues=issues)

    try:
        content = file_path.read_text(encoding="utf-8")

        # Parse AST to extract imports
        try:
            tree = ast.parse(content, filename=str(file_path))
        except Exception as e:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Failed to parse {file_path.name} for import analysis: {str(e)}",
                    str(file_path),
                )
            )
            return ValidationResult(issues=issues)

        # Extract import statements
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                    # Also check individual imported items if needed
                    for alias in node.names:
                        if alias.name != "*":
                            imports.append(f"{node.module}.{alias.name}")

        # Validate each import
        valid_imports = 0
        for import_name in imports:
            if _is_valid_import(import_name, file_path, project_root):
                valid_imports += 1
            else:
                # Check if it's a relative import issue
                if import_name.startswith("."):
                    issues.append(
                        ValidationIssue(
                            IssueSeverity.WARNING,
                            f"Relative import '{import_name}' may not be resolvable: {file_path.name}",
                            str(file_path),
                        )
                    )
                else:
                    issues.append(
                        ValidationIssue(
                            IssueSeverity.ERROR,
                            f"Import '{import_name}' not found: {file_path.name}",
                            str(file_path),
                        )
                    )

        if (
            valid_imports > 0
            and len([i for i in issues if i.severity == IssueSeverity.ERROR]) == 0
        ):
            issues.append(
                ValidationIssue(
                    IssueSeverity.INFO,
                    f"Imports validated successfully in {file_path.name} ({valid_imports} imports checked)",
                    str(file_path),
                )
            )

    except Exception as e:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Error validating imports in {file_path.name}: {str(e)}",
                str(file_path),
            )
        )

    return ValidationResult(issues=issues)


def _is_valid_import(
    import_name: str, file_path: Path, project_root: Optional[str] = None
) -> bool:
    """
    Check if an import name is valid/resolvable.

    Args:
        import_name: Name of the import to validate
        file_path: Path to the file containing the import
        project_root: Root directory of the project

    Returns:
        True if import is valid, False otherwise
    """
    import importlib.util
    import sys
    from pathlib import Path

    # Skip relative imports for now - they're complex to validate properly
    if import_name.startswith("."):
        return True

    # Try to find the module spec
    try:
        spec = importlib.util.find_spec(import_name.split(".")[0])
        return spec is not None
    except (ImportError, AttributeError, ValueError, ModuleNotFoundError):
        # Check if it might be a local module in the project
        if project_root:
            project_path = Path(project_root)
            # Look for the module as a Python file or package
            module_parts = import_name.split(".")

            # Check for .py file
            potential_file = project_path
            for part in module_parts[:-1]:
                potential_file = potential_file / part
            potential_file = potential_file / f"{module_parts[-1]}.py"

            if potential_file.exists():
                return True

            # Check for package (__init__.py)
            potential_package = project_path
            for part in module_parts:
                potential_package = potential_package / part
            potential_init = potential_package / "__init__.py"

            if potential_init.exists():
                return True

        return False


def validate_project_imports(project_root: str) -> ValidationResult:
    """
    Validate imports across an entire project.

    Args:
        project_root: Root directory of the project to validate

    Returns:
        ValidationResult with project-wide import validation issues
    """
    from pathlib import Path

    project_path = Path(project_root)
    issues: List[ValidationIssue] = []

    if not project_path.exists():
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Project root does not exist: {project_root}",
                project_root,
            )
        )
        return ValidationResult(issues=issues)

    # Find all Python files in the project
    python_files = list(project_path.rglob("*.py"))

    if not python_files:
        issues.append(
            ValidationIssue(
                IssueSeverity.WARNING,
                f"No Python files found in project: {project_root}",
                project_root,
            )
        )
        return ValidationResult(issues=issues)

    # Validate imports in each file
    total_files = len(python_files)
    files_with_errors = 0

    for python_file in python_files:
        file_result = validate_python_imports(python_file, project_root)
        issues.extend(file_result.issues)

        if not file_result.is_valid:
            files_with_errors += 1

    # Add summary information
    if files_with_errors == 0:
        issues.append(
            ValidationIssue(
                IssueSeverity.INFO,
                f"Project import validation passed for all {total_files} Python files",
                project_root,
            )
        )
    else:
        issues.append(
            ValidationIssue(
                IssueSeverity.WARNING,
                f"Project import validation found issues in {files_with_errors}/{total_files} Python files",
                project_root,
            )
        )

    return ValidationResult(issues=issues)


def validate_generated_project(project_root: str) -> ValidationResult:
    """
    Comprehensive validation of a generated CrewAI project.

    Validates both project structure and Python code quality including:
    - Python syntax validation for all Python files
    - Import validation and dependency checking
    - Project structure validation

    Args:
        project_root: Root directory of the generated CrewAI project

    Returns:
        ValidationResult with comprehensive validation results
    """
    from pathlib import Path

    project_path = Path(project_root)
    all_issues: List[ValidationIssue] = []

    # 1. Basic project existence check
    if not project_path.exists():
        all_issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Project root does not exist: {project_root}",
                project_root,
            )
        )
        return ValidationResult(issues=all_issues)

    if not project_path.is_dir():
        all_issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Project root is not a directory: {project_root}",
                project_root,
            )
        )
        return ValidationResult(issues=all_issues)

    # 2. Find and validate all Python files
    python_files = list(project_path.rglob("*.py"))

    if not python_files:
        all_issues.append(
            ValidationIssue(
                IssueSeverity.WARNING,
                f"No Python files found in project: {project_root}",
                project_root,
            )
        )
    else:
        # Validate syntax for each Python file
        syntax_errors = 0
        import_errors = 0

        for python_file in python_files:
            # Skip __pycache__ files
            if "__pycache__" in str(python_file):
                continue

            # Validate Python syntax
            syntax_result = validate_python_syntax(python_file)
            all_issues.extend(syntax_result.issues)

            if not syntax_result.is_valid:
                syntax_errors += 1

            # Validate imports (only if syntax is valid)
            if syntax_result.is_valid:
                import_result = validate_python_imports(python_file, project_root)
                all_issues.extend(import_result.issues)

                if not import_result.is_valid:
                    import_errors += 1

        # Add summary statistics
        total_files = len([f for f in python_files if "__pycache__" not in str(f)])

        if syntax_errors == 0 and import_errors == 0:
            all_issues.append(
                ValidationIssue(
                    IssueSeverity.INFO,
                    f"All {total_files} Python files passed validation",
                    project_root,
                )
            )
        else:
            if syntax_errors > 0:
                all_issues.append(
                    ValidationIssue(
                        IssueSeverity.ERROR,
                        f"Syntax errors found in {syntax_errors}/{total_files} Python files",
                        project_root,
                    )
                )
            if import_errors > 0:
                all_issues.append(
                    ValidationIssue(
                        IssueSeverity.WARNING,
                        f"Import issues found in {import_errors}/{total_files} Python files",
                        project_root,
                    )
                )

    # 3. Validate CrewAI project structure
    src_dir = project_path / "src"
    if not src_dir.exists():
        all_issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Missing 'src' directory in CrewAI project: {project_root}",
                project_root,
            )
        )
    else:
        # Check for project subdirectory
        project_subdirs = [
            d for d in src_dir.iterdir() if d.is_dir() and d.name != "__pycache__"
        ]

        if not project_subdirs:
            all_issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"No project subdirectory found in src/: {project_root}",
                    project_root,
                )
            )
        else:
            # Validate key files exist
            project_subdir = project_subdirs[0]  # Use first valid subdirectory

            expected_files = ["main.py", "crew.py"]
            for expected_file in expected_files:
                file_path = project_subdir / expected_file
                if not file_path.exists():
                    all_issues.append(
                        ValidationIssue(
                            IssueSeverity.ERROR,
                            f"Missing required file: {expected_file} in {project_subdir.name}",
                            str(file_path),
                        )
                    )
                else:
                    all_issues.append(
                        ValidationIssue(
                            IssueSeverity.INFO,
                            f"Found required file: {expected_file}",
                            str(file_path),
                        )
                    )

            # Check for config directory
            config_dir = project_subdir / "config"
            if not config_dir.exists():
                all_issues.append(
                    ValidationIssue(
                        IssueSeverity.WARNING,
                        f"Missing config directory in {project_subdir.name}",
                        str(config_dir),
                    )
                )
            else:
                # Check for configuration files and validate their content
                config_files = ["agents.yaml", "tasks.yaml"]
                agents_file = None
                tasks_file = None

                for config_file in config_files:
                    config_path = config_dir / config_file
                    if not config_path.exists():
                        all_issues.append(
                            ValidationIssue(
                                IssueSeverity.WARNING,
                                f"Missing config file: {config_file}",
                                str(config_path),
                            )
                        )
                    else:
                        all_issues.append(
                            ValidationIssue(
                                IssueSeverity.INFO,
                                f"Found config file: {config_file}",
                                str(config_path),
                            )
                        )

                        # Store file paths for validation
                        if config_file == "agents.yaml":
                            agents_file = config_path
                        elif config_file == "tasks.yaml":
                            tasks_file = config_path

                # Validate configuration files if they exist
                if agents_file:
                    agents_result = validate_crewai_agents_config(agents_file)
                    all_issues.extend(agents_result.issues)

                if tasks_file:
                    tasks_result = validate_crewai_tasks_config(tasks_file)
                    all_issues.extend(tasks_result.issues)

                # Validate consistency between agents and tasks if both exist
                if agents_file and tasks_file:
                    try:
                        # Load both configurations for consistency checking even if individual validations failed
                        import yaml

                        agents_content = agents_file.read_text(encoding="utf-8")
                        tasks_content = tasks_file.read_text(encoding="utf-8")

                        try:
                            agents_config = yaml.safe_load(agents_content)
                            tasks_config = yaml.safe_load(tasks_content)

                            # Check agent references even if agents have validation errors
                            if isinstance(agents_config, dict) and isinstance(
                                tasks_config, dict
                            ):
                                available_agents = list(agents_config.keys())
                                consistency_errors_found = False

                                # Check that all task agents reference valid agents
                                for task_name, task_config in tasks_config.items():
                                    if (
                                        isinstance(task_config, dict)
                                        and "agent" in task_config
                                    ):
                                        referenced_agent = task_config["agent"]
                                        if referenced_agent not in available_agents:
                                            all_issues.append(
                                                ValidationIssue(
                                                    IssueSeverity.ERROR,
                                                    f"Task '{task_name}' references unknown agent '{referenced_agent}'. Available agents: {', '.join(available_agents)}",
                                                    f"{tasks_file}[{task_name}].agent",
                                                )
                                            )
                                            consistency_errors_found = True

                                # Add success message if no consistency errors
                                if not consistency_errors_found:
                                    all_issues.append(
                                        ValidationIssue(
                                            IssueSeverity.INFO,
                                            f"Configuration consistency validated: {len(available_agents)} agents, {len(tasks_config)} tasks",
                                            f"{agents_file} + {tasks_file}",
                                        )
                                    )
                        except yaml.YAMLError:
                            # If YAML parsing fails, individual validations already reported this
                            pass
                    except Exception:
                        # If consistency validation fails for other reasons, continue without it
                        pass

    # 4. Check for other important project files
    important_files = ["pyproject.toml", "README.md", ".env"]
    for important_file in important_files:
        file_path = project_path / important_file
        if not file_path.exists():
            all_issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    f"Missing recommended file: {important_file}",
                    str(file_path),
                )
            )
        else:
            all_issues.append(
                ValidationIssue(
                    IssueSeverity.INFO,
                    f"Found project file: {important_file}",
                    str(file_path),
                )
            )

    # 5. Validate workflow execution (only if project structure is valid)
    if src_dir.exists():
        workflow_result = validate_crewai_workflow_execution(project_root)
        all_issues.extend(workflow_result.issues)

    return ValidationResult(issues=all_issues)


# CrewAI Configuration File Validation Functions


def validate_crewai_agents_config(file_path: Union[str, Path]) -> ValidationResult:
    """
    Validate CrewAI agents.yaml configuration file.

    Args:
        file_path: Path to agents.yaml file

    Returns:
        ValidationResult with validation issues
    """
    import yaml

    file_path = Path(file_path)
    issues: List[ValidationIssue] = []

    # Check if file exists
    if not file_path.exists():
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Agents configuration file not found: {file_path}",
                str(file_path),
            )
        )
        return ValidationResult(issues=issues)

    try:
        # Read and parse YAML
        content = file_path.read_text(encoding="utf-8")

        if not content.strip():
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Empty agents configuration file: {file_path.name}",
                    str(file_path),
                )
            )
            return ValidationResult(issues=issues)

        try:
            agents_config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Invalid YAML syntax in {file_path.name}: {str(e)}",
                    str(file_path),
                )
            )
            return ValidationResult(issues=issues)

        # Validate config structure
        if not isinstance(agents_config, dict):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Agents configuration must be a dictionary, got {type(agents_config).__name__}",
                    str(file_path),
                )
            )
            return ValidationResult(issues=issues)

        if not agents_config:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Agents configuration is empty",
                    str(file_path),
                )
            )
            return ValidationResult(issues=issues)

        # Validate each agent
        for agent_name, agent_config in agents_config.items():
            agent_path = f"{file_path}[{agent_name}]"
            _validate_single_agent_config(agent_config, agent_name, agent_path, issues)

        # Success message
        issues.append(
            ValidationIssue(
                IssueSeverity.INFO,
                f"Agents configuration is valid with {len(agents_config)} agents",
                str(file_path),
            )
        )

    except UnicodeDecodeError as e:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Encoding error in {file_path.name}: Cannot decode file as UTF-8",
                str(file_path),
            )
        )
    except Exception as e:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Error reading {file_path.name}: {str(e)}",
                str(file_path),
            )
        )

    return ValidationResult(issues=issues)


def validate_crewai_tasks_config(
    file_path: Union[str, Path], available_agents: Optional[List[str]] = None
) -> ValidationResult:
    """
    Validate CrewAI tasks.yaml configuration file.

    Args:
        file_path: Path to tasks.yaml file
        available_agents: Optional list of valid agent names for validation

    Returns:
        ValidationResult with validation issues
    """
    import yaml

    file_path = Path(file_path)
    issues: List[ValidationIssue] = []

    # Check if file exists
    if not file_path.exists():
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Tasks configuration file not found: {file_path}",
                str(file_path),
            )
        )
        return ValidationResult(issues=issues)

    try:
        # Read and parse YAML
        content = file_path.read_text(encoding="utf-8")

        if not content.strip():
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Empty tasks configuration file: {file_path.name}",
                    str(file_path),
                )
            )
            return ValidationResult(issues=issues)

        try:
            tasks_config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Invalid YAML syntax in {file_path.name}: {str(e)}",
                    str(file_path),
                )
            )
            return ValidationResult(issues=issues)

        # Validate config structure
        if not isinstance(tasks_config, dict):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Tasks configuration must be a dictionary, got {type(tasks_config).__name__}",
                    str(file_path),
                )
            )
            return ValidationResult(issues=issues)

        if not tasks_config:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR, f"Tasks configuration is empty", str(file_path)
                )
            )
            return ValidationResult(issues=issues)

        # Validate each task
        task_names = list(tasks_config.keys())
        for task_name, task_config in tasks_config.items():
            task_path = f"{file_path}[{task_name}]"
            _validate_single_task_config(
                task_config, task_name, task_path, issues, available_agents, task_names
            )

        # Check for circular dependencies
        _validate_task_dependencies(tasks_config, str(file_path), issues)

        # Success message
        issues.append(
            ValidationIssue(
                IssueSeverity.INFO,
                f"Tasks configuration is valid with {len(tasks_config)} tasks",
                str(file_path),
            )
        )

    except UnicodeDecodeError as e:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Encoding error in {file_path.name}: Cannot decode file as UTF-8",
                str(file_path),
            )
        )
    except Exception as e:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Error reading {file_path.name}: {str(e)}",
                str(file_path),
            )
        )

    return ValidationResult(issues=issues)


def validate_crewai_config_consistency(
    agents_file: Union[str, Path], tasks_file: Union[str, Path]
) -> ValidationResult:
    """
    Validate consistency between agents.yaml and tasks.yaml files.

    Args:
        agents_file: Path to agents.yaml file
        tasks_file: Path to tasks.yaml file

    Returns:
        ValidationResult with consistency validation issues
    """
    import yaml

    agents_file = Path(agents_file)
    tasks_file = Path(tasks_file)
    issues: List[ValidationIssue] = []

    # First validate individual files
    agents_result = validate_crewai_agents_config(agents_file)
    tasks_result = validate_crewai_tasks_config(tasks_file)

    # Combine issues from individual validations
    issues.extend(agents_result.issues)
    issues.extend(tasks_result.issues)

    # If individual validations failed, don't proceed with consistency checks
    if not agents_result.is_valid or not tasks_result.is_valid:
        return ValidationResult(issues=issues)

    try:
        # Load both configurations
        agents_content = agents_file.read_text(encoding="utf-8")
        tasks_content = tasks_file.read_text(encoding="utf-8")

        agents_config = yaml.safe_load(agents_content)
        tasks_config = yaml.safe_load(tasks_content)

        # Get list of available agents
        available_agents = list(agents_config.keys())

        # Check that all task agents reference valid agents
        for task_name, task_config in tasks_config.items():
            if isinstance(task_config, dict) and "agent" in task_config:
                referenced_agent = task_config["agent"]
                if referenced_agent not in available_agents:
                    issues.append(
                        ValidationIssue(
                            IssueSeverity.ERROR,
                            f"Task '{task_name}' references unknown agent '{referenced_agent}'. Available agents: {', '.join(available_agents)}",
                            f"{tasks_file}[{task_name}].agent",
                        )
                    )

        # Check for unused agents (warning)
        used_agents = set()
        for task_config in tasks_config.values():
            if isinstance(task_config, dict) and "agent" in task_config:
                used_agents.add(task_config["agent"])

        unused_agents = set(available_agents) - used_agents
        for unused_agent in unused_agents:
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    f"Agent '{unused_agent}' is defined but not used by any tasks",
                    f"{agents_file}[{unused_agent}]",
                )
            )

        # Success message if no consistency errors found
        consistency_errors = [
            issue
            for issue in issues
            if issue.severity == IssueSeverity.ERROR
            and "consistency" in issue.message.lower()
        ]
        if not consistency_errors:
            issues.append(
                ValidationIssue(
                    IssueSeverity.INFO,
                    f"Configuration consistency validated: {len(available_agents)} agents, {len(tasks_config)} tasks",
                    f"{agents_file} + {tasks_file}",
                )
            )

    except Exception as e:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Error validating configuration consistency: {str(e)}",
                f"{agents_file} + {tasks_file}",
            )
        )

    return ValidationResult(issues=issues)


def _validate_single_agent_config(
    agent_config: Any, agent_name: str, agent_path: str, issues: List[ValidationIssue]
) -> None:
    """Helper function to validate a single agent configuration."""

    # Required fields for agents
    required_fields = {"role", "goal", "backstory", "tools"}

    if not isinstance(agent_config, dict):
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Agent '{agent_name}' must be a dictionary, got {type(agent_config).__name__}",
                agent_path,
            )
        )
        return

    # Check required fields
    for field in required_fields:
        if field not in agent_config:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Agent '{agent_name}' is missing required field: {field}",
                    f"{agent_path}.{field}",
                )
            )

    # Validate field types and content
    if "role" in agent_config:
        role = agent_config["role"]
        if not isinstance(role, str):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Agent '{agent_name}' role must be a string, got {type(role).__name__}",
                    f"{agent_path}.role",
                )
            )
        elif role.lower().strip() in {"agent", "assistant", "helper", "bot", "ai"}:
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    f"Agent '{agent_name}' has a vague role: '{role}'. Consider being more specific.",
                    f"{agent_path}.role",
                )
            )

    if "goal" in agent_config:
        goal = agent_config["goal"]
        if not isinstance(goal, str):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Agent '{agent_name}' goal must be a string, got {type(goal).__name__}",
                    f"{agent_path}.goal",
                )
            )
        elif len(goal.strip()) < 20:
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    f"Agent '{agent_name}' has a very short goal. Consider being more descriptive.",
                    f"{agent_path}.goal",
                )
            )
        elif any(vague in goal.lower() for vague in {"help", "assist", "do"}):
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    f"Agent '{agent_name}' has a vague goal containing generic terms. Be more specific about what the agent should accomplish.",
                    f"{agent_path}.goal",
                )
            )

    if "backstory" in agent_config:
        backstory = agent_config["backstory"]
        if not isinstance(backstory, str):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Agent '{agent_name}' backstory must be a string, got {type(backstory).__name__}",
                    f"{agent_path}.backstory",
                )
            )
        elif len(backstory.strip()) < 30:
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    f"Agent '{agent_name}' has a very short backstory. Consider adding more context about the agent's expertise and experience.",
                    f"{agent_path}.backstory",
                )
            )

    if "tools" in agent_config:
        tools = agent_config["tools"]
        if not isinstance(tools, list):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Agent '{agent_name}' tools must be a list, got {type(tools).__name__}",
                    f"{agent_path}.tools",
                )
            )
        elif len(tools) == 0:
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    f"Agent '{agent_name}' has no tools assigned. Consider adding appropriate tools for this agent's role.",
                    f"{agent_path}.tools",
                )
            )

    if "verbose" in agent_config:
        verbose = agent_config["verbose"]
        if not isinstance(verbose, bool):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Agent '{agent_name}' verbose must be a boolean, got {type(verbose).__name__}",
                    f"{agent_path}.verbose",
                )
            )


def _validate_single_task_config(
    task_config: Any,
    task_name: str,
    task_path: str,
    issues: List[ValidationIssue],
    available_agents: Optional[List[str]] = None,
    all_task_names: Optional[List[str]] = None,
) -> None:
    """Helper function to validate a single task configuration."""

    # Required fields for tasks
    required_fields = {"description", "expected_output", "agent"}

    if not isinstance(task_config, dict):
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Task '{task_name}' must be a dictionary, got {type(task_config).__name__}",
                task_path,
            )
        )
        return

    # Check required fields
    for field in required_fields:
        if field not in task_config:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Task '{task_name}' is missing required field: {field}",
                    f"{task_path}.{field}",
                )
            )

    # Validate field types and content
    if "description" in task_config:
        description = task_config["description"]
        if not isinstance(description, str):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Task '{task_name}' description must be a string, got {type(description).__name__}",
                    f"{task_path}.description",
                )
            )
        elif len(description.strip()) < 20:
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    f"Task '{task_name}' has a very short description. Consider being more descriptive.",
                    f"{task_path}.description",
                )
            )
        elif any(
            vague in description.lower() for vague in {"work", "task", "do", "handle"}
        ):
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    f"Task '{task_name}' has a vague description containing generic terms. Be more specific about what should be accomplished.",
                    f"{task_path}.description",
                )
            )

    if "expected_output" in task_config:
        expected_output = task_config["expected_output"]
        if not isinstance(expected_output, str):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Task '{task_name}' expected_output must be a string, got {type(expected_output).__name__}",
                    f"{task_path}.expected_output",
                )
            )
        elif len(expected_output.strip()) < 10:
            issues.append(
                ValidationIssue(
                    IssueSeverity.WARNING,
                    f"Task '{task_name}' has a very short expected_output. Consider being more specific about the deliverable.",
                    f"{task_path}.expected_output",
                )
            )

    if "agent" in task_config:
        agent = task_config["agent"]
        if not isinstance(agent, str):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Task '{task_name}' agent must be a string, got {type(agent).__name__}",
                    f"{task_path}.agent",
                )
            )
        elif available_agents and agent not in available_agents:
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Task '{task_name}' references unknown agent '{agent}'. Available agents: {', '.join(available_agents)}",
                    f"{task_path}.agent",
                )
            )

    # Validate optional context field
    if "context" in task_config:
        context = task_config["context"]
        if not isinstance(context, list):
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Task '{task_name}' context must be a list, got {type(context).__name__}",
                    f"{task_path}.context",
                )
            )
        elif all_task_names:
            for ctx_task in context:
                if ctx_task not in all_task_names:
                    issues.append(
                        ValidationIssue(
                            IssueSeverity.ERROR,
                            f"Task '{task_name}' context references unknown task '{ctx_task}'",
                            f"{task_path}.context",
                        )
                    )

    # Check for generic task name
    if task_name.lower() in {"task", "work", "job", "activity"}:
        issues.append(
            ValidationIssue(
                IssueSeverity.WARNING,
                f"Task name '{task_name}' is very generic. Consider using a more descriptive name.",
                task_path,
            )
        )


def _validate_task_dependencies(
    tasks_config: Dict[str, Any], file_path: str, issues: List[ValidationIssue]
) -> None:
    """Helper function to detect circular dependencies in task context."""

    def has_circular_dependency(
        task_name: str, visited: Set[str], path: List[str]
    ) -> bool:
        if task_name in visited:
            cycle_start = path.index(task_name)
            cycle = " -> ".join(path[cycle_start:] + [task_name])
            issues.append(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    f"Circular dependency detected in task context: {cycle}",
                    f"{file_path}[{task_name}].context",
                )
            )
            return True

        if task_name not in tasks_config:
            return False

        task_config = tasks_config[task_name]
        if not isinstance(task_config, dict) or "context" not in task_config:
            return False

        context = task_config["context"]
        if not isinstance(context, list):
            return False

        visited.add(task_name)
        path.append(task_name)

        for dependency in context:
            if has_circular_dependency(dependency, visited.copy(), path.copy()):
                return True

        return False

    # Check each task for circular dependencies
    for task_name in tasks_config:
        has_circular_dependency(task_name, set(), [])


def validate_crewai_workflow_execution(
    project_root: str, timeout: int = 60
) -> ValidationResult:
    """
    Validate CrewAI project workflow execution.

    Tests if the generated CrewAI project can execute its main workflow
    without errors. This includes validating that the project can run
    its entry point files (main.py, crew.py) and handle dependencies.

    Args:
        project_root: Root directory of the CrewAI project
        timeout: Maximum execution time in seconds (default: 60)

    Returns:
        ValidationResult with execution validation results
    """
    from pathlib import Path

    project_path = Path(project_root)
    issues: List[ValidationIssue] = []

    # Check if project exists
    if not project_path.exists():
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Project not found: {project_root}",
                project_root,
            )
        )
        return ValidationResult(issues=issues)

    # Find project structure and entry points
    src_dir = project_path / "src"
    if not src_dir.exists():
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Missing src directory in project: {project_root}",
                str(src_dir),
            )
        )
        return ValidationResult(issues=issues)

    # Find project subdirectory
    project_subdirs = [
        d for d in src_dir.iterdir() if d.is_dir() and d.name != "__pycache__"
    ]

    if not project_subdirs:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"No project subdirectory found in src/: {project_root}",
                str(src_dir),
            )
        )
        return ValidationResult(issues=issues)

    project_subdir = project_subdirs[0]
    main_py = project_subdir / "main.py"

    if not main_py.exists():
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"No main entry point found: main.py not found in {project_subdir.name}",
                str(main_py),
            )
        )
        return ValidationResult(issues=issues)

    # Prepare environment
    env_file = project_path / ".env"
    env_vars = {}

    if env_file.exists():
        try:
            env_content = env_file.read_text(encoding="utf-8")
            for line in env_content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
        except Exception:
            # Environment file issues won't block execution validation
            pass

    # Add PYTHONPATH to ensure imports work
    env_vars["PYTHONPATH"] = str(src_dir)

    try:
        # Execute the project's main entry point
        # Use sys.executable to get the current Python interpreter
        import sys

        cmd = [sys.executable, str(main_py)]

        # Run subprocess with timeout and capture output
        result = subprocess.run(
            cmd,
            cwd=str(project_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**dict(os.environ), **env_vars} if env_vars else None,
        )

        if result.returncode == 0:
            # Successful execution
            issues.append(
                ValidationIssue(
                    IssueSeverity.INFO,
                    f"CrewAI workflow executed successfully",
                    str(main_py),
                )
            )

            # Add output information if available
            if result.stdout.strip():
                issues.append(
                    ValidationIssue(
                        IssueSeverity.INFO,
                        f"Execution output captured: {len(result.stdout.strip().split())} words",
                        str(main_py),
                    )
                )
        else:
            # Execution failed
            error_msg = (
                result.stderr.strip()
                if result.stderr.strip()
                else "Unknown execution error"
            )

            # Categorize the type of error
            if "SyntaxError" in error_msg or "IndentationError" in error_msg:
                issues.append(
                    ValidationIssue(
                        IssueSeverity.ERROR,
                        f"Syntax error in workflow execution: {error_msg}",
                        str(main_py),
                    )
                )
            elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
                issues.append(
                    ValidationIssue(
                        IssueSeverity.ERROR,
                        f"Import error in workflow execution: {error_msg}",
                        str(main_py),
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        IssueSeverity.ERROR,
                        f"Runtime error in workflow execution: {error_msg}",
                        str(main_py),
                    )
                )

    except subprocess.TimeoutExpired:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Workflow execution timed out after {timeout} seconds",
                str(main_py),
            )
        )
    except FileNotFoundError:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Python interpreter not found. Cannot execute workflow.",
                str(main_py),
            )
        )
    except Exception as e:
        issues.append(
            ValidationIssue(
                IssueSeverity.ERROR,
                f"Execution failed: {str(e)}",
                str(main_py),
            )
        )

    return ValidationResult(issues=issues)

"""
Specification validation and completeness checking system for CrewForge.

This module provides comprehensive validation for parsed project specifications,
ensuring they are complete, valid, and suitable for CrewAI project generation.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
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

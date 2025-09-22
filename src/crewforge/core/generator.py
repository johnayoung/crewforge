"""Agentic AI Generation Engine for CrewForge.

This module implements core generation logic that analyzes prompts and
creates CrewAI configurations using LLM calls with quality controls.
"""

from typing import Any, Dict, List, Optional, Tuple

from .llm import LLMClient
from .progress import StreamingCallbacks
from .templates import TemplateEngine
from ..models import AgentConfig, TaskConfig


class GenerationError(Exception):
    """Custom exception for generation-related errors."""

    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


class GenerationEngine:
    """Core generation engine that coordinates LLM calls to create CrewAI configurations.

    This class implements the agentic AI generation pipeline that:
    1. Analyzes natural language prompts for crew requirements
    2. Generates agent configurations with roles, goals, and backstories
    3. Creates task definitions with clear expected outputs
    4. Selects appropriate tools from CrewAI library
    5. Validates compatibility and quality of generated configurations
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, verbose: bool = False):
        """Initialize GenerationEngine.

        Args:
            llm_client: LLMClient instance for LLM calls. If None, creates default client.
            verbose: If True, enables verbose output for detailed generation steps.
        """
        if llm_client is None:
            llm_client = LLMClient()
        self.llm_client = llm_client
        self.verbose = verbose

        # Initialize template engine for prompt rendering
        self.template_engine = TemplateEngine()

    def analyze_prompt(
        self, prompt: str, streaming_callbacks: Optional[StreamingCallbacks] = None
    ) -> Dict[str, Any]:
        """Extract crew requirements from natural language prompt.

        Uses structured LLM output to identify business context, required roles,
        objectives, and tools needed for the crew.

        Args:
            prompt: Natural language description of desired crew functionality

        Returns:
            Dict containing:
                - business_context: str - Description of business domain/context
                - required_roles: List[str] - List of agent roles needed
                - objectives: List[str] - List of key objectives to accomplish
                - tools_needed: List[str] - List of tool categories required

        Raises:
            GenerationError: If prompt analysis fails
        """
        system_prompt = self.template_engine.render_template(
            "prompts/analyze_prompt_system.j2"
        )

        user_prompt = self.template_engine.render_template(
            "prompts/analyze_prompt_user.j2", prompt=prompt
        )

        try:
            if streaming_callbacks:
                result = self.llm_client.generate_streaming(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    streaming_callbacks=streaming_callbacks,
                    use_json_mode=True,
                    temperature=0.1,
                )
            else:
                result = self.llm_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    use_json_mode=True,
                    temperature=0.1,
                )

            # Validate the response structure
            if not isinstance(result, dict):
                raise GenerationError(f"Expected dict response, got {type(result)}")

            required_keys = [
                "business_context",
                "required_roles",
                "objectives",
                "tools_needed",
            ]
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                raise GenerationError(
                    f"Missing required keys in analysis: {missing_keys}"
                )

            return result

        except Exception as e:
            raise GenerationError(
                f"Failed to analyze prompt: {str(e)}", original_exception=e
            )

    def generate_agents(
        self,
        prompt_analysis: Dict[str, Any],
        streaming_callbacks: Optional[StreamingCallbacks] = None,
    ) -> List[AgentConfig]:
        """Generate agent configurations based on prompt analysis.

        Creates appropriate agent roles, goals, and backstories using LLM
        and validates them against Pydantic models.

        Args:
            prompt_analysis: Analysis result from analyze_prompt()

        Returns:
            List of validated AgentConfig instances

        Raises:
            GenerationError: If agent generation fails
        """
        system_prompt = self.template_engine.render_template(
            "prompts/generate_agents_system.j2"
        )

        business_context = prompt_analysis.get("business_context", "")
        required_roles = prompt_analysis.get("required_roles", [])
        objectives = prompt_analysis.get("objectives", [])
        tools_needed = prompt_analysis.get("tools_needed", [])

        user_prompt = self.template_engine.render_template(
            "prompts/generate_agents_user.j2",
            business_context=business_context,
            required_roles=required_roles,
            objectives=objectives,
            tools_needed=tools_needed,
        )

        try:
            result = self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                use_json_mode=True,
                temperature=0.2,
            )

            if not isinstance(result, dict) or "agents" not in result:
                raise GenerationError("Invalid agent generation response structure")

            agents = []
            for i, agent_data in enumerate(result["agents"]):
                try:
                    # Create AgentConfig instance, which will validate the data
                    agent = AgentConfig(
                        role=agent_data["role"],
                        goal=agent_data["goal"],
                        backstory=agent_data["backstory"],
                    )
                    agents.append(agent)
                except Exception as e:
                    raise GenerationError(
                        f"Invalid agent data at index {i}: {str(e)}",
                        original_exception=e,
                    )

            if not agents:
                raise GenerationError("No valid agents generated")

            return agents

        except Exception as e:
            if isinstance(e, GenerationError):
                raise
            raise GenerationError(
                f"Failed to generate agents: {str(e)}", original_exception=e
            )

    def generate_tasks(
        self,
        agents: List[AgentConfig],
        prompt_analysis: Dict[str, Any],
        streaming_callbacks: Optional[StreamingCallbacks] = None,
    ) -> List[TaskConfig]:
        """Generate task definitions based on agents and requirements.

        Creates task descriptions with clear expected outputs and proper
        agent assignments using LLM.

        Args:
            agents: List of generated AgentConfig instances
            prompt_analysis: Analysis result from analyze_prompt()

        Returns:
            List of validated TaskConfig instances

        Raises:
            GenerationError: If task generation fails
        """
        system_prompt = self.template_engine.render_template(
            "prompts/generate_tasks_system.j2"
        )

        agent_roles = [agent.role for agent in agents]
        objectives = prompt_analysis.get("objectives", [])
        business_context = prompt_analysis.get("business_context", "")

        user_prompt = self.template_engine.render_template(
            "prompts/generate_tasks_user.j2",
            agent_roles=agent_roles,
            objectives=objectives,
            business_context=business_context,
        )

        try:
            result = self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                use_json_mode=True,
                temperature=0.2,
            )

            if not isinstance(result, dict) or "tasks" not in result:
                raise GenerationError("Invalid task generation response structure")

            tasks = []
            for i, task_data in enumerate(result["tasks"]):
                try:
                    # Validate agent assignment
                    agent_role = task_data.get("agent", "")
                    if agent_role not in agent_roles:
                        raise GenerationError(
                            f"Task {i} assigned to unknown agent: {agent_role}"
                        )

                    # Create TaskConfig instance, which will validate the data
                    task = TaskConfig(
                        description=task_data["description"],
                        expected_output=task_data["expected_output"],
                        agent=agent_role,
                        context=task_data.get("context"),
                        async_execution=task_data.get("async_execution", False),
                    )
                    tasks.append(task)
                except Exception as e:
                    raise GenerationError(
                        f"Invalid task data at index {i}: {str(e)}",
                        original_exception=e,
                    )

            if not tasks:
                raise GenerationError("No valid tasks generated")

            return tasks

        except Exception as e:
            if isinstance(e, GenerationError):
                raise
            raise GenerationError(
                f"Failed to generate tasks: {str(e)}", original_exception=e
            )

    def select_tools(
        self,
        tools_needed: List[str],
        streaming_callbacks: Optional[StreamingCallbacks] = None,
    ) -> Dict[str, Any]:
        """Select appropriate CrewAI tools based on requirements.

        Chooses tools from CrewAI library and validates their availability.

        Args:
            tools_needed: List of tool categories/capabilities required

        Returns:
            Dict containing:
                - selected_tools: List[Dict] - Available tools with reasons
                - unavailable_tools: List[str] - Tools that couldn't be found

        Raises:
            GenerationError: If tool selection fails
        """
        system_prompt = self.template_engine.render_template(
            "prompts/select_tools_system.j2"
        )

        user_prompt = self.template_engine.render_template(
            "prompts/select_tools_user.j2",
            tools_needed=tools_needed,
        )

        try:
            result = self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                use_json_mode=True,
                temperature=0.1,
            )

            if not isinstance(result, dict):
                raise GenerationError(f"Expected dict response, got {type(result)}")

            # Validate response structure
            if "selected_tools" not in result:
                result["selected_tools"] = []
            if "unavailable_tools" not in result:
                result["unavailable_tools"] = []

            # Validate selected tools against known CrewAI tools
            selected_tool_names = [
                tool.get("name", "") for tool in result["selected_tools"]
            ]
            available_tools, unavailable_selected = self._validate_tool_availability(
                selected_tool_names
            )

            # Update unavailable tools list
            if unavailable_selected:
                result["unavailable_tools"].extend(unavailable_selected)
                # Remove unavailable tools from selected list
                result["selected_tools"] = [
                    tool
                    for tool in result["selected_tools"]
                    if tool.get("name", "") not in unavailable_selected
                ]

            return result

        except Exception as e:
            if isinstance(e, GenerationError):
                raise
            raise GenerationError(
                f"Failed to select tools: {str(e)}", original_exception=e
            )

    def _validate_agent_task_alignment(
        self, agents: List[AgentConfig], tasks: List[TaskConfig]
    ) -> Tuple[bool, List[str]]:
        """Validate that agents and tasks are properly aligned.

        Checks that all tasks have valid agent assignments and that
        agent capabilities match task requirements.

        Args:
            agents: List of AgentConfig instances
            tasks: List of TaskConfig instances

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        agent_roles = {agent.role for agent in agents}

        for task in tasks:
            if task.agent not in agent_roles:
                errors.append(f"Task assigned to unknown agent: {task.agent}")

        return len(errors) == 0, errors

    def _validate_tool_availability(
        self, tools: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Validate tool availability against CrewAI library.

        Args:
            tools: List of tool names to validate

        Returns:
            Tuple of (available_tools, unavailable_tools)
        """
        # Known CrewAI tools (this would be expanded with actual tool registry)
        known_tools = {
            "SerperDevTool",
            "FileWriterTool",
            "DirectoryReadTool",
            "FileReadTool",
            "CodeInterpreterTool",
            "TXTSearchTool",
            "CSVSearchTool",
            "JSONSearchTool",
            "MDXSearchTool",
            "PDFSearchTool",
            "PGSearchTool",
            "WebsiteSearchTool",
        }

        available = [tool for tool in tools if tool in known_tools]
        unavailable = [tool for tool in tools if tool not in known_tools]

        return available, unavailable

    def _validate_output_format(
        self, config_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate output format consistency.

        Ensures generated configuration data has proper structure and format.

        Args:
            config_data: Generated configuration data

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate required top-level keys
        if "agents" not in config_data:
            errors.append("Missing 'agents' key in configuration")
        if "tasks" not in config_data:
            errors.append("Missing 'tasks' key in configuration")

        # Validate agent structure
        if "agents" in config_data:
            if not isinstance(config_data["agents"], list):
                errors.append("'agents' must be a list")
            else:
                for i, agent in enumerate(config_data["agents"]):
                    if not isinstance(agent, dict):
                        errors.append(f"Agent {i} must be a dictionary")
                        continue
                    required_fields = ["role", "goal", "backstory"]
                    for field in required_fields:
                        if field not in agent:
                            errors.append(f"Agent {i} missing required field: {field}")

        # Validate task structure
        if "tasks" in config_data:
            if not isinstance(config_data["tasks"], list):
                errors.append("'tasks' must be a list")
            else:
                for i, task in enumerate(config_data["tasks"]):
                    if not isinstance(task, dict):
                        errors.append(f"Task {i} must be a dictionary")
                        continue
                    required_fields = ["description", "expected_output", "agent"]
                    for field in required_fields:
                        if field not in task:
                            errors.append(f"Task {i} missing required field: {field}")

        return len(errors) == 0, errors

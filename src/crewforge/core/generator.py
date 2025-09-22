"""Agentic AI Generation Engine for CrewForge.

This module implements core generation logic that analyzes prompts and
creates CrewAI configurations using LLM calls with quality controls.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import click

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
            llm_client = LLMClient(verbose=verbose)
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
            streaming_callbacks: Optional callbacks for streaming LLM responses

        Returns:
            Dict containing:
                - business_context: str - Description of business domain/context
                - required_roles: List[str] - List of agent roles needed
                - objectives: List[str] - List of key objectives to accomplish
                - tools_needed: List[str] - List of tool categories required

        Raises:
            GenerationError: If prompt analysis fails
        """
        if self.verbose:
            click.echo("🔍 Analyzing prompt for requirements...")

        start_time = time.time()

        system_prompt = self.template_engine.render_template(
            "prompts/analyze_prompt_system.j2"
        )

        user_prompt = self.template_engine.render_template(
            "prompts/analyze_prompt_user.j2", prompt=prompt
        )

        # Create default streaming callbacks for verbose mode if none provided
        use_streaming = streaming_callbacks is not None or self.verbose
        callbacks_to_use = streaming_callbacks

        if self.verbose and streaming_callbacks is None:
            # Create verbose streaming callbacks with clear delimiters
            def on_token(token: str) -> None:
                click.echo(f"{token}", nl=False)

            def on_completion(response: str) -> None:
                click.echo("")  # New line after streaming

            callbacks_to_use = StreamingCallbacks(
                on_token=on_token, on_completion=on_completion
            )

        try:
            if use_streaming and callbacks_to_use:
                if self.verbose:
                    click.echo("📡 Sending analysis request with streaming...")
                    click.echo("═══ LLM Response Start ═══")
                result = self.llm_client.generate_streaming(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    streaming_callbacks=callbacks_to_use,
                    use_json_mode=True,
                    temperature=0.1,
                )
                if self.verbose:
                    click.echo("═══ LLM Response End ═══")
            else:
                if self.verbose:
                    click.echo("📡 Sending analysis request...")
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

            duration = time.time() - start_time

            if self.verbose:
                # Show analysis results summary
                click.echo(f"✅ Analysis complete in {duration:.2f}s")
                click.echo(
                    f"   🏢 Business context: {result['business_context'][:50]}..."
                )
                click.echo(
                    f"   👥 Required roles ({len(result['required_roles'])}): {', '.join(result['required_roles'])}"
                )
                click.echo(
                    f"   🎯 Objectives ({len(result['objectives'])}): {len(result['objectives'])} identified"
                )
                click.echo(
                    f"   🛠️  Tools needed ({len(result['tools_needed'])}): {', '.join(result['tools_needed'])}"
                )

            return result

        except Exception as e:
            duration = time.time() - start_time
            if self.verbose:
                click.echo(f"❌ Analysis failed after {duration:.2f}s: {str(e)}")
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
            streaming_callbacks: Optional callbacks for streaming LLM responses

        Returns:
            List of validated AgentConfig instances

        Raises:
            GenerationError: If agent generation fails
        """
        if self.verbose:
            click.echo("🤖 Generating agent configurations...")

        start_time = time.time()

        system_prompt = self.template_engine.render_template(
            "prompts/generate_agents_system.j2"
        )

        business_context = prompt_analysis.get("business_context", "")
        required_roles = prompt_analysis.get("required_roles", [])
        objectives = prompt_analysis.get("objectives", [])
        tools_needed = prompt_analysis.get("tools_needed", [])

        if self.verbose:
            click.echo(
                f"   📝 Creating {len(required_roles)} agents for roles: {', '.join(required_roles)}"
            )

        user_prompt = self.template_engine.render_template(
            "prompts/generate_agents_user.j2",
            business_context=business_context,
            required_roles=required_roles,
            objectives=objectives,
            tools_needed=tools_needed,
        )

        # Create default streaming callbacks for verbose mode if none provided
        use_streaming = streaming_callbacks is not None or self.verbose
        callbacks_to_use = streaming_callbacks

        if self.verbose and streaming_callbacks is None:
            # Create verbose streaming callbacks with clear delimiters
            def on_token(token: str) -> None:
                click.echo(f"{token}", nl=False)

            def on_completion(response: str) -> None:
                click.echo("")  # New line after streaming

            callbacks_to_use = StreamingCallbacks(
                on_token=on_token, on_completion=on_completion
            )

        try:
            if use_streaming and callbacks_to_use:
                if self.verbose:
                    click.echo("📡 Sending agent generation request with streaming...")
                    click.echo("═══ LLM Response Start ═══")
                result = self.llm_client.generate_streaming(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    streaming_callbacks=callbacks_to_use,
                    use_json_mode=True,
                    temperature=0.2,
                )
                if self.verbose:
                    click.echo("═══ LLM Response End ═══")
            else:
                if self.verbose:
                    click.echo("📡 Sending agent generation request...")
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

            duration = time.time() - start_time

            if self.verbose:
                click.echo(f"✅ Agent generation complete in {duration:.2f}s")
                for i, agent in enumerate(agents):
                    click.echo(f"   👤 Agent {i+1}: {agent.role}")
                    click.echo(f"      🎯 Goal: {agent.goal[:60]}...")
                    click.echo(f"      📖 Backstory: {agent.backstory[:60]}...")

            return agents

        except Exception as e:
            duration = time.time() - start_time
            if self.verbose:
                click.echo(
                    f"❌ Agent generation failed after {duration:.2f}s: {str(e)}"
                )
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
            streaming_callbacks: Optional callbacks for streaming LLM responses

        Returns:
            List of validated TaskConfig instances

        Raises:
            GenerationError: If task generation fails
        """
        if self.verbose:
            click.echo("📋 Creating task definitions...")

        start_time = time.time()

        system_prompt = self.template_engine.render_template(
            "prompts/generate_tasks_system.j2"
        )

        agent_roles = [agent.role for agent in agents]
        objectives = prompt_analysis.get("objectives", [])
        business_context = prompt_analysis.get("business_context", "")

        if self.verbose:
            click.echo(
                f"   📝 Creating tasks for {len(agent_roles)} agents with {len(objectives)} objectives"
            )

        user_prompt = self.template_engine.render_template(
            "prompts/generate_tasks_user.j2",
            agent_roles=agent_roles,
            objectives=objectives,
            business_context=business_context,
        )

        # Create default streaming callbacks for verbose mode if none provided
        use_streaming = streaming_callbacks is not None or self.verbose
        callbacks_to_use = streaming_callbacks

        if self.verbose and streaming_callbacks is None:
            # Create verbose streaming callbacks with clear delimiters
            def on_token(token: str) -> None:
                click.echo(f"{token}", nl=False)

            def on_completion(response: str) -> None:
                click.echo("")  # New line after streaming

            callbacks_to_use = StreamingCallbacks(
                on_token=on_token, on_completion=on_completion
            )

        try:
            if use_streaming and callbacks_to_use:
                if self.verbose:
                    click.echo("📡 Sending task generation request with streaming...")
                    click.echo("═══ LLM Response Start ═══")
                result = self.llm_client.generate_streaming(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    streaming_callbacks=callbacks_to_use,
                    use_json_mode=True,
                    temperature=0.2,
                )
                if self.verbose:
                    click.echo("═══ LLM Response End ═══")
            else:
                if self.verbose:
                    click.echo("📡 Sending task generation request...")
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

            duration = time.time() - start_time

            if self.verbose:
                click.echo(f"✅ Task generation complete in {duration:.2f}s")
                for i, task in enumerate(tasks):
                    click.echo(f"   📋 Task {i+1}: {task.description[:50]}...")
                    click.echo(f"      👤 Assigned to: {task.agent}")
                    click.echo(
                        f"      📄 Expected output: {task.expected_output[:60]}..."
                    )

            return tasks

        except Exception as e:
            duration = time.time() - start_time
            if self.verbose:
                click.echo(f"❌ Task generation failed after {duration:.2f}s: {str(e)}")
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
            streaming_callbacks: Optional callbacks for streaming LLM responses

        Returns:
            Dict containing:
                - selected_tools: List[Dict] - Available tools with reasons
                - unavailable_tools: List[str] - Tools that couldn't be found

        Raises:
            GenerationError: If tool selection fails
        """
        if self.verbose:
            click.echo("🛠️  Selecting appropriate tools...")

        start_time = time.time()

        if self.verbose:
            click.echo(f"   🔍 Evaluating tools for needs: {', '.join(tools_needed)}")

        system_prompt = self.template_engine.render_template(
            "prompts/select_tools_system.j2"
        )

        user_prompt = self.template_engine.render_template(
            "prompts/select_tools_user.j2",
            tools_needed=tools_needed,
        )

        # Create default streaming callbacks for verbose mode if none provided
        use_streaming = streaming_callbacks is not None or self.verbose
        callbacks_to_use = streaming_callbacks

        if self.verbose and streaming_callbacks is None:
            # Create verbose streaming callbacks with clear delimiters
            def on_token(token: str) -> None:
                click.echo(f"{token}", nl=False)

            def on_completion(response: str) -> None:
                click.echo("")  # New line after streaming

            callbacks_to_use = StreamingCallbacks(
                on_token=on_token, on_completion=on_completion
            )

        try:
            if use_streaming and callbacks_to_use:
                if self.verbose:
                    click.echo("📡 Sending tool selection request with streaming...")
                    click.echo("═══ LLM Response Start ═══")
                result = self.llm_client.generate_streaming(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    streaming_callbacks=callbacks_to_use,
                    use_json_mode=True,
                    temperature=0.1,
                )
                if self.verbose:
                    click.echo("═══ LLM Response End ═══")
            else:
                if self.verbose:
                    click.echo("📡 Sending tool selection request...")
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

            duration = time.time() - start_time

            if self.verbose:
                click.echo(f"✅ Tool selection complete in {duration:.2f}s")
                selected_count = len(result["selected_tools"])
                unavailable_count = len(result["unavailable_tools"])
                click.echo(f"   ✅ Selected {selected_count} tools")
                if selected_count > 0:
                    for tool in result["selected_tools"]:
                        click.echo(
                            f"      🔧 {tool.get('name', 'Unknown')} - {tool.get('reason', 'No reason')[:50]}..."
                        )
                if unavailable_count > 0:
                    click.echo(
                        f"   ⚠️  {unavailable_count} tools unavailable: {', '.join(result['unavailable_tools'])}"
                    )

            return result

        except Exception as e:
            duration = time.time() - start_time
            if self.verbose:
                click.echo(f"❌ Tool selection failed after {duration:.2f}s: {str(e)}")
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

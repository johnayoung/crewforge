"""
Prompt templates for extracting CrewAI project specifications from natural language.

This module provides structured prompt templates that use liteLLM to extract
agent configurations, task definitions, tool requirements, and dependencies
from user prompts for CrewAI project generation.
"""

from typing import Dict, Any, List, Optional, Union
import json
from jsonschema import validate, ValidationError

from crewforge.llm import LLMClient, LLMError


class PromptTemplateError(Exception):
    """Raised when prompt template processing fails."""

    pass


class PromptTemplates:
    """
    Handles prompt templates for extracting CrewAI project specifications.

    Uses structured prompts with liteLLM to convert natural language project
    descriptions into structured specifications suitable for CrewAI project generation.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize prompt templates with LLM client.

        Args:
            llm_client: Configured LLMClient instance for API interactions
        """
        self.llm_client = llm_client
        self._project_spec_schema = self._build_project_spec_schema()

    @staticmethod
    def get_project_spec_schema() -> Dict[str, Any]:
        """Get the JSON schema for project specifications."""
        return PromptTemplates._build_project_spec_schema()

    @staticmethod
    def _build_project_spec_schema() -> Dict[str, Any]:
        """Build JSON schema for validating project specifications."""
        return {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Project name in kebab-case format",
                },
                "project_description": {
                    "type": "string",
                    "description": "Brief description of the project purpose",
                },
                "agents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "description": "Agent role/title",
                            },
                            "goal": {
                                "type": "string",
                                "description": "What the agent aims to accomplish",
                            },
                            "backstory": {
                                "type": "string",
                                "description": "Agent's background and expertise",
                            },
                            "tools": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of tools the agent uses",
                            },
                        },
                        "required": ["role", "goal", "backstory", "tools"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "What the task accomplishes",
                            },
                            "expected_output": {
                                "type": "string",
                                "description": "Format and content of task output",
                            },
                            "agent": {
                                "type": "string",
                                "description": "Agent role responsible for this task",
                            },
                        },
                        "required": ["description", "expected_output", "agent"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Python packages needed for the project",
                },
            },
            "required": [
                "project_name",
                "project_description",
                "agents",
                "tasks",
                "dependencies",
            ],
            "additionalProperties": False,
        }

    def _build_extraction_prompt(self, user_prompt: str) -> List[Dict[str, str]]:
        """
        Build structured prompt for extracting project specifications.

        Args:
            user_prompt: Natural language project description

        Returns:
            List of message dictionaries for LLM API
        """
        system_prompt = """You are a CrewAI project specification expert. Your task is to analyze natural language project descriptions and extract structured specifications for CrewAI multi-agent systems.

CrewAI is a framework for orchestrating teams of AI agents to work together on complex tasks. Each agent has:
- A specific role and expertise area
- Clear goals for what they accomplish
- A backstory that defines their personality and background  
- Tools they use to complete their work

Extract the following from user prompts:

1. PROJECT BASICS:
   - project_name: Convert to kebab-case (lowercase with hyphens)
   - project_description: Clear 1-2 sentence summary

2. AGENTS: Design 2-5 agents that would work together effectively
   - role: Clear, specific role title (e.g., "Research Analyst", "Content Writer")
   - goal: What the agent aims to accomplish
   - backstory: 1-2 sentences about their expertise and personality
   - tools: List of tools they need (be specific and realistic)

3. TASKS: Define 2-6 tasks that use the agents effectively
   - description: Clear task description
   - expected_output: Specific format and content expected
   - agent: Which agent role handles this task

4. DEPENDENCIES: Python packages needed (always include "crewai" plus relevant packages)

IMPORTANT GUIDELINES:
- Design agents that complement each other and can collaborate
- Ensure tasks flow logically and use agents effectively  
- Choose realistic, commonly available tools
- Include appropriate Python packages for the domain
- Keep agent backstories professional but distinct
- Make sure every agent has at least one task assigned

Respond with valid JSON only, following the exact structure provided."""

        user_message = f"""Project Description: {user_prompt}

Extract a complete CrewAI project specification from this description. Design agents that would work well together and tasks that utilize their unique capabilities effectively."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    async def extract_project_spec(self, user_prompt: str) -> Dict[str, Any]:
        """
        Extract CrewAI project specification from natural language prompt.

        Args:
            user_prompt: Natural language description of the desired project

        Returns:
            Structured project specification with agents, tasks, and dependencies

        Raises:
            PromptTemplateError: If extraction fails or output is invalid
        """
        # Validate input
        if not user_prompt or not user_prompt.strip():
            raise PromptTemplateError("Prompt cannot be empty")

        try:
            # Build structured prompt
            messages = self._build_extraction_prompt(user_prompt.strip())

            # Get structured response from LLM
            response = await self.llm_client.complete_structured(
                prompt=messages,
                schema=self._project_spec_schema,
                temperature=0.7,  # Allow some creativity in agent design
            )

            # Additional validation
            self._validate_project_spec(response)

            return response

        except LLMError as e:
            raise PromptTemplateError(
                f"Failed to extract project specification: {str(e)}"
            )
        except ValidationError as e:
            raise PromptTemplateError(
                f"Invalid project specification structure: {str(e)}"
            )
        except Exception as e:
            raise PromptTemplateError(f"Unexpected error during extraction: {str(e)}")

    def _validate_project_spec(self, spec: Dict[str, Any]) -> None:
        """
        Perform additional validation on extracted project specification.

        Args:
            spec: Project specification to validate

        Raises:
            PromptTemplateError: If validation fails
        """
        # Validate schema first
        try:
            validate(instance=spec, schema=self._project_spec_schema)
        except ValidationError as e:
            raise PromptTemplateError(
                f"Invalid project specification structure: {str(e)}"
            )

        # Additional business logic validation
        agents = spec.get("agents", [])
        tasks = spec.get("tasks", [])

        # Ensure all task agents exist
        agent_roles = {agent["role"] for agent in agents}
        for task in tasks:
            task_agent = task.get("agent")
            if task_agent not in agent_roles:
                raise PromptTemplateError(
                    f"Task references non-existent agent role: '{task_agent}'. "
                    f"Available agents: {list(agent_roles)}"
                )

        # Ensure project name is valid
        project_name = spec.get("project_name", "")
        if not project_name or " " in project_name or "_" in project_name:
            raise PromptTemplateError(
                f"Invalid project name '{project_name}'. Must be kebab-case (lowercase with hyphens)"
            )

        # Ensure dependencies include crewai
        dependencies = spec.get("dependencies", [])
        if "crewai" not in dependencies:
            raise PromptTemplateError("Dependencies must include 'crewai' package")

    @staticmethod
    def _validate_agent_structure(agent: Dict[str, Any]) -> bool:
        """
        Validate individual agent structure.

        Args:
            agent: Agent dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["role", "goal", "backstory", "tools"]

        # Check all required fields exist
        for field in required_fields:
            if field not in agent:
                return False

        # Check field types
        if not isinstance(agent.get("role"), str):
            return False
        if not isinstance(agent.get("goal"), str):
            return False
        if not isinstance(agent.get("backstory"), str):
            return False
        if not isinstance(agent.get("tools"), list):
            return False

        # Check tools are strings
        for tool in agent.get("tools", []):
            if not isinstance(tool, str):
                return False

        return True

    async def enhance_project_spec(
        self, spec: Dict[str, Any], enhancement_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance an existing project specification with additional context.

        Args:
            spec: Existing project specification
            enhancement_context: Additional context for enhancement

        Returns:
            Enhanced project specification

        Raises:
            PromptTemplateError: If enhancement fails
        """
        # Validate input specification
        self._validate_project_spec(spec)

        if not enhancement_context:
            return spec  # No enhancement needed

        try:
            enhancement_prompt = f"""Given this CrewAI project specification:

{json.dumps(spec, indent=2)}

Enhancement Context: {enhancement_context}

Enhance the specification by:
1. Improving agent backstories and goals based on the context
2. Adding more specific tools if appropriate
3. Refining task descriptions and expected outputs
4. Adding relevant dependencies if needed

Return the complete enhanced specification as JSON."""

            messages = [{"role": "user", "content": enhancement_prompt}]

            enhanced_spec = await self.llm_client.complete_structured(
                prompt=messages,
                schema=self._project_spec_schema,
                temperature=0.5,  # Less creativity for enhancement
            )

            # Validate enhanced specification
            self._validate_project_spec(enhanced_spec)

            return enhanced_spec

        except Exception as e:
            raise PromptTemplateError(
                f"Failed to enhance project specification: {str(e)}"
            )

"""
Enhancement Engine - Jinja2 template system for CrewAI project customization

This module provides intelligent customization of generated CrewAI projects
using Jinja2 templates for domain-specific configurations of agents and tasks.
"""

import asyncio
import json
import logging
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader, Template
from jinja2.exceptions import TemplateNotFound, TemplateError

from .llm import LLMClient


class EnhancementError(Exception):
    """Base exception for enhancement engine errors."""

    pass


class TemplateNotFoundError(EnhancementError):
    """Exception raised when a template file cannot be found."""

    pass


class TemplateRenderError(EnhancementError):
    """Exception raised when template rendering fails."""

    pass


class EnhancementEngine:
    """
    Intelligent project enhancement using Jinja2 templates.

    This class provides methods to enhance CrewAI project configurations
    with domain-specific customizations using Jinja2 template rendering.
    """

    # Valid template categories
    VALID_CATEGORIES = {"agents", "tasks"}

    # Required project structure paths
    PROJECT_STRUCTURE_PATHS = [
        "src/{project_name}/config",
        "src/{project_name}/tools",
        "src/{project_name}",
    ]

    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the enhancement engine.

        Args:
            templates_dir: Directory containing Jinja2 templates
            logger: Optional logger instance

        Raises:
            EnhancementError: If templates directory is invalid
        """
        self.logger = logger or logging.getLogger(__name__)

        # Set up templates directory
        if templates_dir is None:
            # Default to templates directory in the package
            package_dir = Path(__file__).parent
            self.templates_dir = package_dir / "templates"
        else:
            self.templates_dir = Path(templates_dir)

        if not self.templates_dir.exists():
            raise EnhancementError(
                f"Templates directory does not exist: {self.templates_dir}"
            )

        # Ensure templates directory structure
        self._ensure_template_directories()

        # Initialize Jinja2 environment
        self.environment = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=False,  # We're generating YAML, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

        self.logger.info(
            f"Enhancement engine initialized with templates from {self.templates_dir}"
        )

    def _ensure_template_directories(self) -> None:
        """Ensure required template directories exist."""
        for category in self.VALID_CATEGORIES:
            category_dir = self.templates_dir / category
            if not category_dir.exists():
                category_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created template directory: {category_dir}")

    def get_available_templates(self, category: str) -> List[str]:
        """
        Get list of available templates for a category.

        Args:
            category: Template category ('agents' or 'tasks')

        Returns:
            List of available template names (without .j2 extension)

        Raises:
            ValueError: If category is invalid
        """
        if category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Invalid template category: {category}. Must be one of {self.VALID_CATEGORIES}"
            )

        category_dir = self.templates_dir / category
        if not category_dir.exists():
            return []

        templates = []
        for template_file in category_dir.glob("*.j2"):
            # Remove .yaml.j2 or .j2 extension to get template name
            template_name = template_file.name
            if template_name.endswith(".yaml.j2"):
                template_name = template_name[:-8]  # Remove .yaml.j2
            elif template_name.endswith(".j2"):
                template_name = template_name[:-3]  # Remove .j2
            templates.append(template_name)

        self.logger.debug(
            f"Found {len(templates)} templates in category '{category}': {templates}"
        )
        return sorted(templates)

    async def generate_agent_role(
        self,
        agent_spec: Dict[str, Any],
        project_spec: Dict[str, Any],
        llm_client: LLMClient,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate an intelligent agent role using liteLLM.

        Args:
            agent_spec: Agent specification with role and description
            project_spec: Project specification with context
            llm_client: LLM client for generation
            **kwargs: Additional parameters for LLM generation

        Returns:
            Dictionary with generated role, goal, and backstory

        Raises:
            json.JSONDecodeError: If LLM returns invalid JSON
            KeyError: If required fields are missing from LLM response
            LLMError: If LLM request fails
        """
        prompt = self._create_agent_generation_prompt(agent_spec, project_spec)

        self.logger.debug(
            f"Generating agent role for '{agent_spec.get('role', 'unknown')}' "
            f"in project '{project_spec.get('project_name', 'unknown')}'"
        )

        # Get LLM response
        response = await llm_client.complete(prompt, **kwargs)

        # Parse JSON response
        try:
            agent_data = json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response from LLM: {response}")
            raise

        # Validate required fields
        required_fields = ["role", "goal", "backstory"]
        for field in required_fields:
            if field not in agent_data:
                self.logger.error(f"Missing required field '{field}' in LLM response")
                raise KeyError(f"Missing required field: {field}")

        self.logger.info(f"Successfully generated agent role: {agent_data['role']}")
        return agent_data

    async def generate_agent_roles(
        self, project_spec: Dict[str, Any], llm_client: LLMClient, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate intelligent agent roles for all agents in the project specification.

        Args:
            project_spec: Project specification with agents list
            llm_client: LLM client for generation
            **kwargs: Additional parameters for LLM generation

        Returns:
            List of generated agent configurations

        Raises:
            ValueError: If project_spec doesn't contain agents
            Various exceptions from generate_agent_role
        """
        agents = project_spec.get("agents", [])
        if not agents:
            raise ValueError("Project specification must contain 'agents' list")

        self.logger.info(f"Generating roles for {len(agents)} agents")

        generated_agents = []
        for agent_spec in agents:
            agent_role = await self.generate_agent_role(
                agent_spec, project_spec, llm_client, **kwargs
            )
            generated_agents.append(agent_role)

        self.logger.info(f"Successfully generated {len(generated_agents)} agent roles")
        return generated_agents

    def _create_agent_generation_prompt(
        self, agent_spec: Dict[str, Any], project_spec: Dict[str, Any]
    ) -> str:
        """
        Create a prompt for LLM agent generation.

        Args:
            agent_spec: Agent specification with role and description
            project_spec: Project specification with context

        Returns:
            Formatted prompt string for LLM
        """
        project_name = project_spec.get("project_name", "Unknown Project")
        project_description = project_spec.get(
            "description", "No description available"
        )
        domain = project_spec.get("domain", "general")
        project_type = project_spec.get("project_type", "general")

        agent_role = agent_spec.get("role", "Unknown Role")
        agent_description = agent_spec.get("description", "No description available")

        prompt = f"""You are an expert at creating CrewAI agent configurations. Generate an intelligent agent role specification for a CrewAI project.

Project Context:
- Project Name: {project_name}
- Description: {project_description}
- Domain: {domain}
- Type: {project_type}

Agent to Generate:
- Role: {agent_role}
- Description: {agent_description}

Requirements:
1. Create a professional, specific role name that fits the project domain
2. Write a clear, actionable goal statement that aligns with the project objectives
3. Generate a compelling backstory that establishes expertise and credibility
4. The backstory should be 2-3 sentences and include relevant experience
5. Make the agent feel like a real professional with domain expertise

Output Format:
Return ONLY a valid JSON object with exactly these fields:
{{
    "role": "Specific professional role title",
    "goal": "Clear, actionable goal statement for this agent",
    "backstory": "Compelling 2-3 sentence backstory establishing expertise and credibility"
}}

Generate the agent configuration now:"""

        return prompt

    async def enhance_project_with_generated_agents(
        self,
        project_path: Path,
        project_spec: Dict[str, Any],
        llm_client: LLMClient,
        template_name: str = "default",
        **kwargs,
    ) -> bool:
        """
        Enhance a CrewAI project by generating intelligent agent roles and applying them.

        Args:
            project_path: Path to the CrewAI project
            project_spec: Project specification
            llm_client: LLM client for agent generation
            template_name: Template to use for rendering
            **kwargs: Additional parameters

        Returns:
            True if enhancement was successful, False otherwise
        """
        try:
            # Generate intelligent agent roles
            generated_agents = await self.generate_agent_roles(
                project_spec, llm_client, **kwargs
            )

            # Create enhancement context with generated agents
            enhancement_context = {
                "agents": generated_agents,
                "project_name": project_spec.get("project_name", "unknown"),
                "project_type": project_spec.get("project_type", "general"),
                "domain": project_spec.get("domain", "general"),
                **kwargs,
            }

            # Apply enhancements using existing template system
            result = self.enhance_agents_config(
                project_path, enhancement_context, template_name
            )

            return result.get("success", False)

        except Exception as e:
            self.logger.error(
                f"Failed to enhance project with generated agents: {str(e)}"
            )
            return False

    async def generate_task_definition(
        self,
        task_spec: Dict[str, Any],
        project_spec: Dict[str, Any],
        llm_client: LLMClient,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate an intelligent task definition using liteLLM.

        Args:
            task_spec: Task specification with name, description, and agent
            project_spec: Project specification with context
            llm_client: LLM client for generation
            **kwargs: Additional parameters for LLM generation

        Returns:
            Dictionary with generated description, expected_output, context, and tools

        Raises:
            json.JSONDecodeError: If LLM returns invalid JSON
            KeyError: If required fields are missing from LLM response
            LLMError: If LLM request fails
        """
        prompt = self._create_task_generation_prompt(task_spec, project_spec)

        self.logger.debug(
            f"Generating task definition for '{task_spec.get('name', 'unknown')}' "
            f"in project '{project_spec.get('project_name', 'unknown')}'"
        )

        # Get LLM response
        response = await llm_client.complete(prompt, **kwargs)

        # Parse JSON response
        try:
            task_data = json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response from LLM: {response}")
            raise

        # Validate required fields
        required_fields = ["description", "expected_output", "context", "tools"]
        for field in required_fields:
            if field not in task_data:
                self.logger.error(f"Missing required field '{field}' in LLM response")
                raise KeyError(f"Missing required field: {field}")

        self.logger.info(
            f"Successfully generated task definition: {task_spec.get('name', 'unknown')}"
        )
        return task_data

    async def generate_task_definitions(
        self, project_spec: Dict[str, Any], llm_client: LLMClient, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate intelligent task definitions for all tasks in the project specification.

        Args:
            project_spec: Project specification with tasks list
            llm_client: LLM client for generation
            **kwargs: Additional parameters for LLM generation

        Returns:
            List of generated task configurations

        Raises:
            ValueError: If project_spec doesn't contain tasks
            Various exceptions from generate_task_definition
        """
        tasks = project_spec.get("tasks", [])
        if not tasks:
            raise ValueError("Project specification must contain 'tasks' list")

        self.logger.info(f"Generating definitions for {len(tasks)} tasks")

        generated_tasks = []
        for i, task in enumerate(tasks):
            # Convert string tasks to task specs if needed
            if isinstance(task, str):
                task_spec = {
                    "name": f"task_{i+1}",
                    "description": task,
                    "agent": f"agent_{(i % len(project_spec.get('agents', ['default'])))+1}",
                }
            else:
                task_spec = task

            task_definition = await self.generate_task_definition(
                task_spec, project_spec, llm_client, **kwargs
            )

            # Combine task spec with generated definition
            enhanced_task = {**task_spec, **task_definition}
            generated_tasks.append(enhanced_task)

        self.logger.info(
            f"Successfully generated {len(generated_tasks)} task definitions"
        )
        return generated_tasks

    def _create_task_generation_prompt(
        self, task_spec: Dict[str, Any], project_spec: Dict[str, Any]
    ) -> str:
        """
        Create a prompt for LLM task generation.

        Args:
            task_spec: Task specification with name, description, and agent
            project_spec: Project specification with context

        Returns:
            Formatted prompt string for LLM
        """
        project_name = project_spec.get("project_name", "Unknown Project")
        project_description = project_spec.get(
            "description", "No description available"
        )
        domain = project_spec.get("domain", "general")
        project_type = project_spec.get("project_type", "general")

        task_name = task_spec.get("name", "Unknown Task")
        task_description = task_spec.get("description", "No description available")
        assigned_agent = task_spec.get("agent", "unknown")

        # Get available agents for context
        agents = project_spec.get("agents", [])
        agent_context = ""
        if agents:
            agent_list = [
                f"- {agent.get('role', agent.get('name', str(agent)))}"
                for agent in agents
                if isinstance(agent, dict)
            ]
            if agent_list:
                agent_context = f"\n\nAvailable Agents:\n" + "\n".join(agent_list)

        # Add additional context fields if they exist
        additional_context = ""
        context_fields = ["industry", "target_audience", "objectives"]
        for field in context_fields:
            if field in project_spec:
                value = project_spec[field]
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                additional_context += f"\n- {field.replace('_', ' ').title()}: {value}"

        if additional_context:
            additional_context = f"\n\nAdditional Context:{additional_context}"

        prompt = f"""You are an expert at creating CrewAI task configurations. Generate an intelligent task definition for a CrewAI project.

Project Context:
- Project Name: {project_name}
- Description: {project_description}
- Domain: {domain}
- Type: {project_type}{additional_context}{agent_context}

Task to Generate:
- Name: {task_name}
- Description: {task_description}
- Assigned Agent: {assigned_agent}

Requirements:
1. Create a comprehensive, actionable task description that clearly defines what needs to be accomplished
2. Specify detailed expected output that describes exactly what deliverables are required
3. Provide relevant context items (as a list) that help the agent understand the task scope and priorities
4. Suggest appropriate tools (as a list) that would be useful for completing this task
5. Make the task definition specific to the project domain and aligned with project objectives

Output Format:
Return ONLY a valid JSON object with exactly these fields:
{{
    "description": "Comprehensive, actionable task description",
    "expected_output": "Detailed specification of expected deliverables and output format",
    "context": ["Context item 1", "Context item 2", "Context item 3"],
    "tools": ["tool1", "tool2", "tool3"]
}}

Generate the task configuration now:"""

        return prompt

    def render_agent_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render an agent configuration template.

        Args:
            template_name: Name of the template to render
            context: Template context data

        Returns:
            Rendered template content as string

        Raises:
            TemplateNotFoundError: If template doesn't exist
            TemplateRenderError: If template rendering fails
        """
        template_path = f"agents/{template_name}.yaml.j2"
        return self._render_template(
            template_path, context, f"Agent template '{template_name}' not found"
        )

    def render_task_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Render a task configuration template.

        Args:
            template_name: Name of the template to render
            context: Template context data

        Returns:
            Rendered template content as string

        Raises:
            TemplateNotFoundError: If template doesn't exist
            TemplateRenderError: If template rendering fails
        """
        template_path = f"tasks/{template_name}.yaml.j2"
        return self._render_template(
            template_path, context, f"Task template '{template_name}' not found"
        )

    def _render_template(
        self, template_path: str, context: Dict[str, Any], not_found_message: str
    ) -> str:
        """
        Internal method to render a template.

        Args:
            template_path: Path to template relative to templates directory
            context: Template context data
            not_found_message: Error message for template not found

        Returns:
            Rendered template content

        Raises:
            TemplateNotFoundError: If template doesn't exist
            TemplateRenderError: If template rendering fails
        """
        try:
            template = self.environment.get_template(template_path)
            rendered = template.render(**context)
            self.logger.debug(f"Successfully rendered template: {template_path}")
            return rendered

        except TemplateNotFound:
            self.logger.error(f"Template not found: {template_path}")
            raise TemplateNotFoundError(not_found_message)

        except TemplateError as e:
            self.logger.error(
                f"Template rendering failed for {template_path}: {str(e)}"
            )
            raise TemplateRenderError(
                f"Failed to render template '{template_path}': {str(e)}"
            )

        except Exception as e:
            self.logger.error(
                f"Unexpected error rendering template {template_path}: {str(e)}"
            )
            raise TemplateRenderError(
                f"Failed to render template '{template_path}': {str(e)}"
            )

    def enhance_agents_config(
        self,
        project_path: Path,
        enhancement_context: Dict[str, Any],
        template_name: str = "default",
    ) -> Dict[str, Any]:
        """
        Enhance the agents.yaml configuration file.

        Args:
            project_path: Path to the CrewAI project
            enhancement_context: Context data for template rendering
            template_name: Name of template to use (default: "default")

        Returns:
            Dictionary with success status and details
        """
        try:
            # Validate project structure
            validation = self._validate_project_structure(project_path)
            if not validation["valid"]:
                self.logger.error(
                    f"Project structure validation failed: {validation['error']}"
                )
                return {
                    "success": False,
                    "error": validation["error"],
                    "backup_created": False,
                }

            config_path = validation["config_path"]
            agents_file = config_path / "agents.yaml"

            if not agents_file.exists():
                error_msg = f"agents.yaml not found in {config_path}"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg, "backup_created": False}

            # Create backup
            backup_path = self._create_backup(agents_file)

            # Render enhanced configuration
            enhanced_content = self.render_agent_template(
                template_name, enhancement_context
            )

            # Write enhanced configuration
            agents_file.write_text(enhanced_content, encoding="utf-8")

            self.logger.info(f"Enhanced agents configuration in {agents_file}")

            return {
                "success": True,
                "enhanced_file": agents_file,
                "backup_file": backup_path,
                "backup_created": True,
                "template_used": template_name,
            }

        except Exception as e:
            self.logger.error(f"Failed to enhance agents configuration: {str(e)}")
            return {"success": False, "error": str(e), "backup_created": False}

    def enhance_tasks_config(
        self,
        project_path: Path,
        enhancement_context: Dict[str, Any],
        template_name: str = "default",
    ) -> Dict[str, Any]:
        """
        Enhance the tasks.yaml configuration file.

        Args:
            project_path: Path to the CrewAI project
            enhancement_context: Context data for template rendering
            template_name: Name of template to use (default: "default")

        Returns:
            Dictionary with success status and details
        """
        try:
            # Validate project structure
            validation = self._validate_project_structure(project_path)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "backup_created": False,
                }

            config_path = validation["config_path"]
            tasks_file = config_path / "tasks.yaml"

            if not tasks_file.exists():
                return {
                    "success": False,
                    "error": f"tasks.yaml not found in {config_path}",
                    "backup_created": False,
                }

            # Create backup
            backup_path = self._create_backup(tasks_file)

            # Render enhanced configuration
            enhanced_content = self.render_task_template(
                template_name, enhancement_context
            )

            # Write enhanced configuration
            tasks_file.write_text(enhanced_content, encoding="utf-8")

            self.logger.info(f"Enhanced tasks configuration in {tasks_file}")

            return {
                "success": True,
                "enhanced_file": tasks_file,
                "backup_file": backup_path,
                "backup_created": True,
                "template_used": template_name,
            }

        except Exception as e:
            self.logger.error(f"Failed to enhance tasks configuration: {str(e)}")
            return {"success": False, "error": str(e), "backup_created": False}

    def _validate_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """
        Validate that the project has the expected CrewAI structure.

        Args:
            project_path: Path to project directory

        Returns:
            Dictionary with validation result
        """
        if not project_path.exists() or not project_path.is_dir():
            return {
                "valid": False,
                "error": f"Project path does not exist or is not a directory: {project_path}",
            }

        # Find project name from directory structure
        src_dir = project_path / "src"
        if not src_dir.exists():
            return {
                "valid": False,
                "error": f"Invalid CrewAI project structure: missing 'src' directory in {project_path}",
            }

        # Look for project subdirectory in src
        project_subdirs = [
            d for d in src_dir.iterdir() if d.is_dir() and d.name != "__pycache__"
        ]
        if not project_subdirs:
            return {
                "valid": False,
                "error": f"Invalid CrewAI project structure: no project subdirectory found in {src_dir}",
            }

        # Use the first project subdirectory (should be only one)
        project_subdir = project_subdirs[0]
        config_path = project_subdir / "config"

        if not config_path.exists():
            return {
                "valid": False,
                "error": f"Invalid CrewAI project structure: missing 'config' directory in {project_subdir}",
            }

        return {
            "valid": True,
            "project_name": project_subdir.name,
            "config_path": config_path,
            "project_subdir": project_subdir,
        }

    def _create_backup(self, file_path: Path) -> Path:
        """
        Create a backup copy of a file.

        Args:
            file_path: Path to file to backup

        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.parent / f"{file_path.stem}_{timestamp}.backup"

        shutil.copy2(file_path, backup_path)
        self.logger.debug(f"Created backup: {backup_path}")

        return backup_path

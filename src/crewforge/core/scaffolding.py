"""CrewAI project scaffolding and file population system.

This module implements the ProjectScaffolder class that integrates CrewAI scaffolding
with complete file population using generated configurations.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .generator import GenerationEngine
from .templates import TemplateEngine
from ..models import AgentConfig, TaskConfig, GenerationRequest


class ScaffoldingError(Exception):
    """Custom exception for scaffolding-related errors."""

    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


class ProjectScaffolder:
    """Manages CrewAI project creation and file population.

    This class orchestrates the complete project generation process:
    1. Uses GenerationEngine to create agent/task configurations from prompts
    2. Calls `crewai create crew <name>` for project scaffolding
    3. Uses TemplateEngine to populate project files with generated content
    4. Provides error handling and cleanup on failures
    """

    def __init__(
        self,
        generation_engine: Optional[GenerationEngine] = None,
        template_engine: Optional[TemplateEngine] = None,
    ):
        """Initialize ProjectScaffolder.

        Args:
            generation_engine: GenerationEngine instance for AI generation.
                              If None, creates default instance.
            template_engine: TemplateEngine instance for file population.
                           If None, creates default instance.
        """
        if generation_engine is None:
            generation_engine = GenerationEngine()
        if template_engine is None:
            template_engine = TemplateEngine()

        self.generation_engine = generation_engine
        self.template_engine = template_engine

    def create_crewai_project(self, project_name: str, parent_dir: Path) -> Path:
        """Create a new CrewAI project using the CrewAI CLI.

        Args:
            project_name: Name for the new CrewAI project
            parent_dir: Directory where the project should be created

        Returns:
            Path to the created project directory

        Raises:
            ScaffoldingError: If project creation fails
        """
        try:
            # Run the crewai create crew command
            result = subprocess.run(
                ["crewai", "create", "crew", project_name],
                cwd=parent_dir,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                raise ScaffoldingError(
                    f"CrewAI project creation failed with exit code {result.returncode}: {error_msg}"
                )

            project_path = parent_dir / project_name
            if not project_path.exists():
                raise ScaffoldingError(
                    f"CrewAI project directory was not created: {project_path}"
                )

            return project_path

        except subprocess.SubprocessError as e:
            raise ScaffoldingError(
                f"Failed to execute CrewAI command: {str(e)}", original_exception=e
            )
        except Exception as e:
            raise ScaffoldingError(
                f"Unexpected error creating CrewAI project: {str(e)}",
                original_exception=e,
            )

    def populate_project_files(
        self,
        project_path: Path,
        agents: List[AgentConfig],
        tasks: List[TaskConfig],
        tools: Dict[str, Any],
    ) -> None:
        """Populate CrewAI project files with generated configurations.

        Args:
            project_path: Path to the CrewAI project directory
            agents: List of generated agent configurations
            tasks: List of generated task configurations
            tools: Dictionary with selected tools information

        Raises:
            ScaffoldingError: If file population fails
        """
        try:
            # Validate project structure
            self._validate_project_structure(project_path)

            # Get the module path for file population
            module_path = self._get_project_module_path(project_path)

            # Populate each file type using templates
            self.template_engine.populate_template(
                "agents.py.j2", module_path / "agents.py", agents=agents
            )

            self.template_engine.populate_template(
                "tasks.py.j2", module_path / "tasks.py", tasks=tasks, agents=agents
            )

            self.template_engine.populate_template(
                "tools.py.j2", module_path / "tools.py", tools=tools["selected_tools"]
            )

            self.template_engine.populate_template(
                "crew.py.j2",
                module_path / "crew.py",
                agents=agents,
                tasks=tasks,
                tools=tools["selected_tools"],
            )

        except Exception as e:
            if isinstance(e, ScaffoldingError):
                raise
            raise ScaffoldingError(
                f"Failed to populate project files: {str(e)}", original_exception=e
            )

    def generate_project(self, request: GenerationRequest, output_dir: Path) -> Path:
        """Generate a complete CrewAI project from a generation request.

        This orchestrates the entire generation pipeline:
        1. Analyze the prompt for requirements
        2. Generate agent configurations
        3. Generate task definitions
        4. Select appropriate tools
        5. Create CrewAI project scaffolding
        6. Populate project files with generated content

        Args:
            request: GenerationRequest with prompt and project name
            output_dir: Directory where the project should be created

        Returns:
            Path to the generated project directory

        Raises:
            ScaffoldingError: If any step of the generation process fails
        """
        project_path = None

        try:
            # Step 1: Analyze prompt for crew requirements
            try:
                prompt_analysis = self.generation_engine.analyze_prompt(request.prompt)
            except Exception as e:
                raise ScaffoldingError(
                    f"Failed to analyze prompt: {str(e)}", original_exception=e
                )

            # Step 2: Generate agent configurations
            try:
                agents = self.generation_engine.generate_agents(prompt_analysis)
            except Exception as e:
                raise ScaffoldingError(
                    f"Failed to generate agents: {str(e)}", original_exception=e
                )

            # Step 3: Generate task definitions
            try:
                tasks = self.generation_engine.generate_tasks(agents, prompt_analysis)
            except Exception as e:
                raise ScaffoldingError(
                    f"Failed to generate tasks: {str(e)}", original_exception=e
                )

            # Step 4: Select appropriate tools
            try:
                tools_needed = prompt_analysis.get("tools_needed", [])
                tools = self.generation_engine.select_tools(tools_needed)
            except Exception as e:
                raise ScaffoldingError(
                    f"Failed to select tools: {str(e)}", original_exception=e
                )

            # Step 5: Create CrewAI project scaffolding
            project_path = self.create_crewai_project(request.project_name, output_dir)

            # Step 6: Populate project files with generated content
            self.populate_project_files(project_path, agents, tasks, tools)

            return project_path

        except Exception as e:
            # Clean up on failure
            if project_path and project_path.exists():
                shutil.rmtree(project_path, ignore_errors=True)

            if isinstance(e, ScaffoldingError):
                raise
            raise ScaffoldingError(
                f"Project generation failed: {str(e)}", original_exception=e
            )

    def _validate_project_structure(self, project_path: Path) -> None:
        """Validate that the CrewAI project has the expected structure.

        Args:
            project_path: Path to the project directory

        Raises:
            ScaffoldingError: If project structure is invalid
        """
        if not project_path.exists():
            raise ScaffoldingError(f"Project directory does not exist: {project_path}")

        src_dir = project_path / "src"
        if not src_dir.exists() or not src_dir.is_dir():
            raise ScaffoldingError(
                f"CrewAI project structure is invalid: missing src directory in {project_path}"
            )

        # Find the module directory (should be the only directory under src)
        module_dirs = [d for d in src_dir.iterdir() if d.is_dir()]
        if not module_dirs:
            raise ScaffoldingError(
                f"CrewAI project structure is invalid: no module directory found in {src_dir}"
            )

    def _get_project_module_path(self, project_path: Path) -> Path:
        """Get the path to the project's Python module directory.

        Args:
            project_path: Path to the project directory

        Returns:
            Path to the module directory under src/

        Raises:
            ScaffoldingError: If module path cannot be determined
        """
        src_dir = project_path / "src"
        module_dirs = [d for d in src_dir.iterdir() if d.is_dir()]

        if not module_dirs:
            raise ScaffoldingError(f"No module directory found in {src_dir}")

        # Return the first (and expected only) module directory
        return module_dirs[0]

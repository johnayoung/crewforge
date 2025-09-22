"""CrewAI project scaffolding and file population system.

This module implements the ProjectScaffolder class that integrates CrewAI scaffolding
with complete file population using generated configurations.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .generator import GenerationEngine
from .progress import (
    ProgressTracker,
    ProgressStatus,
    StreamingCallbacks,
    get_standard_generation_steps,
)
from .templates import TemplateEngine
from ..models import AgentConfig, TaskConfig, GenerationRequest

# Configure logging
logger = logging.getLogger(__name__)


class ScaffoldingError(Exception):
    """Custom exception for scaffolding-related errors."""

    def __init__(
        self,
        message: str,
        original_exception: Exception | None = None,
        error_type: str = "general",
    ):
        super().__init__(message)
        self.original_exception = original_exception
        self.error_type = error_type


class CrewAICommandError(ScaffoldingError):
    """Raised when CrewAI CLI command fails."""

    def __init__(
        self,
        command: List[str],
        return_code: int,
        stderr: str,
        original_exception: Exception | None = None,
    ):
        self.command = command
        self.return_code = return_code
        self.stderr = stderr
        message = (
            f"CrewAI command failed: {' '.join(command)} (exit code {return_code})"
        )
        if stderr:
            message += f"\nError output: {stderr}"
        super().__init__(message, original_exception, "command")


class FileSystemError(ScaffoldingError):
    """Raised when file system operations fail."""

    def __init__(
        self,
        message: str,
        path: Path | None = None,
        original_exception: Exception | None = None,
    ):
        self.path = path
        if path:
            message = f"{message}: {path}"
        super().__init__(message, original_exception, "filesystem")


class ProjectStructureError(ScaffoldingError):
    """Raised when project structure validation fails."""

    def __init__(
        self,
        message: str,
        project_path: Path | None = None,
        original_exception: Exception | None = None,
    ):
        self.project_path = project_path
        if project_path:
            message = f"{message} in project: {project_path}"
        super().__init__(message, original_exception, "structure")


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
        progress_tracker: Optional[ProgressTracker] = None,
    ):
        """Initialize ProjectScaffolder.

        Args:
            generation_engine: GenerationEngine instance for AI generation.
                              If None, creates default instance.
            template_engine: TemplateEngine instance for file population.
                           If None, creates default instance.
            progress_tracker: ProgressTracker instance for progress tracking.
                            If None, creates default with basic steps.
        """
        if generation_engine is None:
            generation_engine = GenerationEngine()
        if template_engine is None:
            template_engine = TemplateEngine()
        if progress_tracker is None:
            # Create default progress tracker with standard generation steps
            progress_tracker = ProgressTracker(get_standard_generation_steps())

        self.generation_engine = generation_engine
        self.template_engine = template_engine
        self.progress_tracker = progress_tracker

    def _normalize_project_name(self, project_name: str) -> str:
        """Convert project name to the directory name that CrewAI will create.

        CrewAI converts hyphens to underscores in directory names.

        Args:
            project_name: Original project name (e.g., "content-research")

        Returns:
            Normalized directory name (e.g., "content_research")
        """
        return project_name.replace("-", "_")

    def create_crewai_project(self, project_name: str, parent_dir: Path) -> Path:
        """Create a new CrewAI project using the CrewAI CLI.

        Args:
            project_name: Name for the new CrewAI project
            parent_dir: Directory where the project should be created

        Returns:
            Path to the created project directory

        Raises:
            CrewAICommandError: If CrewAI CLI command fails
            FileSystemError: If file system operations fail
        """
        # Validate inputs
        if not project_name or not project_name.strip():
            raise ValueError("Project name cannot be empty")

        if not parent_dir.exists():
            raise FileSystemError("Parent directory does not exist", parent_dir)

        if not os.access(parent_dir, os.W_OK):
            raise FileSystemError("Parent directory is not writable", parent_dir)

        # Check if project already exists (using normalized name that CrewAI will create)
        normalized_name = self._normalize_project_name(project_name)
        project_path = parent_dir / normalized_name
        if project_path.exists():
            raise FileSystemError("Project directory already exists", project_path)

        # Check if CrewAI CLI is available
        try:
            result = subprocess.run(
                ["crewai", "--version"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=10,
            )
            if result.returncode != 0:
                raise CrewAICommandError(
                    ["crewai", "--version"],
                    result.returncode,
                    "CrewAI CLI not found or not working properly",
                )
        except subprocess.TimeoutExpired:
            raise CrewAICommandError(
                ["crewai", "--version"], -1, "CrewAI CLI command timed out"
            )
        except FileNotFoundError:
            raise CrewAICommandError(
                ["crewai", "--version"],
                -1,
                "CrewAI CLI not found. Please install with: pip install crewai",
            )
        except subprocess.SubprocessError as e:
            raise CrewAICommandError(
                ["crewai", "--version"],
                -1,
                f"Failed to execute CrewAI command: {str(e)}",
            )

        try:
            logger.info(f"Creating CrewAI project '{project_name}' in {parent_dir}")

            # Run the crewai create crew command with timeout and skip provider prompt
            command = ["crewai", "create", "crew", project_name, "--skip_provider"]
            result = subprocess.run(
                command,
                cwd=parent_dir,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,  # Prevent hanging on stdin reads
                timeout=120,  # 2 minute timeout
                check=False,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"CrewAI command failed: {error_msg}")
                raise CrewAICommandError(command, result.returncode, error_msg)

            # Verify project was created
            if not project_path.exists():
                raise FileSystemError(
                    "Project directory was not created after successful command",
                    project_path,
                )

            # Verify basic project structure
            expected_files = ["pyproject.toml", f"src/{normalized_name}/__init__.py"]
            for expected_file in expected_files:
                file_path = project_path / expected_file
                if not file_path.exists():
                    logger.warning(f"Expected project file not found: {file_path}")

            logger.info(f"Successfully created CrewAI project at {project_path}")
            return project_path

        except subprocess.TimeoutExpired:
            logger.error("CrewAI command timed out")
            raise CrewAICommandError(command, -1, "Command timed out after 2 minutes")

        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error: {e}")
            raise CrewAICommandError(command, -1, f"Subprocess error: {str(e)}", e)

        except CrewAICommandError as e:
            # Re-raise CrewAI command errors as ScaffoldingError with specific message
            logger.error(f"CrewAI project creation failed: {e}")
            raise ScaffoldingError(
                f"CrewAI project creation failed: {str(e)}",
                original_exception=e,
            )

        except Exception as e:
            logger.error(f"Unexpected error creating project: {e}")
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
        project_name: str,
    ) -> None:
        """Populate CrewAI project files with generated configurations.

        Args:
            project_path: Path to the CrewAI project directory
            agents: List of generated agent configurations
            tasks: List of generated task configurations
            tools: Dictionary with selected tools information

        Raises:
            ProjectStructureError: If project structure validation fails
            FileSystemError: If file operations fail
        """
        try:
            logger.info(f"Populating project files in {project_path}")

            # Validate project structure
            self._validate_project_structure(project_path)

            # Get the module path for file population
            module_path = self._get_project_module_path(project_path)

            # Ensure module directory exists and is writable
            if not module_path.exists():
                raise ProjectStructureError(
                    "Project module directory not found", module_path
                )

            if not os.access(module_path, os.W_OK):
                raise FileSystemError(
                    "Project module directory is not writable", module_path
                )

            # Check disk space before operations
            self._check_disk_space(module_path)

            # Populate each file type using templates with individual error handling
            # Modern CrewAI uses YAML configs and a single crew.py file
            config_dir = module_path / "config"
            tools_dir = module_path / "tools"
            file_operations = [
                # YAML configuration files in config/ directory
                ("agents.yaml.j2", config_dir / "agents.yaml", {"agents": agents}),
                (
                    "tasks.yaml.j2",
                    config_dir / "tasks.yaml",
                    {"tasks": tasks, "agents": agents},
                ),
                # Main crew.py file (overwrites the scaffold version)
                (
                    "crew.py.j2",
                    module_path / "crew.py",
                    {
                        "agents": agents,
                        "tasks": tasks,
                        "tools": tools["selected_tools"],
                        "project_name": project_name,
                    },
                ),
                # Custom tools (if any) - placed in tools/ directory
                (
                    "tools.py.j2",
                    tools_dir / "custom_tool.py",
                    {"tools": tools["selected_tools"]},
                ),
            ]

            for template_name, output_path, template_vars in file_operations:
                try:
                    # output_path is now the full path, not relative to module_path
                    logger.debug(f"Populating {output_path}")

                    # Ensure parent directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Backup existing file if it exists
                    backup_path = None
                    if output_path.exists():
                        backup_path = output_path.with_suffix(
                            output_path.suffix + ".bak"
                        )
                        shutil.copy2(output_path, backup_path)
                        logger.debug(f"Created backup: {backup_path}")

                    # Populate template
                    self.template_engine.populate_template(
                        template_name, output_path, **template_vars
                    )

                    # Remove backup on success
                    if backup_path and backup_path.exists():
                        backup_path.unlink()

                    logger.debug(f"Successfully populated {output_path.name}")

                except Exception as e:
                    # Restore backup if it exists
                    if backup_path and backup_path.exists():
                        shutil.move(str(backup_path), str(output_path))
                        logger.warning(f"Restored backup for {output_path.name}")

                    raise FileSystemError(
                        f"Failed to populate {output_path.name}", output_path, e
                    )

            logger.info("Successfully populated all project files")

        except (ProjectStructureError, FileSystemError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error populating project files: {e}")
            raise ScaffoldingError(
                f"Unexpected error populating project files: {str(e)}",
                original_exception=e,
            )

    def generate_project(
        self,
        request: GenerationRequest,
        output_dir: Path,
        streaming_callbacks: Optional[StreamingCallbacks] = None,
        verbose: bool = False,
    ) -> Path:
        """Generate a complete CrewAI project from a generation request.

        This orchestrates the entire generation pipeline:
        1. Create CrewAI project scaffolding first
        2. Analyze the prompt for requirements
        3. Generate agent configurations
        4. Generate task definitions
        5. Select appropriate tools
        6. Populate project files with generated content

        Args:
            request: GenerationRequest with prompt and project name
            output_dir: Directory where the project should be created
            streaming_callbacks: Optional callbacks for LLM streaming
            verbose: If True, enables verbose output for detailed generation steps

        Returns:
            Path to the generated project directory

        Raises:
            ScaffoldingError: If any step of the generation process fails
        """
        project_path = None

        try:
            # If verbose mode is enabled, ensure we have a verbose-capable generation engine
            if verbose and not getattr(self.generation_engine, "verbose", False):
                # Create a new generation engine with verbose mode enabled
                # Create a new LLM client with verbose mode as well
                self.generation_engine = GenerationEngine(
                    llm_client=None,
                    verbose=True,  # Let GenerationEngine create a new verbose LLMClient
                )

            # Step 1: Create CrewAI project scaffolding FIRST
            if verbose:
                logger.info("🏗️  [VERBOSE] Starting CrewAI project scaffolding...")
            self.progress_tracker.start_step("create_scaffold")

            try:
                if not request.project_name:
                    raise ScaffoldingError("Project name is required but not provided")
                project_path = self.create_crewai_project(
                    request.project_name, output_dir
                )
                if verbose:
                    logger.info(
                        f"🏗️  [VERBOSE] Project scaffolding created at: {project_path}"
                    )
                self.progress_tracker.complete_step("create_scaffold")
            except Exception as e:
                self.progress_tracker.fail_step("create_scaffold", str(e))
                raise ScaffoldingError(
                    f"Failed to create project structure: {str(e)}",
                    original_exception=e,
                )

            # Step 2: Analyze prompt for crew requirements
            if verbose:
                logger.info(
                    f"🔍 [VERBOSE] Analyzing prompt: '{request.prompt[:100]}{'...' if len(request.prompt) > 100 else ''}'"
                )
            self.progress_tracker.start_step("analyze_prompt")

            try:
                prompt_analysis = self.generation_engine.analyze_prompt(
                    request.prompt, streaming_callbacks
                )
                if verbose:
                    required_roles = prompt_analysis.get("required_roles", [])
                    logger.info(
                        f"🔍 [VERBOSE] Found {len(required_roles)} required roles: {', '.join(required_roles[:3])}{'...' if len(required_roles) > 3 else ''}"
                    )
                self.progress_tracker.complete_step("analyze_prompt")
            except Exception as e:
                self.progress_tracker.fail_step("analyze_prompt", str(e))
                raise ScaffoldingError(
                    f"Failed to analyze prompt: {str(e)}", original_exception=e
                )

            # Step 3: Generate agent configurations
            if verbose:
                logger.info(
                    "🤖 [VERBOSE] Generating agent configurations based on analysis..."
                )
            self.progress_tracker.start_step("generate_agents")

            try:
                agents = self.generation_engine.generate_agents(
                    prompt_analysis, streaming_callbacks
                )
                if verbose:
                    logger.info(
                        f"🤖 [VERBOSE] Generated {len(agents)} agents: {', '.join([agent.role for agent in agents])}"
                    )
                self.progress_tracker.complete_step("generate_agents")
            except Exception as e:
                self.progress_tracker.fail_step("generate_agents", str(e))
                raise ScaffoldingError(
                    f"Failed to generate agents: {str(e)}", original_exception=e
                )

            # Step 4: Generate task definitions
            if verbose:
                logger.info(
                    f"📋 [VERBOSE] Generating task definitions for {len(agents)} agents..."
                )
            self.progress_tracker.start_step("generate_tasks")

            try:
                tasks = self.generation_engine.generate_tasks(
                    agents, prompt_analysis, streaming_callbacks
                )
                if verbose:
                    logger.info(
                        f"📋 [VERBOSE] Generated {len(tasks)} tasks with clear expected outputs"
                    )
                self.progress_tracker.complete_step("generate_tasks")
            except Exception as e:
                self.progress_tracker.fail_step("generate_tasks", str(e))
                raise ScaffoldingError(
                    f"Failed to generate tasks: {str(e)}", original_exception=e
                )

            # Step 5: Select appropriate tools
            if verbose:
                tools_needed = prompt_analysis.get("tools_needed", [])
                logger.info(
                    f"🔧 [VERBOSE] Selecting tools for categories: {', '.join(tools_needed) if tools_needed else 'basic tools'}"
                )
            self.progress_tracker.start_step("select_tools")

            try:
                tools_needed = prompt_analysis.get("tools_needed", [])
                tools = self.generation_engine.select_tools(
                    tools_needed, streaming_callbacks
                )
                if verbose:
                    logger.info(f"🔧 [VERBOSE] Selected {len(tools)} tools for project")
                self.progress_tracker.complete_step("select_tools")
            except Exception as e:
                self.progress_tracker.fail_step("select_tools", str(e))
                raise ScaffoldingError(
                    f"Failed to select tools: {str(e)}", original_exception=e
                )

            # Step 6: Populate project files with generated content
            if verbose:
                logger.info(
                    f"📁 [VERBOSE] Populating project files with generated configurations..."
                )
            self.progress_tracker.start_step("populate_files")

            try:
                self.populate_project_files(
                    project_path, agents, tasks, tools, request.project_name
                )
                if verbose:
                    logger.info(
                        f"📁 [VERBOSE] Successfully populated all project files in {project_path}"
                    )
                self.progress_tracker.complete_step("populate_files")
            except Exception as e:
                self.progress_tracker.fail_step("populate_files", str(e))
                raise ScaffoldingError(
                    f"Failed to populate project files: {str(e)}", original_exception=e
                )

            return project_path

        except Exception as e:
            # Only clean up project if scaffolding failed or if AI generation succeeded
            # but file population failed. Don't clean up if only AI generation failed.
            should_cleanup = False

            if isinstance(e, ScaffoldingError):
                # Check if this was a scaffolding failure or just AI generation failure
                error_msg = str(e)
                if "Failed to create project structure" in error_msg:
                    # Scaffolding itself failed, clean up
                    should_cleanup = True
                elif "Failed to populate project files" in error_msg:
                    # File population failed after successful scaffolding and AI generation
                    # Clean up to avoid leaving partially populated project
                    should_cleanup = True
                # For AI generation failures (analyze_prompt, generate_agents, etc.)
                # we keep the scaffolding so user has a working project structure

            if should_cleanup and project_path and project_path.exists():
                shutil.rmtree(project_path, ignore_errors=True)
                logger.info(f"Cleaned up failed project at {project_path}")
            elif project_path and project_path.exists():
                logger.info(
                    f"Preserving project scaffolding at {project_path} despite generation failure"
                )

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
            ProjectStructureError: If project structure is invalid
        """
        if not project_path.exists():
            raise ProjectStructureError(
                "Project directory does not exist", project_path
            )

        if not project_path.is_dir():
            raise ProjectStructureError("Project path is not a directory", project_path)

        src_dir = project_path / "src"
        if not src_dir.exists() or not src_dir.is_dir():
            raise ProjectStructureError(
                "Invalid CrewAI project structure: missing src directory", project_path
            )

        # Find the module directory (should be the only directory under src)
        try:
            module_dirs = [d for d in src_dir.iterdir() if d.is_dir()]
        except PermissionError as e:
            raise FileSystemError("Unable to read src directory", src_dir, e)

        if not module_dirs:
            raise ProjectStructureError(
                "Invalid CrewAI project structure: no module directory found", src_dir
            )

        # Verify essential files exist
        essential_files = ["pyproject.toml"]
        for file_name in essential_files:
            file_path = project_path / file_name
            if not file_path.exists():
                logger.warning(f"Expected project file not found: {file_path}")

        logger.debug(f"Project structure validation passed for {project_path}")

    def _get_project_module_path(self, project_path: Path) -> Path:
        """Get the path to the project's Python module directory.

        Args:
            project_path: Path to the project directory

        Returns:
            Path to the module directory under src/

        Raises:
            ProjectStructureError: If module path cannot be determined
        """
        src_dir = project_path / "src"
        module_dirs = [d for d in src_dir.iterdir() if d.is_dir()]

        if not module_dirs:
            raise ProjectStructureError("No module directory found", src_dir)

        # Return the first (and expected only) module directory
        return module_dirs[0]

    def _check_disk_space(self, path: Path, min_space_mb: int = 100) -> None:
        """Check if there's enough disk space for project operations.

        Args:
            path: Path to check disk space for
            min_space_mb: Minimum required space in megabytes

        Raises:
            FileSystemError: If insufficient disk space
        """
        try:
            stat = shutil.disk_usage(path)
            free_space_mb = stat.free / (1024 * 1024)  # Convert to MB

            if free_space_mb < min_space_mb:
                raise FileSystemError(
                    f"Insufficient disk space. Required: {min_space_mb}MB, Available: {free_space_mb:.1f}MB",
                    path,
                )

            logger.debug(f"Disk space check passed: {free_space_mb:.1f}MB available")

        except OSError as e:
            raise FileSystemError(f"Unable to check disk space: {str(e)}", path, e)

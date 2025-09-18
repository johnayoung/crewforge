"""
CrewAI Scaffolder - Native CrewAI project creation using official CLI commands

This module integrates with the CrewAI CLI to create base project structures
using the native `crewai create crew <name>` command, providing error handling
and subprocess management.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
import logging


class CrewAIError(Exception):
    """Exception raised for CrewAI CLI execution errors."""

    pass


class CrewAIScaffolder:
    """
    Handles subprocess execution of CrewAI CLI commands for project scaffolding.

    This class provides a clean interface to the native CrewAI CLI while handling
    error cases, validation, and subprocess management.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the CrewAI scaffolder.

        Args:
            logger: Optional logger instance for debugging and error tracking
        """
        self.logger = logger or logging.getLogger(__name__)

    def check_crewai_available(self) -> bool:
        """
        Check if CrewAI CLI is available and accessible.

        Returns:
            True if CrewAI CLI is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["crewai", "--version"], capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False

    def create_crew(self, project_name: str, target_directory: Path) -> Dict[str, Any]:
        """
        Execute 'crewai create crew <name>' command to create base project structure.

        Args:
            project_name: Name of the crew project to create
            target_directory: Directory where the project should be created

        Returns:
            Dictionary containing execution results and project information

        Raises:
            CrewAIError: If CrewAI CLI execution fails or is not available
            ValueError: If project_name or target_directory are invalid
        """
        # Validate inputs
        if not project_name or not project_name.strip():
            raise ValueError("project_name cannot be empty")

        if not isinstance(target_directory, Path):
            target_directory = Path(target_directory)

        # Check if CrewAI CLI is available
        if not self.check_crewai_available():
            raise CrewAIError("CrewAI CLI is not available or not installed")

        # Ensure target directory exists
        target_directory.mkdir(parents=True, exist_ok=True)

        # Execute CrewAI create command
        try:
            self.logger.info(
                f"Creating CrewAI project '{project_name}' in {target_directory}"
            )

            result = subprocess.run(
                ["crewai", "create", "crew", project_name],
                cwd=target_directory,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout for project creation
            )

            project_path = target_directory / project_name

            if result.returncode != 0:
                error_message = (
                    result.stderr.strip() or result.stdout.strip() or "Unknown error"
                )
                self.logger.error(f"CrewAI CLI failed: {error_message}")
                raise CrewAIError(f"CrewAI project creation failed: {error_message}")

            # Verify the project directory was created
            if not project_path.exists():
                raise CrewAIError(
                    f"Project directory was not created at {project_path}"
                )

            self.logger.info(f"Successfully created CrewAI project at {project_path}")

            return {
                "success": True,
                "project_path": project_path,
                "project_name": project_name,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            raise CrewAIError("CrewAI project creation timed out after 60 seconds")
        except subprocess.SubprocessError as e:
            raise CrewAIError(f"Subprocess error during CrewAI execution: {str(e)}")
        except Exception as e:
            raise CrewAIError(f"Unexpected error during CrewAI execution: {str(e)}")

    def get_version(self) -> Optional[str]:
        """
        Get the version of the installed CrewAI CLI.

        Returns:
            Version string if available, None if CrewAI is not available
        """
        try:
            result = subprocess.run(
                ["crewai", "--version"], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                # Parse version from output (format may vary)
                version_output = result.stdout.strip() or result.stderr.strip()
                return version_output
            return None

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return None

"""
CrewAI Scaffolder - Native CrewAI project creation using official CLI commands

This module integrates with the CrewAI CLI to create base project structures
using the native `crewai create crew <name>` command, providing error handling
and subprocess management.
"""

import subprocess
import shutil
import re
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
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

    def create_crew(
        self,
        project_name: str,
        target_directory: Path,
        min_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute 'crewai create crew <name>' command to create base project structure.

        Args:
            project_name: Name of the crew project to create
            target_directory: Directory where the project should be created
            min_version: Optional minimum CrewAI version requirement

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

        # Validate CrewAI CLI dependency with version checking
        validation_result = self.validate_crewai_dependency(min_version=min_version)
        if not validation_result["valid"]:
            raise CrewAIError(validation_result["error"])

        self.logger.info(
            f"Using CrewAI CLI version {validation_result['installed_version']}"
        )

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

    def parse_version_string(self, version_output: Optional[str]) -> Optional[str]:
        """
        Parse version string from CrewAI CLI output.

        Args:
            version_output: Raw version output from CrewAI CLI

        Returns:
            Parsed semantic version string (e.g., "1.2.3") or None if parsing fails
        """
        if not version_output:
            return None

        # Common version patterns from various CLI tools
        patterns = [
            r"(\d+\.\d+\.\d+(?:-[a-zA-Z0-9\.\-]+)?)",  # Standard semver with optional prerelease
            r"v(\d+\.\d+\.\d+(?:-[a-zA-Z0-9\.\-]+)?)",  # With 'v' prefix
        ]

        for pattern in patterns:
            match = re.search(pattern, version_output)
            if match:
                return match.group(1)

        return None

    def compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two semantic version strings.

        Args:
            version1: First version to compare
            version2: Second version to compare

        Returns:
            -1 if version1 < version2
            0 if version1 == version2
            1 if version1 > version2

        Raises:
            ValueError: If either version string is invalid
        """

        def parse_version_parts(version: str) -> Tuple[int, int, int, str]:
            """Parse version into (major, minor, patch, prerelease)."""
            if not version:
                raise ValueError("Version string cannot be empty")

            # Split on '-' to separate main version from prerelease
            parts = version.split("-", 1)
            main_version = parts[0]
            prerelease = parts[1] if len(parts) > 1 else ""

            # Parse main version numbers
            version_parts = main_version.split(".")
            if len(version_parts) != 3:
                raise ValueError(f"Invalid version format: {version}")

            try:
                major = int(version_parts[0])
                minor = int(version_parts[1])
                patch = int(version_parts[2])
            except ValueError:
                raise ValueError(f"Invalid version format: {version}")

            return (major, minor, patch, prerelease)

        v1_parts = parse_version_parts(version1)
        v2_parts = parse_version_parts(version2)

        # Compare major, minor, patch (only numeric parts)
        v1_major, v1_minor, v1_patch, v1_prerelease = v1_parts
        v2_major, v2_minor, v2_patch, v2_prerelease = v2_parts

        # Compare version numbers
        if v1_major < v2_major:
            return -1
        elif v1_major > v2_major:
            return 1
        elif v1_minor < v2_minor:
            return -1
        elif v1_minor > v2_minor:
            return 1
        elif v1_patch < v2_patch:
            return -1
        elif v1_patch > v2_patch:
            return 1

        # If main versions are equal, compare prerelease
        # Empty prerelease (release version) > any prerelease
        if not v1_prerelease and v2_prerelease:
            return 1
        elif v1_prerelease and not v2_prerelease:
            return -1
        elif v1_prerelease < v2_prerelease:
            return -1
        elif v1_prerelease > v2_prerelease:
            return 1

        return 0

    def check_minimum_version(self, current_version: str, min_version: str) -> bool:
        """
        Check if current version meets minimum version requirement.

        Args:
            current_version: Currently installed version
            min_version: Minimum required version

        Returns:
            True if current version >= minimum version

        Raises:
            ValueError: If either version string is invalid
        """
        return self.compare_versions(current_version, min_version) >= 0

    def validate_crewai_dependency(
        self, min_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate CrewAI CLI dependency with version checking.

        Args:
            min_version: Minimum required version (optional)

        Returns:
            Dictionary containing validation results:
            - valid: Boolean indicating if dependency is valid
            - installed_version: Currently installed version string or None
            - meets_minimum: Boolean indicating if minimum version is met
            - error: Error message if validation fails
        """
        result = {
            "valid": False,
            "installed_version": None,
            "meets_minimum": False,
            "error": None,
        }

        # Check if CrewAI CLI is available
        if not self.check_crewai_available():
            result["error"] = (
                "CrewAI CLI is not installed or not accessible. "
                "Please install CrewAI using: pip install crewai"
            )
            return result

        # Get version information
        version_output = self.get_version()
        if not version_output:
            result["error"] = "Could not determine CrewAI CLI version"
            return result

        # Parse version string
        parsed_version = self.parse_version_string(version_output)
        if not parsed_version:
            result["error"] = f"Could not parse version from output: {version_output}"
            return result

        result["installed_version"] = parsed_version

        # Check minimum version if specified
        if min_version:
            try:
                meets_min = self.check_minimum_version(parsed_version, min_version)
                result["meets_minimum"] = meets_min

                if not meets_min:
                    result["error"] = (
                        f"CrewAI CLI version {parsed_version} is too old. "
                        f"minimum version {min_version} required"
                    )
                    return result
            except ValueError as e:
                result["error"] = f"Version comparison failed: {str(e)}"
                return result
        else:
            result["meets_minimum"] = True

        result["valid"] = True
        return result

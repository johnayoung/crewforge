"""
Dependency management system for CrewForge.

This module provides uv-based dependency management for generated CrewAI projects,
ensuring proper package installation and virtual environment management.
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from crewforge.progress import StatusDisplay


class DependencyError(Exception):
    """Raised when dependency management operations fail."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.__cause__ = cause


class DependencyManager:
    """
    Manages dependencies for generated CrewAI projects using uv.

    Provides comprehensive dependency management including:
    - Virtual environment creation and management
    - Package installation and updates
    - Dependency resolution and conflict detection
    - Project environment setup and validation
    """

    def __init__(self, uv_path: str = "uv", timeout: int = 300):
        """
        Initialize the dependency manager.

        Args:
            uv_path: Path to uv executable
            timeout: Timeout for uv operations in seconds
        """
        self.uv_path = uv_path
        self.timeout = timeout
        self.status = StatusDisplay()

    def check_uv_available(self) -> bool:
        """
        Check if uv is available and accessible.

        Returns:
            True if uv is available, False otherwise
        """
        try:
            result = subprocess.run(
                [self.uv_path, "--version"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.returncode == 0
        except (
            FileNotFoundError,
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
        ):
            return False

    async def install_dependencies(self, project_path: Path) -> Dict[str, Any]:
        """
        Install project dependencies using uv sync.

        Args:
            project_path: Path to the project directory

        Returns:
            Dictionary with success status and message
        """
        if not self._validate_project_structure(project_path):
            raise DependencyError("Invalid project structure: pyproject.toml not found")

        try:
            self.status.info("Installing project dependencies with uv...")

            process = await asyncio.create_subprocess_exec(
                self.uv_path,
                "sync",
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Decode output if needed
            stdout_str = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
            stderr_str = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

            if process.returncode == 0:
                self.status.success("Dependencies installed successfully")
                return {
                    "success": True,
                    "message": "Dependencies installed successfully",
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }
            else:
                error_msg = f"Failed to install dependencies: {stderr_str}"
                self.status.error(error_msg)
                return {
                    "success": False,
                    "message": error_msg,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }

        except (asyncio.TimeoutError, subprocess.SubprocessError) as e:
            error_msg = f"Dependency installation failed: {str(e)}"
            self.status.error(error_msg)
            return {"success": False, "message": error_msg, "error": str(e)}

    async def create_virtual_environment(self, project_path: Path) -> Dict[str, Any]:
        """
        Create a virtual environment for the project using uv.

        Args:
            project_path: Path to the project directory

        Returns:
            Dictionary with success status and message
        """
        try:
            self.status.info("Creating virtual environment...")

            process = await asyncio.create_subprocess_exec(
                self.uv_path,
                "venv",
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Decode output if needed
            stdout_str = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
            stderr_str = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

            if process.returncode == 0:
                self.status.success("Virtual environment created successfully")
                return {
                    "success": True,
                    "message": "Virtual environment created successfully",
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }
            else:
                error_msg = f"Failed to create virtual environment: {stderr_str}"
                self.status.error(error_msg)
                return {
                    "success": False,
                    "message": error_msg,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }

        except (asyncio.TimeoutError, subprocess.SubprocessError) as e:
            error_msg = f"Virtual environment creation failed: {str(e)}"
            self.status.error(error_msg)
            return {"success": False, "message": error_msg, "error": str(e)}

    async def add_dependency(
        self, project_path: Path, package_name: str, version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a dependency to the project using uv.

        Args:
            project_path: Path to the project directory
            package_name: Name of the package to add
            version: Optional version constraint

        Returns:
            Dictionary with success status and message
        """
        try:
            self.status.info(f"Adding dependency: {package_name}")

            cmd = [self.uv_path, "add", package_name]
            if version:
                cmd[-1] = f"{package_name}{version}"

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Decode output if needed
            stdout_str = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
            stderr_str = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

            if process.returncode == 0:
                self.status.success(f"Dependency {package_name} added successfully")
                return {
                    "success": True,
                    "message": f"Dependency {package_name} added successfully",
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }
            else:
                error_msg = f"Failed to add dependency {package_name}: {stderr_str}"
                self.status.error(error_msg)
                return {
                    "success": False,
                    "message": error_msg,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }

        except (asyncio.TimeoutError, subprocess.SubprocessError) as e:
            error_msg = f"Failed to add dependency {package_name}: {str(e)}"
            self.status.error(error_msg)
            return {"success": False, "message": error_msg, "error": str(e)}

    async def remove_dependency(
        self, project_path: Path, package_name: str
    ) -> Dict[str, Any]:
        """
        Remove a dependency from the project using uv.

        Args:
            project_path: Path to the project directory
            package_name: Name of the package to remove

        Returns:
            Dictionary with success status and message
        """
        try:
            self.status.info(f"Removing dependency: {package_name}")

            process = await asyncio.create_subprocess_exec(
                self.uv_path,
                "remove",
                package_name,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Decode output if needed
            stdout_str = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
            stderr_str = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

            if process.returncode == 0:
                self.status.success(f"Dependency {package_name} removed successfully")
                return {
                    "success": True,
                    "message": f"Dependency {package_name} removed successfully",
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }
            else:
                error_msg = f"Failed to remove dependency {package_name}: {stderr_str}"
                self.status.error(error_msg)
                return {
                    "success": False,
                    "message": error_msg,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }

        except (asyncio.TimeoutError, subprocess.SubprocessError) as e:
            error_msg = f"Failed to remove dependency {package_name}: {str(e)}"
            self.status.error(error_msg)
            return {"success": False, "message": error_msg, "error": str(e)}

    async def update_dependencies(self, project_path: Path) -> Dict[str, Any]:
        """
        Update project dependencies using uv.

        Args:
            project_path: Path to the project directory

        Returns:
            Dictionary with success status and message
        """
        try:
            self.status.info("Updating project dependencies...")

            process = await asyncio.create_subprocess_exec(
                self.uv_path,
                "sync",
                "--upgrade",
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Decode output if needed
            stdout_str = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
            stderr_str = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

            if process.returncode == 0:
                self.status.success("Dependencies updated successfully")
                return {
                    "success": True,
                    "message": "Dependencies updated successfully",
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }
            else:
                error_msg = f"Failed to update dependencies: {stderr_str}"
                self.status.error(error_msg)
                return {
                    "success": False,
                    "message": error_msg,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }

        except (asyncio.TimeoutError, subprocess.SubprocessError) as e:
            error_msg = f"Dependency update failed: {str(e)}"
            self.status.error(error_msg)
            return {"success": False, "message": error_msg, "error": str(e)}

    def get_dependency_info(self, project_path: Path) -> Dict[str, Any]:
        """
        Get information about project dependencies.

        Args:
            project_path: Path to the project directory

        Returns:
            Dictionary with dependency information
        """
        info = {
            "pyproject.toml": {
                "exists": False,
                "path": str(project_path / "pyproject.toml"),
                "dependencies": [],
            },
            "uv.lock": {"exists": False, "path": str(project_path / "uv.lock")},
            "venv": {"exists": False, "path": str(project_path / ".venv")},
        }

        # Check pyproject.toml
        pyproject_path = project_path / "pyproject.toml"
        if pyproject_path.exists():
            info["pyproject.toml"]["exists"] = True
            try:
                # Simple parsing to extract dependencies
                content = pyproject_path.read_text()
                # This is a basic extraction - in production you'd use tomllib or similar
                if "dependencies" in content:
                    info["pyproject.toml"]["has_dependencies"] = True
            except Exception:
                pass

        # Check uv.lock
        uv_lock_path = project_path / "uv.lock"
        info["uv.lock"]["exists"] = uv_lock_path.exists()

        # Check virtual environment
        venv_path = project_path / ".venv"
        info["venv"]["exists"] = venv_path.exists()

        return info

    async def setup_project_environment(self, project_path: Path) -> Dict[str, Any]:
        """
        Set up complete project environment (venv + dependencies).

        Args:
            project_path: Path to the project directory

        Returns:
            Dictionary with setup results
        """
        try:
            self.status.info("Setting up complete project environment...")

            # Create virtual environment
            venv_result = await self.create_virtual_environment(project_path)
            if not venv_result["success"]:
                return venv_result

            # Install dependencies
            install_result = await self.install_dependencies(project_path)
            if not install_result["success"]:
                return install_result

            self.status.success("Project environment setup completed successfully")
            return {
                "success": True,
                "message": "Project environment setup completed successfully",
                "venv_result": venv_result,
                "install_result": install_result,
            }

        except Exception as e:
            error_msg = f"Project environment setup failed: {str(e)}"
            self.status.error(error_msg)
            return {"success": False, "message": error_msg, "error": str(e)}

    def _validate_project_structure(self, project_path: Path) -> bool:
        """
        Validate that the project has required structure for dependency management.

        Args:
            project_path: Path to the project directory

        Returns:
            True if structure is valid, False otherwise
        """
        pyproject_path = project_path / "pyproject.toml"
        return pyproject_path.exists() and pyproject_path.is_file()

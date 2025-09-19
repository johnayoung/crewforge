"""
Tests for the DependencyManager component.

This module provides comprehensive tests for the uv-based dependency management
system used in generated CrewAI projects.
"""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from crewforge.dependency_manager import DependencyManager, DependencyError


class TestDependencyManager:
    """Test suite for DependencyManager."""

    @pytest.fixture
    def mock_subprocess(self):
        """Mock subprocess for testing."""
        with (
            patch("subprocess.run") as mock_run,
            patch("asyncio.create_subprocess_exec") as mock_exec,
        ):
            # Set up the mock for async subprocess
            mock_process = Mock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b"Success", b""))
            mock_exec.return_value = mock_process
            yield mock_run, mock_exec

    @pytest.fixture
    def temp_project_dir(self, tmp_path):
        """Create a temporary project directory."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create a basic pyproject.toml
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"
description = "Test project"
dependencies = [
    "crewai",
    "openai"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
        (project_dir / "pyproject.toml").write_text(pyproject_content)

        return project_dir

    def test_initialization(self):
        """Test DependencyManager initialization."""
        manager = DependencyManager()
        assert manager.uv_path == "uv"
        assert manager.timeout == 300

    def test_initialization_with_custom_path(self):
        """Test DependencyManager with custom uv path."""
        manager = DependencyManager(uv_path="/custom/path/uv", timeout=600)
        assert manager.uv_path == "/custom/path/uv"
        assert manager.timeout == 600

    def test_check_uv_available_success(self, mock_subprocess):
        """Test successful uv availability check."""
        mock_run, mock_exec = mock_subprocess
        mock_run.return_value = Mock(returncode=0, stdout="uv 0.1.0")

        manager = DependencyManager()
        result = manager.check_uv_available()

        assert result is True
        mock_run.assert_called_once_with(
            ["uv", "--version"], capture_output=True, text=True, timeout=300
        )

    def test_check_uv_available_not_found(self, mock_subprocess):
        """Test uv not available."""
        mock_run, mock_exec = mock_subprocess
        mock_run.side_effect = FileNotFoundError("uv not found")

        manager = DependencyManager()
        result = manager.check_uv_available()

        assert result is False

    def test_check_uv_available_command_failed(self, mock_subprocess):
        """Test uv command failed."""
        mock_run, mock_exec = mock_subprocess
        mock_run.return_value = Mock(returncode=1, stderr="uv error")

        manager = DependencyManager()
        result = manager.check_uv_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_install_dependencies_success(
        self, mock_subprocess, temp_project_dir
    ):
        """Test successful dependency installation."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Installed dependencies", b"")
        )
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.install_dependencies(temp_project_dir)

        assert result["success"] is True
        assert result["message"] == "Dependencies installed successfully"

    @pytest.mark.asyncio
    async def test_install_dependencies_failure(
        self, mock_subprocess, temp_project_dir
    ):
        """Test dependency installation failure."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Installation failed"))
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.install_dependencies(temp_project_dir)

        assert result["success"] is False
        assert "Installation failed" in result["message"]
        assert "failed" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_install_dependencies_no_pyproject(self, tmp_path):
        """Test dependency installation with no pyproject.toml."""
        project_dir = tmp_path / "no_pyproject"
        project_dir.mkdir()

        manager = DependencyManager()

        with pytest.raises(DependencyError) as exc_info:
            await manager.install_dependencies(project_dir)

        assert "pyproject.toml not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_virtual_environment_success(self, mock_subprocess, tmp_path):
        """Test successful virtual environment creation."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Virtual environment created", b"")
        )
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.create_virtual_environment(tmp_path)

        assert result["success"] is True
        assert result["message"] == "Virtual environment created successfully"

    @pytest.mark.asyncio
    async def test_create_virtual_environment_failure(self, mock_subprocess, tmp_path):
        """Test virtual environment creation failure."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Venv creation failed")
        )
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.create_virtual_environment(tmp_path)

        assert result["success"] is False
        assert "Venv creation failed" in result["message"]
        assert "failed" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_add_dependency_success(self, mock_subprocess, temp_project_dir):
        """Test successful dependency addition."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Added dependency", b""))
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.add_dependency(temp_project_dir, "requests")

        assert result["success"] is True
        assert result["message"] == "Dependency requests added successfully"

    @pytest.mark.asyncio
    async def test_add_dependency_failure(self, mock_subprocess, temp_project_dir):
        """Test dependency addition failure."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Add dependency failed")
        )
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.add_dependency(temp_project_dir, "requests")

        assert result["success"] is False
        assert "Add dependency failed" in result["message"]
        assert "failed" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_remove_dependency_success(self, mock_subprocess, temp_project_dir):
        """Test successful dependency removal."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Removed dependency", b""))
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.remove_dependency(temp_project_dir, "requests")

        assert result["success"] is True
        assert result["message"] == "Dependency requests removed successfully"

    @pytest.mark.asyncio
    async def test_remove_dependency_failure(self, mock_subprocess, temp_project_dir):
        """Test dependency removal failure."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Remove dependency failed")
        )
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.remove_dependency(temp_project_dir, "requests")

        assert result["success"] is False
        assert "Remove dependency failed" in result["message"]
        assert "failed" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_update_dependencies_success(self, mock_subprocess, temp_project_dir):
        """Test successful dependency update."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"Updated dependencies", b"")
        )
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.update_dependencies(temp_project_dir)

        assert result["success"] is True
        assert result["message"] == "Dependencies updated successfully"

    @pytest.mark.asyncio
    async def test_update_dependencies_failure(self, mock_subprocess, temp_project_dir):
        """Test dependency update failure."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Update failed"))
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.update_dependencies(temp_project_dir)

        assert result["success"] is False
        assert "Update failed" in result["message"]

    def test_get_dependency_info_success(self, temp_project_dir):
        """Test successful dependency info retrieval."""
        manager = DependencyManager()
        info = manager.get_dependency_info(temp_project_dir)

        assert "pyproject.toml" in info
        assert info["pyproject.toml"]["exists"] is True
        assert "dependencies" in info["pyproject.toml"]

    def test_get_dependency_info_no_pyproject(self, tmp_path):
        """Test dependency info with no pyproject.toml."""
        project_dir = tmp_path / "no_pyproject"
        project_dir.mkdir()

        manager = DependencyManager()
        info = manager.get_dependency_info(project_dir)

        assert info["pyproject.toml"]["exists"] is False
        assert info["pyproject.toml"]["path"] == str(project_dir / "pyproject.toml")

    @pytest.mark.asyncio
    async def test_setup_project_environment_success(
        self, mock_subprocess, temp_project_dir
    ):
        """Test successful project environment setup."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Success", b""))
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.setup_project_environment(temp_project_dir)

        assert result["success"] is True
        assert result["message"] == "Project environment setup completed successfully"

    @pytest.mark.asyncio
    async def test_setup_project_environment_failure(
        self, mock_subprocess, temp_project_dir
    ):
        """Test project environment setup failure."""
        mock_run, mock_exec = mock_subprocess
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Setup failed"))
        mock_exec.return_value = mock_process

        manager = DependencyManager()
        result = await manager.setup_project_environment(temp_project_dir)

        assert result["success"] is False
        assert "Setup failed" in result["message"]


class TestDependencyError:
    """Test suite for DependencyError."""

    def test_dependency_error_creation(self):
        """Test DependencyError creation."""
        error = DependencyError("Test dependency error")
        assert str(error) == "Test dependency error"

    def test_dependency_error_with_cause(self):
        """Test DependencyError with cause."""
        cause = Exception("Original error")
        error = DependencyError("Test error", cause)
        assert str(error) == "Test error"
        assert error.__cause__ == cause


class TestIntegrationDependencyManager:
    """Integration tests for DependencyManager."""

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self, tmp_path):
        """Test full dependency management workflow simulation."""
        project_dir = tmp_path / "integration_test"
        project_dir.mkdir()

        # Create pyproject.toml
        pyproject_content = """
[project]
name = "integration-test"
version = "0.1.0"
dependencies = ["crewai", "openai"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
        (project_dir / "pyproject.toml").write_text(pyproject_content)

        manager = DependencyManager()

        # Mock all external operations
        with (
            patch.object(manager, "check_uv_available", return_value=True),
            patch.object(manager, "create_virtual_environment") as mock_venv,
            patch.object(manager, "install_dependencies") as mock_install,
        ):

            mock_venv.return_value = {"success": True, "message": "Venv created"}
            mock_install.return_value = {
                "success": True,
                "message": "Dependencies installed",
            }

            # Test the workflow
            venv_result = await manager.create_virtual_environment(project_dir)
            install_result = await manager.install_dependencies(project_dir)

            assert venv_result["success"] is True
            assert install_result["success"] is True

            mock_venv.assert_called_once_with(project_dir)
            mock_install.assert_called_once_with(project_dir)

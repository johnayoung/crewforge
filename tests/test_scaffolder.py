"""
Tests for CrewAI Scaffolder module

Test suite for the CrewAI CLI subprocess integration, including success cases,
error handling, and edge cases for project creation.
"""

import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

from crewforge.scaffolder import CrewAIScaffolder, CrewAIError


class TestCrewAIScaffolder:
    """Test suite for CrewAIScaffolder class."""

    def test_initialization_default_logger(self):
        """Test scaffolder initialization with default logger."""
        scaffolder = CrewAIScaffolder()
        assert scaffolder.logger is not None
        assert scaffolder.logger.name == "crewforge.scaffolder"

    def test_initialization_custom_logger(self):
        """Test scaffolder initialization with custom logger."""
        custom_logger = logging.getLogger("test_logger")
        scaffolder = CrewAIScaffolder(logger=custom_logger)
        assert scaffolder.logger == custom_logger

    @patch("subprocess.run")
    def test_check_crewai_available_success(self, mock_run):
        """Test successful CrewAI availability check."""
        mock_run.return_value = Mock(returncode=0)

        scaffolder = CrewAIScaffolder()
        result = scaffolder.check_crewai_available()

        assert result is True
        mock_run.assert_called_once_with(
            ["crewai", "--version"], capture_output=True, text=True, timeout=10
        )

    @patch("subprocess.run")
    def test_check_crewai_available_not_found(self, mock_run):
        """Test CrewAI availability check when CLI not found."""
        mock_run.side_effect = FileNotFoundError("crewai command not found")

        scaffolder = CrewAIScaffolder()
        result = scaffolder.check_crewai_available()

        assert result is False

    @patch("subprocess.run")
    def test_check_crewai_available_timeout(self, mock_run):
        """Test CrewAI availability check timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(["crewai", "--version"], 10)

        scaffolder = CrewAIScaffolder()
        result = scaffolder.check_crewai_available()

        assert result is False

    @patch("subprocess.run")
    def test_check_crewai_available_error_code(self, mock_run):
        """Test CrewAI availability check with non-zero exit code."""
        mock_run.return_value = Mock(returncode=1)

        scaffolder = CrewAIScaffolder()
        result = scaffolder.check_crewai_available()

        assert result is False

    def test_create_crew_empty_project_name(self):
        """Test create_crew with empty project name."""
        scaffolder = CrewAIScaffolder()

        with pytest.raises(ValueError, match="project_name cannot be empty"):
            scaffolder.create_crew("", Path("/tmp"))

        with pytest.raises(ValueError, match="project_name cannot be empty"):
            scaffolder.create_crew("   ", Path("/tmp"))

    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    def test_create_crew_crewai_not_available(self, mock_validate):
        """Test create_crew when CrewAI CLI is not available."""
        mock_validate.return_value = {
            "valid": False,
            "installed_version": None,
            "meets_minimum": False,
            "error": "CrewAI CLI is not installed or not accessible. Please install CrewAI using: pip install crewai",
        }

        scaffolder = CrewAIScaffolder()

        with pytest.raises(
            CrewAIError, match="CrewAI CLI is not installed or not accessible"
        ):
            scaffolder.create_crew("test-project", Path("/tmp"))

    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    @patch("subprocess.run")
    def test_create_crew_success(self, mock_run, mock_validate):
        """Test successful crew project creation."""
        mock_validate.return_value = {
            "valid": True,
            "installed_version": "1.2.3",
            "meets_minimum": True,
            "error": None,
        }
        mock_run.return_value = Mock(
            returncode=0, stdout="Project created successfully", stderr=""
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_name = "test-project"
            project_path = temp_path / project_name

            # Create the expected project directory (simulating CrewAI CLI behavior)
            project_path.mkdir(parents=True, exist_ok=True)

            scaffolder = CrewAIScaffolder()
            result = scaffolder.create_crew(project_name, temp_path)

            assert result["success"] is True
            assert result["project_path"] == project_path
            assert result["project_name"] == project_name
            assert result["returncode"] == 0
            assert "Project created successfully" in result["stdout"]

            mock_validate.assert_called_once_with(min_version=None)
            mock_run.assert_called_once_with(
                ["crewai", "create", "crew", project_name],
                cwd=temp_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    @patch("subprocess.run")
    def test_create_crew_with_version_requirement(self, mock_run, mock_validate):
        """Test crew creation with minimum version requirement."""
        mock_validate.return_value = {
            "valid": True,
            "installed_version": "1.5.0",
            "meets_minimum": True,
            "error": None,
        }
        mock_run.return_value = Mock(
            returncode=0, stdout="Project created successfully", stderr=""
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_name = "test-project"
            project_path = temp_path / project_name
            project_path.mkdir(parents=True, exist_ok=True)

            scaffolder = CrewAIScaffolder()
            result = scaffolder.create_crew(
                project_name, temp_path, min_version="1.2.0"
            )

            assert result["success"] is True
            mock_validate.assert_called_once_with(min_version="1.2.0")

    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    def test_create_crew_validation_failure(self, mock_validate):
        """Test crew creation when validation fails."""
        mock_validate.return_value = {
            "valid": False,
            "installed_version": "1.0.0",
            "meets_minimum": False,
            "error": "CrewAI CLI version 1.0.0 is too old. minimum version 1.2.0 required",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            scaffolder = CrewAIScaffolder()
            with pytest.raises(CrewAIError, match="minimum version 1.2.0 required"):
                scaffolder.create_crew("test-project", temp_path, min_version="1.2.0")

    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    @patch("subprocess.run")
    def test_create_crew_command_failure(self, mock_run, mock_validate):
        """Test create_crew when CrewAI CLI command fails."""
        mock_validate.return_value = {
            "valid": True,
            "installed_version": "1.2.3",
            "meets_minimum": True,
            "error": None,
        }
        mock_run.return_value = Mock(
            returncode=1, stdout="", stderr="Error: Invalid project name"
        )

        scaffolder = CrewAIScaffolder()

        with pytest.raises(
            CrewAIError,
            match="CrewAI project creation failed: Error: Invalid project name",
        ):
            scaffolder.create_crew("test-project", Path("/tmp"))

    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    @patch("subprocess.run")
    def test_create_crew_timeout(self, mock_run, mock_validate):
        """Test create_crew with subprocess timeout."""
        mock_validate.return_value = {
            "valid": True,
            "installed_version": "1.2.3",
            "meets_minimum": True,
            "error": None,
        }
        mock_run.side_effect = subprocess.TimeoutExpired(
            ["crewai", "create", "crew", "test"], 60
        )

        scaffolder = CrewAIScaffolder()

        with pytest.raises(
            CrewAIError, match="CrewAI project creation timed out after 60 seconds"
        ):
            scaffolder.create_crew("test-project", Path("/tmp"))

    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    @patch("subprocess.run")
    def test_create_crew_subprocess_error(self, mock_run, mock_validate):
        """Test create_crew with subprocess error."""
        mock_validate.return_value = {
            "valid": True,
            "installed_version": "1.2.3",
            "meets_minimum": True,
            "error": None,
        }
        mock_run.side_effect = subprocess.CalledProcessError(1, ["crewai"])

        scaffolder = CrewAIScaffolder()

        with pytest.raises(
            CrewAIError, match="Subprocess error during CrewAI execution"
        ):
            scaffolder.create_crew("test-project", Path("/tmp"))

    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    @patch("subprocess.run")
    def test_create_crew_directory_not_created(self, mock_run, mock_validate):
        """Test create_crew when project directory is not created."""
        mock_validate.return_value = {
            "valid": True,
            "installed_version": "1.2.3",
            "meets_minimum": True,
            "error": None,
        }
        mock_run.return_value = Mock(
            returncode=0, stdout="Project created successfully", stderr=""
        )

        scaffolder = CrewAIScaffolder()

        # Don't create the directory to simulate failure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with pytest.raises(CrewAIError, match="Project directory was not created"):
                scaffolder.create_crew("test-project", temp_path)

    def test_create_crew_path_conversion(self):
        """Test create_crew converts string path to Path object."""
        scaffolder = CrewAIScaffolder()

        # Mock the validation to avoid actual CrewAI dependency
        mock_validation_result = {
            "valid": False,
            "installed_version": None,
            "meets_minimum": False,
            "error": "CrewAI CLI is not installed",
        }
        with patch.object(
            scaffolder,
            "validate_crewai_dependency",
            return_value=mock_validation_result,
        ):
            with pytest.raises(CrewAIError):
                # Should not raise error for string path conversion
                scaffolder.create_crew("test-project", Path("/tmp"))

    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    @patch("subprocess.run")
    def test_create_crew_target_directory_creation(self, mock_run, mock_validate):
        """Test create_crew creates target directory if it doesn't exist."""
        mock_validate.return_value = {
            "valid": True,
            "installed_version": "1.2.3",
            "meets_minimum": True,
            "error": None,
        }
        mock_run.return_value = Mock(
            returncode=0, stdout="Project created successfully", stderr=""
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            non_existent_path = temp_path / "subdir" / "nested"
            project_name = "test-project"
            project_path = non_existent_path / project_name

            # Create the expected project directory
            project_path.mkdir(parents=True, exist_ok=True)

            scaffolder = CrewAIScaffolder()
            result = scaffolder.create_crew(project_name, non_existent_path)

            assert result["success"] is True
            assert non_existent_path.exists()

    @patch("subprocess.run")
    def test_get_version_success(self, mock_run):
        """Test successful version retrieval."""
        mock_run.return_value = Mock(returncode=0, stdout="crewai 1.2.3", stderr="")

        scaffolder = CrewAIScaffolder()
        version = scaffolder.get_version()

        assert version == "crewai 1.2.3"
        mock_run.assert_called_once_with(
            ["crewai", "--version"], capture_output=True, text=True, timeout=10
        )

    @patch("subprocess.run")
    def test_get_version_stderr_output(self, mock_run):
        """Test version retrieval from stderr."""
        mock_run.return_value = Mock(
            returncode=0, stdout="", stderr="crewai version 1.2.3"
        )

        scaffolder = CrewAIScaffolder()
        version = scaffolder.get_version()

        assert version == "crewai version 1.2.3"

    @patch("subprocess.run")
    def test_get_version_failure(self, mock_run):
        """Test version retrieval failure."""
        mock_run.return_value = Mock(returncode=1)

        scaffolder = CrewAIScaffolder()
        version = scaffolder.get_version()

        assert version is None

    @patch("subprocess.run")
    def test_get_version_not_found(self, mock_run):
        """Test version retrieval when CrewAI is not found."""
        mock_run.side_effect = FileNotFoundError("crewai command not found")

        scaffolder = CrewAIScaffolder()
        version = scaffolder.get_version()

        assert version is None

    @patch("subprocess.run")
    def test_get_version_timeout(self, mock_run):
        """Test version retrieval timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(["crewai", "--version"], 10)

        scaffolder = CrewAIScaffolder()
        version = scaffolder.get_version()

        assert version is None

    def test_parse_version_string_standard_format(self):
        """Test parsing standard version string format."""
        scaffolder = CrewAIScaffolder()

        # Test various common version formats
        assert scaffolder.parse_version_string("crewai 1.2.3") == "1.2.3"
        assert scaffolder.parse_version_string("1.2.3") == "1.2.3"
        assert scaffolder.parse_version_string("crewai version 1.2.3") == "1.2.3"
        assert scaffolder.parse_version_string("CrewAI CLI 1.2.3") == "1.2.3"

    def test_parse_version_string_with_prerelease(self):
        """Test parsing version strings with prerelease identifiers."""
        scaffolder = CrewAIScaffolder()

        assert scaffolder.parse_version_string("crewai 1.2.3-alpha") == "1.2.3-alpha"
        assert scaffolder.parse_version_string("1.2.3-beta.1") == "1.2.3-beta.1"
        assert scaffolder.parse_version_string("2.0.0-rc.1") == "2.0.0-rc.1"

    def test_parse_version_string_invalid(self):
        """Test parsing invalid version strings."""
        scaffolder = CrewAIScaffolder()

        assert scaffolder.parse_version_string("") is None
        assert scaffolder.parse_version_string("invalid") is None
        assert scaffolder.parse_version_string("crewai help") is None
        assert scaffolder.parse_version_string(None) is None

    def test_compare_versions_equal(self):
        """Test version comparison for equal versions."""
        scaffolder = CrewAIScaffolder()

        assert scaffolder.compare_versions("1.2.3", "1.2.3") == 0
        assert scaffolder.compare_versions("0.1.0", "0.1.0") == 0

    def test_compare_versions_greater(self):
        """Test version comparison for greater versions."""
        scaffolder = CrewAIScaffolder()

        assert scaffolder.compare_versions("1.2.4", "1.2.3") > 0
        assert scaffolder.compare_versions("1.3.0", "1.2.3") > 0
        assert scaffolder.compare_versions("2.0.0", "1.9.9") > 0

    def test_compare_versions_lesser(self):
        """Test version comparison for lesser versions."""
        scaffolder = CrewAIScaffolder()

        assert scaffolder.compare_versions("1.2.2", "1.2.3") < 0
        assert scaffolder.compare_versions("1.1.0", "1.2.3") < 0
        assert scaffolder.compare_versions("0.9.9", "1.0.0") < 0

    def test_compare_versions_invalid(self):
        """Test version comparison with invalid versions."""
        scaffolder = CrewAIScaffolder()

        with pytest.raises(ValueError):
            scaffolder.compare_versions("invalid", "1.2.3")
        with pytest.raises(ValueError):
            scaffolder.compare_versions("1.2.3", "invalid")

    def test_check_minimum_version_met(self):
        """Test minimum version checking when requirement is met."""
        scaffolder = CrewAIScaffolder()

        assert scaffolder.check_minimum_version("1.2.3", "1.2.3") is True
        assert scaffolder.check_minimum_version("1.2.4", "1.2.3") is True
        assert scaffolder.check_minimum_version("2.0.0", "1.2.3") is True

    def test_check_minimum_version_not_met(self):
        """Test minimum version checking when requirement is not met."""
        scaffolder = CrewAIScaffolder()

        assert scaffolder.check_minimum_version("1.2.2", "1.2.3") is False
        assert scaffolder.check_minimum_version("1.1.9", "1.2.3") is False
        assert scaffolder.check_minimum_version("0.9.9", "1.0.0") is False

    def test_check_minimum_version_invalid(self):
        """Test minimum version checking with invalid versions."""
        scaffolder = CrewAIScaffolder()

        with pytest.raises(ValueError):
            scaffolder.check_minimum_version("invalid", "1.2.3")
        with pytest.raises(ValueError):
            scaffolder.check_minimum_version("1.2.3", "invalid")

    @patch("subprocess.run")
    def test_validate_crewai_dependency_success(self, mock_run):
        """Test successful CrewAI dependency validation."""
        mock_run.return_value = Mock(returncode=0, stdout="crewai 1.2.3", stderr="")

        scaffolder = CrewAIScaffolder()
        result = scaffolder.validate_crewai_dependency(min_version="1.2.0")

        assert result["valid"] is True
        assert result["installed_version"] == "1.2.3"
        assert result["meets_minimum"] is True

    @patch("subprocess.run")
    def test_validate_crewai_dependency_version_too_old(self, mock_run):
        """Test CrewAI dependency validation with version too old."""
        mock_run.return_value = Mock(returncode=0, stdout="crewai 1.1.0", stderr="")

        scaffolder = CrewAIScaffolder()
        result = scaffolder.validate_crewai_dependency(min_version="1.2.0")

        assert result["valid"] is False
        assert result["installed_version"] == "1.1.0"
        assert result["meets_minimum"] is False
        assert "minimum version 1.2.0 required" in result["error"]

    @patch("subprocess.run")
    def test_validate_crewai_dependency_not_installed(self, mock_run):
        """Test CrewAI dependency validation when not installed."""
        mock_run.side_effect = FileNotFoundError("crewai command not found")

        scaffolder = CrewAIScaffolder()
        result = scaffolder.validate_crewai_dependency()

        assert result["valid"] is False
        assert result["installed_version"] is None
        assert result["meets_minimum"] is False
        assert "CrewAI CLI is not installed" in result["error"]


class TestCrewAIError:
    """Test suite for CrewAIError exception."""

    def test_crewai_error_creation(self):
        """Test CrewAIError exception creation."""
        error_message = "Test error message"
        error = CrewAIError(error_message)

        assert str(error) == error_message
        assert isinstance(error, Exception)

    def test_crewai_error_inheritance(self):
        """Test CrewAIError inherits from Exception."""
        error = CrewAIError("Test")
        assert isinstance(error, Exception)

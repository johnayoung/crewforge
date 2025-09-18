"""
Tests for CrewAI Scaffolder module

Test suite for the CrewAI CLI subprocess integration, including success cases,
error handling, and edge cases for project creation.
"""

import pytest
import subprocess
import tempfile
import os
import getpass
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


class TestLLMProviderConfiguration:
    """Test suite for LLM provider selection and API key configuration."""

    def test_get_supported_providers(self):
        """Test getting list of supported LLM providers."""
        scaffolder = CrewAIScaffolder()
        providers = scaffolder.get_supported_providers()

        expected_providers = ["openai", "anthropic", "google", "groq", "sambanova"]
        assert providers == expected_providers
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_validate_provider_valid(self):
        """Test validating supported LLM providers."""
        scaffolder = CrewAIScaffolder()

        valid_providers = ["openai", "anthropic", "google", "groq", "sambanova"]
        for provider in valid_providers:
            result = scaffolder.validate_provider(provider)
            assert result["valid"] is True
            assert result["provider"] == provider
            assert result["error"] is None

    def test_validate_provider_invalid(self):
        """Test validation of unsupported LLM providers."""
        scaffolder = CrewAIScaffolder()

        invalid_providers = ["invalid", "unknown", "fake_provider"]
        for provider in invalid_providers:
            result = scaffolder.validate_provider(provider)
            assert result["valid"] is False
            assert result["provider"] == provider
            assert "not supported" in result["error"].lower()

        # Test empty provider separately
        result = scaffolder.validate_provider("")
        assert result["valid"] is False
        assert result["provider"] == ""
        assert "cannot be empty" in result["error"].lower()

    def test_validate_provider_case_insensitive(self):
        """Test provider validation is case-insensitive."""
        scaffolder = CrewAIScaffolder()

        case_variants = ["OpenAI", "ANTHROPIC", "Google", "gRoQ", "SambaNova"]
        for provider in case_variants:
            result = scaffolder.validate_provider(provider)
            assert result["valid"] is True
            assert result["provider"] == provider.lower()

    def test_get_api_key_env_var_mapping(self):
        """Test getting correct environment variable names for providers."""
        scaffolder = CrewAIScaffolder()

        expected_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "sambanova": "SAMBANOVA_API_KEY",
        }

        for provider, expected_env_var in expected_mapping.items():
            env_var = scaffolder.get_api_key_env_var(provider)
            assert env_var == expected_env_var

    def test_get_api_key_env_var_invalid_provider(self):
        """Test getting environment variable for invalid provider."""
        scaffolder = CrewAIScaffolder()

        with pytest.raises(ValueError, match="Unsupported provider"):
            scaffolder.get_api_key_env_var("invalid_provider")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key_123"})
    def test_check_api_key_from_environment(self):
        """Test checking API key from environment variables."""
        scaffolder = CrewAIScaffolder()

        result = scaffolder.check_api_key("openai")
        assert result["available"] is True
        assert result["source"] == "environment"
        assert result["key"].startswith("test_key")
        assert result["error"] is None

    def test_check_api_key_missing_from_environment(self):
        """Test checking API key when not in environment."""
        scaffolder = CrewAIScaffolder()

        with patch.dict("os.environ", {}, clear=True):
            result = scaffolder.check_api_key("openai")
            assert result["available"] is False
            assert result["source"] is None
            assert result["key"] is None
            assert "not found in environment" in result["error"]

    def test_validate_api_key_format_openai(self):
        """Test API key format validation for OpenAI."""
        scaffolder = CrewAIScaffolder()

        # Valid OpenAI API key format (starts with sk-)
        valid_key = "sk-1234567890abcdef1234567890abcdef12345678"
        result = scaffolder.validate_api_key_format("openai", valid_key)
        assert result["valid"] is True
        assert result["error"] is None

        # Invalid format
        invalid_key = "invalid_key_format"
        result = scaffolder.validate_api_key_format("openai", invalid_key)
        assert result["valid"] is False
        assert "invalid format" in result["error"].lower()

    def test_validate_api_key_format_anthropic(self):
        """Test API key format validation for Anthropic."""
        scaffolder = CrewAIScaffolder()

        # Valid Anthropic API key format (starts with sk-ant-)
        valid_key = "sk-ant-api03-1234567890abcdef"
        result = scaffolder.validate_api_key_format("anthropic", valid_key)
        assert result["valid"] is True

        # Invalid format
        invalid_key = "sk-1234567890"
        result = scaffolder.validate_api_key_format("anthropic", invalid_key)
        assert result["valid"] is False

    def test_configure_provider_with_api_key(self):
        """Test configuring LLM provider with API key."""
        scaffolder = CrewAIScaffolder()

        config = {
            "provider": "openai",
            "api_key": "sk-1234567890abcdef1234567890abcdef12345678",
            "model": "gpt-3.5-turbo",
        }

        result = scaffolder.configure_provider(config)
        assert result["success"] is True
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-3.5-turbo"
        assert result["error"] is None

    def test_configure_provider_invalid_provider(self):
        """Test configuring invalid LLM provider."""
        scaffolder = CrewAIScaffolder()

        config = {"provider": "invalid_provider", "api_key": "test_key"}

        result = scaffolder.configure_provider(config)
        assert result["success"] is False
        assert "not supported" in result["error"].lower()

    def test_configure_provider_invalid_api_key(self):
        """Test configuring provider with invalid API key."""
        scaffolder = CrewAIScaffolder()

        config = {"provider": "openai", "api_key": "invalid_key_format"}

        result = scaffolder.configure_provider(config)
        assert result["success"] is False
        assert "invalid format" in result["error"].lower()

    def test_generate_provider_config_file(self):
        """Test generating provider configuration file for CrewAI project."""
        scaffolder = CrewAIScaffolder()

        config = {"provider": "openai", "api_key": "sk-test123456789", "model": "gpt-4"}

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()

            result = scaffolder.generate_provider_config_file(project_path, config)

            assert result["success"] is True
            config_file = project_path / ".env"
            assert config_file.exists()

            # Check config file contents
            content = config_file.read_text()
            assert "OPENAI_API_KEY=sk-test123456789" in content
            assert "LLM_MODEL=gpt-4" in content

    def test_create_crew_with_provider_config(self):
        """Test creating CrewAI project with LLM provider configuration."""
        scaffolder = CrewAIScaffolder()

        with tempfile.TemporaryDirectory() as temp_dir:
            target_directory = Path(temp_dir)

            provider_config = {
                "provider": "openai",
                "api_key": "sk-test123456789",
                "model": "gpt-3.5-turbo",
            }

            with patch.object(scaffolder, "check_crewai_available", return_value=True):
                with patch.object(
                    scaffolder, "validate_crewai_dependency"
                ) as mock_validate:
                    mock_validate.return_value = {
                        "valid": True,
                        "installed_version": "1.0.0",
                        "meets_minimum": True,
                        "error": None,
                    }

                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

                        # Mock project directory creation
                        project_path = target_directory / "test_project"
                        project_path.mkdir(parents=True)

                        result = scaffolder.create_crew_with_provider(
                            "test_project", target_directory, provider_config
                        )

                        assert result["success"] is True
                        assert result["provider_configured"] is True
                        assert result["project_path"] == project_path

    def test_interactive_provider_selection(self):
        """Test interactive provider selection workflow."""
        scaffolder = CrewAIScaffolder()

        # Mock user input for provider selection
        with patch("builtins.input", side_effect=["2"]):  # Select anthropic (index 2)
            result = scaffolder.interactive_provider_selection()

            assert result["provider"] == "anthropic"
            assert result["cancelled"] is False

    def test_interactive_provider_selection_invalid_choice(self):
        """Test interactive provider selection with invalid choice."""
        scaffolder = CrewAIScaffolder()

        # Mock user input: invalid choice, then valid choice
        with patch(
            "builtins.input", side_effect=["99", "1"]
        ):  # Invalid, then select openai
            result = scaffolder.interactive_provider_selection()

            assert result["provider"] == "openai"
            assert result["cancelled"] is False

    def test_interactive_api_key_input(self):
        """Test interactive API key input workflow."""
        scaffolder = CrewAIScaffolder()

        test_key = "sk-1234567890abcdef1234567890abcdef12345678"

        with patch("getpass.getpass", return_value=test_key):
            result = scaffolder.interactive_api_key_input("openai")

            assert result["api_key"] == test_key
            assert result["cancelled"] is False
            assert result["error"] is None

    def test_interactive_api_key_input_invalid_format(self):
        """Test interactive API key input with invalid format."""
        scaffolder = CrewAIScaffolder()

        invalid_key = "invalid_key"
        valid_key = "sk-1234567890abcdef1234567890abcdef12345678"

        with patch("getpass.getpass", side_effect=[invalid_key, valid_key]):
            result = scaffolder.interactive_api_key_input("openai")

            assert result["api_key"] == valid_key
            assert result["cancelled"] is False

    def test_full_provider_configuration_workflow(self):
        """Test the complete provider configuration workflow."""
        scaffolder = CrewAIScaffolder()

        # Mock interactive selections
        with patch.object(
            scaffolder, "interactive_provider_selection"
        ) as mock_provider:
            mock_provider.return_value = {"provider": "openai", "cancelled": False}

            with patch.object(scaffolder, "interactive_api_key_input") as mock_api_key:
                mock_api_key.return_value = {
                    "api_key": "sk-test123456789",
                    "cancelled": False,
                    "error": None,
                }

                result = scaffolder.configure_llm_provider()

                assert result["success"] is True
                assert result["provider"] == "openai"
                assert result["api_key"] == "sk-test123456789"
                assert result["error"] is None


class TestDirectoryManagement:
    """Test suite for directory management and file system operations."""

    def test_ensure_project_directory_creation(self):
        """Test creating project directory with proper permissions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scaffolder = CrewAIScaffolder()
            project_path = Path(temp_dir) / "test_project"

            # Test directory creation
            result = scaffolder.ensure_project_directory(project_path)

            assert result["success"] is True
            assert project_path.exists()
            assert project_path.is_dir()
            assert result["created"] is True
            assert result["path"] == project_path

    def test_ensure_project_directory_exists(self):
        """Test handling existing project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scaffolder = CrewAIScaffolder()
            project_path = Path(temp_dir) / "existing_project"
            project_path.mkdir()

            # Test with existing directory
            result = scaffolder.ensure_project_directory(project_path)

            assert result["success"] is True
            assert project_path.exists()
            assert result["created"] is False
            assert result["path"] == project_path

    def test_ensure_project_directory_file_conflict(self):
        """Test handling file conflict when creating directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scaffolder = CrewAIScaffolder()
            project_path = Path(temp_dir) / "conflicting_file"

            # Create a file where directory should be
            project_path.touch()

            # Test directory creation should fail
            result = scaffolder.ensure_project_directory(project_path)

            assert result["success"] is False
            assert "exists but is not a directory" in result["error"]

    @patch("pathlib.Path.mkdir")
    def test_ensure_project_directory_permission_error(self, mock_mkdir):
        """Test handling permission errors during directory creation."""
        mock_mkdir.side_effect = PermissionError("Permission denied")

        scaffolder = CrewAIScaffolder()
        project_path = Path("/tmp/test_project")

        result = scaffolder.ensure_project_directory(project_path)

        assert result["success"] is False
        assert "Permission denied" in result["error"]

    def test_validate_project_path_valid(self):
        """Test validation of valid project paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scaffolder = CrewAIScaffolder()
            project_path = Path(temp_dir) / "valid_project"

            result = scaffolder.validate_project_path(project_path)

            assert result["valid"] is True
            assert result["error"] is None

    def test_validate_project_path_reserved_name(self):
        """Test validation rejects reserved directory names."""
        scaffolder = CrewAIScaffolder()

        reserved_names = ["con", "prn", "aux", "nul", "com1", "lpt1"]
        for name in reserved_names:
            project_path = Path("/tmp") / name
            result = scaffolder.validate_project_path(project_path)

            assert result["valid"] is False
            assert "reserved name" in result["error"].lower()

    def test_validate_project_path_invalid_characters(self):
        """Test validation rejects paths with invalid characters."""
        scaffolder = CrewAIScaffolder()

        invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
        for char in invalid_chars:
            project_path = Path("/tmp") / f"project{char}name"
            result = scaffolder.validate_project_path(project_path)

            assert result["valid"] is False
            assert "invalid characters" in result["error"].lower()

    def test_validate_project_path_too_long(self):
        """Test validation rejects paths that are too long."""
        scaffolder = CrewAIScaffolder()

        # Create a very long path
        long_name = "a" * 256
        project_path = Path("/tmp") / long_name

        result = scaffolder.validate_project_path(project_path)

        assert result["valid"] is False
        assert "too long" in result["error"].lower()

    def test_safe_cleanup_directory(self):
        """Test safe cleanup of project directory on failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scaffolder = CrewAIScaffolder()
            project_path = Path(temp_dir) / "cleanup_test"
            project_path.mkdir()

            # Create some files in the directory
            (project_path / "test_file.txt").write_text("test content")
            (project_path / "subdir").mkdir()
            (project_path / "subdir" / "nested_file.txt").write_text("nested content")

            # Test cleanup
            result = scaffolder.safe_cleanup_directory(project_path)

            assert result["success"] is True
            assert not project_path.exists()

    def test_safe_cleanup_directory_not_exists(self):
        """Test safe cleanup when directory doesn't exist."""
        scaffolder = CrewAIScaffolder()
        project_path = Path("/tmp/nonexistent_directory")

        result = scaffolder.safe_cleanup_directory(project_path)

        assert result["success"] is True  # Should not fail for non-existent directory

    def test_safe_cleanup_directory_permission_error(self):
        """Test safe cleanup handles permission errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scaffolder = CrewAIScaffolder()
            project_path = Path(temp_dir) / "permission_test"
            project_path.mkdir()  # Create the directory so it exists

            # Use a more specific patch to avoid affecting tempfile cleanup
            with patch.object(
                scaffolder,
                "safe_cleanup_directory",
                wraps=scaffolder.safe_cleanup_directory,
            ) as mock_method:
                with patch(
                    "shutil.rmtree", side_effect=PermissionError("Permission denied")
                ):
                    result = scaffolder.safe_cleanup_directory(project_path)

                    assert result["success"] is False
                    assert "Permission denied" in result["error"]

    def test_create_backup_directory(self):
        """Test creating backup of existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scaffolder = CrewAIScaffolder()
            original_path = Path(temp_dir) / "original_project"
            original_path.mkdir()

            # Create content in original directory
            (original_path / "config.yaml").write_text("original: config")

            # Create backup
            result = scaffolder.create_backup_directory(original_path)

            assert result["success"] is True
            assert result["backup_path"].exists()
            assert result["backup_path"].name.startswith("original_project.backup.")
            assert (result["backup_path"] / "config.yaml").exists()

    def test_get_safe_project_name(self):
        """Test generating safe project names from user input."""
        scaffolder = CrewAIScaffolder()

        test_cases = [
            ("My Cool Project", "my_cool_project"),
            ("project-with-dashes", "project_with_dashes"),
            ("Project123", "project123"),
            ("project with spaces", "project_with_spaces"),
            ("Project@#$%", "project"),
            ("", "untitled_project"),
            ("123project", "project_123"),
        ]

        for input_name, expected in test_cases:
            result = scaffolder.get_safe_project_name(input_name)
            assert result == expected


class TestCrewAIErrorHandling:
    """Test suite for enhanced CrewAI CLI error handling and recovery."""

    def test_categorize_cli_error_network_issues(self):
        """Test error categorization for network-related CLI failures."""
        scaffolder = CrewAIScaffolder()

        network_errors = [
            "ConnectionError: Unable to reach server",
            "Network timeout occurred",
            "Failed to resolve hostname",
            "Connection refused",
            "SSL certificate verification failed",
        ]

        for error_msg in network_errors:
            category = scaffolder.categorize_cli_error(error_msg, returncode=1)
            assert category["type"] == "network"
            assert category["retryable"] is True
            assert "network" in category["user_message"].lower()

    def test_categorize_cli_error_permission_issues(self):
        """Test error categorization for permission-related CLI failures."""
        scaffolder = CrewAIScaffolder()

        permission_errors = [
            "Permission denied",
            "Access denied",
            "Insufficient privileges",
            "Operation not permitted",
            "You don't have permission",
        ]

        for error_msg in permission_errors:
            category = scaffolder.categorize_cli_error(error_msg, returncode=1)
            assert category["type"] == "permission"
            assert category["retryable"] is False
            assert "permission" in category["user_message"].lower()

    def test_categorize_cli_error_disk_space(self):
        """Test error categorization for disk space issues."""
        scaffolder = CrewAIScaffolder()

        disk_errors = [
            "No space left on device",
            "Disk full",
            "Not enough free space",
            "Unable to write file: disk full",
        ]

        for error_msg in disk_errors:
            category = scaffolder.categorize_cli_error(error_msg, returncode=1)
            assert category["type"] == "disk_space"
            assert category["retryable"] is False
            assert "disk space" in category["user_message"].lower()

    def test_categorize_cli_error_corrupted_installation(self):
        """Test error categorization for corrupted CLI installation."""
        scaffolder = CrewAIScaffolder()

        corruption_errors = [
            "Command not found: crewai",
            "ImportError: No module named 'crewai'",
            "ModuleNotFoundError: crewai",
            "crewai: command not found",
            "Python module 'crewai' is corrupted",
        ]

        for error_msg in corruption_errors:
            category = scaffolder.categorize_cli_error(error_msg, returncode=127)
            assert category["type"] == "installation"
            assert category["retryable"] is False
            assert "install" in category["user_message"].lower()

    def test_categorize_cli_error_invalid_command(self):
        """Test error categorization for invalid CLI commands."""
        scaffolder = CrewAIScaffolder()

        invalid_errors = [
            "Invalid command: create",
            "Unknown option: --invalid",
            "Error: unrecognized arguments",
            "usage: crewai [-h]",
            "Invalid project name format",
        ]

        for error_msg in invalid_errors:
            category = scaffolder.categorize_cli_error(error_msg, returncode=2)
            assert category["type"] == "invalid_command"
            assert category["retryable"] is False
            assert "command" in category["user_message"].lower()

    def test_categorize_cli_error_transient_failures(self):
        """Test error categorization for transient failures."""
        scaffolder = CrewAIScaffolder()

        transient_errors = [
            "Temporary failure in name resolution",
            "Resource temporarily unavailable",
            "Server temporarily unavailable",
            "Rate limit exceeded, please try again",
        ]

        for error_msg in transient_errors:
            category = scaffolder.categorize_cli_error(error_msg, returncode=1)
            assert category["type"] == "transient"
            assert category["retryable"] is True
            assert "try again" in category["user_message"].lower()

    @patch("subprocess.run")
    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    def test_create_crew_with_retry_logic_success_after_retry(
        self, mock_validate, mock_run
    ):
        """Test retry logic succeeds after initial failure."""
        mock_validate.return_value = {
            "valid": True,
            "installed_version": "1.2.3",
            "meets_minimum": True,
            "error": None,
        }

        # First call fails with transient error, second succeeds
        mock_run.side_effect = [
            Mock(returncode=1, stderr="Network timeout occurred", stdout=""),
            Mock(returncode=0, stdout="Project created successfully", stderr=""),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            scaffolder = CrewAIScaffolder()
            target_directory = Path(temp_dir)
            project_name = "test_project"
            project_path = target_directory / project_name

            # Create the project directory for the second attempt
            project_path.mkdir(parents=True)

            result = scaffolder.create_crew_with_retry(
                project_name, target_directory, max_retries=2
            )

            assert result["success"] is True
            assert result["attempts"] == 2
            assert mock_run.call_count == 2

    @patch("subprocess.run")
    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    def test_create_crew_with_retry_logic_max_retries_exceeded(
        self, mock_validate, mock_run
    ):
        """Test retry logic fails after max retries exceeded."""
        mock_validate.return_value = {
            "valid": True,
            "installed_version": "1.2.3",
            "meets_minimum": True,
            "error": None,
        }

        # Always fail with transient error
        mock_run.return_value = Mock(
            returncode=1, stderr="Network timeout occurred", stdout=""
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            scaffolder = CrewAIScaffolder()
            target_directory = Path(temp_dir)

            with pytest.raises(CrewAIError) as exc_info:
                scaffolder.create_crew_with_retry(
                    "test_project", target_directory, max_retries=2
                )

            assert "Max retries exceeded" in str(exc_info.value)
            assert mock_run.call_count == 2

    @patch("subprocess.run")
    @patch("crewforge.scaffolder.CrewAIScaffolder.validate_crewai_dependency")
    def test_create_crew_with_retry_logic_non_retryable_error(
        self, mock_validate, mock_run
    ):
        """Test retry logic doesn't retry non-retryable errors."""
        mock_validate.return_value = {
            "valid": True,
            "installed_version": "1.2.3",
            "meets_minimum": True,
            "error": None,
        }

        # Fail with permission error (non-retryable)
        mock_run.return_value = Mock(
            returncode=1, stderr="Permission denied: cannot create directory", stdout=""
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            scaffolder = CrewAIScaffolder()
            target_directory = Path(temp_dir)

            with pytest.raises(CrewAIError) as exc_info:
                scaffolder.create_crew_with_retry(
                    "test_project", target_directory, max_retries=3
                )

            assert "Permission denied" in str(exc_info.value)
            assert mock_run.call_count == 1  # No retry attempted

    def test_exponential_backoff_timing(self):
        """Test exponential backoff timing calculation."""
        scaffolder = CrewAIScaffolder()

        # Test backoff calculation
        assert scaffolder.calculate_backoff_delay(0) == 1.0  # First retry
        assert scaffolder.calculate_backoff_delay(1) == 2.0  # Second retry
        assert scaffolder.calculate_backoff_delay(2) == 4.0  # Third retry
        assert scaffolder.calculate_backoff_delay(3) == 8.0  # Fourth retry

        # Test max delay cap
        assert scaffolder.calculate_backoff_delay(10) == 60.0  # Capped at 60 seconds

    @patch("time.sleep")
    def test_retry_with_backoff_timing(self, mock_sleep):
        """Test that retry logic uses exponential backoff timing."""
        scaffolder = CrewAIScaffolder()

        with patch.object(scaffolder, "create_crew") as mock_create:
            # First call fails, second succeeds
            mock_create.side_effect = [
                CrewAIError("Network timeout occurred"),
                {"success": True, "project_path": Path("/test")},
            ]

            with patch.object(scaffolder, "categorize_cli_error") as mock_categorize:
                mock_categorize.return_value = {
                    "type": "network",
                    "retryable": True,
                    "user_message": "Network error occurred",
                }

                result = scaffolder.create_crew_with_retry(
                    "test", Path("/tmp"), max_retries=2
                )

                # Should have slept once between retries
                mock_sleep.assert_called_once_with(1.0)
                assert result["success"] is True

    def test_create_detailed_error_report(self):
        """Test creation of detailed error reports for troubleshooting."""
        scaffolder = CrewAIScaffolder()

        error_context = {
            "command": ["crewai", "create", "crew", "test_project"],
            "stderr": "Permission denied: cannot create directory '/protected'",
            "stdout": "",
            "returncode": 1,
            "cwd": "/tmp",
            "timeout": 60,
            "attempt": 1,
        }

        report = scaffolder.create_error_report(error_context)

        assert "Command" in report
        assert "crewai create crew test_project" in report
        assert "Permission denied" in report
        assert "Return Code: 1" in report
        assert "Working Directory: /tmp" in report
        assert "Timeout: 60 seconds" in report

    def test_suggest_recovery_actions_permission_error(self):
        """Test recovery action suggestions for permission errors."""
        scaffolder = CrewAIScaffolder()

        error_category = {
            "type": "permission",
            "retryable": False,
            "user_message": "Permission denied error occurred",
        }

        suggestions = scaffolder.suggest_recovery_actions(
            error_category, Path("/protected/dir")
        )

        assert len(suggestions) > 0
        assert any(
            "chmod" in suggestion.lower() or "permission" in suggestion.lower()
            for suggestion in suggestions
        )
        assert any(
            "different" in suggestion.lower() and "directory" in suggestion.lower()
            for suggestion in suggestions
        )

    def test_suggest_recovery_actions_installation_error(self):
        """Test recovery action suggestions for installation errors."""
        scaffolder = CrewAIScaffolder()

        error_category = {
            "type": "installation",
            "retryable": False,
            "user_message": "CrewAI CLI not found or corrupted",
        }

        suggestions = scaffolder.suggest_recovery_actions(error_category, Path("/tmp"))

        assert len(suggestions) > 0
        assert any(
            "pip install" in suggestion.lower() or "install" in suggestion.lower()
            for suggestion in suggestions
        )
        assert any(
            "virtual environment" in suggestion.lower() for suggestion in suggestions
        )

    def test_graceful_degradation_fallback_directory(self):
        """Test graceful degradation when target directory is inaccessible."""
        scaffolder = CrewAIScaffolder()

        inaccessible_dir = Path("/root/protected")  # Typically inaccessible

        fallback_result = scaffolder.suggest_fallback_directory(inaccessible_dir)

        assert fallback_result["success"] is True
        assert fallback_result["fallback_path"] != inaccessible_dir
        assert (
            fallback_result["fallback_path"].exists()
            or fallback_result["fallback_path"].parent.exists()
        )

    @patch("subprocess.run")
    def test_cli_health_check(self, mock_run):
        """Test CLI health check functionality."""
        scaffolder = CrewAIScaffolder()

        # Mock successful health check
        mock_run.return_value = Mock(returncode=0, stdout="crewai 1.2.3", stderr="")

        health = scaffolder.perform_cli_health_check()

        assert health["healthy"] is True
        assert health["version"] is not None
        assert "issues" not in health or len(health["issues"]) == 0

    @patch("subprocess.run")
    def test_cli_health_check_with_issues(self, mock_run):
        """Test CLI health check detects issues."""
        scaffolder = CrewAIScaffolder()

        # Mock CLI with issues
        mock_run.side_effect = [
            Mock(returncode=0, stdout="crewai 0.5.0", stderr=""),  # Version check
            Mock(
                returncode=1, stderr="ImportError: missing dependency", stdout=""
            ),  # Health check
        ]

        health = scaffolder.perform_cli_health_check()

        assert health["healthy"] is False
        assert len(health["issues"]) > 0
        assert any("dependency" in issue.lower() for issue in health["issues"])

    def test_error_context_collection(self):
        """Test comprehensive error context collection."""
        scaffolder = CrewAIScaffolder()

        with tempfile.TemporaryDirectory() as temp_dir:
            context = scaffolder.collect_error_context(
                command=["crewai", "create", "crew", "test"],
                cwd=temp_dir,
                error_msg="Test error message",
                returncode=1,
            )

            assert "command" in context
            assert "cwd" in context
            assert "error_msg" in context
            assert "returncode" in context
            assert "timestamp" in context
            assert "system_info" in context

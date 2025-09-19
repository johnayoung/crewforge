"""
CrewAI Scaffolder - Native CrewAI project creation using official CLI commands

This module integrates with the CrewAI CLI to create base project structures
using the native `crewai create crew <name>` command, providing error handling
and subprocess management.
"""

import subprocess
import shutil
import re
import os
import getpass
import datetime
import time
import platform
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
import logging

# Import enhancement engine for project customization
try:
    from .enhancement import EnhancementEngine, EnhancementError
except ImportError:
    # Handle import error gracefully during testing
    EnhancementEngine = None
    EnhancementError = Exception


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

    # LLM Provider Configuration Methods

    def get_supported_providers(self) -> List[str]:
        """
        Get list of supported LLM providers compatible with liteLLM.

        Returns:
            List of supported provider names
        """
        return ["openai", "anthropic", "google", "groq", "sambanova"]

    def validate_provider(self, provider: str) -> Dict[str, Any]:
        """
        Validate if the given provider is supported.

        Args:
            provider: Provider name to validate

        Returns:
            Dictionary containing validation results
        """
        result = {
            "valid": False,
            "provider": provider.lower() if provider else "",
            "error": None,
        }

        if not provider or not provider.strip():
            result["error"] = "Provider name cannot be empty"
            return result

        normalized_provider = provider.lower().strip()
        supported_providers = self.get_supported_providers()

        if normalized_provider not in supported_providers:
            result["error"] = (
                f"Provider '{provider}' is not supported. Supported providers: {supported_providers}"
            )
            return result

        result["valid"] = True
        result["provider"] = normalized_provider
        return result

    def get_api_key_env_var(self, provider: str) -> str:
        """
        Get the environment variable name for the given provider's API key.

        Args:
            provider: Provider name

        Returns:
            Environment variable name

        Raises:
            ValueError: If provider is not supported
        """
        env_var_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "sambanova": "SAMBANOVA_API_KEY",
        }

        if provider not in env_var_mapping:
            raise ValueError(f"Unsupported provider: {provider}")

        return env_var_mapping[provider]

    def check_api_key(self, provider: str) -> Dict[str, Any]:
        """
        Check if API key is available for the given provider.

        Args:
            provider: Provider name

        Returns:
            Dictionary containing API key availability information
        """
        result = {"available": False, "source": None, "key": None, "error": None}

        try:
            env_var = self.get_api_key_env_var(provider)
            api_key = os.getenv(env_var)

            if api_key:
                result["available"] = True
                result["source"] = "environment"
                result["key"] = api_key
            else:
                result["error"] = f"API key not found in environment variable {env_var}"

        except ValueError as e:
            result["error"] = str(e)

        return result

    def validate_api_key_format(self, provider: str, api_key: str) -> Dict[str, Any]:
        """
        Validate API key format for the given provider.

        Args:
            provider: Provider name
            api_key: API key to validate

        Returns:
            Dictionary containing validation results
        """
        result = {"valid": False, "error": None}

        if not api_key or not api_key.strip():
            result["error"] = "API key cannot be empty"
            return result

        # Provider-specific validation patterns
        patterns = {
            "openai": r"^sk-[a-zA-Z0-9]{20,}$",
            "anthropic": r"^sk-ant-[a-zA-Z0-9\-_]{20,}$",
            "google": r"^[a-zA-Z0-9\-_]{20,}$",
            "groq": r"^gsk_[a-zA-Z0-9]{50,}$",
            "sambanova": r"^[a-zA-Z0-9\-_]{20,}$",
        }

        pattern = patterns.get(provider)
        if not pattern:
            # If no specific pattern, just check it's not empty
            result["valid"] = True
            return result

        import re

        if re.match(pattern, api_key.strip()):
            result["valid"] = True
        else:
            result["error"] = f"API key has invalid format for {provider} provider"

        return result

    def configure_provider(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure LLM provider with validation.

        Args:
            config: Configuration dictionary with provider, api_key, and optional model

        Returns:
            Dictionary containing configuration results
        """
        result = {"success": False, "provider": None, "model": None, "error": None}

        # Validate provider
        provider = config.get("provider", "").strip()
        provider_validation = self.validate_provider(provider)
        if not provider_validation["valid"]:
            result["error"] = provider_validation["error"]
            return result

        provider = provider_validation["provider"]
        result["provider"] = provider

        # Validate API key
        api_key = config.get("api_key", "").strip()
        key_validation = self.validate_api_key_format(provider, api_key)
        if not key_validation["valid"]:
            result["error"] = key_validation["error"]
            return result

        # Set model (use default if not provided)
        default_models = {
            "openai": "gpt-3.5-turbo",
            "anthropic": "claude-3-sonnet-20240229",
            "google": "gemini-pro",
            "groq": "mixtral-8x7b-32768",
            "sambanova": "Meta-Llama-3.1-8B-Instruct",
        }

        model = config.get("model") or default_models.get(provider)
        result["model"] = model

        result["success"] = True
        return result

    def generate_provider_config_file(
        self, project_path: Path, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate provider configuration file for CrewAI project.

        Args:
            project_path: Path to the CrewAI project directory
            config: Configuration dictionary with provider settings

        Returns:
            Dictionary containing generation results
        """
        result = {"success": False, "config_file": None, "error": None}

        try:
            config_file = project_path / ".env"

            # Generate environment variables content
            env_content = []

            # Add API key
            provider = config["provider"]
            api_key = config["api_key"]
            env_var = self.get_api_key_env_var(provider)
            env_content.append(f"{env_var}={api_key}")

            # Add model configuration
            if config.get("model"):
                env_content.append(f"LLM_MODEL={config['model']}")

            # Add provider configuration
            env_content.append(f"LLM_PROVIDER={provider}")

            # Write to .env file
            with open(config_file, "w") as f:
                f.write("\n".join(env_content) + "\n")

            result["success"] = True
            result["config_file"] = config_file

        except Exception as e:
            result["error"] = f"Failed to generate config file: {str(e)}"

        return result

    def create_crew_with_provider(
        self,
        project_name: str,
        target_directory: Path,
        provider_config: Dict[str, Any],
        min_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create CrewAI project with LLM provider configuration.

        Args:
            project_name: Name of the crew project
            target_directory: Directory where project should be created
            provider_config: LLM provider configuration
            min_version: Optional minimum CrewAI version requirement

        Returns:
            Dictionary containing creation results
        """
        result = {
            "success": False,
            "project_path": None,
            "provider_configured": False,
            "error": None,
        }

        try:
            # First create the basic CrewAI project
            crew_result = self.create_crew(project_name, target_directory, min_version)
            if not crew_result["success"]:
                result["error"] = (
                    f"Failed to create CrewAI project: {crew_result.get('stderr', 'Unknown error')}"
                )
                return result

            project_path = crew_result["project_path"]
            result["project_path"] = project_path

            # Configure provider
            config_result = self.generate_provider_config_file(
                project_path, provider_config
            )
            if not config_result["success"]:
                result["error"] = config_result["error"]
                return result

            result["success"] = True
            result["provider_configured"] = True

        except Exception as e:
            result["error"] = f"Failed to create project with provider: {str(e)}"

        return result

    def interactive_provider_selection(self) -> Dict[str, Any]:
        """
        Interactive provider selection workflow.

        Returns:
            Dictionary containing selected provider or cancellation
        """
        result = {"provider": None, "cancelled": False}

        providers = self.get_supported_providers()

        print("\nAvailable LLM Providers:")
        for i, provider in enumerate(providers, 1):
            print(f"{i}. {provider}")

        while True:
            try:
                choice = input("\nSelect a provider (number): ").strip()

                if not choice:
                    result["cancelled"] = True
                    return result

                provider_index = int(choice) - 1
                if 0 <= provider_index < len(providers):
                    result["provider"] = providers[provider_index]
                    return result
                else:
                    print(f"Invalid choice. Please select 1-{len(providers)}")

            except (ValueError, KeyboardInterrupt):
                print(
                    "\nInvalid input. Please enter a number or press Ctrl+C to cancel."
                )
                continue

    def interactive_api_key_input(self, provider: str) -> Dict[str, Any]:
        """
        Interactive API key input workflow.

        Args:
            provider: Provider name

        Returns:
            Dictionary containing API key or error information
        """
        result = {"api_key": None, "cancelled": False, "error": None}

        while True:
            try:
                print(f"\nEnter API key for {provider}:")
                api_key = getpass.getpass("API Key: ").strip()

                if not api_key:
                    result["cancelled"] = True
                    return result

                # Validate format
                validation = self.validate_api_key_format(provider, api_key)
                if validation["valid"]:
                    result["api_key"] = api_key
                    return result
                else:
                    print(f"Error: {validation['error']}")
                    print("Please try again.")

            except KeyboardInterrupt:
                result["cancelled"] = True
                return result

    def configure_llm_provider(self) -> Dict[str, Any]:
        """
        Full LLM provider configuration workflow.

        Returns:
            Dictionary containing configuration results
        """
        result = {"success": False, "provider": None, "api_key": None, "error": None}

        # Select provider
        provider_selection = self.interactive_provider_selection()
        if provider_selection["cancelled"]:
            result["error"] = "Provider selection cancelled"
            return result

        provider = provider_selection["provider"]
        result["provider"] = provider

        # Get API key
        api_key_input = self.interactive_api_key_input(provider)
        if api_key_input["cancelled"]:
            result["error"] = "API key input cancelled"
            return result

        if api_key_input["error"]:
            result["error"] = api_key_input["error"]
            return result

        result["api_key"] = api_key_input["api_key"]
        result["success"] = True
        return result

    def create_crew_with_advanced_directory_management(
        self,
        project_name: str,
        target_directory: Path,
        min_version: Optional[str] = None,
        handle_existing: str = "backup",  # "backup", "overwrite", "error"
    ) -> Dict[str, Any]:
        """
        Execute 'crewai create crew <name>' command with advanced directory management.

        Args:
            project_name: Name of the crew project to create
            target_directory: Directory where the project should be created
            min_version: Optional minimum CrewAI version requirement
            handle_existing: How to handle existing project directories

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

        # Ensure target directory exists with proper validation
        dir_result = self.ensure_project_directory(target_directory)
        if not dir_result["success"]:
            raise CrewAIError(
                f"Failed to create target directory: {dir_result['error']}"
            )

        # Generate safe project name
        safe_project_name = self.get_safe_project_name(project_name)
        project_path = target_directory / safe_project_name

        # Handle existing project directory
        if project_path.exists():
            if handle_existing == "error":
                raise CrewAIError(f"Project directory {project_path} already exists")
            elif handle_existing == "backup":
                self.logger.warning(f"Project directory {project_path} already exists")
                backup_result = self.create_backup_directory(project_path)
                if backup_result["success"]:
                    self.logger.info(
                        f"Created backup at {backup_result['backup_path']}"
                    )
                # Clean up existing directory
                cleanup_result = self.safe_cleanup_directory(project_path)
                if not cleanup_result["success"]:
                    raise CrewAIError(
                        f"Failed to clean up existing directory: {cleanup_result['error']}"
                    )
            elif handle_existing == "overwrite":
                cleanup_result = self.safe_cleanup_directory(project_path)
                if not cleanup_result["success"]:
                    raise CrewAIError(
                        f"Failed to clean up existing directory: {cleanup_result['error']}"
                    )

        # Execute CrewAI create command
        try:
            self.logger.info(
                f"Creating CrewAI project '{safe_project_name}' in {target_directory}"
            )

            result = subprocess.run(
                ["crewai", "create", "crew", safe_project_name],
                cwd=target_directory,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout for project creation
            )

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
                "project_name": safe_project_name,
                "original_name": project_name,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            raise CrewAIError("CrewAI project creation timed out after 60 seconds")
        except subprocess.SubprocessError as e:
            raise CrewAIError(f"Subprocess error during CrewAI execution: {str(e)}")
        except Exception as e:
            # Cleanup on failure
            if project_path.exists():
                cleanup_result = self.safe_cleanup_directory(project_path)
                if not cleanup_result["success"]:
                    self.logger.error(
                        f"Failed to cleanup after error: {cleanup_result['error']}"
                    )
            raise CrewAIError(f"Unexpected error during CrewAI execution: {str(e)}")

    def ensure_project_directory(self, project_path: Path) -> Dict[str, Any]:
        """
        Ensure project directory exists with proper validation and error handling.

        Args:
            project_path: Path to the project directory to create

        Returns:
            Dictionary containing creation results and metadata
        """
        result = {
            "success": False,
            "created": False,
            "path": project_path,
            "error": None,
        }

        try:
            # Validate project path first
            validation = self.validate_project_path(project_path)
            if not validation["valid"]:
                result["error"] = validation["error"]
                return result

            # Check if path exists and is not a directory
            if project_path.exists() and not project_path.is_dir():
                result["error"] = f"Path {project_path} exists but is not a directory"
                return result

            # Create directory if it doesn't exist
            if not project_path.exists():
                project_path.mkdir(parents=True, exist_ok=True)
                result["created"] = True
                self.logger.info(f"Created project directory: {project_path}")
            else:
                result["created"] = False
                self.logger.info(f"Using existing project directory: {project_path}")

            result["success"] = True
            return result

        except PermissionError as e:
            result["error"] = (
                f"Permission denied creating directory {project_path}: {str(e)}"
            )
            self.logger.error(result["error"])
            return result
        except OSError as e:
            result["error"] = f"OS error creating directory {project_path}: {str(e)}"
            self.logger.error(result["error"])
            return result
        except Exception as e:
            result["error"] = (
                f"Unexpected error creating directory {project_path}: {str(e)}"
            )
            self.logger.error(result["error"])
            return result

    def validate_project_path(self, project_path: Path) -> Dict[str, Any]:
        """
        Validate project path for safety and filesystem compatibility.

        Args:
            project_path: Path to validate

        Returns:
            Dictionary containing validation results
        """
        result = {"valid": True, "error": None}

        try:
            # Check for reserved names on Windows (cross-platform safety)
            reserved_names = {
                "con",
                "prn",
                "aux",
                "nul",
                "com1",
                "com2",
                "com3",
                "com4",
                "com5",
                "com6",
                "com7",
                "com8",
                "com9",
                "lpt1",
                "lpt2",
                "lpt3",
                "lpt4",
                "lpt5",
                "lpt6",
                "lpt7",
                "lpt8",
                "lpt9",
            }

            if project_path.name.lower() in reserved_names:
                result["valid"] = False
                result["error"] = (
                    f"Project name '{project_path.name}' is a reserved name"
                )
                return result

            # Check for invalid characters
            invalid_chars = set('<>:"|?*')
            if any(char in str(project_path.name) for char in invalid_chars):
                result["valid"] = False
                result["error"] = (
                    f"Project name contains invalid characters: {invalid_chars & set(project_path.name)}"
                )
                return result

            # Check path length (filesystem limits)
            if len(str(project_path)) > 260:  # Windows MAX_PATH limitation
                result["valid"] = False
                result["error"] = (
                    f"Project path too long ({len(str(project_path))} characters, max 260)"
                )
                return result

            # Check for empty or whitespace-only name
            if not project_path.name.strip():
                result["valid"] = False
                result["error"] = "Project name cannot be empty or whitespace only"
                return result

            return result

        except Exception as e:
            result["valid"] = False
            result["error"] = f"Error validating project path: {str(e)}"
            return result

    def safe_cleanup_directory(self, project_path: Path) -> Dict[str, Any]:
        """
        Safely cleanup project directory on failure.

        Args:
            project_path: Path to the directory to cleanup

        Returns:
            Dictionary containing cleanup results
        """
        result = {"success": False, "error": None}

        try:
            if not project_path.exists():
                result["success"] = True
                return result

            if not project_path.is_dir():
                project_path.unlink()
                self.logger.info(f"Removed file: {project_path}")
            else:
                shutil.rmtree(project_path)
                self.logger.info(f"Cleaned up directory: {project_path}")

            result["success"] = True
            return result

        except PermissionError as e:
            result["error"] = f"Permission denied cleaning up {project_path}: {str(e)}"
            self.logger.error(result["error"])
            return result
        except OSError as e:
            result["error"] = f"OS error cleaning up {project_path}: {str(e)}"
            self.logger.error(result["error"])
            return result
        except Exception as e:
            result["error"] = f"Unexpected error cleaning up {project_path}: {str(e)}"
            self.logger.error(result["error"])
            return result

    def create_backup_directory(self, original_path: Path) -> Dict[str, Any]:
        """
        Create backup of existing directory before overwriting.

        Args:
            original_path: Path to the original directory to backup

        Returns:
            Dictionary containing backup results and backup path
        """
        result = {"success": False, "backup_path": None, "error": None}

        try:
            if not original_path.exists():
                result["error"] = f"Original directory {original_path} does not exist"
                return result

            # Generate backup name with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = (
                original_path.parent / f"{original_path.name}.backup.{timestamp}"
            )

            # Ensure backup path doesn't exist
            counter = 1
            original_backup_path = backup_path
            while backup_path.exists():
                backup_path = Path(f"{original_backup_path}_{counter}")
                counter += 1

            # Copy directory
            shutil.copytree(original_path, backup_path)
            result["backup_path"] = backup_path
            result["success"] = True

            self.logger.info(f"Created backup: {original_path} -> {backup_path}")
            return result

        except PermissionError as e:
            result["error"] = f"Permission denied creating backup: {str(e)}"
            self.logger.error(result["error"])
            return result
        except OSError as e:
            result["error"] = f"OS error creating backup: {str(e)}"
            self.logger.error(result["error"])
            return result
        except Exception as e:
            result["error"] = f"Unexpected error creating backup: {str(e)}"
            self.logger.error(result["error"])
            return result

    def get_safe_project_name(self, user_input: str) -> str:
        """
        Generate a safe, filesystem-compatible project name from user input.

        Args:
            user_input: Raw user input for project name

        Returns:
            Safe project name suitable for filesystem use
        """
        if not user_input or not user_input.strip():
            return "untitled_project"

        # Convert to lowercase and replace spaces and dashes with underscores
        safe_name = user_input.lower().strip()
        safe_name = re.sub(r"[\s\-]+", "_", safe_name)

        # Remove invalid characters, keeping only alphanumeric and underscores
        safe_name = re.sub(r"[^a-z0-9_]", "", safe_name)

        # Ensure it doesn't start with a number
        if safe_name and safe_name[0].isdigit():
            # Move the numbers to the end
            match = re.match(r"^(\d+)(.*)$", safe_name)
            if match:
                numbers, rest = match.groups()
                safe_name = f"project_{numbers}" if not rest else f"{rest}_{numbers}"
            else:
                safe_name = f"project_{safe_name}"  # Ensure minimum length
        if not safe_name:
            return "untitled_project"

        return safe_name

    def categorize_cli_error(
        self, error_message: str, returncode: int
    ) -> Dict[str, Any]:
        """
        Categorize CrewAI CLI errors for appropriate handling and recovery.

        Args:
            error_message: Error message from CLI execution
            returncode: Process return code

        Returns:
            Dictionary containing error category, retryability, and user message
        """
        error_message_lower = error_message.lower()

        # Network-related errors (retryable)
        if any(
            keyword in error_message_lower
            for keyword in [
                "connection",
                "network",
                "timeout",
                "hostname",
                "ssl",
                "certificate",
                "resolve",
            ]
        ):
            return {
                "type": "network",
                "retryable": True,
                "user_message": "Network connectivity issue detected. Check your internet connection and try again.",
                "retry_delay": 2.0,
            }

        # Permission errors (not retryable)
        if any(
            keyword in error_message_lower
            for keyword in [
                "permission denied",
                "access denied",
                "insufficient privileges",
                "operation not permitted",
                "you don't have permission",
            ]
        ):
            return {
                "type": "permission",
                "retryable": False,
                "user_message": "Permission denied. Check directory permissions or try a different location.",
                "retry_delay": 0,
            }

        # Disk space errors (not retryable)
        if any(
            keyword in error_message_lower
            for keyword in [
                "no space left",
                "disk full",
                "not enough free space",
                "unable to write",
            ]
        ):
            return {
                "type": "disk_space",
                "retryable": False,
                "user_message": "Insufficient disk space. Free up space and try again.",
                "retry_delay": 0,
            }

        # Installation/corruption errors (not retryable)
        if returncode == 127 or any(
            keyword in error_message_lower
            for keyword in [
                "command not found",
                "importerror",
                "modulenotfounderror",
                "corrupted",
                "not found",
            ]
        ):
            return {
                "type": "installation",
                "retryable": False,
                "user_message": "CrewAI CLI not found or corrupted. Please reinstall with 'pip install crewai'.",
                "retry_delay": 0,
            }

        # Invalid command errors (not retryable)
        if returncode == 2 or any(
            keyword in error_message_lower
            for keyword in [
                "invalid command",
                "unknown option",
                "unrecognized arguments",
                "usage:",
                "invalid project name",
            ]
        ):
            return {
                "type": "invalid_command",
                "retryable": False,
                "user_message": "Invalid command or arguments. Check the command syntax.",
                "retry_delay": 0,
            }

        # Transient errors (retryable)
        if any(
            keyword in error_message_lower
            for keyword in [
                "temporary failure",
                "temporarily unavailable",
                "rate limit",
                "try again",
            ]
        ):
            return {
                "type": "transient",
                "retryable": True,
                "user_message": "Temporary service issue. Please try again in a moment.",
                "retry_delay": 5.0,
            }

        # Default: unknown error (potentially retryable)
        return {
            "type": "unknown",
            "retryable": True,
            "user_message": f"Unexpected error occurred: {error_message}",
            "retry_delay": 1.0,
        }

    def create_crew_with_retry(
        self,
        project_name: str,
        target_directory: Path,
        min_version: Optional[str] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute crew creation with retry logic and enhanced error handling.

        Args:
            project_name: Name of the crew project to create
            target_directory: Directory where the project should be created
            min_version: Optional minimum CrewAI version requirement
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary containing execution results and attempt information

        Raises:
            CrewAIError: If all retry attempts fail or non-retryable error occurs
        """
        last_error = None
        attempts = 0

        for attempt in range(max_retries):
            attempts = attempt + 1
            try:
                self.logger.info(
                    f"Attempt {attempts}/{max_retries} to create CrewAI project"
                )

                # Use the existing create_crew method
                result = self.create_crew(project_name, target_directory, min_version)
                result["attempts"] = attempts
                return result

            except CrewAIError as e:
                last_error = e
                error_msg = str(e)

                # Categorize the error
                error_category = self.categorize_cli_error(
                    error_msg, getattr(e, "returncode", 1)
                )

                # Log the error
                self.logger.warning(
                    f"Attempt {attempts} failed: {error_category['user_message']}"
                )

                # If error is not retryable, fail immediately
                if not error_category["retryable"]:
                    self.logger.error(
                        f"Non-retryable error: {error_category['user_message']}"
                    )
                    raise CrewAIError(f"{error_category['user_message']}: {error_msg}")

                # If this was the last attempt, fail
                if attempt == max_retries - 1:
                    break

                # Wait before retry with exponential backoff
                retry_delay = self.calculate_backoff_delay(attempt)
                self.logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        # All retries failed
        error_context = self.collect_error_context(
            command=["crewai", "create", "crew", project_name],
            cwd=str(target_directory),
            error_msg=str(last_error),
            returncode=getattr(last_error, "returncode", 1),
        )

        error_report = self.create_error_report(error_context)
        self.logger.error(f"Max retries exceeded after {attempts} attempts")

        raise CrewAIError(
            f"Max retries exceeded ({attempts} attempts). Last error: {last_error}\n\n{error_report}"
        )

    def calculate_backoff_delay(
        self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
    ) -> float:
        """
        Calculate exponential backoff delay for retry attempts.

        Args:
            attempt: Current attempt number (0-based)
            base_delay: Base delay in seconds
            max_delay: Maximum delay cap in seconds

        Returns:
            Delay in seconds for the next retry
        """
        delay = base_delay * (2**attempt)
        return min(delay, max_delay)

    def create_error_report(self, error_context: Dict[str, Any]) -> str:
        """
        Create a detailed error report for troubleshooting.

        Args:
            error_context: Context information about the error

        Returns:
            Formatted error report string
        """
        report_lines = [
            "=== CrewAI CLI Error Report ===",
            f"Command: {' '.join(error_context.get('command', []))}",
            f"Working Directory: {error_context.get('cwd', 'N/A')}",
            f"Return Code: {error_context.get('returncode', 'N/A')}",
            f"Timeout: {error_context.get('timeout', 60)} seconds",
            f"Timestamp: {error_context.get('timestamp', 'N/A')}",
            "",
            "Error Output:",
            error_context.get("error_msg", "")
            or error_context.get("stderr", "No error message available"),
            "",
            "System Information:",
        ]

        system_info = error_context.get("system_info", {})
        for key, value in system_info.items():
            report_lines.append(f"  {key}: {value}")

        return "\n".join(report_lines)

    def collect_error_context(
        self, command: List[str], cwd: str, error_msg: str, returncode: int
    ) -> Dict[str, Any]:
        """
        Collect comprehensive error context for debugging.

        Args:
            command: Command that failed
            cwd: Working directory
            error_msg: Error message
            returncode: Process return code

        Returns:
            Dictionary containing error context
        """
        return {
            "command": command,
            "cwd": cwd,
            "error_msg": error_msg,
            "returncode": returncode,
            "timestamp": datetime.datetime.now().isoformat(),
            "timeout": 60,
            "system_info": {
                "platform": platform.system(),
                "platform_version": platform.release(),
                "python_version": platform.python_version(),
                "architecture": platform.machine(),
            },
        }

    def suggest_recovery_actions(
        self, error_category: Dict[str, Any], target_path: Path
    ) -> List[str]:
        """
        Suggest actionable recovery steps based on error type.

        Args:
            error_category: Categorized error information
            target_path: Target directory path

        Returns:
            List of suggested recovery actions
        """
        suggestions = []
        error_type = error_category["type"]

        if error_type == "permission":
            suggestions.extend(
                [
                    f"Check permissions: chmod 755 {target_path.parent}",
                    "Try using a different target directory (e.g., ~/projects)",
                    "Run with elevated privileges if necessary (not recommended)",
                    "Ensure you have write access to the target location",
                ]
            )

        elif error_type == "installation":
            suggestions.extend(
                [
                    "Reinstall CrewAI: pip install --upgrade crewai",
                    "Create a new virtual environment: python -m venv crewai_env",
                    "Activate virtual environment and install: source crewai_env/bin/activate && pip install crewai",
                    "Check Python PATH: python -m site --user-site",
                ]
            )

        elif error_type == "disk_space":
            suggestions.extend(
                [
                    f"Free up disk space in {target_path.parent}",
                    "Use a different drive or directory with more space",
                    "Clean up temporary files: rm -rf /tmp/*",
                    "Check disk usage: df -h",
                ]
            )

        elif error_type == "network":
            suggestions.extend(
                [
                    "Check internet connectivity",
                    "Try again in a few minutes",
                    "Check firewall/proxy settings",
                    "Use a different network if available",
                ]
            )

        else:  # transient, unknown, invalid_command
            suggestions.extend(
                [
                    "Try the command again",
                    "Check the CrewAI documentation for correct usage",
                    "Update CrewAI to the latest version: pip install --upgrade crewai",
                ]
            )

        return suggestions

    def suggest_fallback_directory(self, original_path: Path) -> Dict[str, Any]:
        """
        Suggest fallback directory when original is inaccessible.

        Args:
            original_path: Original target directory path

        Returns:
            Dictionary containing fallback suggestion
        """
        fallback_candidates = [
            Path.home() / "crewai_projects",
            Path.home() / "projects",
            Path.home() / "Desktop" / "crewai_projects",
            Path("/tmp") / "crewai_projects",
        ]

        for candidate in fallback_candidates:
            try:
                # Check if we can create the directory
                candidate.mkdir(parents=True, exist_ok=True)
                if candidate.exists() and os.access(candidate, os.W_OK):
                    return {
                        "success": True,
                        "fallback_path": candidate,
                        "reason": f"Original path {original_path} is inaccessible",
                    }
            except (OSError, PermissionError):
                continue

        return {
            "success": False,
            "fallback_path": None,
            "reason": "No suitable fallback directory found",
        }

    def perform_cli_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of CrewAI CLI installation.

        Returns:
            Dictionary containing health status and issues
        """
        health_result = {"healthy": True, "version": None, "issues": []}

        try:
            # Check if CLI is available and get version
            version = self.get_version()
            if version:
                health_result["version"] = version
            else:
                health_result["healthy"] = False
                health_result["issues"].append("CrewAI CLI not found or not responding")

            # Test basic CLI functionality
            try:
                result = subprocess.run(
                    ["crewai", "--help"], capture_output=True, text=True, timeout=10
                )

                if result.returncode != 0:
                    health_result["healthy"] = False
                    health_result["issues"].append(
                        f"CLI help command failed: {result.stderr}"
                    )

            except subprocess.TimeoutExpired:
                health_result["healthy"] = False
                health_result["issues"].append("CLI commands are timing out")
            except Exception as e:
                health_result["healthy"] = False
                health_result["issues"].append(f"CLI execution error: {str(e)}")

        except Exception as e:
            health_result["healthy"] = False
            health_result["issues"].append(f"Health check failed: {str(e)}")

        return health_result

    def create_crew_with_enhancement(
        self,
        project_name: str,
        target_directory: Path,
        enhancement_context: Optional[Dict[str, Any]] = None,
        template_name: str = "default",
        min_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a CrewAI project and enhance it with intelligent configurations.

        This method combines the native CrewAI scaffolding with the enhancement
        engine to provide domain-specific customizations.

        Args:
            project_name: Name of the crew project to create
            target_directory: Directory where the project should be created
            enhancement_context: Context data for template rendering
            template_name: Name of template to use for enhancement
            min_version: Optional minimum CrewAI version requirement

        Returns:
            Dictionary containing execution results and enhancement information

        Raises:
            CrewAIError: If CrewAI CLI execution fails or is not available
            EnhancementError: If enhancement fails
        """
        if EnhancementEngine is None:
            # Fallback to basic creation if enhancement engine not available
            self.logger.warning(
                "Enhancement engine not available, using basic scaffolding"
            )
            return self.create_crew(project_name, target_directory, min_version)

        # Step 1: Create basic project structure
        scaffold_result = self.create_crew(project_name, target_directory, min_version)

        if not scaffold_result.get("success", False):
            return scaffold_result

        # Step 2: Enhancement phase
        if not enhancement_context:
            # No enhancement requested, return scaffolding result
            scaffold_result["enhancement"] = {
                "attempted": False,
                "reason": "No enhancement context provided",
            }
            return scaffold_result

        try:
            # Initialize enhancement engine
            enhancement_engine = EnhancementEngine(logger=self.logger)
            project_path = scaffold_result["project_path"]

            self.logger.info(f"Enhancing project with template '{template_name}'")

            # Enhance agents configuration
            agents_result = enhancement_engine.enhance_agents_config(
                project_path, enhancement_context, template_name
            )

            # Enhance tasks configuration
            tasks_result = enhancement_engine.enhance_tasks_config(
                project_path, enhancement_context, template_name
            )

            enhancement_summary = {
                "attempted": True,
                "agents_enhanced": agents_result.get("success", False),
                "tasks_enhanced": tasks_result.get("success", False),
                "template_used": template_name,
                "errors": [],
            }

            # Collect any enhancement errors
            if not agents_result.get("success", False):
                enhancement_summary["errors"].append(
                    f"Agents enhancement failed: {agents_result.get('error', 'Unknown error')}"
                )

            if not tasks_result.get("success", False):
                enhancement_summary["errors"].append(
                    f"Tasks enhancement failed: {tasks_result.get('error', 'Unknown error')}"
                )

            # Overall enhancement success
            enhancement_summary["success"] = agents_result.get(
                "success", False
            ) and tasks_result.get("success", False)

            if enhancement_summary["success"]:
                self.logger.info(
                    f"Successfully enhanced project '{project_name}' with template '{template_name}'"
                )
            else:
                self.logger.warning(
                    f"Partial enhancement of project '{project_name}': {enhancement_summary['errors']}"
                )

            # Add enhancement details to the result
            scaffold_result["enhancement"] = enhancement_summary
            scaffold_result["agents_backup"] = agents_result.get("backup_file")
            scaffold_result["tasks_backup"] = tasks_result.get("backup_file")

            return scaffold_result

        except EnhancementError as e:
            self.logger.error(
                f"Enhancement failed for project '{project_name}': {str(e)}"
            )
            scaffold_result["enhancement"] = {
                "attempted": True,
                "success": False,
                "error": str(e),
                "agents_enhanced": False,
                "tasks_enhanced": False,
            }
            return scaffold_result

        except Exception as e:
            self.logger.error(f"Unexpected error during enhancement: {str(e)}")
            scaffold_result["enhancement"] = {
                "attempted": True,
                "success": False,
                "error": f"Unexpected enhancement error: {str(e)}",
                "agents_enhanced": False,
                "tasks_enhanced": False,
            }
            return scaffold_result

    def get_available_enhancement_templates(
        self, category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get available enhancement templates by category.

        Args:
            category: Optional category filter ('agents' or 'tasks')

        Returns:
            Dictionary mapping categories to lists of available templates or error information
        """
        if EnhancementEngine is None:
            return {"error": "Enhancement engine not available"}

        try:
            enhancement_engine = EnhancementEngine(logger=self.logger)

            if category:
                return {category: enhancement_engine.get_available_templates(category)}
            else:
                return {
                    "agents": enhancement_engine.get_available_templates("agents"),
                    "tasks": enhancement_engine.get_available_templates("tasks"),
                }

        except Exception as e:
            self.logger.error(f"Failed to get available templates: {str(e)}")
            return {"error": str(e)}

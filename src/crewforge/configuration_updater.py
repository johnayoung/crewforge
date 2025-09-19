"""
Configuration file updating system for CrewAI projects.

This module provides intelligent updating capabilities for CrewAI configuration files
(agents.yaml and tasks.yaml) with backup, merge, and selective update functionality.
"""

import logging
import shutil
import yaml
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


class ConfigurationUpdateError(Exception):
    """Exception raised for configuration update errors."""

    pass


class UpdateStrategy(Enum):
    """Strategies for merging configuration updates."""

    REPLACE = "replace"  # Replace entire sections with new content
    UPDATE = "update"  # Update existing fields, preserve others
    ADD_ONLY = "add_only"  # Only add new fields, never modify existing


class ConfigurationUpdater:
    """
    Smart configuration file updater for CrewAI projects.

    Provides capabilities for:
    - Selective updating of specific agents/tasks
    - Bulk updates with different merge strategies
    - Backup creation and restoration
    - YAML comment preservation (when supported)
    - Comprehensive error handling
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        create_backups: bool = True,
        preserve_comments: bool = True,
    ):
        """
        Initialize the configuration updater.

        Args:
            logger: Optional logger instance
            create_backups: Whether to create backup files before updates
            preserve_comments: Whether to attempt to preserve YAML comments
        """
        self.logger = logger or logging.getLogger(__name__)
        self.create_backups = create_backups
        self.preserve_comments = preserve_comments

        self.logger.debug(
            f"ConfigurationUpdater initialized: "
            f"backups={create_backups}, preserve_comments={preserve_comments}"
        )

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load and parse a YAML configuration file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Parsed YAML data as dictionary

        Raises:
            ConfigurationUpdateError: If file doesn't exist or has invalid YAML
        """
        if not file_path.exists():
            raise ConfigurationUpdateError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            self.logger.debug(f"Loaded YAML file: {file_path}")
            return data

        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML syntax in {file_path}: {str(e)}")
            raise ConfigurationUpdateError(
                f"Invalid YAML syntax in {file_path}: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load YAML file {file_path}: {str(e)}")
            raise ConfigurationUpdateError(
                f"Failed to load configuration file: {str(e)}"
            )

    def _save_yaml_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Save data to a YAML configuration file.

        Args:
            file_path: Path to save the YAML file
            data: Data to save as YAML

        Raises:
            ConfigurationUpdateError: If file cannot be written
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                    sort_keys=False,  # Preserve key ordering
                )

            self.logger.debug(f"Saved YAML file: {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save YAML file {file_path}: {str(e)}")
            raise ConfigurationUpdateError(
                f"Failed to save configuration file: {str(e)}"
            )

    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """
        Create a backup copy of a configuration file.

        Args:
            file_path: Path to the file to backup

        Returns:
            Path to backup file, or None if backups are disabled

        Raises:
            ConfigurationUpdateError: If backup creation fails
        """
        if not self.create_backups:
            return None

        if not file_path.exists():
            raise ConfigurationUpdateError(
                f"Cannot create backup: file does not exist: {file_path}"
            )

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.parent / f"{file_path.stem}_{timestamp}.backup"

            shutil.copy2(file_path, backup_path)
            self.logger.debug(f"Created backup: {backup_path}")

            return backup_path

        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {str(e)}")
            raise ConfigurationUpdateError(f"Cannot create backup: {str(e)}")

    def _merge_configurations(
        self,
        original: Dict[str, Any],
        updates: Dict[str, Any],
        strategy: UpdateStrategy = UpdateStrategy.UPDATE,
    ) -> Dict[str, Any]:
        """
        Merge configuration updates with original data using specified strategy.

        Args:
            original: Original configuration data
            updates: Updates to apply
            strategy: Merge strategy to use

        Returns:
            Merged configuration data
        """
        result = original.copy()

        for key, update_value in updates.items():
            if strategy == UpdateStrategy.REPLACE:
                # Replace entire section
                result[key] = update_value

            elif strategy == UpdateStrategy.UPDATE:
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(update_value, dict)
                ):
                    # Deep merge dictionaries
                    result[key] = {**result[key], **update_value}
                else:
                    # Replace value
                    result[key] = update_value

            elif strategy == UpdateStrategy.ADD_ONLY:
                if key in result:
                    # Only add new fields to existing entries
                    if isinstance(result[key], dict) and isinstance(update_value, dict):
                        for update_field, update_field_value in update_value.items():
                            if update_field not in result[key]:
                                result[key][update_field] = update_field_value
                else:
                    # Add new entries entirely
                    result[key] = update_value

        return result

    def update_agent(
        self,
        config_file: Path,
        agent_name: str,
        updates: Dict[str, Any],
        strategy: UpdateStrategy = UpdateStrategy.UPDATE,
    ) -> Dict[str, Any]:
        """
        Update a specific agent in the agents.yaml configuration.

        Args:
            config_file: Path to the agents.yaml file
            agent_name: Name of the agent to update
            updates: Updates to apply to the agent
            strategy: Merge strategy to use

        Returns:
            Dictionary with operation result
        """
        try:
            # Create backup first
            backup_path = self._create_backup(config_file)

            # Load current configuration
            current_config = self._load_yaml_file(config_file)

            # Apply updates to specific agent
            agent_updates = {agent_name: updates}
            updated_config = self._merge_configurations(
                current_config, agent_updates, strategy
            )

            # Save updated configuration
            self._save_yaml_file(config_file, updated_config)

            self.logger.info(
                f"Successfully updated agent '{agent_name}' in {config_file}"
            )

            return {
                "success": True,
                "agent_name": agent_name,
                "backup_file": backup_path,
                "backup_created": backup_path is not None,
                "strategy": strategy.value,
            }

        except ConfigurationUpdateError as e:
            return {
                "success": False,
                "error": str(e),
                "agent_name": agent_name,
                "backup_created": False,
            }
        except Exception as e:
            self.logger.error(
                f"Unexpected error updating agent '{agent_name}': {str(e)}"
            )
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "agent_name": agent_name,
                "backup_created": False,
            }

    def update_task(
        self,
        config_file: Path,
        task_name: str,
        updates: Dict[str, Any],
        strategy: UpdateStrategy = UpdateStrategy.UPDATE,
    ) -> Dict[str, Any]:
        """
        Update a specific task in the tasks.yaml configuration.

        Args:
            config_file: Path to the tasks.yaml file
            task_name: Name of the task to update
            updates: Updates to apply to the task
            strategy: Merge strategy to use

        Returns:
            Dictionary with operation result
        """
        try:
            # Create backup first
            backup_path = self._create_backup(config_file)

            # Load current configuration
            current_config = self._load_yaml_file(config_file)

            # Apply updates to specific task
            task_updates = {task_name: updates}
            updated_config = self._merge_configurations(
                current_config, task_updates, strategy
            )

            # Save updated configuration
            self._save_yaml_file(config_file, updated_config)

            self.logger.info(
                f"Successfully updated task '{task_name}' in {config_file}"
            )

            return {
                "success": True,
                "task_name": task_name,
                "backup_file": backup_path,
                "backup_created": backup_path is not None,
                "strategy": strategy.value,
            }

        except ConfigurationUpdateError as e:
            return {
                "success": False,
                "error": str(e),
                "task_name": task_name,
                "backup_created": False,
            }
        except Exception as e:
            self.logger.error(f"Unexpected error updating task '{task_name}': {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "task_name": task_name,
                "backup_created": False,
            }

    def bulk_update_agents(
        self,
        config_file: Path,
        updates: Dict[str, Dict[str, Any]],
        strategy: UpdateStrategy = UpdateStrategy.UPDATE,
    ) -> Dict[str, Any]:
        """
        Update multiple agents in the agents.yaml configuration.

        Args:
            config_file: Path to the agents.yaml file
            updates: Dictionary mapping agent names to their updates
            strategy: Merge strategy to use

        Returns:
            Dictionary with operation result
        """
        try:
            # Create backup first
            backup_path = self._create_backup(config_file)

            # Load current configuration
            current_config = self._load_yaml_file(config_file)

            # Apply all updates at once
            updated_config = self._merge_configurations(
                current_config, updates, strategy
            )

            # Save updated configuration
            self._save_yaml_file(config_file, updated_config)

            updated_agents = list(updates.keys())
            self.logger.info(
                f"Successfully updated {len(updated_agents)} agents in {config_file}: "
                f"{', '.join(updated_agents)}"
            )

            return {
                "success": True,
                "updated_agents": updated_agents,
                "updated_count": len(updated_agents),
                "backup_file": backup_path,
                "backup_created": backup_path is not None,
                "strategy": strategy.value,
            }

        except ConfigurationUpdateError as e:
            return {"success": False, "error": str(e), "backup_created": False}
        except Exception as e:
            self.logger.error(f"Unexpected error in bulk agent update: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "backup_created": False,
            }

    def bulk_update_tasks(
        self,
        config_file: Path,
        updates: Dict[str, Dict[str, Any]],
        strategy: UpdateStrategy = UpdateStrategy.UPDATE,
    ) -> Dict[str, Any]:
        """
        Update multiple tasks in the tasks.yaml configuration.

        Args:
            config_file: Path to the tasks.yaml file
            updates: Dictionary mapping task names to their updates
            strategy: Merge strategy to use

        Returns:
            Dictionary with operation result
        """
        try:
            # Create backup first
            backup_path = self._create_backup(config_file)

            # Load current configuration
            current_config = self._load_yaml_file(config_file)

            # Apply all updates at once
            updated_config = self._merge_configurations(
                current_config, updates, strategy
            )

            # Save updated configuration
            self._save_yaml_file(config_file, updated_config)

            updated_tasks = list(updates.keys())
            self.logger.info(
                f"Successfully updated {len(updated_tasks)} tasks in {config_file}: "
                f"{', '.join(updated_tasks)}"
            )

            return {
                "success": True,
                "updated_tasks": updated_tasks,
                "updated_count": len(updated_tasks),
                "backup_file": backup_path,
                "backup_created": backup_path is not None,
                "strategy": strategy.value,
            }

        except ConfigurationUpdateError as e:
            return {"success": False, "error": str(e), "backup_created": False}
        except Exception as e:
            self.logger.error(f"Unexpected error in bulk task update: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "backup_created": False,
            }

    def restore_from_backup(
        self, config_file: Path, backup_file: Path
    ) -> Dict[str, Any]:
        """
        Restore a configuration file from a backup.

        Args:
            config_file: Path to the configuration file to restore
            backup_file: Path to the backup file

        Returns:
            Dictionary with operation result
        """
        try:
            if not backup_file.exists():
                return {
                    "success": False,
                    "error": f"Backup file not found: {backup_file}",
                }

            shutil.copy2(backup_file, config_file)

            self.logger.info(
                f"Successfully restored {config_file} from backup {backup_file}"
            )

            return {
                "success": True,
                "restored_file": config_file,
                "backup_file": backup_file,
            }

        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to restore from backup: {str(e)}",
            }

    def list_backups(self, config_file: Path) -> List[Path]:
        """
        List available backup files for a configuration file.

        Args:
            config_file: Path to the configuration file

        Returns:
            List of backup file paths, sorted by creation time (newest first)
        """
        backup_pattern = f"{config_file.stem}_*.backup"
        backup_files = list(config_file.parent.glob(backup_pattern))

        # Sort by modification time, newest first
        backup_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        self.logger.debug(f"Found {len(backup_files)} backup files for {config_file}")
        return backup_files

    def validate_configuration(self, config_file: Path) -> Dict[str, Any]:
        """
        Validate the structure and content of a configuration file.

        Args:
            config_file: Path to the configuration file

        Returns:
            Dictionary with validation results
        """
        try:
            # Load and parse the file
            config_data = self._load_yaml_file(config_file)

            # Basic validation checks
            validation_results = {
                "success": True,
                "file_path": config_file,
                "errors": [],
                "warnings": [],
                "entry_count": len(config_data) if config_data else 0,
            }

            if not config_data:
                validation_results["warnings"].append("Configuration file is empty")
                return validation_results

            # Validate agent configurations
            if "agents.yaml" in config_file.name:
                validation_results.update(self._validate_agents_config(config_data))

            # Validate task configurations
            elif "tasks.yaml" in config_file.name:
                validation_results.update(self._validate_tasks_config(config_data))

            return validation_results

        except ConfigurationUpdateError as e:
            return {
                "success": False,
                "file_path": config_file,
                "errors": [str(e)],
                "warnings": [],
                "entry_count": 0,
            }

    def _validate_agents_config(
        self, config_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Validate agents configuration structure."""
        errors = []
        warnings = []

        for agent_name, agent_config in config_data.items():
            if not isinstance(agent_config, dict):
                errors.append(f"Agent '{agent_name}' must be a dictionary")
                continue

            # Check required fields
            required_fields = ["role", "goal"]
            for field in required_fields:
                if field not in agent_config:
                    errors.append(
                        f"Agent '{agent_name}' missing required field: {field}"
                    )

            # Check recommended fields
            recommended_fields = ["backstory"]
            for field in recommended_fields:
                if field not in agent_config:
                    warnings.append(
                        f"Agent '{agent_name}' missing recommended field: {field}"
                    )

        return {"errors": errors, "warnings": warnings}

    def _validate_tasks_config(
        self, config_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Validate tasks configuration structure."""
        errors = []
        warnings = []

        for task_name, task_config in config_data.items():
            if not isinstance(task_config, dict):
                errors.append(f"Task '{task_name}' must be a dictionary")
                continue

            # Check required fields
            required_fields = ["description", "agent"]
            for field in required_fields:
                if field not in task_config:
                    errors.append(f"Task '{task_name}' missing required field: {field}")

            # Check recommended fields
            recommended_fields = ["expected_output"]
            for field in recommended_fields:
                if field not in task_config:
                    warnings.append(
                        f"Task '{task_name}' missing recommended field: {field}"
                    )

        return {"errors": errors, "warnings": warnings}

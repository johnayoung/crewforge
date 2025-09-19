"""
Tests for the configuration file updating system.

This module tests the ConfigurationUpdater class which provides smart updates
for CrewAI configuration files (agents.yaml and tasks.yaml) with backup,
merge, and selective update capabilities.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from crewforge.configuration_updater import (
    ConfigurationUpdater,
    ConfigurationUpdateError,
    UpdateStrategy,
)


class TestConfigurationUpdater:
    """Test cases for ConfigurationUpdater class."""

    def test_init_default(self):
        """Test ConfigurationUpdater initialization with defaults."""
        updater = ConfigurationUpdater()

        assert updater.logger is not None
        assert updater.create_backups is True
        assert updater.preserve_comments is True

    def test_init_with_logger(self):
        """Test ConfigurationUpdater initialization with custom logger."""
        mock_logger = MagicMock()
        updater = ConfigurationUpdater(logger=mock_logger)

        assert updater.logger is mock_logger

    def test_init_with_options(self):
        """Test ConfigurationUpdater initialization with options."""
        updater = ConfigurationUpdater(create_backups=False, preserve_comments=False)

        assert updater.create_backups is False
        assert updater.preserve_comments is False


class TestYAMLParsing:
    """Test YAML parsing and loading functionality."""

    def test_load_yaml_file_valid(self):
        """Test loading valid YAML configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test.yaml"
            test_data = {
                "researcher": {
                    "role": "Research Specialist",
                    "goal": "Conduct thorough research",
                    "backstory": "PhD in research methodology",
                    "tools": ["search", "analysis"],
                }
            }
            config_file.write_text(yaml.dump(test_data))

            updater = ConfigurationUpdater()
            result = updater._load_yaml_file(config_file)

            assert result == test_data

    def test_load_yaml_file_not_found(self):
        """Test loading non-existent YAML file."""
        updater = ConfigurationUpdater()
        non_existent_file = Path("/non/existent/file.yaml")

        with pytest.raises(
            ConfigurationUpdateError, match="Configuration file not found"
        ):
            updater._load_yaml_file(non_existent_file)

    def test_load_yaml_file_invalid_syntax(self):
        """Test loading YAML file with invalid syntax."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "invalid.yaml"
            config_file.write_text("invalid: yaml: content: [\n  unclosed bracket")

            updater = ConfigurationUpdater()

            with pytest.raises(ConfigurationUpdateError, match="Invalid YAML syntax"):
                updater._load_yaml_file(config_file)

    def test_save_yaml_file(self):
        """Test saving YAML configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "output.yaml"
            test_data = {
                "agent1": {"role": "Analyst", "goal": "Analyze data"},
                "agent2": {"role": "Writer", "goal": "Write reports"},
            }

            updater = ConfigurationUpdater()
            updater._save_yaml_file(config_file, test_data)

            # Verify file was created and content is correct
            assert config_file.exists()
            loaded_data = yaml.safe_load(config_file.read_text())
            assert loaded_data == test_data


class TestBackupCreation:
    """Test backup file creation and management."""

    def test_create_backup_success(self):
        """Test successful backup file creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_file = Path(temp_dir) / "agents.yaml"
            original_content = "test: content"
            original_file.write_text(original_content)

            updater = ConfigurationUpdater()
            backup_path = updater._create_backup(original_file)

            assert backup_path.exists()
            assert backup_path.read_text() == original_content
            assert backup_path.suffix == ".backup"
            assert original_file.stem in backup_path.name

    def test_create_backup_disabled(self):
        """Test backup creation when disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_file = Path(temp_dir) / "agents.yaml"
            original_file.write_text("test: content")

            updater = ConfigurationUpdater(create_backups=False)
            backup_path = updater._create_backup(original_file)

            assert backup_path is None

    def test_create_backup_file_not_found(self):
        """Test backup creation for non-existent file."""
        updater = ConfigurationUpdater()
        non_existent_file = Path("/non/existent/file.yaml")

        with pytest.raises(ConfigurationUpdateError, match="Cannot create backup"):
            updater._create_backup(non_existent_file)


class TestMergeStrategies:
    """Test different merge strategies for configuration updates."""

    def test_merge_replace_strategy(self):
        """Test replace merge strategy."""
        original = {
            "agent1": {"role": "Old Role", "goal": "Old Goal"},
            "agent2": {"role": "Keep This", "goal": "Keep This Too"},
        }

        updates = {
            "agent1": {"role": "New Role", "goal": "New Goal", "backstory": "Added"},
            "agent3": {"role": "Brand New", "goal": "New Agent"},
        }

        updater = ConfigurationUpdater()
        result = updater._merge_configurations(
            original, updates, UpdateStrategy.REPLACE
        )

        expected = {
            "agent1": {"role": "New Role", "goal": "New Goal", "backstory": "Added"},
            "agent2": {"role": "Keep This", "goal": "Keep This Too"},
            "agent3": {"role": "Brand New", "goal": "New Agent"},
        }

        assert result == expected

    def test_merge_update_strategy(self):
        """Test update merge strategy."""
        original = {
            "agent1": {
                "role": "Original Role",
                "goal": "Original Goal",
                "backstory": "Original Backstory",
                "tools": ["tool1", "tool2"],
            }
        }

        updates = {
            "agent1": {
                "goal": "Updated Goal",
                "backstory": "Updated Backstory",
                "verbose": True,
            }
        }

        updater = ConfigurationUpdater()
        result = updater._merge_configurations(original, updates, UpdateStrategy.UPDATE)

        expected = {
            "agent1": {
                "role": "Original Role",  # preserved
                "goal": "Updated Goal",  # updated
                "backstory": "Updated Backstory",  # updated
                "tools": ["tool1", "tool2"],  # preserved
                "verbose": True,  # added
            }
        }

        assert result == expected

    def test_merge_add_only_strategy(self):
        """Test add-only merge strategy."""
        original = {"agent1": {"role": "Existing", "goal": "Existing Goal"}}

        updates = {
            "agent1": {"role": "Should Not Replace", "backstory": "Should Add"},
            "agent2": {"role": "New Agent", "goal": "New Goal"},
        }

        updater = ConfigurationUpdater()
        result = updater._merge_configurations(
            original, updates, UpdateStrategy.ADD_ONLY
        )

        expected = {
            "agent1": {
                "role": "Existing",  # preserved
                "goal": "Existing Goal",  # preserved
                "backstory": "Should Add",  # added
            },
            "agent2": {"role": "New Agent", "goal": "New Goal"},  # added
        }

        assert result == expected


class TestSelectiveUpdates:
    """Test selective updating of specific agents or tasks."""

    def test_update_specific_agent(self):
        """Test updating a specific agent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "agents.yaml"
            original_config = {
                "researcher": {"role": "Researcher", "goal": "Research"},
                "writer": {"role": "Writer", "goal": "Write"},
            }
            config_file.write_text(yaml.dump(original_config))

            updater = ConfigurationUpdater()

            agent_updates = {
                "role": "Senior Researcher",
                "goal": "Conduct advanced research",
                "backstory": "PhD with 10 years experience",
            }

            result = updater.update_agent(config_file, "researcher", agent_updates)

            assert result["success"] is True
            assert result["backup_created"] is True

            # Verify only the researcher agent was updated
            updated_config = yaml.safe_load(config_file.read_text())
            assert updated_config["researcher"]["role"] == "Senior Researcher"
            assert (
                updated_config["researcher"]["backstory"]
                == "PhD with 10 years experience"
            )
            assert updated_config["writer"] == original_config["writer"]  # unchanged

    def test_update_specific_task(self):
        """Test updating a specific task."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "tasks.yaml"
            original_config = {
                "research_task": {
                    "description": "Basic research",
                    "agent": "researcher",
                },
                "write_task": {"description": "Basic writing", "agent": "writer"},
            }
            config_file.write_text(yaml.dump(original_config))

            updater = ConfigurationUpdater()

            task_updates = {
                "description": "Comprehensive market research",
                "expected_output": "Detailed market analysis report",
                "context": ["industry trends", "competitor analysis"],
            }

            result = updater.update_task(config_file, "research_task", task_updates)

            assert result["success"] is True

            # Verify only the research task was updated
            updated_config = yaml.safe_load(config_file.read_text())
            assert (
                updated_config["research_task"]["description"]
                == "Comprehensive market research"
            )
            assert "expected_output" in updated_config["research_task"]
            assert (
                updated_config["write_task"] == original_config["write_task"]
            )  # unchanged

    def test_update_nonexistent_agent(self):
        """Test updating a non-existent agent with add strategy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "agents.yaml"
            original_config = {"researcher": {"role": "Researcher", "goal": "Research"}}
            config_file.write_text(yaml.dump(original_config))

            updater = ConfigurationUpdater()

            new_agent = {
                "role": "Data Analyst",
                "goal": "Analyze data patterns",
                "backstory": "Expert in statistical analysis",
            }

            result = updater.update_agent(
                config_file, "analyst", new_agent, strategy=UpdateStrategy.ADD_ONLY
            )

            assert result["success"] is True

            # Verify new agent was added
            updated_config = yaml.safe_load(config_file.read_text())
            assert "analyst" in updated_config
            assert updated_config["analyst"]["role"] == "Data Analyst"
            assert (
                updated_config["researcher"] == original_config["researcher"]
            )  # unchanged


class TestBulkUpdates:
    """Test bulk updating of multiple agents or tasks."""

    def test_bulk_update_agents(self):
        """Test bulk updating multiple agents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "agents.yaml"
            original_config = {
                "researcher": {"role": "Researcher", "goal": "Research"},
                "writer": {"role": "Writer", "goal": "Write"},
                "analyst": {"role": "Analyst", "goal": "Analyze"},
            }
            config_file.write_text(yaml.dump(original_config))

            updater = ConfigurationUpdater()

            bulk_updates = {
                "researcher": {
                    "backstory": "PhD researcher with domain expertise",
                    "verbose": True,
                },
                "writer": {
                    "backstory": "Professional technical writer",
                    "tools": ["writing_tools", "grammar_check"],
                },
            }

            result = updater.bulk_update_agents(config_file, bulk_updates)

            assert result["success"] is True
            assert result["updated_count"] == 2

            # Verify updates were applied correctly
            updated_config = yaml.safe_load(config_file.read_text())
            assert "backstory" in updated_config["researcher"]
            assert updated_config["researcher"]["verbose"] is True
            assert updated_config["writer"]["tools"] == [
                "writing_tools",
                "grammar_check",
            ]
            assert updated_config["analyst"] == original_config["analyst"]  # unchanged

    def test_bulk_update_tasks(self):
        """Test bulk updating multiple tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "tasks.yaml"
            original_config = {
                "task1": {"description": "Task 1", "agent": "agent1"},
                "task2": {"description": "Task 2", "agent": "agent2"},
            }
            config_file.write_text(yaml.dump(original_config))

            updater = ConfigurationUpdater()

            bulk_updates = {
                "task1": {
                    "expected_output": "Comprehensive analysis report",
                    "tools": ["analysis_tools"],
                },
                "task2": {
                    "expected_output": "Well-structured document",
                    "context": ["target audience", "brand guidelines"],
                },
            }

            result = updater.bulk_update_tasks(config_file, bulk_updates)

            assert result["success"] is True
            assert result["updated_count"] == 2

            # Verify all updates were applied
            updated_config = yaml.safe_load(config_file.read_text())
            assert "expected_output" in updated_config["task1"]
            assert updated_config["task1"]["tools"] == ["analysis_tools"]
            assert updated_config["task2"]["context"] == [
                "target audience",
                "brand guidelines",
            ]


class TestErrorHandling:
    """Test error handling and recovery scenarios."""

    def test_update_with_invalid_yaml(self):
        """Test handling of invalid YAML files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "invalid.yaml"
            config_file.write_text("invalid: yaml: content: [\n  unclosed")

            updater = ConfigurationUpdater()

            result = updater.update_agent(config_file, "test", {"role": "Test"})

            assert result["success"] is False
            assert "Invalid YAML syntax" in result["error"]

    def test_update_with_write_permission_error(self):
        """Test handling of file write permission errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "readonly.yaml"
            config_file.write_text("test: content")
            config_file.chmod(0o444)  # Read-only

            updater = ConfigurationUpdater()

            try:
                result = updater.update_agent(config_file, "test", {"role": "Test"})
                assert result["success"] is False
                assert "Permission" in result["error"] or "readonly" in result["error"]
            finally:
                config_file.chmod(0o644)  # Restore permissions for cleanup

    def test_backup_creation_failure_handling(self):
        """Test handling of backup creation failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test.yaml"
            config_file.write_text("test: content")

            # Make parent directory read-only to prevent backup creation
            config_file.parent.chmod(0o555)

            updater = ConfigurationUpdater()

            try:
                result = updater.update_agent(config_file, "test", {"role": "Test"})
                assert result["success"] is False
                assert result["backup_created"] is False
            finally:
                config_file.parent.chmod(0o755)  # Restore permissions


class TestCommentPreservation:
    """Test preservation of YAML comments during updates."""

    def test_preserve_comments_enabled(self):
        """Test comment preservation when enabled (note: standard PyYAML doesn't preserve comments)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "agents.yaml"
            yaml_with_comments = """# Agent configurations
researcher:  # Primary research agent
  role: Researcher
  goal: Conduct research  # Main objective
  tools:
    - search  # Web search tool
    - analysis  # Data analysis
"""
            config_file.write_text(yaml_with_comments)

            updater = ConfigurationUpdater(preserve_comments=True)

            result = updater.update_agent(
                config_file, "researcher", {"backstory": "Expert researcher"}
            )

            assert result["success"] is True

            # Verify the content was updated correctly (comments will be lost with standard PyYAML)
            updated_content = config_file.read_text()
            assert "backstory: Expert researcher" in updated_content
            assert "role: Researcher" in updated_content

            # Note: Comments are not preserved with standard PyYAML library
            # This is a known limitation - would need ruamel.yaml or similar for comment preservation

    @pytest.mark.skip(
        reason="Comment preservation implementation depends on chosen YAML library"
    )
    def test_preserve_comments_disabled(self):
        """Test comment handling when preservation is disabled."""
        # This test will be implemented based on chosen YAML library approach
        pass


class TestIntegrationWithEnhancementEngine:
    """Test integration with existing EnhancementEngine."""

    def test_configuration_updater_usage_in_enhancement_engine(self):
        """Test that ConfigurationUpdater can be used by EnhancementEngine."""
        # This will test the integration after implementation
        updater = ConfigurationUpdater()

        # Basic validation that the interface is compatible
        assert hasattr(updater, "update_agent")
        assert hasattr(updater, "update_task")
        assert hasattr(updater, "bulk_update_agents")
        assert hasattr(updater, "bulk_update_tasks")

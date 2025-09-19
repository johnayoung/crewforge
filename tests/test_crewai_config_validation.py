"""
Tests for CrewAI configuration file validation functionality.

This module tests validation for agents.yaml and tasks.yaml files
to ensure they conform to CrewAI specifications and can be processed correctly.
"""

import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Any
import pytest
import yaml

from crewforge.validation import (
    ValidationIssue,
    ValidationResult,
    IssueSeverity,
)


class TestCrewAIConfigValidation:
    """Test cases for CrewAI configuration file validation."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for test config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def valid_agents_config(self) -> Dict[str, Any]:
        """Sample valid agents configuration."""
        return {
            "research_agent": {
                "role": "Senior Research Analyst",
                "goal": "Research and analyze market trends and opportunities",
                "backstory": "You are an expert research analyst with 10+ years of experience in market analysis.",
                "tools": ["SerperDevTool", "WebsiteSearchTool"],
                "verbose": True,
            },
            "writer_agent": {
                "role": "Content Writer",
                "goal": "Create engaging and informative content based on research findings",
                "backstory": "You are a skilled content writer who excels at transforming complex data into readable content.",
                "tools": ["WebsiteSearchTool"],
                "verbose": False,
            },
        }

    @pytest.fixture
    def valid_tasks_config(self) -> Dict[str, Any]:
        """Sample valid tasks configuration."""
        return {
            "research_task": {
                "description": "Research the latest trends in artificial intelligence and machine learning",
                "expected_output": "A comprehensive report with 5 key trends and supporting data",
                "agent": "research_agent",
            },
            "writing_task": {
                "description": "Write a blog post based on the research findings",
                "expected_output": "A 1500-word blog post in markdown format",
                "agent": "writer_agent",
                "context": ["research_task"],
            },
        }

    def test_validate_valid_agents_yaml(self, temp_config_dir, valid_agents_config):
        """Test validation of a valid agents.yaml file."""
        from crewforge.validation import validate_crewai_agents_config

        # Create valid agents.yaml file
        agents_file = temp_config_dir / "agents.yaml"
        with open(agents_file, "w") as f:
            yaml.dump(valid_agents_config, f)

        result = validate_crewai_agents_config(agents_file)

        assert result.is_valid
        assert len(result.errors) == 0
        assert any(issue.severity == IssueSeverity.INFO for issue in result.issues)

    def test_validate_valid_tasks_yaml(self, temp_config_dir, valid_tasks_config):
        """Test validation of a valid tasks.yaml file."""
        from crewforge.validation import validate_crewai_tasks_config

        # Create valid tasks.yaml file
        tasks_file = temp_config_dir / "tasks.yaml"
        with open(tasks_file, "w") as f:
            yaml.dump(valid_tasks_config, f)

        result = validate_crewai_tasks_config(tasks_file)

        assert result.is_valid
        assert len(result.errors) == 0
        assert any(issue.severity == IssueSeverity.INFO for issue in result.issues)

    def test_validate_agents_missing_required_fields(self, temp_config_dir):
        """Test validation of agents.yaml with missing required fields."""
        from crewforge.validation import validate_crewai_agents_config

        invalid_config = {
            "incomplete_agent": {
                "role": "Research Analyst",
                # Missing: goal, backstory, tools
                "verbose": True,
            }
        }

        agents_file = temp_config_dir / "agents.yaml"
        with open(agents_file, "w") as f:
            yaml.dump(invalid_config, f)

        result = validate_crewai_agents_config(agents_file)

        assert not result.is_valid
        assert len(result.errors) >= 3  # Missing goal, backstory, tools

        error_messages = [error.message for error in result.errors]
        assert any("goal" in msg.lower() for msg in error_messages)
        assert any("backstory" in msg.lower() for msg in error_messages)
        assert any("tools" in msg.lower() for msg in error_messages)

    def test_validate_tasks_missing_required_fields(self, temp_config_dir):
        """Test validation of tasks.yaml with missing required fields."""
        from crewforge.validation import validate_crewai_tasks_config

        invalid_config = {
            "incomplete_task": {
                "description": "Do some work",
                # Missing: expected_output, agent
            }
        }

        tasks_file = temp_config_dir / "tasks.yaml"
        with open(tasks_file, "w") as f:
            yaml.dump(invalid_config, f)

        result = validate_crewai_tasks_config(tasks_file)

        assert not result.is_valid
        assert len(result.errors) >= 2  # Missing expected_output, agent

        error_messages = [error.message for error in result.errors]
        assert any("expected_output" in msg.lower() for msg in error_messages)
        assert any("agent" in msg.lower() for msg in error_messages)

    def test_validate_agents_invalid_data_types(self, temp_config_dir):
        """Test validation of agents.yaml with invalid data types."""
        from crewforge.validation import validate_crewai_agents_config

        invalid_config = {
            "bad_agent": {
                "role": 123,  # Should be string
                "goal": "Valid goal",
                "backstory": "Valid backstory",
                "tools": "not_a_list",  # Should be list
                "verbose": "not_a_boolean",  # Should be boolean
            }
        }

        agents_file = temp_config_dir / "agents.yaml"
        with open(agents_file, "w") as f:
            yaml.dump(invalid_config, f)

        result = validate_crewai_agents_config(agents_file)

        assert not result.is_valid
        assert len(result.errors) >= 3  # Invalid types for role, tools, verbose

        error_messages = [error.message for error in result.errors]
        assert any(
            "role" in msg.lower() and "string" in msg.lower() for msg in error_messages
        )
        assert any(
            "tools" in msg.lower() and "list" in msg.lower() for msg in error_messages
        )
        assert any(
            "verbose" in msg.lower() and "boolean" in msg.lower()
            for msg in error_messages
        )

    def test_validate_tasks_invalid_agent_reference(self, temp_config_dir):
        """Test validation of tasks.yaml with invalid agent references."""
        from crewforge.validation import validate_crewai_tasks_config

        invalid_config = {
            "orphaned_task": {
                "description": "Task with invalid agent reference",
                "expected_output": "Some output",
                "agent": "nonexistent_agent",
            }
        }

        tasks_file = temp_config_dir / "tasks.yaml"
        with open(tasks_file, "w") as f:
            yaml.dump(invalid_config, f)

        # Test with known agent list
        available_agents = ["research_agent", "writer_agent"]
        result = validate_crewai_tasks_config(
            tasks_file, available_agents=available_agents
        )

        assert not result.is_valid
        error_messages = [error.message for error in result.errors]
        assert any("nonexistent_agent" in msg.lower() for msg in error_messages)

    def test_validate_tasks_circular_dependencies(self, temp_config_dir):
        """Test detection of circular dependencies in tasks."""
        from crewforge.validation import validate_crewai_tasks_config

        circular_config = {
            "task_a": {
                "description": "Task A",
                "expected_output": "Output A",
                "agent": "agent1",
                "context": ["task_b"],
            },
            "task_b": {
                "description": "Task B",
                "expected_output": "Output B",
                "agent": "agent2",
                "context": ["task_a"],  # Circular dependency
            },
        }

        tasks_file = temp_config_dir / "tasks.yaml"
        with open(tasks_file, "w") as f:
            yaml.dump(circular_config, f)

        result = validate_crewai_tasks_config(tasks_file)

        assert not result.is_valid
        error_messages = [error.message for error in result.errors]
        assert any("circular" in msg.lower() for msg in error_messages)

    def test_validate_empty_config_files(self, temp_config_dir):
        """Test validation of empty config files."""
        from crewforge.validation import (
            validate_crewai_agents_config,
            validate_crewai_tasks_config,
        )

        # Empty agents file
        empty_agents_file = temp_config_dir / "empty_agents.yaml"
        empty_agents_file.write_text("")

        result = validate_crewai_agents_config(empty_agents_file)
        assert not result.is_valid
        assert any("empty" in error.message.lower() for error in result.errors)

        # Empty tasks file
        empty_tasks_file = temp_config_dir / "empty_tasks.yaml"
        empty_tasks_file.write_text("")

        result = validate_crewai_tasks_config(empty_tasks_file)
        assert not result.is_valid
        assert any("empty" in error.message.lower() for error in result.errors)

    def test_validate_invalid_yaml_syntax(self, temp_config_dir):
        """Test validation of files with invalid YAML syntax."""
        from crewforge.validation import validate_crewai_agents_config

        # Invalid YAML syntax
        invalid_yaml = temp_config_dir / "invalid.yaml"
        invalid_yaml.write_text(
            textwrap.dedent(
                """
        invalid_agent:
          role: "Research Analyst"
          goal: "Do research
          # Missing closing quote - invalid YAML
        """
            )
        )

        result = validate_crewai_agents_config(invalid_yaml)
        assert not result.is_valid
        assert any(
            "yaml" in error.message.lower() or "syntax" in error.message.lower()
            for error in result.errors
        )

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent files."""
        from crewforge.validation import validate_crewai_agents_config

        nonexistent_file = Path("/nonexistent/path/agents.yaml")
        result = validate_crewai_agents_config(nonexistent_file)

        assert not result.is_valid
        assert any("not found" in error.message.lower() for error in result.errors)

    def test_validate_agents_with_warnings(self, temp_config_dir):
        """Test validation generates appropriate warnings for agents."""
        from crewforge.validation import validate_crewai_agents_config

        warning_config = {
            "agent": {
                "role": "agent",  # Vague role should trigger warning
                "goal": "help users",  # Vague goal should trigger warning
                "backstory": "A helpful agent",  # Short backstory should trigger warning
                "tools": [],  # Empty tools list should trigger warning
                "verbose": True,
            }
        }

        agents_file = temp_config_dir / "agents.yaml"
        with open(agents_file, "w") as f:
            yaml.dump(warning_config, f)

        result = validate_crewai_agents_config(agents_file)

        # Should be valid but have warnings
        assert result.is_valid
        assert len(result.warnings) > 0

        warning_messages = [w.message for w in result.warnings]
        assert any(
            "vague" in msg.lower() or "generic" in msg.lower()
            for msg in warning_messages
        )

    def test_validate_tasks_with_warnings(self, temp_config_dir):
        """Test validation generates appropriate warnings for tasks."""
        from crewforge.validation import validate_crewai_tasks_config

        warning_config = {
            "task": {  # Generic task name should trigger warning
                "description": "Do work",  # Vague description should trigger warning
                "expected_output": "output",  # Vague output should trigger warning
                "agent": "some_agent",
            }
        }

        tasks_file = temp_config_dir / "tasks.yaml"
        with open(tasks_file, "w") as f:
            yaml.dump(warning_config, f)

        result = validate_crewai_tasks_config(tasks_file)

        # Should be valid but have warnings
        assert result.is_valid
        assert len(result.warnings) > 0

        warning_messages = [w.message for w in result.warnings]
        assert any(
            "vague" in msg.lower() or "generic" in msg.lower()
            for msg in warning_messages
        )

    def test_integrated_agents_and_tasks_validation(
        self, temp_config_dir, valid_agents_config, valid_tasks_config
    ):
        """Test validation of agents and tasks together for consistency."""
        from crewforge.validation import validate_crewai_config_consistency

        # Create both config files
        agents_file = temp_config_dir / "agents.yaml"
        tasks_file = temp_config_dir / "tasks.yaml"

        with open(agents_file, "w") as f:
            yaml.dump(valid_agents_config, f)
        with open(tasks_file, "w") as f:
            yaml.dump(valid_tasks_config, f)

        result = validate_crewai_config_consistency(agents_file, tasks_file)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_integrated_validation_agent_mismatch(
        self, temp_config_dir, valid_agents_config
    ):
        """Test integrated validation detects agent mismatches."""
        from crewforge.validation import validate_crewai_config_consistency

        # Tasks reference agents not in agents.yaml
        mismatched_tasks = {
            "task1": {
                "description": "Task using unknown agent",
                "expected_output": "Some output",
                "agent": "unknown_agent",  # Not in valid_agents_config
            }
        }

        agents_file = temp_config_dir / "agents.yaml"
        tasks_file = temp_config_dir / "tasks.yaml"

        with open(agents_file, "w") as f:
            yaml.dump(valid_agents_config, f)
        with open(tasks_file, "w") as f:
            yaml.dump(mismatched_tasks, f)

        result = validate_crewai_config_consistency(agents_file, tasks_file)

        assert not result.is_valid
        error_messages = [error.message for error in result.errors]
        assert any("unknown_agent" in msg.lower() for msg in error_messages)

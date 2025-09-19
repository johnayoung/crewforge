"""
Tests for integration of CrewAI config validation with main validation flow.

This module tests that the main project validation function correctly
calls and integrates CrewAI configuration validation results.
"""

import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Any
import pytest
import yaml

from crewforge.validation import validate_generated_project, IssueSeverity


class TestCrewAIConfigIntegration:
    """Test cases for CrewAI config validation integration."""

    @pytest.fixture
    def sample_project_structure(self):
        """Create a sample CrewAI project structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create basic project structure
            src_dir = project_root / "src" / "test_project"
            config_dir = src_dir / "config"
            tools_dir = src_dir / "tools"

            src_dir.mkdir(parents=True)
            config_dir.mkdir()
            tools_dir.mkdir()

            # Create basic Python files
            (src_dir / "__init__.py").write_text("")
            (src_dir / "main.py").write_text('print("Hello, CrewAI!")')
            (src_dir / "crew.py").write_text(
                "from crewai import Crew\n\nclass TestCrew:\n    pass"
            )
            (tools_dir / "__init__.py").write_text("")
            (tools_dir / "custom_tool.py").write_text("# Custom tools")

            # Create project files
            (project_root / "pyproject.toml").write_text(
                "[project]\nname = 'test-project'"
            )
            (project_root / "README.md").write_text("# Test Project")
            (project_root / ".env").write_text("# Environment variables")

            yield project_root, config_dir

    def test_validate_project_with_valid_config(self, sample_project_structure):
        """Test project validation with valid configuration files."""
        project_root, config_dir = sample_project_structure

        # Create valid configuration files
        valid_agents = {
            "research_agent": {
                "role": "Senior Research Analyst",
                "goal": "Research and analyze market trends and opportunities",
                "backstory": "You are an expert research analyst with 10+ years of experience.",
                "tools": ["SerperDevTool", "WebsiteSearchTool"],
                "verbose": True,
            }
        }

        valid_tasks = {
            "research_task": {
                "description": "Research the latest trends in artificial intelligence",
                "expected_output": "A comprehensive report with 5 key trends and supporting data",
                "agent": "research_agent",
            }
        }

        # Write configuration files
        with open(config_dir / "agents.yaml", "w") as f:
            yaml.dump(valid_agents, f)
        with open(config_dir / "tasks.yaml", "w") as f:
            yaml.dump(valid_tasks, f)

        # Create Python files that don't import external modules
        src_dir = project_root / "src" / "test_project"
        (src_dir / "__init__.py").write_text("# Test project init")
        (src_dir / "main.py").write_text('print("Hello, CrewAI!")')
        (src_dir / "crew.py").write_text("class TestCrew:\n    pass")
        (src_dir / "tools" / "__init__.py").write_text("# Tools init")

        # Validate the project
        result = validate_generated_project(str(project_root))

        # Should be valid overall (config is good, Python syntax is good)
        assert result.is_valid

        # Should contain config validation success messages
        info_messages = [issue.message for issue in result.info_messages]
        assert any("Agents configuration is valid" in msg for msg in info_messages)
        assert any("Tasks configuration is valid" in msg for msg in info_messages)
        assert any(
            "Configuration consistency validated" in msg for msg in info_messages
        )

    def test_validate_project_with_invalid_config(self, sample_project_structure):
        """Test project validation with invalid configuration files."""
        project_root, config_dir = sample_project_structure

        # Create Python files that don't import external modules
        src_dir = project_root / "src" / "test_project"
        (src_dir / "__init__.py").write_text("# Test project init")
        (src_dir / "main.py").write_text('print("Hello, CrewAI!")')
        (src_dir / "crew.py").write_text("class TestCrew:\n    pass")
        (src_dir / "tools" / "__init__.py").write_text("# Tools init")

        # Create invalid configuration files
        invalid_agents = {
            "incomplete_agent": {
                "role": "Research Analyst",
                # Missing: goal, backstory, tools
            }
        }

        invalid_tasks = {
            "orphaned_task": {
                "description": "Task with invalid agent reference",
                "expected_output": "Some output",
                "agent": "nonexistent_agent",  # Not in agents config
            }
        }

        # Write configuration files
        with open(config_dir / "agents.yaml", "w") as f:
            yaml.dump(invalid_agents, f)
        with open(config_dir / "tasks.yaml", "w") as f:
            yaml.dump(invalid_tasks, f)

        # Validate the project
        result = validate_generated_project(str(project_root))

        # Should be invalid due to config errors
        assert not result.is_valid

        # Should contain config validation error messages
        error_messages = [error.message for error in result.errors]
        assert any("missing required field" in msg.lower() for msg in error_messages)
        assert any("references unknown agent" in msg.lower() for msg in error_messages)

    def test_validate_project_with_missing_config_files(self, sample_project_structure):
        """Test project validation with missing configuration files."""
        project_root, config_dir = sample_project_structure

        # Create Python files that don't import external modules
        src_dir = project_root / "src" / "test_project"
        (src_dir / "__init__.py").write_text("# Test project init")
        (src_dir / "main.py").write_text('print("Hello, CrewAI!")')
        (src_dir / "crew.py").write_text("class TestCrew:\n    pass")
        (src_dir / "tools" / "__init__.py").write_text("# Tools init")

        # Don't create any config files - they should be reported as missing

        # Validate the project
        result = validate_generated_project(str(project_root))

        # Should still be valid overall (missing config files are warnings, not errors)
        assert result.is_valid

        # Should contain warnings about missing config files
        warning_messages = [warning.message for warning in result.warnings]
        assert any(
            "Missing config file: agents.yaml" in msg for msg in warning_messages
        )
        assert any("Missing config file: tasks.yaml" in msg for msg in warning_messages)

    def test_validate_project_with_syntax_errors_and_config(
        self, sample_project_structure
    ):
        """Test project validation when Python files have syntax errors but config is valid."""
        project_root, config_dir = sample_project_structure

        # Create Python file with syntax error
        src_dir = project_root / "src" / "test_project"
        (src_dir / "broken.py").write_text(
            "def broken_function(\n    # Missing closing parenthesis"
        )

        # Create valid configuration files
        valid_agents = {
            "agent": {
                "role": "Test Agent",
                "goal": "Test goal for the agent",
                "backstory": "Test backstory for the agent",
                "tools": ["SerperDevTool"],
                "verbose": True,
            }
        }

        valid_tasks = {
            "task": {
                "description": "Test task description",
                "expected_output": "Test expected output",
                "agent": "agent",
            }
        }

        # Write configuration files
        with open(config_dir / "agents.yaml", "w") as f:
            yaml.dump(valid_agents, f)
        with open(config_dir / "tasks.yaml", "w") as f:
            yaml.dump(valid_tasks, f)

        # Validate the project
        result = validate_generated_project(str(project_root))

        # Should be invalid due to Python syntax errors
        assert not result.is_valid

        # Should still contain config validation (which should be successful)
        info_messages = [issue.message for issue in result.info_messages]
        assert any("Agents configuration is valid" in msg for msg in info_messages)

        # Should contain syntax error messages
        error_messages = [error.message for error in result.errors]
        assert any("syntax error" in msg.lower() for msg in error_messages)

    def test_validate_project_without_config_directory(self, sample_project_structure):
        """Test project validation when config directory doesn't exist."""
        project_root, config_dir = sample_project_structure

        # Create Python files that don't import external modules
        src_dir = project_root / "src" / "test_project"
        (src_dir / "__init__.py").write_text("# Test project init")
        (src_dir / "main.py").write_text('print("Hello, CrewAI!")')
        (src_dir / "crew.py").write_text("class TestCrew:\n    pass")
        (src_dir / "tools" / "__init__.py").write_text("# Tools init")

        # Remove the config directory
        import shutil

        shutil.rmtree(config_dir)

        # Validate the project
        result = validate_generated_project(str(project_root))

        # Should still be valid overall (missing config directory is a warning)
        assert result.is_valid

        # Should contain warning about missing config directory
        warning_messages = [warning.message for warning in result.warnings]
        assert any("Missing config directory" in msg for msg in warning_messages)

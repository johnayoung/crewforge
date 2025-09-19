"""
Integration tests for ConfigurationUpdater with EnhancementEngine.

This module tests the integration between the new selective configuration
updating system and the existing EnhancementEngine functionality.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock

from crewforge.enhancement import EnhancementEngine
from crewforge.configuration_updater import UpdateStrategy


class TestConfigurationUpdaterIntegration:
    """Test integration of ConfigurationUpdater with EnhancementEngine."""

    def test_enhancement_engine_has_config_updater(self):
        """Test that EnhancementEngine initializes with ConfigurationUpdater."""
        engine = EnhancementEngine()

        assert hasattr(engine, "config_updater")
        assert engine.config_updater is not None

    def test_update_agent_configuration(self):
        """Test updating a specific agent configuration through EnhancementEngine."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project structure
            project_path = Path(temp_dir) / "test_project"
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            # Create agents.yaml
            agents_config = {
                "researcher": {
                    "role": "Research Analyst",
                    "goal": "Conduct thorough research",
                    "backstory": "Experienced researcher",
                },
                "writer": {"role": "Content Writer", "goal": "Create engaging content"},
            }
            agents_file = config_path / "agents.yaml"
            agents_file.write_text(yaml.dump(agents_config))

            engine = EnhancementEngine()

            # Update specific agent
            agent_updates = {
                "backstory": "Senior research analyst with 10 years experience",
                "verbose": True,
                "tools": ["web_search", "data_analysis"],
            }

            result = engine.update_agent_configuration(
                project_path, "researcher", agent_updates
            )

            assert result["success"] is True
            assert result["agent_name"] == "researcher"
            assert result["backup_created"] is True
            assert result["strategy"] == "update"

            # Verify the update was applied correctly
            updated_config = yaml.safe_load(agents_file.read_text())
            researcher = updated_config["researcher"]

            assert researcher["role"] == "Research Analyst"  # preserved
            assert researcher["goal"] == "Conduct thorough research"  # preserved
            assert (
                researcher["backstory"]
                == "Senior research analyst with 10 years experience"
            )  # updated
            assert researcher["verbose"] is True  # added
            assert researcher["tools"] == ["web_search", "data_analysis"]  # added

            # Verify other agent wasn't affected
            assert updated_config["writer"] == agents_config["writer"]

    def test_update_task_configuration(self):
        """Test updating a specific task configuration through EnhancementEngine."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project structure
            project_path = Path(temp_dir) / "test_project"
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            # Create tasks.yaml
            tasks_config = {
                "research_task": {
                    "description": "Conduct market research",
                    "agent": "researcher",
                },
                "writing_task": {
                    "description": "Write report",
                    "agent": "writer",
                    "expected_output": "Written report",
                },
            }
            tasks_file = config_path / "tasks.yaml"
            tasks_file.write_text(yaml.dump(tasks_config))

            engine = EnhancementEngine()

            # Update specific task
            task_updates = {
                "expected_output": "Comprehensive market analysis report",
                "context": ["industry trends", "competitor analysis"],
                "tools": ["research_tools", "analysis_tools"],
            }

            result = engine.update_task_configuration(
                project_path, "research_task", task_updates
            )

            assert result["success"] is True
            assert result["task_name"] == "research_task"
            assert result["backup_created"] is True

            # Verify the update was applied correctly
            updated_config = yaml.safe_load(tasks_file.read_text())
            research_task = updated_config["research_task"]

            assert (
                research_task["description"] == "Conduct market research"
            )  # preserved
            assert research_task["agent"] == "researcher"  # preserved
            assert (
                research_task["expected_output"]
                == "Comprehensive market analysis report"
            )  # updated
            assert research_task["context"] == [
                "industry trends",
                "competitor analysis",
            ]  # added
            assert research_task["tools"] == [
                "research_tools",
                "analysis_tools",
            ]  # added

            # Verify other task wasn't affected
            assert updated_config["writing_task"] == tasks_config["writing_task"]

    def test_bulk_update_agents_configuration(self):
        """Test bulk updating multiple agents through EnhancementEngine."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project structure
            project_path = Path(temp_dir) / "test_project"
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            # Create agents.yaml
            agents_config = {
                "researcher": {"role": "Researcher", "goal": "Research"},
                "writer": {"role": "Writer", "goal": "Write"},
                "analyst": {"role": "Analyst", "goal": "Analyze"},
            }
            agents_file = config_path / "agents.yaml"
            agents_file.write_text(yaml.dump(agents_config))

            engine = EnhancementEngine()

            # Bulk update multiple agents
            bulk_updates = {
                "researcher": {
                    "backstory": "Senior research specialist",
                    "verbose": True,
                },
                "writer": {
                    "backstory": "Professional technical writer",
                    "tools": ["writing_tools", "grammar_check"],
                },
            }

            result = engine.bulk_update_agents_configuration(project_path, bulk_updates)

            assert result["success"] is True
            assert result["updated_count"] == 2
            assert set(result["updated_agents"]) == {"researcher", "writer"}

            # Verify updates were applied correctly
            updated_config = yaml.safe_load(agents_file.read_text())

            # Check researcher updates
            researcher = updated_config["researcher"]
            assert researcher["backstory"] == "Senior research specialist"
            assert researcher["verbose"] is True

            # Check writer updates
            writer = updated_config["writer"]
            assert writer["backstory"] == "Professional technical writer"
            assert writer["tools"] == ["writing_tools", "grammar_check"]

            # Check analyst wasn't affected
            assert updated_config["analyst"] == agents_config["analyst"]

    def test_bulk_update_tasks_configuration(self):
        """Test bulk updating multiple tasks through EnhancementEngine."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project structure
            project_path = Path(temp_dir) / "test_project"
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            # Create tasks.yaml
            tasks_config = {
                "task1": {"description": "Task 1", "agent": "agent1"},
                "task2": {"description": "Task 2", "agent": "agent2"},
                "task3": {"description": "Task 3", "agent": "agent3"},
            }
            tasks_file = config_path / "tasks.yaml"
            tasks_file.write_text(yaml.dump(tasks_config))

            engine = EnhancementEngine()

            # Bulk update multiple tasks
            bulk_updates = {
                "task1": {
                    "expected_output": "Detailed analysis report",
                    "tools": ["analysis_tools"],
                },
                "task2": {
                    "expected_output": "Well-structured document",
                    "context": ["target audience", "brand guidelines"],
                },
            }

            result = engine.bulk_update_tasks_configuration(project_path, bulk_updates)

            assert result["success"] is True
            assert result["updated_count"] == 2
            assert set(result["updated_tasks"]) == {"task1", "task2"}

            # Verify updates were applied correctly
            updated_config = yaml.safe_load(tasks_file.read_text())

            # Check task1 updates
            task1 = updated_config["task1"]
            assert task1["expected_output"] == "Detailed analysis report"
            assert task1["tools"] == ["analysis_tools"]

            # Check task2 updates
            task2 = updated_config["task2"]
            assert task2["expected_output"] == "Well-structured document"
            assert task2["context"] == ["target audience", "brand guidelines"]

            # Check task3 wasn't affected
            assert updated_config["task3"] == tasks_config["task3"]

    def test_validate_project_configuration(self):
        """Test project configuration validation through EnhancementEngine."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project structure
            project_path = Path(temp_dir) / "test_project"
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            # Create valid agents.yaml
            agents_config = {
                "researcher": {
                    "role": "Research Analyst",
                    "goal": "Conduct research",
                    "backstory": "Experienced researcher",
                }
            }
            agents_file = config_path / "agents.yaml"
            agents_file.write_text(yaml.dump(agents_config))

            # Create tasks.yaml missing required fields
            tasks_config = {
                "research_task": {
                    "description": "Research task",
                    # Missing required "agent" field
                }
            }
            tasks_file = config_path / "tasks.yaml"
            tasks_file.write_text(yaml.dump(tasks_config))

            engine = EnhancementEngine()

            result = engine.validate_project_configuration(project_path)

            assert result["success"] is False  # Overall validation failed

            # Check agents validation (should pass)
            agents_validation = result["agents_validation"]
            assert agents_validation["success"] is True
            assert len(agents_validation["errors"]) == 0
            assert agents_validation["entry_count"] == 1

            # Check tasks validation (should fail due to missing field)
            tasks_validation = result["tasks_validation"]
            assert tasks_validation["success"] is True  # File loads successfully
            assert len(tasks_validation["errors"]) == 1  # Missing required field
            assert "missing required field: agent" in tasks_validation["errors"][0]

    def test_update_strategies_integration(self):
        """Test different update strategies through EnhancementEngine."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project structure
            project_path = Path(temp_dir) / "test_project"
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            # Create agents.yaml
            agents_config = {
                "researcher": {
                    "role": "Original Role",
                    "goal": "Original Goal",
                    "existing_field": "Keep This",
                }
            }
            agents_file = config_path / "agents.yaml"
            agents_file.write_text(yaml.dump(agents_config))

            engine = EnhancementEngine()

            # Test ADD_ONLY strategy
            agent_updates = {
                "role": "Should Not Replace",  # Should be ignored
                "backstory": "Should Be Added",  # Should be added
            }

            result = engine.update_agent_configuration(
                project_path, "researcher", agent_updates, UpdateStrategy.ADD_ONLY
            )

            assert result["success"] is True
            assert result["strategy"] == "add_only"

            # Verify ADD_ONLY behavior
            updated_config = yaml.safe_load(agents_file.read_text())
            researcher = updated_config["researcher"]
            assert researcher["role"] == "Original Role"  # preserved
            assert researcher["goal"] == "Original Goal"  # preserved
            assert researcher["existing_field"] == "Keep This"  # preserved
            assert researcher["backstory"] == "Should Be Added"  # added

    def test_invalid_project_structure_handling(self):
        """Test handling of invalid project structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid project (missing src directory)
            project_path = Path(temp_dir) / "invalid_project"
            project_path.mkdir()

            engine = EnhancementEngine()

            result = engine.update_agent_configuration(
                project_path, "researcher", {"role": "Test"}
            )

            assert result["success"] is False
            assert "Invalid CrewAI project structure" in result["error"]

    def test_missing_config_file_handling(self):
        """Test handling of missing configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project structure but no agents.yaml
            project_path = Path(temp_dir) / "test_project"
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            engine = EnhancementEngine()

            result = engine.update_agent_configuration(
                project_path, "researcher", {"role": "Test"}
            )

            assert result["success"] is False
            assert "agents.yaml not found" in result["error"]

    def test_backward_compatibility_with_templates(self):
        """Test that existing template-based methods still work."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project structure
            project_path = Path(temp_dir) / "test_project"
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            # Create agents.yaml
            agents_config = {"researcher": {"role": "Researcher", "goal": "Research"}}
            agents_file = config_path / "agents.yaml"
            agents_file.write_text(yaml.dump(agents_config))

            # Create template directory
            templates_dir = Path(temp_dir) / "templates"
            agents_templates_dir = templates_dir / "agents"
            agents_templates_dir.mkdir(parents=True)

            template_content = """researcher:
  role: Enhanced Researcher
  goal: {{ enhanced_goal }}
  backstory: {{ enhanced_backstory }}
"""
            (agents_templates_dir / "default.yaml.j2").write_text(template_content)

            engine = EnhancementEngine(templates_dir=templates_dir)

            # Test that template-based enhancement still works
            enhancement_context = {
                "enhanced_goal": "Advanced research with AI",
                "enhanced_backstory": "AI research specialist",
            }

            result = engine.enhance_agents_config(project_path, enhancement_context)

            assert result["success"] is True

            # Verify template was applied
            updated_content = agents_file.read_text()
            assert "Enhanced Researcher" in updated_content
            assert "Advanced research with AI" in updated_content

    def test_integration_with_tool_patterns(self):
        """Test that new methods work with existing tool patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project structure
            project_path = Path(temp_dir) / "test_project"
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            # Create agents.yaml
            agents_config = {"researcher": {"role": "Researcher", "goal": "Research"}}
            agents_file = config_path / "agents.yaml"
            agents_file.write_text(yaml.dump(agents_config))

            engine = EnhancementEngine()

            # Generate tool configuration (existing functionality)
            project_spec = {
                "project_name": "test_project",
                "description": "A research project",
                "project_type": "research",
            }
            tool_config = engine.generate_tool_configuration(project_spec)

            # Use selective update to add tool information to agent
            agent_updates = {
                "tools": tool_config["all_tools"][:3],  # Use first 3 tools
                "project_type": tool_config["project_type"],
            }

            result = engine.update_agent_configuration(
                project_path, "researcher", agent_updates
            )

            assert result["success"] is True

            # Verify tools were added to the agent
            updated_config = yaml.safe_load(agents_file.read_text())
            researcher = updated_config["researcher"]
            assert "tools" in researcher
            assert "project_type" in researcher
            assert researcher["project_type"] == "research"

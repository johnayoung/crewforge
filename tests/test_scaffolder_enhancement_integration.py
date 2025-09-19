"""
Tests for Scaffolder Enhancement Integration

Test suite for the integration between CrewAI Scaffolder and Enhancement Engine
to validate the end-to-end project creation and customization workflow.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from crewforge.scaffolder import CrewAIScaffolder, CrewAIError


class TestScaffolderEnhancementIntegration:
    """Test suite for scaffolder and enhancement engine integration."""

    def test_create_crew_with_enhancement_success(self):
        """Test successful project creation with enhancement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir)
            project_name = "test-enhanced-project"

            # Mock successful scaffolding
            with patch.object(CrewAIScaffolder, "create_crew") as mock_create:
                project_path = target_dir / project_name
                project_path.mkdir()

                # Create basic project structure
                config_path = project_path / "src" / project_name / "config"
                config_path.mkdir(parents=True)

                # Create basic config files
                basic_agents = {
                    "researcher": {
                        "role": "Researcher",
                        "goal": "Basic research",
                        "backstory": "Basic backstory",
                        "tools": ["search"],
                    }
                }
                basic_tasks = {
                    "research": {
                        "description": "Basic research task",
                        "expected_output": "Results",
                        "agent": "researcher",
                    }
                }

                (config_path / "agents.yaml").write_text(yaml.dump(basic_agents))
                (config_path / "tasks.yaml").write_text(yaml.dump(basic_tasks))

                mock_create.return_value = {
                    "success": True,
                    "project_path": project_path,
                    "project_name": project_name,
                    "stdout": "Project created successfully",
                    "stderr": "",
                    "returncode": 0,
                }

                scaffolder = CrewAIScaffolder()

                enhancement_context = {
                    "agents": basic_agents,
                    "tasks": basic_tasks,
                    "enhanced_goals": {
                        "researcher": "Conduct comprehensive research using advanced methodologies"
                    },
                    "enhanced_backstories": {
                        "researcher": "Senior researcher with 10 years of experience"
                    },
                    "enhanced_tools": {
                        "researcher": [
                            "advanced_search",
                            "data_analysis",
                            "report_generation",
                        ]
                    },
                }

                result = scaffolder.create_crew_with_enhancement(
                    project_name, target_dir, enhancement_context, "default"
                )

                # Verify basic scaffolding succeeded
                assert result["success"] is True
                assert result["project_name"] == project_name
                assert result["project_path"] == project_path

                # Verify enhancement was attempted and succeeded
                assert result["enhancement"]["attempted"] is True
                assert result["enhancement"]["success"] is True
                assert result["enhancement"]["agents_enhanced"] is True
                assert result["enhancement"]["tasks_enhanced"] is True
                assert result["enhancement"]["template_used"] == "default"

                # Verify enhanced content was written
                enhanced_agents_content = (config_path / "agents.yaml").read_text()
                assert (
                    "comprehensive research using advanced methodologies"
                    in enhanced_agents_content.lower()
                )
                assert "10 years of experience" in enhanced_agents_content

                # Verify backups were created
                assert result.get("agents_backup") is not None
                assert result.get("tasks_backup") is not None

    def test_create_crew_with_enhancement_no_context(self):
        """Test project creation without enhancement context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir)
            project_name = "test-basic-project"

            with patch.object(CrewAIScaffolder, "create_crew") as mock_create:
                project_path = target_dir / project_name

                mock_create.return_value = {
                    "success": True,
                    "project_path": project_path,
                    "project_name": project_name,
                    "stdout": "Project created successfully",
                    "stderr": "",
                    "returncode": 0,
                }

                scaffolder = CrewAIScaffolder()

                result = scaffolder.create_crew_with_enhancement(
                    project_name, target_dir
                )

                # Verify basic scaffolding succeeded
                assert result["success"] is True

                # Verify enhancement was not attempted
                assert result["enhancement"]["attempted"] is False
                assert (
                    result["enhancement"]["reason"] == "No enhancement context provided"
                )

    def test_create_crew_with_enhancement_scaffolding_fails(self):
        """Test enhancement when basic scaffolding fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir)
            project_name = "test-failed-project"

            with patch.object(CrewAIScaffolder, "create_crew") as mock_create:
                mock_create.return_value = {
                    "success": False,
                    "error": "CrewAI CLI not available",
                    "project_path": None,
                    "project_name": project_name,
                    "returncode": 1,
                }

                scaffolder = CrewAIScaffolder()

                enhancement_context = {"agents": {"test": "data"}}

                result = scaffolder.create_crew_with_enhancement(
                    project_name, target_dir, enhancement_context
                )

                # Verify scaffolding failure is returned as-is
                assert result["success"] is False
                assert result["error"] == "CrewAI CLI not available"

                # Enhancement should not be attempted if scaffolding fails
                assert "enhancement" not in result

    def test_create_crew_with_enhancement_partial_failure(self):
        """Test enhancement when one configuration fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir)
            project_name = "test-partial-project"

            with patch.object(CrewAIScaffolder, "create_crew") as mock_create:
                project_path = target_dir / project_name
                project_path.mkdir()

                # Create partial project structure (missing tasks.yaml)
                config_path = project_path / "src" / project_name / "config"
                config_path.mkdir(parents=True)

                basic_agents = {
                    "researcher": {
                        "role": "Researcher",
                        "goal": "Basic research",
                        "backstory": "Basic backstory",
                        "tools": ["search"],
                    }
                }

                (config_path / "agents.yaml").write_text(yaml.dump(basic_agents))
                # Note: tasks.yaml is intentionally missing

                mock_create.return_value = {
                    "success": True,
                    "project_path": project_path,
                    "project_name": project_name,
                    "stdout": "Project created successfully",
                    "stderr": "",
                    "returncode": 0,
                }

                scaffolder = CrewAIScaffolder()

                enhancement_context = {
                    "agents": basic_agents,
                    "tasks": {"research": {"description": "test"}},
                    "enhanced_goals": {"researcher": "Enhanced goal"},
                }

                result = scaffolder.create_crew_with_enhancement(
                    project_name, target_dir, enhancement_context, "default"
                )

                # Verify basic scaffolding succeeded
                assert result["success"] is True

                # Verify enhancement was attempted but partially failed
                assert result["enhancement"]["attempted"] is True
                assert (
                    result["enhancement"]["success"] is False
                )  # Overall failure due to tasks
                assert (
                    result["enhancement"]["agents_enhanced"] is True
                )  # Agents should succeed
                assert (
                    result["enhancement"]["tasks_enhanced"] is False
                )  # Tasks should fail
                assert len(result["enhancement"]["errors"]) > 0
                assert "tasks.yaml not found" in str(result["enhancement"]["errors"])

    def test_get_available_enhancement_templates(self):
        """Test getting available enhancement templates."""
        scaffolder = CrewAIScaffolder()

        templates = scaffolder.get_available_enhancement_templates()

        # Should have both agents and tasks categories
        assert "agents" in templates
        assert "tasks" in templates

        # Should find our default templates
        assert "default" in templates["agents"]
        assert "default" in templates["tasks"]

        # Should find specialized templates
        assert "content_team" in templates["agents"]
        assert "data_team" in templates["agents"]
        assert "marketing_team" in templates["agents"]

    def test_get_available_enhancement_templates_by_category(self):
        """Test getting templates for specific category."""
        scaffolder = CrewAIScaffolder()

        agents_templates = scaffolder.get_available_enhancement_templates("agents")
        tasks_templates = scaffolder.get_available_enhancement_templates("tasks")

        assert "agents" in agents_templates
        assert "default" in agents_templates["agents"]

        assert "tasks" in tasks_templates
        assert "default" in tasks_templates["tasks"]

    @patch("crewforge.scaffolder.EnhancementEngine", None)
    def test_enhancement_engine_not_available(self):
        """Test graceful handling when enhancement engine is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir)
            project_name = "test-no-enhancement"

            with patch.object(CrewAIScaffolder, "create_crew") as mock_create:
                project_path = target_dir / project_name

                mock_create.return_value = {
                    "success": True,
                    "project_path": project_path,
                    "project_name": project_name,
                    "stdout": "Project created successfully",
                    "stderr": "",
                    "returncode": 0,
                }

                scaffolder = CrewAIScaffolder()

                # Should fall back to basic creation
                result = scaffolder.create_crew_with_enhancement(
                    project_name, target_dir, {"test": "context"}
                )

                assert result["success"] is True
                mock_create.assert_called_once()

        # Test template listing when engine not available
        scaffolder = CrewAIScaffolder()
        templates = scaffolder.get_available_enhancement_templates()
        assert "error" in templates
        assert "Enhancement engine not available" in templates["error"]


class TestScaffolderEnhancementErrorHandling:
    """Test suite for error handling in scaffolder enhancement integration."""

    def test_enhancement_error_handling(self):
        """Test handling of enhancement errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir)
            project_name = "test-error-project"

            with patch.object(CrewAIScaffolder, "create_crew") as mock_create:
                project_path = target_dir / project_name

                mock_create.return_value = {
                    "success": True,
                    "project_path": project_path,
                    "project_name": project_name,
                    "stdout": "Project created successfully",
                    "stderr": "",
                    "returncode": 0,
                }

                # Mock enhancement engine to raise error
                from crewforge.enhancement import EnhancementEngine, EnhancementError

                with patch.object(
                    EnhancementEngine,
                    "__init__",
                    side_effect=EnhancementError("Template error"),
                ):
                    scaffolder = CrewAIScaffolder()

                    result = scaffolder.create_crew_with_enhancement(
                        project_name, target_dir, {"test": "context"}
                    )

                    # Scaffolding should succeed, enhancement should fail gracefully
                    assert result["success"] is True  # Basic scaffolding succeeded
                    assert result["enhancement"]["attempted"] is True
                    assert result["enhancement"]["success"] is False
                    assert "Template error" in result["enhancement"]["error"]

    def test_unexpected_enhancement_error(self):
        """Test handling of unexpected errors during enhancement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir)
            project_name = "test-unexpected-error"

            with patch.object(CrewAIScaffolder, "create_crew") as mock_create:
                project_path = target_dir / project_name

                mock_create.return_value = {
                    "success": True,
                    "project_path": project_path,
                    "project_name": project_name,
                    "stdout": "Project created successfully",
                    "stderr": "",
                    "returncode": 0,
                }

                # Mock enhancement engine to raise unexpected error
                from crewforge.enhancement import EnhancementEngine

                with patch.object(
                    EnhancementEngine,
                    "__init__",
                    side_effect=ValueError("Unexpected error"),
                ):
                    scaffolder = CrewAIScaffolder()

                    result = scaffolder.create_crew_with_enhancement(
                        project_name, target_dir, {"test": "context"}
                    )

                    # Scaffolding should succeed, enhancement should fail gracefully
                    assert result["success"] is True
                    assert result["enhancement"]["attempted"] is True
                    assert result["enhancement"]["success"] is False
                    assert (
                        "Unexpected enhancement error" in result["enhancement"]["error"]
                    )

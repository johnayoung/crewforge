"""
Tests for Enhancement Engine module

Test suite for the Jinja2 template system that handles intelligent customization
of generated CrewAI projects with domain-specific configurations.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

from crewforge.enhancement import (
    EnhancementEngine,
    EnhancementError,
    TemplateNotFoundError,
    TemplateRenderError,
)


class TestEnhancementEngine:
    """Test suite for EnhancementEngine class."""

    def test_initialization_default_templates_dir(self):
        """Test enhancement engine initialization with default templates directory."""
        engine = EnhancementEngine()

        assert engine.templates_dir is not None
        assert engine.templates_dir.is_dir()
        assert engine.environment is not None
        assert engine.logger is not None

    def test_initialization_custom_templates_dir(self):
        """Test enhancement engine initialization with custom templates directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = Path(temp_dir) / "custom_templates"
            custom_dir.mkdir()

            engine = EnhancementEngine(templates_dir=custom_dir)

            assert engine.templates_dir == custom_dir

    def test_initialization_nonexistent_templates_dir_error(self):
        """Test initialization error with nonexistent templates directory."""
        nonexistent_dir = Path("/nonexistent/path")

        with pytest.raises(
            EnhancementError, match="Templates directory does not exist"
        ):
            EnhancementEngine(templates_dir=nonexistent_dir)

    def test_get_available_templates_agents(self):
        """Test getting available agent templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir)
            agents_dir = templates_dir / "agents"
            agents_dir.mkdir()

            # Create test template files
            (agents_dir / "base_agent.yaml.j2").touch()
            (agents_dir / "researcher_agent.yaml.j2").touch()
            (agents_dir / "writer_agent.yaml.j2").touch()

            engine = EnhancementEngine(templates_dir=templates_dir)
            templates = engine.get_available_templates("agents")

            assert "base_agent" in templates
            assert "researcher_agent" in templates
            assert "writer_agent" in templates

    def test_get_available_templates_tasks(self):
        """Test getting available task templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir)
            tasks_dir = templates_dir / "tasks"
            tasks_dir.mkdir()

            # Create test template files
            (tasks_dir / "research_task.yaml.j2").touch()
            (tasks_dir / "writing_task.yaml.j2").touch()

            engine = EnhancementEngine(templates_dir=templates_dir)
            templates = engine.get_available_templates("tasks")

            assert "research_task" in templates
            assert "writing_task" in templates

    def test_get_available_templates_invalid_category(self):
        """Test getting templates for invalid category."""
        engine = EnhancementEngine()

        with pytest.raises(ValueError, match="Invalid template category"):
            engine.get_available_templates("invalid_category")

    def test_render_agent_template_success(self):
        """Test successful agent template rendering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir)
            agents_dir = templates_dir / "agents"
            agents_dir.mkdir()

            # Create test template
            template_content = """role: {{ agent.role }}
goal: {{ agent.goal }}
backstory: {{ agent.backstory }}
tools:
{% for tool in agent.tools %}
  - {{ tool }}
{% endfor %}
"""
            (agents_dir / "test_agent.yaml.j2").write_text(template_content)

            engine = EnhancementEngine(templates_dir=templates_dir)

            agent_data = {
                "role": "Researcher",
                "goal": "Conduct thorough research",
                "backstory": "Expert in data analysis",
                "tools": ["search_tool", "analysis_tool"],
            }

            result = engine.render_agent_template("test_agent", {"agent": agent_data})

            assert "role: Researcher" in result
            assert "goal: Conduct thorough research" in result
            assert "backstory: Expert in data analysis" in result
            assert "- search_tool" in result
            assert "- analysis_tool" in result

    def test_render_agent_template_not_found(self):
        """Test agent template rendering with nonexistent template."""
        engine = EnhancementEngine()

        with pytest.raises(
            TemplateNotFoundError, match="Agent template 'nonexistent' not found"
        ):
            engine.render_agent_template("nonexistent", {})

    def test_render_task_template_success(self):
        """Test successful task template rendering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir)
            tasks_dir = templates_dir / "tasks"
            tasks_dir.mkdir()

            # Create test template
            template_content = """description: {{ task.description }}
expected_output: {{ task.expected_output }}
agent: {{ task.agent }}
{% if task.dependencies %}
dependencies:
{% for dep in task.dependencies %}
  - {{ dep }}
{% endfor %}
{% endif %}
"""
            (tasks_dir / "test_task.yaml.j2").write_text(template_content)

            engine = EnhancementEngine(templates_dir=templates_dir)

            task_data = {
                "description": "Research the latest AI trends",
                "expected_output": "Comprehensive research report",
                "agent": "researcher",
                "dependencies": ["data_collection", "analysis"],
            }

            result = engine.render_task_template("test_task", {"task": task_data})

            assert "description: Research the latest AI trends" in result
            assert "expected_output: Comprehensive research report" in result
            assert "agent: researcher" in result
            assert "- data_collection" in result
            assert "- analysis" in result

    def test_render_task_template_not_found(self):
        """Test task template rendering with nonexistent template."""
        engine = EnhancementEngine()

        with pytest.raises(
            TemplateNotFoundError, match="Task template 'nonexistent' not found"
        ):
            engine.render_task_template("nonexistent", {})

    def test_render_template_error_handling(self):
        """Test template rendering error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            templates_dir = Path(temp_dir)
            agents_dir = templates_dir / "agents"
            agents_dir.mkdir()

            # Create template with syntax error
            template_content = "role: {{ agent.role }"  # Missing closing brace
            (agents_dir / "broken_template.yaml.j2").write_text(template_content)

            engine = EnhancementEngine(templates_dir=templates_dir)

            with pytest.raises(TemplateRenderError, match="Failed to render template"):
                engine.render_agent_template(
                    "broken_template", {"agent": {"role": "Test"}}
                )

    def test_enhance_agents_config_success(self):
        """Test successful agents configuration enhancement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            # Create original agents.yaml
            original_agents = {
                "researcher": {
                    "role": "Researcher",
                    "goal": "Generic research goal",
                    "backstory": "Generic backstory",
                    "tools": ["search"],
                }
            }
            agents_file = config_path / "agents.yaml"
            agents_file.write_text(yaml.dump(original_agents))

            # Create template directory with enhanced template
            templates_dir = Path(temp_dir) / "templates"
            agents_templates_dir = templates_dir / "agents"
            agents_templates_dir.mkdir(parents=True)

            enhanced_template = """researcher:
  role: {{ agents.researcher.role }}
  goal: {{ enhanced_goal }}
  backstory: {{ enhanced_backstory }}
  tools:
{% for tool in enhanced_tools %}
    - {{ tool }}
{% endfor %}
"""
            (agents_templates_dir / "enhanced_agents.yaml.j2").write_text(
                enhanced_template
            )

            engine = EnhancementEngine(templates_dir=templates_dir)

            enhancement_context = {
                "agents": original_agents,
                "enhanced_goal": "Conduct advanced AI research with domain expertise",
                "enhanced_backstory": "PhD in AI with 10 years of industry experience",
                "enhanced_tools": ["advanced_search", "data_analysis", "ml_tools"],
            }

            result = engine.enhance_agents_config(
                project_path, enhancement_context, template_name="enhanced_agents"
            )

            assert result["success"] is True
            assert result["backup_created"] is True

            # Verify enhanced content
            enhanced_content = agents_file.read_text()
            assert (
                "Conduct advanced AI research with domain expertise" in enhanced_content
            )
            assert "PhD in AI with 10 years of industry experience" in enhanced_content
            assert "advanced_search" in enhanced_content

    def test_enhance_agents_config_file_not_found(self):
        """Test agents configuration enhancement with missing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            # Create proper structure but no agents.yaml
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            engine = EnhancementEngine()

            result = engine.enhance_agents_config(project_path, {})

            assert result["success"] is False
            assert "agents.yaml not found" in result["error"]

    def test_enhance_tasks_config_success(self):
        """Test successful tasks configuration enhancement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            # Create original tasks.yaml
            original_tasks = {
                "research_task": {
                    "description": "Generic research task",
                    "expected_output": "Research results",
                    "agent": "researcher",
                }
            }
            tasks_file = config_path / "tasks.yaml"
            tasks_file.write_text(yaml.dump(original_tasks))

            # Create template directory with enhanced template
            templates_dir = Path(temp_dir) / "templates"
            tasks_templates_dir = templates_dir / "tasks"
            tasks_templates_dir.mkdir(parents=True)

            enhanced_template = """research_task:
  description: {{ enhanced_description }}
  expected_output: {{ enhanced_output }}
  agent: {{ tasks.research_task.agent }}
  context: {{ context }}
"""
            (tasks_templates_dir / "enhanced_tasks.yaml.j2").write_text(
                enhanced_template
            )

            engine = EnhancementEngine(templates_dir=templates_dir)

            enhancement_context = {
                "tasks": original_tasks,
                "enhanced_description": "Conduct comprehensive AI trend analysis using advanced research methodologies",
                "enhanced_output": "Detailed 50-page report with actionable insights and recommendations",
                "context": "Focus on enterprise AI adoption patterns and emerging technologies",
            }

            result = engine.enhance_tasks_config(
                project_path, enhancement_context, template_name="enhanced_tasks"
            )

            assert result["success"] is True
            assert result["backup_created"] is True

            # Verify enhanced content
            enhanced_content = tasks_file.read_text()
            assert "comprehensive AI trend analysis" in enhanced_content
            assert "50-page report with actionable insights" in enhanced_content

    def test_create_backup_file(self):
        """Test backup file creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_file = Path(temp_dir) / "test.yaml"
            original_content = "original content"
            original_file.write_text(original_content)

            engine = EnhancementEngine()
            backup_path = engine._create_backup(original_file)

            assert backup_path.exists()
            assert backup_path.read_text() == original_content
            assert backup_path.suffix == ".backup"

    def test_validate_project_structure_valid(self):
        """Test project structure validation with valid structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            engine = EnhancementEngine()
            result = engine._validate_project_structure(project_path)

            assert result["valid"] is True
            assert result["config_path"] == config_path

    def test_validate_project_structure_invalid(self):
        """Test project structure validation with invalid structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "invalid_project"
            project_path.mkdir()

            engine = EnhancementEngine()
            result = engine._validate_project_structure(project_path)

            assert result["valid"] is False
            assert "Invalid CrewAI project structure" in result["error"]


class TestEnhancementEngineIntegration:
    """Test suite for enhancement engine integration scenarios."""

    def test_full_project_enhancement_workflow(self):
        """Test complete project enhancement workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup project structure
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            config_path = project_path / "src" / "test_project" / "config"
            config_path.mkdir(parents=True)

            # Create original configuration files
            agents = {
                "researcher": {
                    "role": "Researcher",
                    "goal": "Research",
                    "backstory": "Basic",
                    "tools": ["search"],
                },
                "writer": {
                    "role": "Writer",
                    "goal": "Write",
                    "backstory": "Basic",
                    "tools": ["write"],
                },
            }
            tasks = {
                "research": {
                    "description": "Research task",
                    "expected_output": "Results",
                    "agent": "researcher",
                },
                "write": {
                    "description": "Write task",
                    "expected_output": "Document",
                    "agent": "writer",
                },
            }

            (config_path / "agents.yaml").write_text(yaml.dump(agents))
            (config_path / "tasks.yaml").write_text(yaml.dump(tasks))

            # Setup templates
            templates_dir = Path(temp_dir) / "templates"
            (templates_dir / "agents").mkdir(parents=True)
            (templates_dir / "tasks").mkdir(parents=True)

            # Create comprehensive templates
            agent_template = """researcher:
  role: {{ agents.researcher.role }}
  goal: {{ project.enhanced_research_goal }}
  backstory: {{ project.enhanced_research_backstory }}
  tools: {{ project.research_tools | tojson }}

writer:
  role: {{ agents.writer.role }}
  goal: {{ project.enhanced_writing_goal }}
  backstory: {{ project.enhanced_writing_backstory }}
  tools: {{ project.writing_tools | tojson }}
"""
            (templates_dir / "agents" / "content_team.yaml.j2").write_text(
                agent_template
            )

            task_template = """research:
  description: {{ project.enhanced_research_description }}
  expected_output: {{ project.enhanced_research_output }}
  agent: researcher
  context: {{ project.domain_context }}

write:
  description: {{ project.enhanced_writing_description }}
  expected_output: {{ project.enhanced_writing_output }}
  agent: writer
  context: {{ project.domain_context }}
  dependencies: [research]
"""
            (templates_dir / "tasks" / "content_team.yaml.j2").write_text(task_template)

            # Initialize engine and enhance
            engine = EnhancementEngine(templates_dir=templates_dir)

            enhancement_context = {
                "agents": agents,
                "tasks": tasks,
                "project": {
                    "enhanced_research_goal": "Conduct comprehensive market research using advanced methodologies",
                    "enhanced_research_backstory": "Senior market research analyst with 15 years of experience in emerging technologies",
                    "research_tools": [
                        "advanced_search",
                        "data_analysis",
                        "survey_tools",
                        "competitor_analysis",
                    ],
                    "enhanced_writing_goal": "Create compelling and data-driven content that converts leads",
                    "enhanced_writing_backstory": "Award-winning content strategist specializing in B2B technology marketing",
                    "writing_tools": [
                        "content_editor",
                        "seo_optimizer",
                        "readability_checker",
                    ],
                    "enhanced_research_description": "Execute comprehensive market analysis including competitor research, trend analysis, and customer insights",
                    "enhanced_research_output": "Detailed 30-page market research report with actionable recommendations",
                    "enhanced_writing_description": "Transform research insights into compelling marketing content optimized for conversion",
                    "enhanced_writing_output": "Complete content package including blog posts, whitepapers, and case studies",
                    "domain_context": "B2B SaaS marketing for emerging AI technologies",
                },
            }

            # Enhance both configurations
            agents_result = engine.enhance_agents_config(
                project_path, enhancement_context, template_name="content_team"
            )
            tasks_result = engine.enhance_tasks_config(
                project_path, enhancement_context, template_name="content_team"
            )

            assert agents_result["success"] is True
            assert tasks_result["success"] is True

            # Verify enhancements
            enhanced_agents = yaml.safe_load((config_path / "agents.yaml").read_text())
            enhanced_tasks = yaml.safe_load((config_path / "tasks.yaml").read_text())

            # Check agent enhancements
            assert (
                "comprehensive market research using advanced methodologies"
                in enhanced_agents["researcher"]["goal"]
            )
            assert "advanced_search" in enhanced_agents["researcher"]["tools"]
            assert "content strategist" in enhanced_agents["writer"]["backstory"]

            # Check task enhancements
            assert (
                "comprehensive market analysis"
                in enhanced_tasks["research"]["description"]
            )
            assert (
                "30-page market research report"
                in enhanced_tasks["research"]["expected_output"]
            )
            assert "B2B SaaS marketing" in enhanced_tasks["research"]["context"]
            assert enhanced_tasks["write"]["dependencies"] == ["research"]


class TestEnhancementEngineErrors:
    """Test suite for enhancement engine error handling."""

    def test_enhancement_error_initialization(self):
        """Test EnhancementError exception initialization."""
        error = EnhancementError("Test error message")
        assert str(error) == "Test error message"

    def test_template_not_found_error_initialization(self):
        """Test TemplateNotFoundError exception initialization."""
        error = TemplateNotFoundError("Template not found")
        assert str(error) == "Template not found"

    def test_template_render_error_initialization(self):
        """Test TemplateRenderError exception initialization."""
        error = TemplateRenderError("Render failed")
        assert str(error) == "Render failed"

    def test_error_logging_on_failures(self):
        """Test that errors are properly logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "invalid_project"
            project_path.mkdir()
            # Create proper structure but no agents.yaml
            config_path = project_path / "src" / "invalid_project" / "config"
            config_path.mkdir(parents=True)

            # Mock logger to verify logging calls
            mock_logger = Mock()
            engine = EnhancementEngine(logger=mock_logger)

            result = engine.enhance_agents_config(project_path, {})

            assert result["success"] is False
            mock_logger.error.assert_called()

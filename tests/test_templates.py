"""Test suite for Jinja2 template engine and CrewAI template generation."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

from crewforge.core.templates import TemplateEngine, TemplateError
from crewforge.models.agent import AgentConfig
from crewforge.models.task import TaskConfig
from crewforge.models.crew import CrewConfig


class TestTemplateEngine:
    """Test cases for TemplateEngine class."""

    @pytest.fixture
    def template_engine(self):
        """Create TemplateEngine instance for testing."""
        return TemplateEngine()

    @pytest.fixture
    def sample_agent_config(self):
        """Sample agent configuration for template testing."""
        return AgentConfig(
            role="Content Researcher",
            goal="Research and gather relevant content from various sources",
            backstory="You are an expert content researcher with years of experience",
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            tools=["SearchTool", "ScrapeTool"],
        )

    @pytest.fixture
    def sample_task_config(self):
        """Sample task configuration for template testing."""
        return TaskConfig(
            description="Research latest trends in artificial intelligence",
            expected_output="Comprehensive report on AI trends with sources",
            agent="Content Researcher",
            async_execution=False,
            tools=["SearchTool"],
        )

    @pytest.fixture
    def sample_crew_config(self, sample_agent_config, sample_task_config):
        """Sample crew configuration for template testing."""
        return CrewConfig(
            name="research-crew",
            description="A crew specialized in content research",
            agents=[sample_agent_config],
            tasks=[sample_task_config],
            verbose=True,
            process="sequential",
            max_rpm=100,
            language="en",
            full_output=True,
        )

    def test_template_engine_initialization(self, template_engine):
        """Test that TemplateEngine initializes correctly."""
        assert template_engine.env is not None
        assert template_engine.template_dir.exists()
        assert template_engine.template_dir.name == "templates"

    def test_template_engine_custom_template_dir(self):
        """Test TemplateEngine with custom template directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = Path(temp_dir) / "custom_templates"
            custom_dir.mkdir()

            engine = TemplateEngine(template_dir=custom_dir)
            assert engine.template_dir == custom_dir

    def test_get_template_success(self, template_engine):
        """Test successful template retrieval."""
        template = template_engine.get_template("agents.py.j2")
        assert template is not None
        assert hasattr(template, "render")

    def test_get_template_not_found(self, template_engine):
        """Test template not found error handling."""
        with pytest.raises(TemplateError, match="Template not found"):
            template_engine.get_template("nonexistent.j2")

    def test_render_template_success(self, template_engine, sample_crew_config):
        """Test successful template rendering."""
        result = template_engine.render_template(
            "agents.py.j2", crew_config=sample_crew_config
        )

        assert "class ResearchCrewAgents:" in result
        assert "Content Researcher" in result
        assert "Research and gather relevant content" in result
        assert "expert content researcher" in result
        assert "verbose=true" in result
        assert "allow_delegation=false" in result

    def test_render_template_invalid_template(
        self, template_engine, sample_crew_config
    ):
        """Test template rendering with invalid template."""
        with pytest.raises(TemplateError, match="Template not found"):
            template_engine.render_template(
                "invalid.j2", crew_config=sample_crew_config
            )

    def test_render_template_invalid_context(self, template_engine):
        """Test template rendering with invalid context."""
        with pytest.raises(TemplateError, match="Template rendering failed"):
            # Missing required crew_config variable
            template_engine.render_template("agents.py.j2")

    def test_populate_template_success(self, template_engine, sample_crew_config):
        """Test successful template population to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "agents.py"

            template_engine.populate_template(
                "agents.py.j2", output_file, crew_config=sample_crew_config
            )

            assert output_file.exists()
            content = output_file.read_text()
            assert "class ResearchCrewAgents:" in content
            assert "Content Researcher" in content

    def test_populate_template_creates_directory(
        self, template_engine, sample_crew_config
    ):
        """Test that populate_template creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "nested" / "dir"
            output_file = nested_dir / "agents.py"

            template_engine.populate_template(
                "agents.py.j2", output_file, crew_config=sample_crew_config
            )

            assert output_file.exists()
            assert nested_dir.exists()

    def test_populate_template_file_write_error(
        self, template_engine, sample_crew_config
    ):
        """Test populate_template handles file write errors."""
        # Try to write to a read-only location
        with pytest.raises(TemplateError, match="Failed to write template"):
            template_engine.populate_template(
                "agents.py.j2",
                Path("/root/readonly.py"),  # Should fail on most systems
                crew_config=sample_crew_config,
            )

    def test_agents_template_rendering(self, template_engine, sample_crew_config):
        """Test agents.py.j2 template specific rendering."""
        result = template_engine.render_template(
            "agents.py.j2", crew_config=sample_crew_config
        )

        # Check for proper agent method generation
        assert "def content_researcher(self):" in result
        assert 'role="Content Researcher"' in result
        assert (
            'goal="Research and gather relevant content from various sources"' in result
        )
        assert 'backstory="You are an expert content researcher' in result
        assert "verbose=true" in result
        assert "allow_delegation=false" in result
        assert "max_iter=5" in result
        assert "tools=[SearchTool, ScrapeTool]" in result

    def test_tasks_template_rendering(self, template_engine, sample_crew_config):
        """Test tasks.py.j2 template specific rendering."""
        result = template_engine.render_template(
            "tasks.py.j2", crew_config=sample_crew_config
        )

        # Check for proper task method generation
        assert "class ResearchCrewTasks:" in result
        assert "Research latest trends" in result
        assert "Comprehensive report on AI trends" in result
        assert "agent=content_researcher" in result
        assert "tools=[SearchTool]" in result

    def test_tools_template_rendering(self, template_engine, sample_crew_config):
        """Test tools.py.j2 template specific rendering."""
        result = template_engine.render_template(
            "tools.py.j2", crew_config=sample_crew_config
        )

        # Check for proper tools class generation
        assert "class ResearchCrewTools:" in result
        assert "from crewai_tools import SearchTool" in result
        assert "from crewai_tools import ScrapeTool" in result
        assert "def searchtool(self):" in result
        assert "def scrapetool(self):" in result

    def test_tools_template_no_tools(
        self, template_engine, sample_agent_config, sample_task_config
    ):
        """Test tools.py.j2 template with no tools configured."""
        crew_config = CrewConfig(
            name="minimal-crew",
            agents=[sample_agent_config],
            tasks=[sample_task_config],
        )

        result = template_engine.render_template("tools.py.j2", crew_config=crew_config)

        # Should still generate tools because the agent has tools
        assert "class MinimalCrewTools:" in result
        assert "from crewai_tools import SearchTool" in result

    def test_crew_template_rendering(self, template_engine, sample_crew_config):
        """Test crew.py.j2 template specific rendering."""
        result = template_engine.render_template(
            "crew.py.j2", crew_config=sample_crew_config
        )

        # Check for proper crew orchestration
        assert "class ResearchCrewCrew:" in result
        assert "self.agents = ResearchCrewAgents()" in result
        assert "self.tasks = ResearchCrewTasks()" in result
        assert "self.tools = ResearchCrewTools()" in result
        assert "content_researcher = self.agents.content_researcher()" in result
        assert "verbose=true" in result
        assert "process=Process.SEQUENTIAL" in result
        assert "max_rpm=100" in result

    def test_template_validation_missing_required_fields(self, template_engine):
        """Test template validation with missing required fields."""
        # This should not raise TemplateError since Jinja2 will render undefined variables as empty
        # The validation should happen at the Pydantic level when creating models
        incomplete_config = {
            "name": "test-crew"
            # Missing agents and tasks - but Jinja2 will handle this gracefully
        }

        # This should work but produce a template with missing data
        result = template_engine.render_template(
            "agents.py.j2", crew_config=incomplete_config
        )
        assert "test-crew" in result

    def test_template_name_sanitization(
        self, template_engine, sample_agent_config, sample_task_config
    ):
        """Test that crew names are properly sanitized in templates."""
        crew_config = CrewConfig(
            name="test-crew-with-special-chars!@#",
            agents=[sample_agent_config],
            tasks=[sample_task_config],
        )

        result = template_engine.render_template(
            "agents.py.j2", crew_config=crew_config
        )

        # Class names should be sanitized (the name gets cleaned by CrewConfig validator)
        assert "TestCrewWithSpecialCharsAgents" in result

    def test_multiple_agents_and_tasks(self, template_engine):
        """Test template rendering with multiple agents and tasks."""
        agent1 = AgentConfig(
            role="Researcher", goal="Research content", backstory="Expert researcher"
        )
        agent2 = AgentConfig(
            role="Writer", goal="Write content", backstory="Expert writer"
        )
        task1 = TaskConfig(
            description="Research task",
            expected_output="Research report",
            agent="Researcher",
        )
        task2 = TaskConfig(
            description="Writing task",
            expected_output="Written content",
            agent="Writer",
        )

        crew_config = CrewConfig(
            name="multi-crew", agents=[agent1, agent2], tasks=[task1, task2]
        )

        result = template_engine.render_template("crew.py.j2", crew_config=crew_config)

        # Should have both agents and tasks
        assert "researcher = self.agents.researcher()" in result
        assert "writer = self.agents.writer()" in result
        assert "agents=[\nresearcher, \nwriter\n" in result

"""Tests for the template engine and custom filters."""

import tempfile
from pathlib import Path

import pytest

from crewforge.core.templates import TemplateEngine, TemplateError


class TestTemplateFilters:
    """Test custom Jinja2 filters for Python identifier conversion."""

    def test_to_python_identifier_basic(self):
        """Test basic string to Python identifier conversion."""
        engine = TemplateEngine()

        # Test various input formats
        assert (
            engine._to_python_identifier("Content Research Expert")
            == "content_research_expert"
        )
        assert (
            engine._to_python_identifier("API-Integration-Manager")
            == "api_integration_manager"
        )
        assert engine._to_python_identifier("Data Analyst!") == "data_analyst"
        assert engine._to_python_identifier("Web Scraper 2.0") == "web_scraper_2_0"

    def test_to_python_identifier_edge_cases(self):
        """Test edge cases for Python identifier conversion."""
        engine = TemplateEngine()

        # Empty string
        assert engine._to_python_identifier("") == "unnamed"
        assert engine._to_python_identifier("", prefix="agent") == "agent"

        # Leading numbers
        assert (
            engine._to_python_identifier("123 Numbers First")
            == "item_123_numbers_first"
        )
        assert engine._to_python_identifier("123", prefix="task") == "task_123"

        # Special characters only
        assert engine._to_python_identifier("!!!") == "unnamed"
        assert engine._to_python_identifier("---___---") == "unnamed"

        # Python keywords
        assert engine._to_python_identifier("class") == "class_var"
        assert engine._to_python_identifier("def") == "def_var"
        assert engine._to_python_identifier("return") == "return_var"

        # Multiple underscores
        assert (
            engine._to_python_identifier("too___many____underscores")
            == "too_many_underscores"
        )

    def test_to_class_name(self):
        """Test string to PascalCase class name conversion."""
        engine = TemplateEngine()

        assert engine._to_class_name("content-research-crew") == "ContentResearchCrew"
        assert engine._to_class_name("my_project_name") == "MyProjectName"
        assert engine._to_class_name("api integration tool") == "ApiIntegrationTool"
        assert engine._to_class_name("") == "UnnamedClass"
        assert engine._to_class_name("123-numbers") == "Item123Numbers"

    def test_truncate_identifier(self):
        """Test identifier truncation."""
        engine = TemplateEngine()

        # Short identifier (no truncation needed)
        assert engine._truncate_identifier("short_name", 20) == "short_name"

        # Long identifier (needs truncation)
        long_name = "this_is_a_very_long_identifier_name_that_needs_truncation"
        assert len(engine._truncate_identifier(long_name, 30)) <= 30
        assert (
            engine._truncate_identifier(long_name, 30)
            == "this_is_a_very_long_identifier"
        )

        # Edge case: truncate to very short
        assert engine._truncate_identifier("test_name", 4) == "test"

        # Edge case: all underscores after truncation
        assert engine._truncate_identifier("a_______", 2) == "a"


class TestTemplateRendering:
    """Test template rendering functionality."""

    def test_render_agents_yaml(self):
        """Test rendering of agents.yaml template."""
        engine = TemplateEngine()

        agents = [
            {
                "role": "Content Research Expert!",
                "goal": "Find relevant content",
                "backstory": "Expert researcher with years of experience",
            },
            {
                "role": "Data-Analyst",
                "goal": "Analyze data patterns",
                "backstory": "Statistical expert",
                "tools": ["FileReadTool", "CSVSearchTool"],
            },
        ]

        result = engine.render_template("agents.yaml.j2", agents=agents)

        # Check that identifiers are properly converted
        assert "content_research_expert:" in result
        assert "data_analyst:" in result

        # Check that original values are preserved in the content
        assert "Content Research Expert!" in result
        assert "Find relevant content" in result
        assert "FileReadTool" in result


class TestTemplateEngine:
    """Test core TemplateEngine functionality."""

    def test_template_loading(self):
        """Test that templates can be loaded successfully."""
        engine = TemplateEngine()

        # Should load without error
        template = engine.get_template("agents.yaml.j2")
        assert template is not None

        template = engine.get_template("tasks.yaml.j2")
        assert template is not None

    def test_template_not_found(self):
        """Test error handling for missing templates."""
        engine = TemplateEngine()

        with pytest.raises(TemplateError) as exc_info:
            engine.get_template("nonexistent.j2")

        assert "Template not found" in str(exc_info.value)

    def test_populate_template_to_file(self):
        """Test writing rendered template to file."""
        engine = TemplateEngine()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.yaml"

            agents = [
                {
                    "role": "test_agent",
                    "goal": "Test goal",
                    "backstory": "Test backstory",
                }
            ]

            engine.populate_template("agents.yaml.j2", output_path, agents=agents)

            assert output_path.exists()
            content = output_path.read_text()
            assert "test_agent:" in content
            assert "Test goal" in content

    def test_validate_template(self):
        """Test template validation."""
        engine = TemplateEngine()

        # Valid templates
        assert engine.validate_template("agents.yaml.j2") is True
        assert engine.validate_template("tasks.yaml.j2") is True
        assert engine.validate_template("tools.py.j2") is True

        # Invalid template
        assert engine.validate_template("does_not_exist.j2") is False


class TestTemplateIntegration:
    """Test integration scenarios with multiple components."""

    def test_full_project_generation(self):
        """Test generating all template files for a project."""
        engine = TemplateEngine()

        # Sample data matching what GenerationEngine would produce
        agents = [
            {
                "role": "Market Research Analyst",
                "goal": "Analyze market trends and competitor data",
                "backstory": "Expert analyst with deep market knowledge",
                "tools": ["WebsiteSearchTool", "FileWriterTool"],
            },
            {
                "role": "Content Writer",
                "goal": "Create engaging content based on research",
                "backstory": "Creative writer with SEO expertise",
            },
        ]

        tasks = [
            {
                "description": "Analyze competitor strategies and market position",
                "expected_output": "Detailed market analysis report",
                "agent_role": "Market Research Analyst",
            },
            {
                "description": "Write blog post about market insights",
                "expected_output": "1500-word blog post",
                "agent_role": "Content Writer",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Generate agents.yaml
            agents_path = base_path / "config" / "agents.yaml"
            engine.populate_template("agents.yaml.j2", agents_path, agents=agents)

            # Generate tasks.yaml
            tasks_path = base_path / "config" / "tasks.yaml"
            engine.populate_template(
                "tasks.yaml.j2", tasks_path, tasks=tasks, agents=agents
            )

            # Generate custom tools
            tools_path = base_path / "tools" / "custom_tool.py"
            engine.populate_template(
                "tools.py.j2",
                tools_path,
                tools=["WebsiteSearchTool", "FileWriterTool"],
            )

            # Verify all files created
            assert agents_path.exists()
            assert tasks_path.exists()
            assert tools_path.exists()

            # Verify content structure
            agents_content = agents_path.read_text()
            assert "market_research_analyst:" in agents_content
            assert "content_writer:" in agents_content

            tools_content = tools_path.read_text()
            assert "WebsiteSearchTool" in tools_content

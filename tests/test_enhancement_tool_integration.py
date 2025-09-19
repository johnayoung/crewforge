"""
Integration tests for tool patterns with enhancement engine.

Tests the integration between the ToolPatternRegistry and EnhancementEngine
to ensure tools are correctly applied to projects.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from crewforge.enhancement import EnhancementEngine
from crewforge.tool_patterns import ToolPatternRegistry


class TestEnhancementEngineToolIntegration:
    """Test suite for tool pattern integration with enhancement engine."""

    @pytest.fixture
    def temp_templates_dir(self):
        """Create temporary templates directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create template structure
        agents_dir = temp_dir / "agents"
        tasks_dir = temp_dir / "tasks"
        agents_dir.mkdir(parents=True)
        tasks_dir.mkdir(parents=True)

        # Create basic templates
        default_agent_template = """# Test Agent Template
{% if agents %}
{% for agent in agents %}
{{ agent.role | lower | replace(' ', '_') }}:
  role: {{ agent.role }}
  goal: {{ agent.goal }}
  backstory: {{ agent.backstory }}
  tools:
    {% if all_tools %}
    {% for tool in all_tools %}
    - {{ tool }}
    {% endfor %}
    {% else %}
    - default_tool
    {% endif %}
{% endfor %}
{% endif %}"""

        (agents_dir / "default.yaml.j2").write_text(default_agent_template)

        default_task_template = """# Test Task Template
test_task:
  description: Test task
  expected_output: Test output
  tools:
    {% if all_tools %}
    {% for tool in all_tools %}
    - {{ tool }}
    {% endfor %}
    {% else %}
    - default_tool
    {% endif %}"""

        (tasks_dir / "default.yaml.j2").write_text(default_task_template)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary CrewAI project directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create project structure
        project_name = "test_project"
        src_dir = temp_dir / "src" / project_name
        config_dir = src_dir / "config"
        tools_dir = src_dir / "tools"

        src_dir.mkdir(parents=True)
        config_dir.mkdir(parents=True)
        tools_dir.mkdir(parents=True)

        # Create basic files
        (config_dir / "agents.yaml").write_text("# Initial agents config")
        (config_dir / "tasks.yaml").write_text("# Initial tasks config")
        (src_dir / "crew.py").write_text("# Initial crew config")
        (tools_dir / "__init__.py").write_text("# Initial tools")

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def enhancement_engine(self, temp_templates_dir):
        """Create enhancement engine with temporary templates."""
        return EnhancementEngine(templates_dir=temp_templates_dir)

    def test_tool_registry_initialization(self, enhancement_engine):
        """Test that enhancement engine initializes tool registry."""
        assert hasattr(enhancement_engine, "tool_registry")
        assert isinstance(enhancement_engine.tool_registry, ToolPatternRegistry)

    def test_detect_project_type_and_get_tools(self, enhancement_engine):
        """Test project type detection and tool retrieval."""
        project_spec = {
            "description": "Research project for market analysis",
            "project_name": "market_research",
        }

        result = enhancement_engine.detect_project_type_and_get_tools(project_spec)

        assert "project_type" in result
        assert "tools" in result
        assert "pattern_info" in result
        assert "tool_names" in result
        assert "tool_imports" in result
        assert "tool_dependencies" in result

        assert result["project_type"] == "research"
        assert len(result["tools"]) > 0
        assert "SerperDevTool" in result["tool_names"]

    def test_generate_tool_configuration(self, enhancement_engine):
        """Test tool configuration generation."""
        project_spec = {
            "description": "Content creation project",
            "project_name": "content_team",
        }

        config = enhancement_engine.generate_tool_configuration(project_spec)

        assert config["project_type"] == "content"
        assert "all_tools" in config
        assert "tool_imports" in config
        assert "tool_descriptions" in config
        assert "tool_dependencies" in config
        assert "tools_by_context" in config
        assert "research_tools" in config  # Content project specific
        assert "writing_tools" in config  # Content project specific

        assert len(config["all_tools"]) > 0
        assert isinstance(config["tool_imports"], dict)
        assert isinstance(config["tool_descriptions"], dict)

    def test_generate_tool_configuration_data_analysis(self, enhancement_engine):
        """Test tool configuration for data analysis project."""
        project_spec = {
            "description": "Data analytics and reporting project",
            "project_name": "analytics_team",
        }

        config = enhancement_engine.generate_tool_configuration(project_spec)

        assert config["project_type"] == "data_analysis"
        assert "data_tools" in config
        assert "analysis_tools" in config

        # Check for data-specific tools
        assert any("CSV" in tool for tool in config["all_tools"])
        assert any("Excel" in tool for tool in config["all_tools"])

    async def test_enhance_project_with_generated_agents_includes_tools(
        self, enhancement_engine, temp_project_dir
    ):
        """Test that agent enhancement includes tool configuration."""
        project_spec = {
            "description": "Research team for market analysis",
            "project_name": "test_project",
            "agents": [
                {"role": "Market Researcher", "description": "Research market trends"}
            ],
        }

        # Mock LLM client
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = """{"role": "Market Researcher", "goal": "Research market trends", "backstory": "Expert researcher"}"""

        result = await enhancement_engine.enhance_project_with_generated_agents(
            temp_project_dir, project_spec, mock_llm
        )

        assert result is True

        # Check that agents.yaml was updated
        agents_file = (
            temp_project_dir / "src" / "test_project" / "config" / "agents.yaml"
        )
        content = agents_file.read_text()

        # Should contain tool references
        assert "SerperDevTool" in content or "WebsiteSearchTool" in content

    def test_generate_tools_file(self, enhancement_engine, temp_project_dir):
        """Test generation of tools/__init__.py file."""
        project_spec = {
            "description": "Content creation project",
            "project_name": "test_project",
        }

        result = enhancement_engine.generate_tools_file(temp_project_dir, project_spec)

        assert result["success"] is True
        assert "tools_file" in result
        assert "tool_count" in result
        assert "dependencies" in result

        tools_file = result["tools_file"]
        assert tools_file.exists()

        content = tools_file.read_text()
        assert "from crewai_tools" in content
        assert "tools_registry" in content
        assert "get_tools_for_agent" in content
        assert "get_tool" in content

    def test_enhance_project_with_tools(self, enhancement_engine, temp_project_dir):
        """Test complete project enhancement with tools."""
        project_spec = {
            "description": "Data analysis project",
            "project_name": "test_project",
        }

        result = enhancement_engine.enhance_project_with_tools(
            temp_project_dir, project_spec, update_crew_file=True
        )

        assert result["success"] is True
        assert result["tools_generated"] is True
        assert result["tool_count"] > 0
        assert len(result["dependencies"]) > 0

        # Check that tools file exists
        tools_file = temp_project_dir / "src" / "test_project" / "tools" / "__init__.py"
        assert tools_file.exists()

        content = tools_file.read_text()
        assert "CSVSearchTool" in content  # Data analysis tools
        assert "ExcelSearchTool" in content

    def test_update_crew_file_with_tools(self, enhancement_engine, temp_project_dir):
        """Test updating crew.py with tools import."""
        project_spec = {
            "description": "General project",
            "project_name": "test_project",
        }

        # First generate tools
        tools_result = enhancement_engine.generate_tools_file(
            temp_project_dir, project_spec
        )
        assert tools_result["success"] is True

        # Then update crew file
        crew_result = enhancement_engine._update_crew_file_with_tools(
            temp_project_dir, project_spec
        )

        assert crew_result["success"] is True

        crew_file = temp_project_dir / "src" / "test_project" / "crew.py"
        content = crew_file.read_text()

        assert "from .tools import get_tools_for_agent, get_tool" in content

    def test_tool_configuration_different_project_types(self, enhancement_engine):
        """Test that different project types get appropriate tool configurations."""
        test_cases = [
            ("research project", "research", ["SerperDevTool", "WebsiteSearchTool"]),
            ("content creation blog", "content", ["FileReadTool", "DirectoryReadTool"]),
            (
                "data analytics dashboard",
                "data_analysis",
                ["CSVSearchTool", "ExcelSearchTool"],
            ),
            (
                "customer support system",
                "customer_service",
                ["FileReadTool", "DirectorySearchTool"],
            ),
            (
                "software development",
                "development",
                ["FileReadTool", "GithubSearchTool"],
            ),
        ]

        for description, expected_type, expected_tools in test_cases:
            project_spec = {
                "description": description,
                "project_name": f"{expected_type}_project",
            }

            config = enhancement_engine.generate_tool_configuration(project_spec)

            assert (
                config["project_type"] == expected_type
            ), f"Expected {expected_type} for '{description}'"

            for tool in expected_tools:
                assert (
                    tool in config["all_tools"]
                ), f"Expected {tool} in tools for {expected_type}"

    def test_tool_configuration_handles_empty_project_spec(self, enhancement_engine):
        """Test tool configuration with minimal project specification."""
        project_spec = {}

        config = enhancement_engine.generate_tool_configuration(project_spec)

        assert config["project_type"] == "general"
        assert len(config["all_tools"]) > 0
        assert "SerperDevTool" in config["all_tools"]

    def test_error_handling_invalid_project_path(self, enhancement_engine):
        """Test error handling for invalid project paths."""
        invalid_path = Path("/nonexistent/path")
        project_spec = {"description": "test project"}

        result = enhancement_engine.generate_tools_file(invalid_path, project_spec)

        assert result["success"] is False
        assert "error" in result

    def test_backup_creation_during_enhancement(
        self, enhancement_engine, temp_project_dir
    ):
        """Test that backups are created during enhancement."""
        project_spec = {"description": "Test project", "project_name": "test_project"}

        # Add some initial content to tools file
        tools_file = temp_project_dir / "src" / "test_project" / "tools" / "__init__.py"
        tools_file.write_text("# Original content")

        result = enhancement_engine.generate_tools_file(temp_project_dir, project_spec)

        assert result["success"] is True
        assert result["backup_created"] is True
        assert "backup_file" in result

        backup_file = result["backup_file"]
        assert backup_file.exists()
        assert "Original content" in backup_file.read_text()

    def test_tool_dependencies_aggregation(self, enhancement_engine):
        """Test that tool dependencies are properly aggregated."""
        project_spec = {
            "description": "Research and content creation project",
            "project_name": "mixed_project",
        }

        config = enhancement_engine.generate_tool_configuration(project_spec)

        # Should aggregate all unique dependencies
        dependencies = config["tool_dependencies"]
        assert isinstance(dependencies, list)
        assert len(dependencies) > 0

        # Check for specific dependencies that should be present
        dependency_patterns = ["crewai-tools[", "crewai-tools"]
        assert any(
            any(pattern in dep for pattern in dependency_patterns)
            for dep in dependencies
        )

    def test_tools_by_context_grouping(self, enhancement_engine):
        """Test that tools are properly grouped by usage context."""
        project_spec = {
            "description": "Data analysis research project",
            "project_name": "context_test",
        }

        config = enhancement_engine.generate_tool_configuration(project_spec)

        tools_by_context = config["tools_by_context"]
        assert isinstance(tools_by_context, dict)
        assert len(tools_by_context) > 0

        # Should have different contexts
        contexts = list(tools_by_context.keys())
        assert len(contexts) > 0

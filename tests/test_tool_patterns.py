"""
Test suite for tool integration patterns module.

Tests the ToolPatternRegistry functionality including pattern detection,
tool selection, and integration with project specifications.
"""

import pytest
from unittest.mock import Mock, patch
from crewforge.tool_patterns import (
    ToolPattern,
    ProjectTypePattern,
    ToolPatternRegistry,
)


class TestToolPattern:
    """Test suite for ToolPattern dataclass."""

    def test_tool_pattern_creation(self):
        """Test creating a ToolPattern with all fields."""
        tool = ToolPattern(
            name="TestTool",
            import_path="test.tool.TestTool",
            description="A test tool",
            dependencies=["test-package"],
            config_params={"api_key": "TEST_KEY"},
            usage_context="testing",
        )

        assert tool.name == "TestTool"
        assert tool.import_path == "test.tool.TestTool"
        assert tool.description == "A test tool"
        assert tool.dependencies == ["test-package"]
        assert tool.config_params == {"api_key": "TEST_KEY"}
        assert tool.usage_context == "testing"

    def test_tool_pattern_defaults(self):
        """Test ToolPattern with minimal required fields."""
        tool = ToolPattern(
            name="MinimalTool", import_path="minimal.Tool", description="Minimal tool"
        )

        assert tool.name == "MinimalTool"
        assert tool.import_path == "minimal.Tool"
        assert tool.description == "Minimal tool"
        assert tool.dependencies == []
        assert tool.config_params == {}
        assert tool.usage_context == ""


class TestProjectTypePattern:
    """Test suite for ProjectTypePattern dataclass."""

    def test_project_type_pattern_creation(self):
        """Test creating a ProjectTypePattern with all fields."""
        tools = [
            ToolPattern("Tool1", "path.Tool1", "First tool"),
            ToolPattern("Tool2", "path.Tool2", "Second tool"),
        ]

        pattern = ProjectTypePattern(
            project_type="test_type",
            description="Test project type",
            tools=tools,
            agent_roles=["tester", "validator"],
            common_tasks=["run_tests", "validate"],
            keywords=["test", "validation"],
        )

        assert pattern.project_type == "test_type"
        assert pattern.description == "Test project type"
        assert len(pattern.tools) == 2
        assert pattern.agent_roles == ["tester", "validator"]
        assert pattern.common_tasks == ["run_tests", "validate"]
        assert pattern.keywords == ["test", "validation"]

    def test_project_type_pattern_defaults(self):
        """Test ProjectTypePattern with minimal fields."""
        pattern = ProjectTypePattern(
            project_type="minimal", description="Minimal pattern"
        )

        assert pattern.project_type == "minimal"
        assert pattern.description == "Minimal pattern"
        assert pattern.tools == []
        assert pattern.agent_roles == []
        assert pattern.common_tasks == []
        assert pattern.keywords == []


class TestToolPatternRegistry:
    """Test suite for ToolPatternRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh ToolPatternRegistry for each test."""
        return ToolPatternRegistry()

    def test_registry_initialization(self, registry):
        """Test that registry initializes with expected patterns."""
        patterns = registry.get_available_patterns()

        expected_patterns = [
            "research",
            "content",
            "data_analysis",
            "customer_service",
            "development",
            "general",
        ]

        for pattern in expected_patterns:
            assert pattern in patterns

    def test_get_pattern_existing(self, registry):
        """Test retrieving an existing pattern."""
        pattern = registry.get_pattern("research")

        assert pattern is not None
        assert pattern.project_type == "research"
        assert len(pattern.tools) > 0
        assert "researcher" in pattern.agent_roles
        assert "research" in pattern.keywords

    def test_get_pattern_non_existing(self, registry):
        """Test retrieving a non-existing pattern returns None."""
        pattern = registry.get_pattern("nonexistent")
        assert pattern is None

    def test_detect_project_type_research(self, registry):
        """Test detection of research project type."""
        project_spec = {
            "description": "Market research and competitive analysis project",
            "prompt": "Build a team to analyze market trends",
            "project_name": "market_research_team",
            "agents": [{"role": "researcher"}, {"role": "analyst"}],
            "tasks": [{"name": "gather_market_data"}],
        }

        detected_type = registry.detect_project_type(project_spec)
        assert detected_type == "research"

    def test_detect_project_type_content(self, registry):
        """Test detection of content project type."""
        project_spec = {
            "description": "Content creation team for blog writing and SEO",
            "prompt": "Create a content marketing team",
            "project_name": "content_creation",
            "agents": [{"role": "writer"}, {"role": "editor"}],
            "tasks": [{"name": "write_blog_posts"}],
        }

        detected_type = registry.detect_project_type(project_spec)
        assert detected_type == "content"

    def test_detect_project_type_data_analysis(self, registry):
        """Test detection of data analysis project type."""
        project_spec = {
            "description": "Data analytics and business intelligence dashboard",
            "prompt": "Build a data analysis team for reporting",
            "project_name": "analytics_dashboard",
            "agents": [{"role": "data_analyst"}],
            "tasks": [{"name": "analyze_sales_data"}],
        }

        detected_type = registry.detect_project_type(project_spec)
        assert detected_type == "data_analysis"

    def test_detect_project_type_customer_service(self, registry):
        """Test detection of customer service project type."""
        project_spec = {
            "description": "Customer support automation system",
            "prompt": "Create a helpdesk support team",
            "project_name": "customer_support",
            "agents": [{"role": "support_agent"}],
            "tasks": [{"name": "handle_customer_inquiries"}],
        }

        detected_type = registry.detect_project_type(project_spec)
        assert detected_type == "customer_service"

    def test_detect_project_type_development(self, registry):
        """Test detection of development project type."""
        project_spec = {
            "description": "Software development and code review team",
            "prompt": "Build a development team for technical documentation",
            "project_name": "dev_team",
            "agents": [{"role": "developer"}, {"role": "code_reviewer"}],
            "tasks": [{"name": "review_code_changes"}],
        }

        detected_type = registry.detect_project_type(project_spec)
        assert detected_type == "development"

    def test_detect_project_type_general_fallback(self, registry):
        """Test fallback to general type for unclear projects."""
        project_spec = {
            "description": "Some generic project",
            "prompt": "Create a basic team",
            "project_name": "generic_project",
        }

        detected_type = registry.detect_project_type(project_spec)
        assert detected_type == "general"

    def test_detect_project_type_empty_spec(self, registry):
        """Test detection with empty project specification."""
        project_spec = {}

        detected_type = registry.detect_project_type(project_spec)
        assert detected_type == "general"

    def test_get_tools_for_project_research(self, registry):
        """Test getting tools for research project."""
        project_spec = {
            "description": "Market research project for competitive analysis"
        }

        tools = registry.get_tools_for_project(project_spec)

        assert len(tools) > 0
        tool_names = [tool.name for tool in tools]
        assert "SerperDevTool" in tool_names
        assert "WebsiteSearchTool" in tool_names

    def test_get_tools_for_project_content(self, registry):
        """Test getting tools for content project."""
        project_spec = {"description": "Content creation and blog writing project"}

        tools = registry.get_tools_for_project(project_spec)

        assert len(tools) > 0
        tool_names = [tool.name for tool in tools]
        assert "SerperDevTool" in tool_names
        assert "FileReadTool" in tool_names

    def test_get_tools_for_project_data_analysis(self, registry):
        """Test getting tools for data analysis project."""
        project_spec = {
            "description": "Data analytics and business intelligence project"
        }

        tools = registry.get_tools_for_project(project_spec)

        assert len(tools) > 0
        tool_names = [tool.name for tool in tools]
        assert "CSVSearchTool" in tool_names
        assert "ExcelSearchTool" in tool_names

    def test_get_tools_for_project_general_fallback(self, registry):
        """Test getting tools for unrecognized project type."""
        project_spec = {"description": "Some random project"}

        tools = registry.get_tools_for_project(project_spec)

        assert len(tools) > 0
        # Should get general tools
        tool_names = [tool.name for tool in tools]
        assert "SerperDevTool" in tool_names
        assert "WebsiteSearchTool" in tool_names

    def test_get_pattern_info_existing(self, registry):
        """Test getting detailed pattern information."""
        info = registry.get_pattern_info("research")

        assert info["project_type"] == "research"
        assert "research" in info["description"].lower()
        assert info["tool_count"] > 0
        assert len(info["tools"]) == info["tool_count"]
        assert len(info["agent_roles"]) > 0
        assert len(info["common_tasks"]) > 0
        assert len(info["keywords"]) > 0

    def test_get_pattern_info_non_existing(self, registry):
        """Test getting info for non-existing pattern."""
        info = registry.get_pattern_info("nonexistent")
        assert info == {}

    def test_keyword_matching_case_insensitive(self, registry):
        """Test that keyword matching is case-insensitive."""
        project_spec = {"description": "RESEARCH and ANALYSIS project with DATA"}

        detected_type = registry.detect_project_type(project_spec)
        assert detected_type in ["research", "data_analysis"]  # Could match either

    def test_keyword_matching_word_boundaries(self, registry):
        """Test that keyword matching respects word boundaries."""
        project_spec = {"description": "Researching development of a new product"}

        detected_type = registry.detect_project_type(project_spec)
        # Should match "research" and "development" keywords
        assert detected_type in ["research", "development"]

    def test_multiple_keyword_matches_highest_score(self, registry):
        """Test that pattern with most keyword matches is selected."""
        project_spec = {
            "description": "Data analysis research for analytics insights and reporting metrics"
        }

        detected_type = registry.detect_project_type(project_spec)
        # Should match data_analysis due to multiple data/analytics keywords
        assert detected_type == "data_analysis"

    def test_tool_pattern_attributes_complete(self, registry):
        """Test that all tool patterns have required attributes."""
        for pattern_name in registry.get_available_patterns():
            pattern = registry.get_pattern(pattern_name)
            assert pattern is not None

            for tool in pattern.tools:
                assert tool.name
                assert tool.import_path
                assert tool.description
                assert isinstance(tool.dependencies, list)
                assert isinstance(tool.config_params, dict)
                assert isinstance(tool.usage_context, str)

    def test_project_type_patterns_complete(self, registry):
        """Test that all project type patterns have required attributes."""
        for pattern_name in registry.get_available_patterns():
            pattern = registry.get_pattern(pattern_name)
            assert pattern is not None

            assert pattern.project_type
            assert pattern.description
            assert isinstance(pattern.tools, list)
            assert isinstance(pattern.agent_roles, list)
            assert isinstance(pattern.common_tasks, list)
            assert isinstance(pattern.keywords, list)

            # All non-general patterns should have tools and keywords
            if pattern_name != "general":
                assert len(pattern.tools) > 0
                assert len(pattern.keywords) > 0

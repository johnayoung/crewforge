"""
Tool Integration Patterns - Common project type tool configurations

This module defines tool patterns for different project types, providing
structured mappings between project characteristics and appropriate tool sets.
"""

from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass, field


@dataclass
class ToolPattern:
    """Represents a tool with its configuration and usage context."""

    name: str
    import_path: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    config_params: Dict[str, Any] = field(default_factory=dict)
    usage_context: str = ""


@dataclass
class ProjectTypePattern:
    """Defines tool patterns and configurations for a specific project type."""

    project_type: str
    description: str
    tools: List[ToolPattern] = field(default_factory=list)
    agent_roles: List[str] = field(default_factory=list)
    common_tasks: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class ToolPatternRegistry:
    """
    Registry for tool integration patterns across different project types.

    Provides systematic tool selection and configuration based on project
    characteristics and requirements.
    """

    def __init__(self):
        """Initialize the registry with predefined patterns."""
        self._patterns = {}
        self._initialize_patterns()

    def _initialize_patterns(self) -> None:
        """Initialize predefined tool integration patterns."""

        # Research & Analysis Pattern
        research_tools = [
            ToolPattern(
                name="SerperDevTool",
                import_path="crewai_tools.SerperDevTool",
                description="Web search and research capabilities",
                dependencies=["crewai-tools[serper]"],
                config_params={"api_key": "SERPER_API_KEY"},
                usage_context="web_search",
            ),
            ToolPattern(
                name="WebsiteSearchTool",
                import_path="crewai_tools.WebsiteSearchTool",
                description="Search within specific websites",
                dependencies=["crewai-tools[web]"],
                usage_context="targeted_search",
            ),
            ToolPattern(
                name="ScrapeWebsiteTool",
                import_path="crewai_tools.ScrapeWebsiteTool",
                description="Extract content from web pages",
                dependencies=["crewai-tools[web]"],
                usage_context="data_extraction",
            ),
            ToolPattern(
                name="PDFSearchTool",
                import_path="crewai_tools.PDFSearchTool",
                description="Search and extract information from PDF files",
                dependencies=["crewai-tools[pdf]"],
                usage_context="document_analysis",
            ),
        ]

        self._patterns["research"] = ProjectTypePattern(
            project_type="research",
            description="Scientific research, market analysis, competitive intelligence",
            tools=research_tools,
            agent_roles=["researcher", "analyst", "data_collector"],
            common_tasks=["gather_information", "analyze_data", "compile_report"],
            keywords=[
                "research",
                "analysis",
                "investigation",
                "study",
                "intel",
                "market",
                "competitive",
            ],
        )

        # Content Creation Pattern
        content_tools = [
            ToolPattern(
                name="SerperDevTool",
                import_path="crewai_tools.SerperDevTool",
                description="Research content topics and trends",
                dependencies=["crewai-tools[serper]"],
                config_params={"api_key": "SERPER_API_KEY"},
                usage_context="topic_research",
            ),
            ToolPattern(
                name="FileReadTool",
                import_path="crewai_tools.FileReadTool",
                description="Read and process content files",
                dependencies=["crewai-tools[file]"],
                usage_context="content_processing",
            ),
            ToolPattern(
                name="DirectoryReadTool",
                import_path="crewai_tools.DirectoryReadTool",
                description="Scan directories for content assets",
                dependencies=["crewai-tools[file]"],
                usage_context="asset_management",
            ),
            ToolPattern(
                name="WebsiteSearchTool",
                import_path="crewai_tools.WebsiteSearchTool",
                description="Research competitor content strategies",
                dependencies=["crewai-tools[web]"],
                usage_context="competitor_analysis",
            ),
        ]

        self._patterns["content"] = ProjectTypePattern(
            project_type="content",
            description="Content creation, blogging, copywriting, marketing materials",
            tools=content_tools,
            agent_roles=["writer", "editor", "content_strategist", "researcher"],
            common_tasks=[
                "research_topics",
                "write_content",
                "edit_content",
                "optimize_seo",
            ],
            keywords=[
                "content",
                "blog",
                "writing",
                "copy",
                "articles",
                "marketing",
                "social",
                "seo",
            ],
        )

        # Data Analysis Pattern
        data_tools = [
            ToolPattern(
                name="CSVSearchTool",
                import_path="crewai_tools.CSVSearchTool",
                description="Search and analyze CSV datasets",
                dependencies=["crewai-tools[data]"],
                usage_context="structured_data_analysis",
            ),
            ToolPattern(
                name="ExcelSearchTool",
                import_path="crewai_tools.ExcelSearchTool",
                description="Process Excel spreadsheets and workbooks",
                dependencies=["crewai-tools[data]"],
                usage_context="spreadsheet_analysis",
            ),
            ToolPattern(
                name="JSONSearchTool",
                import_path="crewai_tools.JSONSearchTool",
                description="Parse and analyze JSON data structures",
                dependencies=["crewai-tools[data]"],
                usage_context="api_data_analysis",
            ),
            ToolPattern(
                name="DatabaseTool",
                import_path="crewai_tools.DatabaseTool",
                description="Query and analyze database records",
                dependencies=["crewai-tools[database]"],
                usage_context="database_analysis",
            ),
        ]

        self._patterns["data_analysis"] = ProjectTypePattern(
            project_type="data_analysis",
            description="Data processing, analytics, business intelligence, reporting",
            tools=data_tools,
            agent_roles=[
                "data_analyst",
                "data_scientist",
                "business_analyst",
                "report_generator",
            ],
            common_tasks=[
                "process_data",
                "analyze_patterns",
                "generate_insights",
                "create_visualizations",
            ],
            keywords=[
                "data",
                "analytics",
                "analysis",
                "reporting",
                "insights",
                "metrics",
                "dashboard",
                "bi",
            ],
        )

        # Customer Service Pattern
        customer_service_tools = [
            ToolPattern(
                name="FileReadTool",
                import_path="crewai_tools.FileReadTool",
                description="Access knowledge base and documentation",
                dependencies=["crewai-tools[file]"],
                usage_context="knowledge_retrieval",
            ),
            ToolPattern(
                name="DirectorySearchTool",
                import_path="crewai_tools.DirectorySearchTool",
                description="Search through support documentation",
                dependencies=["crewai-tools[file]"],
                usage_context="document_search",
            ),
            ToolPattern(
                name="WebsiteSearchTool",
                import_path="crewai_tools.WebsiteSearchTool",
                description="Search company website and resources",
                dependencies=["crewai-tools[web]"],
                usage_context="resource_lookup",
            ),
        ]

        self._patterns["customer_service"] = ProjectTypePattern(
            project_type="customer_service",
            description="Customer support, helpdesk, service automation",
            tools=customer_service_tools,
            agent_roles=["support_agent", "escalation_handler", "knowledge_expert"],
            common_tasks=[
                "handle_inquiry",
                "provide_solution",
                "escalate_issue",
                "update_knowledge",
            ],
            keywords=[
                "support",
                "customer",
                "service",
                "helpdesk",
                "tickets",
                "assistance",
                "help",
            ],
        )

        # Development Pattern
        development_tools = [
            ToolPattern(
                name="FileReadTool",
                import_path="crewai_tools.FileReadTool",
                description="Read and analyze code files",
                dependencies=["crewai-tools[file]"],
                usage_context="code_analysis",
            ),
            ToolPattern(
                name="DirectoryReadTool",
                import_path="crewai_tools.DirectoryReadTool",
                description="Scan project structure and codebase",
                dependencies=["crewai-tools[file]"],
                usage_context="project_exploration",
            ),
            ToolPattern(
                name="GithubSearchTool",
                import_path="crewai_tools.GithubSearchTool",
                description="Search GitHub repositories and code",
                dependencies=["crewai-tools[github]"],
                config_params={"github_token": "GITHUB_TOKEN"},
                usage_context="code_research",
            ),
            ToolPattern(
                name="SerperDevTool",
                import_path="crewai_tools.SerperDevTool",
                description="Research development solutions and documentation",
                dependencies=["crewai-tools[serper]"],
                config_params={"api_key": "SERPER_API_KEY"},
                usage_context="tech_research",
            ),
        ]

        self._patterns["development"] = ProjectTypePattern(
            project_type="development",
            description="Software development, code review, technical documentation",
            tools=development_tools,
            agent_roles=["developer", "code_reviewer", "technical_writer", "architect"],
            common_tasks=[
                "analyze_code",
                "write_documentation",
                "review_changes",
                "plan_architecture",
            ],
            keywords=[
                "development",
                "code",
                "programming",
                "software",
                "technical",
                "documentation",
                "review",
            ],
        )

        # General/Default Pattern
        general_tools = [
            ToolPattern(
                name="SerperDevTool",
                import_path="crewai_tools.SerperDevTool",
                description="General web search and research",
                dependencies=["crewai-tools[serper]"],
                config_params={"api_key": "SERPER_API_KEY"},
                usage_context="general_search",
            ),
            ToolPattern(
                name="WebsiteSearchTool",
                import_path="crewai_tools.WebsiteSearchTool",
                description="Search specific websites for information",
                dependencies=["crewai-tools[web]"],
                usage_context="targeted_search",
            ),
        ]

        self._patterns["general"] = ProjectTypePattern(
            project_type="general",
            description="General purpose projects with basic tool requirements",
            tools=general_tools,
            agent_roles=["assistant", "researcher", "coordinator"],
            common_tasks=[
                "gather_information",
                "provide_assistance",
                "coordinate_activities",
            ],
            keywords=["general", "assistant", "help", "basic", "simple"],
        )

    def detect_project_type(self, project_spec: Dict[str, Any]) -> str:
        """
        Detect project type based on project specifications.

        Args:
            project_spec: Project specification dictionary containing description,
                         requirements, agents, tasks, etc.

        Returns:
            Detected project type string, defaults to "general" if no match
        """
        # Extract text for analysis
        text_sources = [
            project_spec.get("description", ""),
            project_spec.get("prompt", ""),
            project_spec.get("project_name", ""),
            " ".join(project_spec.get("requirements", [])),
            " ".join(
                [agent.get("role", "") for agent in project_spec.get("agents", [])]
            ),
            " ".join([task.get("name", "") for task in project_spec.get("tasks", [])]),
        ]

        full_text = " ".join(text_sources).lower()

        # Score each pattern based on keyword matches
        pattern_scores = {}

        for pattern_name, pattern in self._patterns.items():
            if pattern_name == "general":  # Skip general pattern in scoring
                continue

            score = 0
            for keyword in pattern.keywords:
                # Count keyword occurrences with word boundaries
                matches = len(
                    re.findall(r"\b" + re.escape(keyword.lower()) + r"\b", full_text)
                )
                score += matches

            pattern_scores[pattern_name] = score

        # Return pattern with highest score, or "general" if no clear match
        if not pattern_scores or max(pattern_scores.values()) == 0:
            return "general"

        return max(pattern_scores.items(), key=lambda x: x[1])[0]

    def get_pattern(self, project_type: str) -> Optional[ProjectTypePattern]:
        """
        Get tool pattern for a specific project type.

        Args:
            project_type: Project type string

        Returns:
            ProjectTypePattern if found, None otherwise
        """
        return self._patterns.get(project_type)

    def get_tools_for_project(self, project_spec: Dict[str, Any]) -> List[ToolPattern]:
        """
        Get recommended tools for a project based on its specifications.

        Args:
            project_spec: Project specification dictionary

        Returns:
            List of recommended ToolPattern objects
        """
        project_type = self.detect_project_type(project_spec)
        pattern = self.get_pattern(project_type)

        if pattern:
            return pattern.tools

        return self._patterns["general"].tools

    def get_available_patterns(self) -> List[str]:
        """Get list of available project type patterns."""
        return list(self._patterns.keys())

    def get_pattern_info(self, project_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a project type pattern.

        Args:
            project_type: Project type string

        Returns:
            Dictionary with pattern information
        """
        pattern = self.get_pattern(project_type)
        if not pattern:
            return {}

        return {
            "project_type": pattern.project_type,
            "description": pattern.description,
            "tool_count": len(pattern.tools),
            "tools": [
                {"name": tool.name, "description": tool.description}
                for tool in pattern.tools
            ],
            "agent_roles": pattern.agent_roles,
            "common_tasks": pattern.common_tasks,
            "keywords": pattern.keywords,
        }

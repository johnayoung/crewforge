"""
Tests for intelligent task definition generation.

This module tests the integration of liteLLM with the enhancement engine
to generate contextual task definitions based on project specifications.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from crewforge.enhancement import EnhancementEngine
from crewforge.llm import LLMClient


class TestTaskGeneration:
    """Test cases for intelligent task generation functionality."""

    @pytest.fixture
    def sample_project_spec(self):
        """Sample project specification for testing."""
        return {
            "project_name": "content_research_team",
            "description": "A team that researches and analyzes content trends",
            "project_type": "research",
            "domain": "content marketing",
            "agents": [
                {"role": "researcher", "description": "Finds and gathers information"},
                {"role": "analyst", "description": "Analyzes collected data"},
            ],
            "tasks": [
                {
                    "name": "research_trends",
                    "description": "Research content trends",
                    "agent": "researcher",
                },
                {
                    "name": "analyze_data",
                    "description": "Analyze competitor content",
                    "agent": "analyst",
                },
                {
                    "name": "generate_report",
                    "description": "Generate insights report",
                    "agent": "analyst",
                },
            ],
        }

    @pytest.fixture
    def enhancement_engine(self, tmp_path):
        """Create an enhancement engine for testing."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "tasks").mkdir()
        return EnhancementEngine(templates_dir=templates_dir)

    @pytest.fixture
    def llm_client(self):
        """Create a mock LLM client for testing."""
        with patch("crewforge.llm.LLMClient") as MockLLMClient:
            client = MockLLMClient.return_value
            client.complete = AsyncMock()
            yield client

    @pytest.mark.asyncio
    async def test_generate_task_definition_basic(
        self, enhancement_engine, llm_client, sample_project_spec
    ):
        """Test basic task definition generation functionality."""
        # Mock LLM response for task definition generation
        expected_response = json.dumps(
            {
                "description": "Conduct comprehensive content trend analysis by researching emerging topics, viral content patterns, and audience engagement metrics across multiple platforms and demographics",
                "expected_output": "A detailed research report containing trending content topics, engagement analytics, audience insights, and strategic recommendations for content marketing campaigns",
                "context": [
                    "Focus on Q4 2024 trends",
                    "Include social media platform analysis",
                    "Prioritize B2B content marketing sector",
                ],
                "tools": ["web_search_tool", "analytics_api", "social_media_scraper"],
            }
        )

        llm_client.complete.return_value = expected_response

        # Test the task generation method
        result = await enhancement_engine.generate_task_definition(
            task_spec={
                "name": "research_trends",
                "description": "Research content trends",
                "agent": "researcher",
            },
            project_spec=sample_project_spec,
            llm_client=llm_client,
        )

        # Verify the result structure
        assert "description" in result
        assert "expected_output" in result
        assert "context" in result
        assert "tools" in result
        assert "comprehensive content trend analysis" in result["description"].lower()
        assert "research report" in result["expected_output"].lower()
        assert isinstance(result["context"], list)
        assert isinstance(result["tools"], list)

        # Verify LLM was called with appropriate context
        llm_client.complete.assert_called_once()
        call_args = llm_client.complete.call_args[0][0]
        assert "content_research_team" in call_args
        assert "content marketing" in call_args
        assert "research_trends" in call_args

    @pytest.mark.asyncio
    async def test_generate_task_definition_with_context(
        self, enhancement_engine, llm_client, sample_project_spec
    ):
        """Test task definition generation with rich context."""
        # Enhanced project specification with more context
        enhanced_project_spec = {
            **sample_project_spec,
            "industry": "SaaS",
            "target_audience": "B2B marketing managers",
            "objectives": ["increase lead generation", "improve content ROI"],
        }

        expected_response = json.dumps(
            {
                "description": "Execute advanced competitor analysis by examining content strategies, engagement patterns, and performance metrics of leading B2B SaaS companies to identify opportunities and best practices",
                "expected_output": "Comprehensive competitor analysis report with performance benchmarks, content gap analysis, and strategic recommendations tailored for B2B SaaS marketing",
                "context": [
                    "Focus on B2B SaaS competitors",
                    "Analyze content performance metrics",
                    "Identify content gaps and opportunities",
                ],
                "tools": [
                    "competitor_analysis_tool",
                    "performance_metrics_api",
                    "content_scraper",
                ],
            }
        )

        llm_client.complete.return_value = expected_response

        # Test enhanced task generation
        result = await enhancement_engine.generate_task_definition(
            task_spec={
                "name": "analyze_data",
                "description": "Analyze competitor content",
                "agent": "analyst",
            },
            project_spec=enhanced_project_spec,
            llm_client=llm_client,
        )

        # Verify the enhanced context is used
        assert (
            "b2b saas" in result["description"].lower()
            or "saas" in result["expected_output"].lower()
        )
        assert "competitor" in result["description"].lower()

        # Verify LLM prompt included enhanced context
        call_args = llm_client.complete.call_args[0][0]
        assert "SaaS" in call_args
        assert "B2B marketing managers" in call_args

    @pytest.mark.asyncio
    async def test_generate_task_definitions_multiple(
        self, enhancement_engine, llm_client, sample_project_spec
    ):
        """Test generation of multiple task definitions."""
        # Mock multiple LLM responses
        responses = [
            json.dumps(
                {
                    "description": "Research trending content topics and viral patterns across social media platforms",
                    "expected_output": "Trending topics report with engagement metrics and viral content analysis",
                    "context": ["Social media trends", "Viral content patterns"],
                    "tools": ["social_media_api", "trend_analysis_tool"],
                }
            ),
            json.dumps(
                {
                    "description": "Analyze competitor content strategies and performance metrics",
                    "expected_output": "Competitor analysis report with performance benchmarks and insights",
                    "context": ["Competitor content audit", "Performance analysis"],
                    "tools": ["competitor_analysis_tool", "metrics_api"],
                }
            ),
            json.dumps(
                {
                    "description": "Generate comprehensive insights report with strategic recommendations",
                    "expected_output": "Strategic insights report with actionable recommendations and data visualizations",
                    "context": ["Strategic recommendations", "Data visualization"],
                    "tools": ["report_generator", "visualization_tool"],
                }
            ),
        ]

        llm_client.complete.side_effect = responses

        # Test generating all task definitions
        result = await enhancement_engine.generate_task_definitions(
            project_spec=sample_project_spec,
            llm_client=llm_client,
        )

        # Verify all tasks were processed
        assert len(result) == 3
        assert all("description" in task for task in result)
        assert all("expected_output" in task for task in result)
        assert all("context" in task for task in result)
        assert all("tools" in task for task in result)

        # Verify LLM was called for each task
        assert llm_client.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_task_definition_error_handling(
        self, enhancement_engine, llm_client, sample_project_spec
    ):
        """Test error handling in task definition generation."""
        # Test JSON decode error
        llm_client.complete.return_value = "Invalid JSON response"

        with pytest.raises(json.JSONDecodeError):
            await enhancement_engine.generate_task_definition(
                task_spec={"name": "test", "description": "test", "agent": "test"},
                project_spec=sample_project_spec,
                llm_client=llm_client,
            )

        # Test missing required fields
        llm_client.complete.return_value = json.dumps(
            {"description": "Only description"}
        )

        with pytest.raises(KeyError, match="Missing required field"):
            await enhancement_engine.generate_task_definition(
                task_spec={"name": "test", "description": "test", "agent": "test"},
                project_spec=sample_project_spec,
                llm_client=llm_client,
            )

    @pytest.mark.asyncio
    async def test_generate_task_definition_validation(
        self, enhancement_engine, llm_client, sample_project_spec
    ):
        """Test validation of generated task definitions."""
        valid_response = json.dumps(
            {
                "description": "Detailed task description with comprehensive approach",
                "expected_output": "Clear expected output specification with deliverables",
                "context": ["Context item 1", "Context item 2"],
                "tools": ["tool1", "tool2"],
            }
        )

        llm_client.complete.return_value = valid_response

        result = await enhancement_engine.generate_task_definition(
            task_spec={
                "name": "test_task",
                "description": "Test",
                "agent": "test_agent",
            },
            project_spec=sample_project_spec,
            llm_client=llm_client,
        )

        # Verify all required fields are present and valid
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 20  # Substantial description
        assert isinstance(result["expected_output"], str)
        assert len(result["expected_output"]) > 20  # Substantial output
        assert isinstance(result["context"], list)
        assert isinstance(result["tools"], list)

    @pytest.mark.asyncio
    async def test_generate_task_definitions_no_tasks(
        self, enhancement_engine, llm_client
    ):
        """Test handling of project spec with no tasks."""
        project_spec_no_tasks = {"project_name": "test_project", "tasks": []}

        with pytest.raises(
            ValueError, match="Project specification must contain 'tasks' list"
        ):
            await enhancement_engine.generate_task_definitions(
                project_spec=project_spec_no_tasks,
                llm_client=llm_client,
            )

    @pytest.mark.asyncio
    async def test_create_task_generation_prompt(
        self, enhancement_engine, sample_project_spec
    ):
        """Test creation of task generation prompt."""
        task_spec = {
            "name": "research_trends",
            "description": "Research content trends",
            "agent": "researcher",
        }

        # This tests the internal prompt creation method
        prompt = enhancement_engine._create_task_generation_prompt(
            task_spec, sample_project_spec
        )

        # Verify prompt contains key information
        assert "content_research_team" in prompt
        assert "content marketing" in prompt
        assert "research_trends" in prompt
        assert "Research content trends" in prompt
        assert "researcher" in prompt
        assert "JSON" in prompt  # Should request JSON output

        # Verify required fields are mentioned in prompt
        assert "description" in prompt
        assert "expected_output" in prompt
        assert "context" in prompt
        assert "tools" in prompt

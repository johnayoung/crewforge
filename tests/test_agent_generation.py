"""
Tests for intelligent agent role and backstory generation.

This module tests the integration of liteLLM with the enhancement engine
to generate contextual agent roles and backstories for CrewAI projects.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from crewforge.enhancement import EnhancementEngine
from crewforge.llm import LLMClient


class TestAgentGeneration:
    """Test cases for intelligent agent generation functionality."""

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
                "Research content trends",
                "Analyze competitor content",
                "Generate insights report",
            ],
        }

    @pytest.fixture
    def enhancement_engine(self, tmp_path):
        """Create an enhancement engine for testing."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "agents").mkdir()
        return EnhancementEngine(templates_dir=templates_dir)

    @pytest.fixture
    def llm_client(self):
        """Create a mock LLM client for testing."""
        with patch("crewforge.llm.LLMClient") as MockLLMClient:
            client = MockLLMClient.return_value
            client.complete = AsyncMock()
            yield client

    @pytest.mark.asyncio
    async def test_generate_agent_role_basic(
        self, enhancement_engine, llm_client, sample_project_spec
    ):
        """Test basic agent role generation functionality."""
        # Mock LLM response for agent role generation
        expected_response = json.dumps(
            {
                "role": "Senior Content Researcher",
                "goal": "Research and identify trending content topics and patterns in the content marketing domain",
                "backstory": "You are an experienced content researcher with 8+ years in content marketing analysis. You have worked with major brands to identify viral content patterns and consumer engagement trends. Your expertise lies in data-driven content discovery and trend prediction.",
            }
        )

        llm_client.complete.return_value = expected_response

        # Test the agent generation method
        result = await enhancement_engine.generate_agent_role(
            agent_spec={
                "role": "researcher",
                "description": "Finds and gathers information",
            },
            project_spec=sample_project_spec,
            llm_client=llm_client,
        )

        # Verify the result structure
        assert "role" in result
        assert "goal" in result
        assert "backstory" in result
        assert result["role"] == "Senior Content Researcher"
        assert "content marketing" in result["goal"].lower()
        assert len(result["backstory"]) > 50  # Substantial backstory

        # Verify LLM was called with appropriate context
        llm_client.complete.assert_called_once()
        call_args = llm_client.complete.call_args[0][0]
        assert "content_research_team" in call_args
        assert "content marketing" in call_args
        assert "researcher" in call_args

    @pytest.mark.asyncio
    async def test_generate_agent_role_with_context(
        self, enhancement_engine, llm_client, sample_project_spec
    ):
        """Test agent role generation with rich context."""
        expected_response = json.dumps(
            {
                "role": "Marketing Data Analyst",
                "goal": "Analyze content performance metrics and competitor strategies to provide actionable insights",
                "backstory": "You are a detail-oriented data analyst specializing in content marketing metrics. With a background in statistics and marketing analytics, you excel at turning raw data into strategic recommendations. Your analytical approach has helped numerous companies optimize their content strategies.",
            }
        )

        llm_client.complete.return_value = expected_response

        # Test with analyst role
        result = await enhancement_engine.generate_agent_role(
            agent_spec={"role": "analyst", "description": "Analyzes collected data"},
            project_spec=sample_project_spec,
            llm_client=llm_client,
        )

        # Verify context-appropriate generation
        assert "analyst" in result["role"].lower()
        assert "analyze" in result["goal"].lower()
        assert "data" in result["backstory"].lower()

    @pytest.mark.asyncio
    async def test_generate_multiple_agent_roles(
        self, enhancement_engine, llm_client, sample_project_spec
    ):
        """Test generating roles for multiple agents."""
        responses = [
            json.dumps(
                {
                    "role": "Senior Content Researcher",
                    "goal": "Research trending topics and content opportunities",
                    "backstory": "Expert researcher with deep content marketing knowledge.",
                }
            ),
            json.dumps(
                {
                    "role": "Content Performance Analyst",
                    "goal": "Analyze content metrics and performance data",
                    "backstory": "Data-driven analyst specializing in content performance optimization.",
                }
            ),
        ]

        llm_client.complete.side_effect = responses

        # Generate roles for all agents
        results = await enhancement_engine.generate_agent_roles(
            project_spec=sample_project_spec, llm_client=llm_client
        )

        # Verify results
        assert len(results) == 2
        assert results[0]["role"] == "Senior Content Researcher"
        assert results[1]["role"] == "Content Performance Analyst"
        assert llm_client.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_agent_role_error_handling(
        self, enhancement_engine, llm_client, sample_project_spec
    ):
        """Test error handling in agent role generation."""
        # Test invalid JSON response
        llm_client.complete.return_value = "Invalid JSON response"

        with pytest.raises(json.JSONDecodeError):
            await enhancement_engine.generate_agent_role(
                agent_spec={"role": "researcher", "description": "Test"},
                project_spec=sample_project_spec,
                llm_client=llm_client,
            )

        # Test missing required fields
        llm_client.complete.return_value = json.dumps(
            {"role": "Test Role"}
        )  # Missing goal and backstory

        with pytest.raises(KeyError):
            await enhancement_engine.generate_agent_role(
                agent_spec={"role": "researcher", "description": "Test"},
                project_spec=sample_project_spec,
                llm_client=llm_client,
            )

    @pytest.mark.asyncio
    async def test_generate_agent_role_validation(
        self, enhancement_engine, llm_client, sample_project_spec
    ):
        """Test validation of generated agent roles."""
        # Test with valid response
        valid_response = json.dumps(
            {
                "role": "Senior Content Researcher",
                "goal": "Research and identify trending content topics",
                "backstory": "Experienced researcher with expertise in content marketing trends and data analysis.",
            }
        )

        llm_client.complete.return_value = valid_response

        result = await enhancement_engine.generate_agent_role(
            agent_spec={"role": "researcher", "description": "Test"},
            project_spec=sample_project_spec,
            llm_client=llm_client,
        )

        # Verify validation passes
        assert len(result["role"]) > 5
        assert len(result["goal"]) > 10
        assert len(result["backstory"]) > 20

    @pytest.mark.asyncio
    async def test_generate_domain_specific_roles(self, enhancement_engine, llm_client):
        """Test generation of domain-specific agent roles."""
        project_specs = [
            {
                "project_name": "ecommerce_optimizer",
                "domain": "e-commerce",
                "project_type": "optimization",
                "agents": [
                    {
                        "role": "optimizer",
                        "description": "Optimizes e-commerce performance",
                    }
                ],
            },
            {
                "project_name": "financial_analyzer",
                "domain": "finance",
                "project_type": "analysis",
                "agents": [
                    {"role": "analyst", "description": "Analyzes financial data"}
                ],
            },
        ]

        # Mock responses for different domains
        ecommerce_response = json.dumps(
            {
                "role": "E-commerce Performance Optimizer",
                "goal": "Optimize online store conversion rates and customer experience",
                "backstory": "E-commerce specialist with expertise in conversion optimization and customer journey analysis.",
            }
        )

        finance_response = json.dumps(
            {
                "role": "Financial Data Analyst",
                "goal": "Analyze financial trends and provide investment insights",
                "backstory": "CFA-certified analyst with experience in financial modeling and risk assessment.",
            }
        )

        llm_client.complete.side_effect = [ecommerce_response, finance_response]

        # Test e-commerce domain
        ecommerce_result = await enhancement_engine.generate_agent_role(
            agent_spec=project_specs[0]["agents"][0],
            project_spec=project_specs[0],
            llm_client=llm_client,
        )

        assert (
            "e-commerce" in ecommerce_result["role"].lower()
            or "ecommerce" in ecommerce_result["role"].lower()
        )
        assert "conversion" in ecommerce_result["goal"].lower()

        # Test finance domain
        finance_result = await enhancement_engine.generate_agent_role(
            agent_spec=project_specs[1]["agents"][0],
            project_spec=project_specs[1],
            llm_client=llm_client,
        )

        assert "financial" in finance_result["role"].lower()
        assert (
            "financial" in finance_result["goal"].lower()
            or "investment" in finance_result["goal"].lower()
        )

    def test_agent_generation_prompt_template(self, enhancement_engine):
        """Test the prompt template used for agent generation."""
        template = enhancement_engine._create_agent_generation_prompt(
            agent_spec={
                "role": "researcher",
                "description": "Researches content trends",
            },
            project_spec={
                "project_name": "content_team",
                "domain": "marketing",
                "description": "Content marketing team",
            },
        )

        # Verify prompt contains necessary context
        assert "researcher" in template
        assert "content_team" in template
        assert "marketing" in template
        assert "JSON" in template  # Should request JSON format
        assert "role" in template and "goal" in template and "backstory" in template

    @pytest.mark.asyncio
    async def test_integration_with_enhancement_engine(
        self, enhancement_engine, llm_client, sample_project_spec, tmp_path
    ):
        """Test integration of agent generation with the existing enhancement engine."""
        # Create a simple agent template
        agent_template_content = """
{% for agent in agents %}
{{ agent.role }}:
  role: >
    {{ agent.role }}
  goal: >
    {{ agent.goal }}
  verbose: {{ agent.verbose | default(true) | lower }}
  memory: {{ agent.memory | default(true) | lower }}
  backstory: >
    {{ agent.backstory }}
{% endfor %}
"""

        agent_template_path = (
            enhancement_engine.templates_dir / "agents" / "default.yaml.j2"
        )
        agent_template_path.write_text(agent_template_content)

        # Mock agent generation
        generated_agents = [
            {
                "role": "Senior Content Researcher",
                "goal": "Research trending content topics and opportunities",
                "backstory": "Expert researcher with deep content marketing knowledge.",
                "verbose": True,
                "memory": True,
            }
        ]

        # Mock the generate_agent_roles method
        with patch.object(
            enhancement_engine, "generate_agent_roles", return_value=generated_agents
        ):
            # Test enhanced project generation
            project_path = tmp_path / "test_project"
            project_path.mkdir()

            # Create basic project structure
            config_dir = project_path / "src" / "test_project" / "config"
            config_dir.mkdir(parents=True)

            # Create initial agents.yaml file
            agents_config_path = config_dir / "agents.yaml"
            agents_config_path.write_text("# Initial agents config\n")

            result = await enhancement_engine.enhance_project_with_generated_agents(
                project_path=project_path,
                project_spec=sample_project_spec,
                llm_client=llm_client,
            )

            # Verify integration worked
            assert result is True
            agents_config_path = config_dir / "agents.yaml"
            assert agents_config_path.exists()

            agents_content = agents_config_path.read_text()
            assert "Senior Content Researcher" in agents_content
            assert "Research trending content topics" in agents_content

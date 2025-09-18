"""
Tests for prompt template functionality.

Tests the PromptTemplates class that extracts CrewAI project specifications
from natural language prompts using the liteLLM client.
"""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch

from crewforge.llm import LLMClient, LLMError
from crewforge.prompt_templates import PromptTemplates, PromptTemplateError


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = Mock(spec=LLMClient)
    client.complete_structured = AsyncMock()
    return client


@pytest.fixture
def prompt_templates(mock_llm_client):
    """Create PromptTemplates instance with mock LLM client."""
    return PromptTemplates(llm_client=mock_llm_client)


@pytest.mark.asyncio
class TestPromptTemplatesBasic:
    """Test basic functionality of PromptTemplates."""

    async def test_extract_project_spec_simple_prompt(
        self, prompt_templates, mock_llm_client
    ):
        """Test extracting project specs from simple natural language prompt."""
        # Arrange
        user_prompt = "Create a content research team that writes blog posts"
        expected_response = {
            "project_name": "content-research-team",
            "project_description": "A team that researches and writes blog posts",
            "agents": [
                {
                    "role": "Research Agent",
                    "goal": "Research topics for blog posts",
                    "backstory": "An experienced researcher with expertise in content discovery",
                    "tools": ["web_search", "content_scraper"],
                },
                {
                    "role": "Writer Agent",
                    "goal": "Write high-quality blog posts",
                    "backstory": "A skilled content writer with expertise in engaging writing",
                    "tools": ["text_editor", "grammar_checker"],
                },
            ],
            "tasks": [
                {
                    "description": "Research trending topics and gather information",
                    "expected_output": "List of trending topics with research notes",
                    "agent": "Research Agent",
                },
                {
                    "description": "Write engaging blog posts based on research",
                    "expected_output": "Complete blog post with title, content, and metadata",
                    "agent": "Writer Agent",
                },
            ],
            "dependencies": ["crewai", "requests", "beautifulsoup4"],
        }

        mock_llm_client.complete_structured.return_value = expected_response

        # Act
        result = await prompt_templates.extract_project_spec(user_prompt)

        # Assert
        assert result == expected_response
        mock_llm_client.complete_structured.assert_called_once()

        # Verify the prompt contains key instructions
        call_args = mock_llm_client.complete_structured.call_args
        # The prompt is passed as keyword argument 'prompt'
        prompt_messages = call_args.kwargs.get("prompt") or call_args[0][0]
        # Convert messages to string for validation
        if isinstance(prompt_messages, list):
            prompt_text = " ".join([msg.get("content", "") for msg in prompt_messages])
        else:
            prompt_text = str(prompt_messages)
        assert "CrewAI project" in prompt_text
        assert "agents" in prompt_text
        assert "tasks" in prompt_text
        assert "tools" in prompt_text

    async def test_extract_project_spec_complex_prompt(
        self, prompt_templates, mock_llm_client
    ):
        """Test extracting specs from complex multi-domain prompt."""
        user_prompt = """
        Build an AI-powered customer support system with multiple agents:
        - A ticket classifier that categorizes incoming support requests
        - A knowledge base agent that searches for relevant solutions
        - A response generator that crafts personalized replies
        - An escalation agent that determines when to involve humans
        The system should integrate with email and chat platforms.
        """

        expected_response = {
            "project_name": "ai-customer-support-system",
            "project_description": "AI-powered multi-agent customer support system",
            "agents": [
                {
                    "role": "Ticket Classifier",
                    "goal": "Categorize and prioritize incoming support requests",
                    "backstory": "An AI specialist trained in customer service triage",
                    "tools": ["text_classifier", "priority_scorer"],
                },
                {
                    "role": "Knowledge Base Agent",
                    "goal": "Find relevant solutions from knowledge base",
                    "backstory": "An expert in information retrieval and knowledge management",
                    "tools": ["vector_search", "knowledge_retriever"],
                },
                {
                    "role": "Response Generator",
                    "goal": "Generate personalized customer responses",
                    "backstory": "A customer service expert skilled in empathetic communication",
                    "tools": ["text_generator", "template_engine"],
                },
                {
                    "role": "Escalation Agent",
                    "goal": "Determine when human intervention is needed",
                    "backstory": "A senior support manager with escalation expertise",
                    "tools": ["complexity_analyzer", "urgency_detector"],
                },
            ],
            "tasks": [
                {
                    "description": "Classify and prioritize incoming support tickets",
                    "expected_output": "Categorized tickets with priority scores",
                    "agent": "Ticket Classifier",
                },
                {
                    "description": "Search knowledge base for relevant solutions",
                    "expected_output": "List of relevant articles and solutions",
                    "agent": "Knowledge Base Agent",
                },
            ],
            "dependencies": ["crewai", "scikit-learn", "transformers", "langchain"],
        }

        mock_llm_client.complete_structured.return_value = expected_response

        # Act
        result = await prompt_templates.extract_project_spec(user_prompt)

        # Assert
        assert result == expected_response
        assert len(result["agents"]) == 4
        assert "ai-customer-support-system" == result["project_name"]

    async def test_extract_project_spec_schema_validation(
        self, prompt_templates, mock_llm_client
    ):
        """Test that returned structure matches expected schema."""
        user_prompt = "Create a simple data analysis team"

        mock_llm_client.complete_structured.return_value = {
            "project_name": "data-analysis-team",
            "project_description": "Simple data analysis team",
            "agents": [
                {
                    "role": "Data Analyst",
                    "goal": "Analyze data",
                    "backstory": "Expert analyst",
                    "tools": ["pandas"],
                }
            ],
            "tasks": [
                {
                    "description": "Analyze dataset",
                    "expected_output": "Analysis report",
                    "agent": "Data Analyst",
                }
            ],
            "dependencies": ["crewai", "pandas", "numpy"],
        }

        # Act
        result = await prompt_templates.extract_project_spec(user_prompt)

        # Assert - verify schema structure
        assert "project_name" in result
        assert "project_description" in result
        assert "agents" in result
        assert "tasks" in result
        assert "dependencies" in result

        # Verify agent structure
        for agent in result["agents"]:
            assert "role" in agent
            assert "goal" in agent
            assert "backstory" in agent
            assert "tools" in agent

        # Verify task structure
        for task in result["tasks"]:
            assert "description" in task
            assert "expected_output" in task
            assert "agent" in task


@pytest.mark.asyncio
class TestPromptTemplatesErrorHandling:
    """Test error handling and edge cases."""

    async def test_extract_project_spec_llm_error(
        self, prompt_templates, mock_llm_client
    ):
        """Test handling LLM client errors."""
        mock_llm_client.complete_structured.side_effect = LLMError("API request failed")

        with pytest.raises(
            PromptTemplateError, match="Failed to extract project specification"
        ):
            await prompt_templates.extract_project_spec("Test prompt")

    async def test_extract_project_spec_invalid_response_structure(
        self, prompt_templates, mock_llm_client
    ):
        """Test handling invalid response structure from LLM."""
        # Invalid response missing required fields
        mock_llm_client.complete_structured.return_value = {
            "project_name": "test",
            # Missing other required fields
        }

        with pytest.raises(
            PromptTemplateError, match="Invalid project specification structure"
        ):
            await prompt_templates.extract_project_spec("Test prompt")

    async def test_extract_project_spec_empty_prompt(self, prompt_templates):
        """Test handling empty prompts."""
        with pytest.raises(PromptTemplateError, match="Prompt cannot be empty"):
            await prompt_templates.extract_project_spec("")

        with pytest.raises(PromptTemplateError, match="Prompt cannot be empty"):
            await prompt_templates.extract_project_spec("   ")

    async def test_extract_project_spec_malformed_agents(
        self, prompt_templates, mock_llm_client
    ):
        """Test handling malformed agent definitions."""
        mock_llm_client.complete_structured.return_value = {
            "project_name": "test-project",
            "project_description": "Test project",
            "agents": [
                {
                    "role": "Test Agent",
                    # Missing required fields: goal, backstory, tools
                }
            ],
            "tasks": [],
            "dependencies": [],
        }

        with pytest.raises(
            PromptTemplateError, match="Invalid project specification structure"
        ):
            await prompt_templates.extract_project_spec("Test prompt")


@pytest.mark.asyncio
class TestPromptTemplatesIntegration:
    """Test integration with real LLM client."""

    async def test_prompt_template_with_real_client(self):
        """Test PromptTemplates with actual LLM client (integration test)."""
        # This test would run against a real LLM API
        # Skip in CI/CD unless API keys are available

        import os

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available for integration test")

        # Create real LLM client
        llm_client = LLMClient(provider="openai", model="gpt-3.5-turbo")
        prompt_templates = PromptTemplates(llm_client=llm_client)

        # Test with simple prompt
        result = await prompt_templates.extract_project_spec(
            "Create a team that analyzes stock market data and makes trading recommendations"
        )

        # Verify structure (not specific content since LLM responses vary)
        assert isinstance(result, dict)
        assert "project_name" in result
        assert "agents" in result
        assert "tasks" in result
        assert isinstance(result["agents"], list)
        assert len(result["agents"]) > 0
        assert isinstance(result["tasks"], list)
        assert len(result["tasks"]) > 0


class TestPromptTemplateSchemas:
    """Test JSON schemas used by PromptTemplates."""

    def test_project_spec_schema_structure(self):
        """Test that the project spec schema is well-formed."""
        from crewforge.prompt_templates import PromptTemplates

        schema = PromptTemplates.get_project_spec_schema()

        # Verify top-level schema structure
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Verify required fields
        required_fields = [
            "project_name",
            "project_description",
            "agents",
            "tasks",
            "dependencies",
        ]
        for field in required_fields:
            assert field in schema["required"]
            assert field in schema["properties"]

        # Verify agents schema
        agents_schema = schema["properties"]["agents"]
        assert agents_schema["type"] == "array"
        assert "items" in agents_schema

        agent_properties = agents_schema["items"]["properties"]
        required_agent_fields = ["role", "goal", "backstory", "tools"]
        for field in required_agent_fields:
            assert field in agent_properties
            assert field in agents_schema["items"]["required"]

    def test_agent_schema_validation(self):
        """Test agent schema validation logic."""
        from crewforge.prompt_templates import PromptTemplates

        valid_agent = {
            "role": "Test Agent",
            "goal": "Test goal",
            "backstory": "Test backstory",
            "tools": ["test_tool"],
        }

        invalid_agent = {
            "role": "Test Agent"
            # Missing required fields
        }

        # This would be tested through the main validation logic
        assert PromptTemplates._validate_agent_structure(valid_agent) == True
        assert PromptTemplates._validate_agent_structure(invalid_agent) == False

"""Test suite for Agentic AI Generation Engine."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from crewforge.models import AgentConfig, TaskConfig, CrewConfig
from crewforge.core.llm import LLMClient


class TestGenerationEngine:
    """Test cases for GenerationEngine class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLMClient for testing."""
        mock_client = Mock(spec=LLMClient)
        return mock_client

    @pytest.fixture
    def sample_prompt(self):
        """Sample user prompt for testing."""
        return "Create a content research crew that finds and summarizes articles"

    @pytest.fixture
    def sample_prompt_analysis(self):
        """Sample prompt analysis result."""
        return {
            "business_context": "content research and summarization",
            "required_roles": ["researcher", "summarizer"],
            "objectives": ["find articles", "summarize content"],
            "tools_needed": ["web_search", "text_analysis"]
        }

    @pytest.fixture
    def sample_generated_agents(self):
        """Sample generated agent configurations."""
        return [
            {
                "role": "Content Researcher",
                "goal": "Find relevant articles and sources on specified topics",
                "backstory": "You are an experienced researcher with expertise in finding high-quality content sources.",
                "tools": ["web_search"]
            },
            {
                "role": "Content Summarizer", 
                "goal": "Create concise and accurate summaries of research findings",
                "backstory": "You are a skilled writer who excels at distilling complex information into clear summaries.",
                "tools": ["text_analysis"]
            }
        ]

    @pytest.fixture
    def sample_agent_configs(self):
        """Sample AgentConfig objects for testing."""
        return [
            AgentConfig(
                role="Content Researcher",
                goal="Find relevant articles and sources on specified topics",
                backstory="You are an experienced researcher with expertise in finding high-quality content sources."
            ),
            AgentConfig(
                role="Content Summarizer", 
                goal="Create concise and accurate summaries of research findings",
                backstory="You are a skilled writer who excels at distilling complex information into clear summaries."
            )
        ]

    @pytest.fixture
    def sample_generated_tasks(self):
        """Sample generated task configurations."""
        return [
            {
                "description": "Search for and collect relevant articles on the specified topic",
                "expected_output": "A list of high-quality articles with URLs, titles, and brief descriptions",
                "agent": "Content Researcher"
            },
            {
                "description": "Analyze and summarize the collected articles",
                "expected_output": "A comprehensive summary report highlighting key findings and insights",
                "agent": "Content Summarizer",
                "context": ["Search for and collect relevant articles on the specified topic"]
            }
        ]

    def test_generation_engine_initialization(self, mock_llm_client):
        """Test GenerationEngine initializes correctly."""
        from crewforge.core.generator import GenerationEngine
        
        engine = GenerationEngine(llm_client=mock_llm_client)
        assert engine.llm_client == mock_llm_client
        assert hasattr(engine, 'analyze_prompt')
        assert hasattr(engine, 'generate_agents')
        assert hasattr(engine, 'generate_tasks')
        assert hasattr(engine, 'select_tools')

    def test_analyze_prompt_extracts_requirements(self, mock_llm_client, sample_prompt, sample_prompt_analysis):
        """Test analyze_prompt method extracts crew requirements from natural language."""
        from crewforge.core.generator import GenerationEngine
        
        # Mock LLM response
        mock_llm_client.generate.return_value = sample_prompt_analysis
        
        engine = GenerationEngine(llm_client=mock_llm_client)
        result = engine.analyze_prompt(sample_prompt)
        
        assert result["business_context"] == "content research and summarization"
        assert "researcher" in result["required_roles"]
        assert "summarizer" in result["required_roles"]
        assert "find articles" in result["objectives"]
        assert "web_search" in result["tools_needed"]
        
        # Verify LLM was called with appropriate prompt
        mock_llm_client.generate.assert_called_once()
        call_kwargs = mock_llm_client.generate.call_args.kwargs
        assert sample_prompt in call_kwargs["user_prompt"]
        assert call_kwargs["use_json_mode"] is True

    def test_generate_agents_creates_valid_configurations(self, mock_llm_client, sample_prompt_analysis, sample_generated_agents):
        """Test generate_agents creates valid AgentConfig objects."""
        from crewforge.core.generator import GenerationEngine
        
        # Mock LLM response
        mock_llm_client.generate.return_value = {"agents": sample_generated_agents}
        
        engine = GenerationEngine(llm_client=mock_llm_client)
        result = engine.generate_agents(sample_prompt_analysis)
        
        assert len(result) == 2
        assert all(isinstance(agent, AgentConfig) for agent in result)
        
        # Check first agent
        assert result[0].role == "Content Researcher"
        assert "find" in result[0].goal.lower() or "articles" in result[0].goal.lower()
        assert "experienced researcher" in result[0].backstory
        
        # Verify LLM was called with prompt analysis
        mock_llm_client.generate.assert_called_once()

    def test_generate_tasks_creates_valid_configurations(self, mock_llm_client, sample_agent_configs, sample_generated_tasks):
        """Test generate_tasks creates valid TaskConfig objects with proper agent assignment."""
        from crewforge.core.generator import GenerationEngine
        
        # Mock LLM response
        mock_llm_client.generate.return_value = {"tasks": sample_generated_tasks}
        
        engine = GenerationEngine(llm_client=mock_llm_client)
        result = engine.generate_tasks(sample_agent_configs, {"objectives": ["find articles", "summarize content"]})
        
        assert len(result) == 2
        assert all(isinstance(task, TaskConfig) for task in result)
        
        # Check first task
        assert result[0].agent == "Content Researcher"
        assert "search" in result[0].description.lower()
        assert "articles" in result[0].expected_output.lower()
        
        # Check second task has context dependency
        assert result[1].agent == "Content Summarizer"
        assert result[1].context is not None
        
        # Verify LLM was called
        mock_llm_client.generate.assert_called_once()

    def test_select_tools_chooses_appropriate_tools(self, mock_llm_client):
        """Test select_tools chooses appropriate CrewAI tools from library."""
        from crewforge.core.generator import GenerationEngine
        
        tools_needed = ["web_search", "text_analysis", "file_writing"]
        mock_llm_client.generate.return_value = {
            "selected_tools": [
                {"name": "SerperDevTool", "reason": "For web search capabilities"},
                {"name": "FileWriterTool", "reason": "For saving output files"}
            ],
            "unavailable_tools": ["CustomTextAnalyzer"]
        }
        
        engine = GenerationEngine(llm_client=mock_llm_client)
        result = engine.select_tools(tools_needed)
        
        assert "selected_tools" in result
        assert "unavailable_tools" in result
        assert len(result["selected_tools"]) >= 1
        
        # Verify tools are validated for availability
        mock_llm_client.generate.assert_called_once()

    def test_generation_pipeline_quality_controls(self, mock_llm_client):
        """Test generation pipeline includes quality controls for compatibility."""
        from crewforge.core.generator import GenerationEngine
        
        engine = GenerationEngine(llm_client=mock_llm_client)
        
        # Should have methods for quality controls
        assert hasattr(engine, '_validate_agent_task_alignment')
        assert hasattr(engine, '_validate_tool_availability') 
        assert hasattr(engine, '_validate_output_format')

    def test_agent_task_alignment_validation(self, mock_llm_client):
        """Test that agents and tasks are properly aligned."""
        from crewforge.core.generator import GenerationEngine
        
        engine = GenerationEngine(llm_client=mock_llm_client)
        
        agents = [
            AgentConfig(role="Researcher", goal="Find articles", backstory="Expert researcher"),
        ]
        tasks = [
            TaskConfig(description="Search for articles", expected_output="Article list", agent="Researcher"),
            TaskConfig(description="Analyze data", expected_output="Analysis report", agent="NonexistentAgent"),
        ]
        
        # Should detect misaligned agent assignment
        is_valid, errors = engine._validate_agent_task_alignment(agents, tasks)
        assert not is_valid
        assert any("NonexistentAgent" in error for error in errors)

    def test_tool_availability_validation(self, mock_llm_client):
        """Test tool availability validation against CrewAI library."""
        from crewforge.core.generator import GenerationEngine
        
        engine = GenerationEngine(llm_client=mock_llm_client)
        
        # Test with mix of available and unavailable tools
        tools = ["SerperDevTool", "FileWriterTool", "NonexistentTool"]
        available_tools, unavailable_tools = engine._validate_tool_availability(tools)
        
        assert "NonexistentTool" in unavailable_tools
        # Should have some mechanism to check real tool availability

    def test_output_format_consistency(self, mock_llm_client):
        """Test output format consistency validation."""
        from crewforge.core.generator import GenerationEngine
        
        engine = GenerationEngine(llm_client=mock_llm_client)
        
        # Test with valid configuration
        config_data = {
            "agents": [{"role": "Test", "goal": "Test goal", "backstory": "Test story"}],
            "tasks": [{"description": "Test task", "expected_output": "Test output", "agent": "Test"}]
        }
        
        is_valid, errors = engine._validate_output_format(config_data)
        assert is_valid
        assert len(errors) == 0

    def test_generation_engine_error_handling(self, mock_llm_client):
        """Test GenerationEngine handles errors gracefully."""
        from crewforge.core.generator import GenerationEngine, GenerationError
        
        # Mock LLM client to raise exception
        mock_llm_client.generate.side_effect = Exception("LLM API Error")
        
        engine = GenerationEngine(llm_client=mock_llm_client)
        
        with pytest.raises(GenerationError):
            engine.analyze_prompt("test prompt")

    def test_generation_engine_with_default_llm_client(self):
        """Test GenerationEngine can be initialized with default LLM client."""
        from crewforge.core.generator import GenerationEngine
        
        with patch('crewforge.core.generator.LLMClient') as mock_llm_class:
            mock_client = Mock()
            mock_llm_class.return_value = mock_client
            
            engine = GenerationEngine()
            assert engine.llm_client == mock_client
            mock_llm_class.assert_called_once()
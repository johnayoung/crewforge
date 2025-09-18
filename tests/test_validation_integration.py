"""
Integration tests for specification validation with existing prompt parsing functionality.

These tests ensure the validation system works correctly with the actual
prompt parsing output from the LLM client.
"""

import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any

from crewforge.llm import LLMClient, LLMError
from crewforge.prompt_templates import PromptTemplates, PromptTemplateError
from crewforge.validation import SpecificationValidator, ValidationError


class TestValidationIntegration:
    """Test integration between validation and existing components."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client for testing."""
        mock_client = AsyncMock(spec=LLMClient)
        return mock_client

    @pytest.fixture
    def prompt_templates(self, mock_llm_client):
        """Create PromptTemplates instance with mock LLM client."""
        return PromptTemplates(mock_llm_client)

    @pytest.fixture
    def validator(self):
        """Create SpecificationValidator instance."""
        return SpecificationValidator()

    @pytest.fixture
    def valid_llm_response(self) -> Dict[str, Any]:
        """Create a valid LLM response that should pass validation."""
        return {
            "project_name": "content-research-team",
            "project_description": "AI-powered content research and analysis team",
            "agents": [
                {
                    "role": "Research Specialist",
                    "goal": "Gather comprehensive information on specified topics",
                    "backstory": "Expert researcher with extensive experience in data collection and analysis",
                    "tools": ["web_search", "document_analyzer", "data_extractor"],
                },
                {
                    "role": "Content Analyst",
                    "goal": "Analyze and synthesize research findings into actionable insights",
                    "backstory": "Analytical expert specializing in content evaluation and insight generation",
                    "tools": [
                        "text_analyzer",
                        "sentiment_analyzer",
                        "report_generator",
                    ],
                },
            ],
            "tasks": [
                {
                    "description": "Research current trends in artificial intelligence",
                    "expected_output": "Comprehensive report with key findings and trend analysis",
                    "agent": "Research Specialist",
                },
                {
                    "description": "Analyze research findings and create executive summary",
                    "expected_output": "Executive summary with key insights and recommendations",
                    "agent": "Content Analyst",
                },
            ],
            "dependencies": [
                "crewai",
                "requests",
                "beautifulsoup4",
                "pandas",
                "matplotlib",
            ],
        }

    async def test_valid_llm_response_passes_validation(
        self, prompt_templates, validator, mock_llm_client, valid_llm_response
    ):
        """Test that valid LLM response passes validation."""
        # Mock the LLM client to return valid response
        mock_llm_client.complete_structured.return_value = valid_llm_response

        # Extract specification using prompt templates
        spec = await prompt_templates.extract_project_spec(
            "Build a content research team"
        )

        # Validate the specification
        result = validator.validate(spec)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.completeness_score > 0.9  # Should be high quality

    async def test_invalid_llm_response_fails_validation(
        self, prompt_templates, validator, mock_llm_client
    ):
        """Test that invalid LLM response fails validation."""
        # Mock LLM client to return invalid response
        invalid_response = {
            "project_name": "Invalid Name With Spaces",  # Invalid format
            "project_description": "",  # Empty description
            "agents": [],  # Empty agents list
            "tasks": [],  # Empty tasks list
            "dependencies": [],  # Missing crewai dependency
        }
        mock_llm_client.complete_structured.return_value = invalid_response

        # Extract specification - this should fail with PromptTemplateError
        with pytest.raises(PromptTemplateError):
            await prompt_templates.extract_project_spec("Bad prompt")

        # But if we bypass prompt template validation and validate directly
        result = validator.validate(invalid_response)

        assert result.is_valid is False
        assert len(result.errors) >= 4  # Multiple errors expected
        assert result.completeness_score < 0.5  # Poor quality

    async def test_validation_with_prompt_template_validation(
        self, prompt_templates, validator, mock_llm_client
    ):
        """Test that our validation system works alongside existing prompt template validation."""
        # Mock LLM response with task referencing non-existent agent
        problematic_response = {
            "project_name": "test-project",
            "project_description": "Test project",
            "agents": [
                {
                    "role": "Worker",
                    "goal": "Do work",
                    "backstory": "A worker",
                    "tools": ["tool1"],
                }
            ],
            "tasks": [
                {
                    "description": "Do something",
                    "expected_output": "Something done",
                    "agent": "Non-Existent Worker",  # This should be caught
                }
            ],
            "dependencies": ["crewai"],
        }
        mock_llm_client.complete_structured.return_value = problematic_response

        # The prompt template should catch this during its own validation
        with pytest.raises(PromptTemplateError) as exc_info:
            await prompt_templates.extract_project_spec("Test prompt")

        # Verify the error message mentions the agent reference issue
        assert "Non-Existent Worker" in str(exc_info.value)

    async def test_validation_identifies_issues_missed_by_prompt_templates(
        self, validator, valid_llm_response
    ):
        """Test that our validator catches issues that might be missed by prompt templates."""
        # Modify the response to have issues not caught by basic validation
        response_with_warnings = valid_llm_response.copy()

        # Add duplicate agent role (should generate warning)
        duplicate_agent = response_with_warnings["agents"][0].copy()
        duplicate_agent["backstory"] = "Another agent with same role"
        response_with_warnings["agents"].append(duplicate_agent)

        # Add unused agent
        unused_agent = {
            "role": "Unused Agent",
            "goal": "Never used",
            "backstory": "This agent is not used in any task",
            "tools": [],
        }
        response_with_warnings["agents"].append(unused_agent)

        # Validate the specification
        result = validator.validate(response_with_warnings)

        # Should still be valid but have warnings
        assert result.is_valid is True
        assert len(result.warnings) >= 2  # Duplicate role + unused agent
        assert any(
            "duplicate" in warning.message.lower() for warning in result.warnings
        )
        assert any("unused" in warning.message.lower() for warning in result.warnings)

    def test_validation_completeness_scoring(self, validator):
        """Test that completeness scoring works correctly for different quality levels."""
        # High-quality specification
        high_quality = {
            "project_name": "excellent-project",
            "project_description": "Well-described project with clear goals",
            "agents": [
                {
                    "role": "Expert Agent",
                    "goal": "Detailed goal description",
                    "backstory": "Comprehensive backstory with relevant experience",
                    "tools": ["tool1", "tool2", "tool3"],
                }
            ],
            "tasks": [
                {
                    "description": "Detailed task description with clear requirements",
                    "expected_output": "Specific output format with measurable criteria",
                    "agent": "Expert Agent",
                }
            ],
            "dependencies": ["crewai", "requests", "pandas", "matplotlib", "numpy"],
        }

        result_high = validator.validate(high_quality)
        assert result_high.completeness_score > 0.9

        # Low-quality specification (but still valid)
        low_quality = {
            "project_name": "basic-project",
            "project_description": "Basic project",
            "agents": [
                {
                    "role": "Agent",
                    "goal": "Do stuff",
                    "backstory": "An agent",
                    "tools": [],
                }
            ],
            "tasks": [
                {"description": "Task", "expected_output": "Output", "agent": "Agent"}
            ],
            "dependencies": ["crewai"],
        }

        result_low = validator.validate(low_quality)
        assert result_low.is_valid is True  # Still valid
        assert result_low.completeness_score < result_high.completeness_score

    def test_validation_error_reporting_format(self, validator):
        """Test that validation errors are reported in a user-friendly format."""
        invalid_spec = {
            "project_name": "Invalid Name",
            "project_description": "",
            "agents": [{"role": "Agent"}],  # Missing required fields
            "tasks": [],
            "dependencies": [],
        }

        result = validator.validate(invalid_spec)

        # Check error reporting format
        result_str = str(result)
        assert "FAILED" in result_str
        assert "Completeness Score" in result_str
        assert "Errors" in result_str

        # Verify specific error messages
        error_messages = [error.message for error in result.errors]
        assert any("project name" in msg.lower() for msg in error_messages)
        assert any(
            "description cannot be empty" in msg.lower() for msg in error_messages
        )
        assert any("task" in msg.lower() for msg in error_messages)

    def test_validation_schema_compatibility_with_prompt_templates(self, validator):
        """Test that validation schema is compatible with prompt template schema."""
        from crewforge.prompt_templates import PromptTemplates

        # Get the schema from prompt templates
        prompt_schema = PromptTemplates.get_project_spec_schema()

        # Create a specification that matches the prompt template schema
        spec_matching_schema = {
            "project_name": "schema-test",
            "project_description": "Test specification matching the prompt template schema",
            "agents": [
                {
                    "role": "Test Agent",
                    "goal": "Test the schema compatibility",
                    "backstory": "Agent designed to test schema compatibility",
                    "tools": ["test_tool"],
                }
            ],
            "tasks": [
                {
                    "description": "Validate schema compatibility",
                    "expected_output": "Confirmation that schemas are compatible",
                    "agent": "Test Agent",
                }
            ],
            "dependencies": ["crewai", "jsonschema"],
        }

        # This should pass validation if schemas are compatible
        result = validator.validate(spec_matching_schema)
        assert result.is_valid is True


class TestValidationWithRealLLMScenarios:
    """Test validation with realistic LLM output scenarios."""

    @pytest.fixture
    def validator(self):
        """Create SpecificationValidator instance."""
        return SpecificationValidator()

    def test_validation_with_llm_hallucinations(self, validator):
        """Test validation handles common LLM hallucinations gracefully."""
        # LLM might hallucinate invalid field types
        hallucinated_spec = {
            "project_name": "test-project",
            "project_description": "Test description",
            "agents": [
                {
                    "role": "Agent",
                    "goal": "Do work",
                    "backstory": "Worker agent",
                    "tools": "invalid_string_instead_of_list",  # Common hallucination
                }
            ],
            "tasks": [
                {
                    "description": "Task description",
                    "expected_output": "Output",
                    "agent": "Agent",
                }
            ],
            "dependencies": ["crewai"],
        }

        result = validator.validate(hallucinated_spec)
        assert result.is_valid is False
        assert any("tools must be a list" in error.message for error in result.errors)

    def test_validation_with_incomplete_llm_output(self, validator):
        """Test validation with incomplete LLM responses."""
        # Simulate LLM providing incomplete response
        incomplete_spec = {
            "project_name": "incomplete-project",
            # Missing project_description
            "agents": [
                {
                    "role": "Partial Agent",
                    "goal": "Partially defined goal",
                    # Missing backstory and tools
                }
            ],
            "tasks": [
                {
                    "description": "Incomplete task"
                    # Missing expected_output and agent
                }
            ],
            # Missing dependencies
        }

        result = validator.validate(incomplete_spec)
        assert result.is_valid is False

        # Check that all missing fields are reported
        error_messages = [error.message for error in result.errors]
        assert any("project_description" in msg for msg in error_messages)
        assert any("backstory" in msg for msg in error_messages)
        assert any("tools" in msg for msg in error_messages)
        assert any("expected_output" in msg for msg in error_messages)
        assert any("dependencies" in msg for msg in error_messages)

    def test_validation_performance_with_large_specification(self, validator):
        """Test validation performance with large specifications."""
        # Create a large specification
        large_spec = {
            "project_name": "large-project",
            "project_description": "Large project with many agents and tasks",
            "agents": [
                {
                    "role": f"Agent {i}",
                    "goal": f"Goal for agent {i}",
                    "backstory": f"Backstory for agent {i}",
                    "tools": [f"tool_{i}_1", f"tool_{i}_2"],
                }
                for i in range(10)  # 10 agents
            ],
            "tasks": [
                {
                    "description": f"Task {i} description",
                    "expected_output": f"Output for task {i}",
                    "agent": f"Agent {i % 10}",  # Distribute tasks among agents
                }
                for i in range(20)  # 20 tasks
            ],
            "dependencies": ["crewai"] + [f"dependency_{i}" for i in range(10)],
        }

        import time

        start_time = time.time()
        result = validator.validate(large_spec)
        validation_time = time.time() - start_time

        # Validation should complete quickly (under 1 second)
        assert validation_time < 1.0
        assert result.is_valid is True
        assert result.completeness_score > 0.8

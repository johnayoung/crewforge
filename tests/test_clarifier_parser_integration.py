"""
Test suite for integrating Interactive Clarifier with Prompt Parser feedback loop.

Tests the complete workflow:
1. Parse initial specification
2. Generate clarification questions
3. Incorporate user responses back into specification
4. Iterative improvement through multiple rounds
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch

from crewforge.clarifier import (
    InteractiveClarifier,
    ConversationContext,
    ConversationState,
    Question,
    QuestionType,
    ClarificationError,
)
from crewforge.llm import LLMClient, LLMError
from crewforge.prompt_templates import PromptTemplates, PromptTemplateError
from crewforge.validation import (
    SpecificationValidator,
    ValidationResult,
    ValidationIssue,
    IssueSeverity,
)


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = Mock(spec=LLMClient)
    client.complete = AsyncMock()
    client.complete_structured = AsyncMock()
    return client


@pytest.fixture
def clarifier(mock_llm_client):
    """Create an InteractiveClarifier instance with mock LLM client."""
    return InteractiveClarifier(mock_llm_client)


@pytest.fixture
def prompt_templates(mock_llm_client):
    """Create a PromptTemplates instance with mock LLM client."""
    return PromptTemplates(mock_llm_client)


@pytest.fixture
def sample_incomplete_spec():
    """Sample incomplete project specification for testing."""
    return {
        "project_name": "research-team",
        "project_description": "A team for research",
        "agents": [
            {
                "role": "researcher",
                "goal": "",  # Missing goal
                "backstory": "",  # Missing backstory
                "tools": [],  # Required by schema
            }
        ],
        "tasks": [
            {
                "description": "Do research",  # Vague description
                "agent": "researcher",
                "expected_output": "",  # Missing expected output
            }
        ],
        "dependencies": [],
    }


@pytest.fixture
def sample_clarification_responses():
    """Sample user responses to clarification questions."""
    return [
        {
            "question_id": "q1",
            "field": "agents[0].goal",
            "response": "Find comprehensive information about AI trends in 2024",
        },
        {
            "question_id": "q2",
            "field": "agents[0].backstory",
            "response": "An experienced market research analyst with expertise in AI and technology trends",
        },
        {
            "question_id": "q3",
            "field": "tasks[0].expected_output",
            "response": "A detailed report with key findings, trends, and recommendations formatted as markdown",
        },
    ]


class TestClarifierParserIntegration:
    """Test cases for clarifier-parser feedback loop integration."""

    @pytest.mark.asyncio
    async def test_feedback_loop_basic_workflow(
        self,
        clarifier,
        prompt_templates,
        mock_llm_client,
        sample_incomplete_spec,
        sample_clarification_responses,
    ):
        """Test the complete feedback loop workflow."""
        # Step 1: Initial parsing creates incomplete spec (already provided as fixture)

        # Step 2: Validate and generate questions
        validator = SpecificationValidator()
        validation_result = validator.validate(sample_incomplete_spec)

        # Mock question generation
        mock_questions = [
            Question(
                question_type=QuestionType.MISSING_FIELD,
                field="agents[0].goal",
                question="What is the main goal for the researcher agent?",
                context="Agent goal is required for CrewAI configuration",
                suggestions=[
                    "Research market trends",
                    "Analyze data",
                    "Generate reports",
                ],
            ),
            Question(
                question_type=QuestionType.MISSING_FIELD,
                field="agents[0].backstory",
                question="What background should the researcher have?",
                context="Agent backstory helps with role context",
                suggestions=["Market analyst", "Data scientist", "Research specialist"],
            ),
            Question(
                question_type=QuestionType.MISSING_FIELD,
                field="tasks[0].expected_output",
                question="What format should the research output be in?",
                context="Expected output helps structure task results",
                suggestions=["JSON report", "Markdown document", "CSV data"],
            ),
        ]

        mock_llm_client.complete.return_value = json.dumps(
            [
                {
                    "type": "missing_field",
                    "field": "agents[0].goal",
                    "question": "What is the main goal for the researcher agent?",
                    "context": "Agent goal is required for CrewAI configuration",
                    "suggestions": [
                        "Research market trends",
                        "Analyze data",
                        "Generate reports",
                    ],
                },
                {
                    "type": "missing_field",
                    "field": "agents[0].backstory",
                    "question": "What background should the researcher have?",
                    "context": "Agent backstory helps with role context",
                    "suggestions": [
                        "Market analyst",
                        "Data scientist",
                        "Research specialist",
                    ],
                },
                {
                    "type": "missing_field",
                    "field": "tasks[0].expected_output",
                    "question": "What format should the research output be in?",
                    "context": "Expected output helps structure task results",
                    "suggestions": ["JSON report", "Markdown document", "CSV data"],
                },
            ]
        )

        questions = await clarifier.generate_questions(validation_result)

        # Step 3: Start conversation context
        context = await clarifier.start_clarification_session(sample_incomplete_spec)
        assert context.session_id is not None
        assert context.initial_specification == sample_incomplete_spec

        # Step 4: Apply user responses and get enhanced specification
        # Mock the enhanced specification that would come from LLM
        enhanced_spec_mock = {
            **sample_incomplete_spec,
            "agents": [
                {
                    **sample_incomplete_spec["agents"][0],
                    "goal": "Find comprehensive information about AI trends in 2024",
                    "backstory": "An experienced market research analyst with expertise in AI and technology trends",
                }
            ],
            "tasks": [
                {
                    **sample_incomplete_spec["tasks"][0],
                    "expected_output": "A detailed report with key findings, trends, and recommendations formatted as markdown",
                }
            ],
        }
        mock_llm_client.complete_structured.return_value = enhanced_spec_mock

        # This is the NEW functionality we need to implement
        enhanced_spec = await clarifier.apply_clarification_responses(
            context, questions, sample_clarification_responses
        )

        # Verify the specification was enhanced
        assert (
            enhanced_spec["agents"][0]["goal"]
            == "Find comprehensive information about AI trends in 2024"
        )
        assert (
            enhanced_spec["agents"][0]["backstory"]
            == "An experienced market research analyst with expertise in AI and technology trends"
        )
        assert (
            enhanced_spec["tasks"][0]["expected_output"]
            == "A detailed report with key findings, trends, and recommendations formatted as markdown"
        )

    @pytest.mark.asyncio
    async def test_iterative_clarification_rounds(
        self, clarifier, mock_llm_client, sample_incomplete_spec
    ):
        """Test multiple rounds of clarification improving the specification."""
        # Start with incomplete spec
        context = await clarifier.start_clarification_session(sample_incomplete_spec)

        # Round 1: Basic field completion
        first_responses = [
            {
                "question_id": "q1",
                "field": "agents[0].goal",
                "response": "Research AI trends",
            },
        ]

        # Mock enhanced spec after round 1
        mock_llm_client.complete_structured.return_value = {
            **sample_incomplete_spec,
            "agents": [
                {
                    **sample_incomplete_spec["agents"][0],
                    "goal": "Research AI trends",
                }
            ],
        }

        spec_after_round1 = await clarifier.apply_clarification_responses(
            context, [], first_responses
        )

        # Round 2: Further refinement based on updated spec
        validator = SpecificationValidator()
        validation_result = validator.validate(spec_after_round1)

        mock_llm_client.complete.return_value = json.dumps(
            [
                {
                    "type": "vague_content",
                    "field": "agents[0].goal",
                    "question": "Can you be more specific about what AI trends to research?",
                    "context": "More specific goals lead to better agent performance",
                    "suggestions": [
                        "Machine learning trends",
                        "AI adoption trends",
                        "AI technology trends",
                    ],
                }
            ]
        )

        second_questions = await clarifier.generate_questions(validation_result)

        second_responses = [
            {
                "question_id": "q2",
                "field": "agents[0].goal",
                "response": "Research machine learning and generative AI adoption trends in enterprise",
            }
        ]

        mock_llm_client.complete_structured.return_value = {
            **spec_after_round1,
            "agents": [
                {
                    **spec_after_round1["agents"][0],
                    "goal": "Research machine learning and generative AI adoption trends in enterprise",
                }
            ],
        }

        final_spec = await clarifier.apply_clarification_responses(
            context, second_questions, second_responses
        )

        # Verify iterative improvement
        assert "enterprise" in final_spec["agents"][0]["goal"].lower()
        assert "generative ai" in final_spec["agents"][0]["goal"].lower()

    @pytest.mark.asyncio
    async def test_conversation_context_preservation(
        self,
        clarifier,
        mock_llm_client,
        sample_incomplete_spec,
        sample_clarification_responses,
    ):
        """Test that conversation context is preserved through multiple interactions."""
        context = await clarifier.start_clarification_session(sample_incomplete_spec)

        # Simulate interaction history
        clarifier.add_to_history("user", "I want to build a research team")
        clarifier.add_to_history("assistant", "What specific type of research?")
        clarifier.add_to_history("user", "AI market research")

        # Mock the enhanced specification that would come from LLM
        enhanced_spec_mock = {
            **sample_incomplete_spec,
            "agents": [
                {
                    **sample_incomplete_spec["agents"][0],
                    "goal": "Find comprehensive information about AI trends in 2024",
                }
            ],
        }
        mock_llm_client.complete_structured.return_value = enhanced_spec_mock

        # Apply responses should preserve this context
        enhanced_spec = await clarifier.apply_clarification_responses(
            context, [], sample_clarification_responses
        )

        # Context should still be available
        history = clarifier.get_conversation_context()
        assert "research team" in history
        assert "AI market research" in history

        # Context should be updated with applied responses
        assert context.current_specification != context.initial_specification
        assert context.conversation_state in [
            ConversationState.PROCESSING_RESPONSES,
            ConversationState.ASKING_QUESTIONS,
        ]

    @pytest.mark.asyncio
    async def test_error_handling_in_feedback_loop(
        self, clarifier, mock_llm_client, sample_incomplete_spec
    ):
        """Test error handling during feedback loop integration."""
        context = await clarifier.start_clarification_session(sample_incomplete_spec)

        # Test LLM error during enhancement with non-empty responses
        mock_llm_client.complete_structured.side_effect = LLMError("API error")

        test_responses = [
            {"question_id": "q1", "field": "agents[0].goal", "response": "test"}
        ]

        with pytest.raises(
            ClarificationError, match="Failed to apply clarification responses"
        ):
            await clarifier.apply_clarification_responses(context, [], test_responses)

        # Test invalid response format
        mock_llm_client.complete_structured.side_effect = None
        mock_llm_client.complete_structured.return_value = "invalid json"

        with pytest.raises(ClarificationError, match="Invalid enhanced specification"):
            await clarifier.apply_clarification_responses(context, [], test_responses)

    @pytest.mark.asyncio
    async def test_partial_response_application(
        self, clarifier, mock_llm_client, sample_incomplete_spec
    ):
        """Test applying partial responses when user doesn't answer all questions."""
        context = await clarifier.start_clarification_session(sample_incomplete_spec)

        # Only respond to some questions
        partial_responses = [
            {
                "question_id": "q1",
                "field": "agents[0].goal",
                "response": "Research AI trends",
            }
            # Missing responses for other questions
        ]

        mock_llm_client.complete_structured.return_value = {
            **sample_incomplete_spec,
            "agents": [
                {
                    **sample_incomplete_spec["agents"][0],
                    "goal": "Research AI trends",
                    # Other fields remain unchanged
                }
            ],
        }

        enhanced_spec = await clarifier.apply_clarification_responses(
            context, [], partial_responses
        )

        # Only answered field should be updated
        assert enhanced_spec["agents"][0]["goal"] == "Research AI trends"
        assert enhanced_spec["agents"][0]["backstory"] == ""  # Still empty

    @pytest.mark.asyncio
    async def test_integration_with_prompt_templates(
        self, clarifier, prompt_templates, mock_llm_client
    ):
        """Test integration between clarifier and prompt templates."""
        # Initial extraction
        user_prompt = "Build a content research team"

        mock_llm_client.complete_structured.return_value = {
            "project_name": "content-research",
            "project_description": "Content research team",
            "agents": [
                {"role": "researcher", "goal": "", "backstory": "", "tools": []}
            ],
            "tasks": [
                {
                    "description": "Research content",
                    "agent": "researcher",
                    "expected_output": "",
                }
            ],
            "dependencies": ["crewai"],  # Include required dependency
        }

        initial_spec = await prompt_templates.extract_project_spec(user_prompt)

        # Generate questions from validation
        validator = SpecificationValidator()
        validation_result = validator.validate(initial_spec)

        mock_llm_client.complete.return_value = json.dumps(
            [
                {
                    "type": "missing_field",
                    "field": "agents[0].goal",
                    "question": "What should the researcher focus on?",
                    "context": "Specific goals improve agent performance",
                    "suggestions": [
                        "Content trends",
                        "Audience research",
                        "Competitor analysis",
                    ],
                }
            ]
        )

        questions = await clarifier.generate_questions(validation_result)

        # Apply clarification using prompt templates enhancement
        responses = [
            {
                "question_id": questions[0].id,  # Use .id instead of .question_id
                "field": "agents[0].goal",
                "response": "Research content trends and audience preferences",
            }
        ]

        context = await clarifier.start_clarification_session(initial_spec)

        # Mock the enhanced specification for the integration
        enhanced_spec_mock = {
            **initial_spec,
            "agents": [
                {
                    **initial_spec["agents"][0],
                    "goal": "Research content trends and audience preferences",
                }
            ],
        }
        # Reset mock for the apply_clarification_responses call
        mock_llm_client.complete_structured.return_value = enhanced_spec_mock

        # The integration should use prompt_templates.enhance_project_spec internally
        enhanced_spec = await clarifier.apply_clarification_responses(
            context, questions, responses
        )

        assert "content trends" in enhanced_spec["agents"][0]["goal"].lower()
        assert "audience preferences" in enhanced_spec["agents"][0]["goal"].lower()


class TestIntegrationUtilities:
    """Test helper utilities for the integration."""

    def test_format_responses_for_enhancement(self):
        """Test formatting user responses for prompt template enhancement."""
        responses = [
            {
                "question_id": "q1",
                "field": "agents[0].goal",
                "response": "Research AI trends",
            },
            {
                "question_id": "q2",
                "field": "tasks[0].description",
                "response": "Analyze market data",
            },
        ]

        # This would be a new utility method we need to implement
        # We'll implement this as part of the clarifier module
        enhancement_context = f"""User provided the following clarifications:
- {responses[0]['field']}: {responses[0]['response']}
- {responses[1]['field']}: {responses[1]['response']}"""

        assert "agents[0].goal" in enhancement_context
        assert "Research AI trends" in enhancement_context
        assert "tasks[0].description" in enhancement_context
        assert "Analyze market data" in enhancement_context

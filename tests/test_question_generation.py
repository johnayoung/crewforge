"""
Tests for the Interactive Clarifier's question generation functionality.

This module tests the generation of targeted questions for missing or ambiguous
requirements in CrewAI project specifications.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from crewforge.validation import (
    ValidationResult,
    ValidationIssue,
    IssueSeverity,
    SpecificationValidator,
)
from crewforge.llm import LLMClient, LLMError
from crewforge.clarifier import (
    InteractiveClarifier,
    ClarificationError,
    Question,
    QuestionType,
)


class TestInteractiveClarifier:
    """Test the Interactive Clarifier component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm_client = Mock(spec=LLMClient)
        self.clarifier = InteractiveClarifier(llm_client=self.llm_client)

    def test_clarifier_initialization(self):
        """Test that clarifier initializes correctly."""
        assert self.clarifier.llm_client is self.llm_client
        assert self.clarifier.conversation_history == []

    @pytest.mark.asyncio
    async def test_generate_questions_for_missing_fields(self):
        """Test question generation for missing required fields."""
        # Create validation result with missing field errors
        issues = [
            ValidationIssue(
                IssueSeverity.ERROR,
                "Required field 'project_description' is missing",
                "project_description",
            ),
            ValidationIssue(
                IssueSeverity.ERROR, "At least one agent is required", "agents"
            ),
        ]
        validation_result = ValidationResult(issues=issues, completeness_score=0.3)

        # Mock LLM response
        self.llm_client.complete = AsyncMock(
            return_value="""
        [
            {
                "type": "missing_field",
                "field": "project_description", 
                "question": "Can you provide a brief description of what your CrewAI project should accomplish?",
                "context": "A project description helps define the overall goals and scope.",
                "suggestions": ["e.g., 'Analyze market trends for content strategy'", "e.g., 'Process customer support tickets automatically'"]
            },
            {
                "type": "missing_field",
                "field": "agents",
                "question": "What types of agents (roles) do you need for this project?",
                "context": "Agents are the AI workers that will perform tasks in your CrewAI project.",
                "suggestions": ["Research Agent", "Content Writer", "Data Analyst", "Social Media Manager"]
            }
        ]
        """
        )

        questions = await self.clarifier.generate_questions(validation_result)

        assert len(questions) == 2
        assert questions[0].question_type == QuestionType.MISSING_FIELD
        assert questions[0].field == "project_description"
        assert "brief description" in questions[0].question
        assert len(questions[0].suggestions) >= 2

        assert questions[1].question_type == QuestionType.MISSING_FIELD
        assert questions[1].field == "agents"
        assert "types of agents" in questions[1].question

    @pytest.mark.asyncio
    async def test_generate_questions_for_vague_content(self):
        """Test question generation for vague or ambiguous content."""
        issues = [
            ValidationIssue(
                IssueSeverity.WARNING,
                "Agent role 'assistant' is vague",
                "agents[0].role",
            ),
            ValidationIssue(
                IssueSeverity.WARNING,
                "Task description appears vague",
                "tasks[0].description",
            ),
        ]
        validation_result = ValidationResult(issues=issues, completeness_score=0.7)

        self.llm_client.complete = AsyncMock(
            return_value="""
        [
            {
                "type": "vague_content",
                "field": "agents[0].role",
                "question": "What specific role should this agent perform? 'Assistant' is quite general.",
                "context": "Specific roles like 'Research Analyst' or 'Content Writer' work better than generic names.",
                "suggestions": ["Research Analyst", "Content Writer", "Data Processor", "Quality Reviewer"]
            },
            {
                "type": "vague_content", 
                "field": "tasks[0].description",
                "question": "Can you provide more specific details about what this task should accomplish?",
                "context": "Clear task descriptions help agents understand exactly what they need to do.",
                "suggestions": ["Analyze competitor pricing data", "Write blog post about market trends", "Process customer feedback data"]
            }
        ]
        """
        )

        questions = await self.clarifier.generate_questions(validation_result)

        assert len(questions) == 2
        assert questions[0].question_type == QuestionType.VAGUE_CONTENT
        assert questions[0].field == "agents[0].role"
        assert "specific role" in questions[0].question

        assert questions[1].question_type == QuestionType.VAGUE_CONTENT
        assert questions[1].field == "tasks[0].description"
        assert "specific details" in questions[1].question

    @pytest.mark.asyncio
    async def test_generate_questions_for_tool_mismatches(self):
        """Test question generation for tool-role mismatches."""
        issues = [
            ValidationIssue(
                IssueSeverity.WARNING,
                "Agent role 'writer' may not match assigned tools",
                "agents[0].tools",
            ),
        ]
        validation_result = ValidationResult(issues=issues, completeness_score=0.8)

        self.llm_client.complete = AsyncMock(
            return_value="""
        [
            {
                "type": "tool_mismatch",
                "field": "agents[0].tools", 
                "question": "The tools assigned to your 'writer' agent might not be the best fit. What specific writing tasks will this agent perform?",
                "context": "Writers typically need tools like document editors, grammar checkers, or content generators.",
                "suggestions": ["document_writer", "grammar_checker", "content_generator", "text_editor"]
            }
        ]
        """
        )

        questions = await self.clarifier.generate_questions(validation_result)

        assert len(questions) == 1
        assert questions[0].question_type == QuestionType.TOOL_MISMATCH
        assert questions[0].field == "agents[0].tools"
        assert "writing tasks" in questions[0].question
        assert "document_writer" in questions[0].suggestions

    @pytest.mark.asyncio
    async def test_generate_questions_with_no_issues(self):
        """Test that no questions are generated when specification is complete."""
        validation_result = ValidationResult(issues=[], completeness_score=0.95)

        questions = await self.clarifier.generate_questions(validation_result)

        assert questions == []
        # Should not call LLM when no issues to clarify
        self.llm_client.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_questions_llm_error_handling(self):
        """Test error handling when LLM fails to generate questions."""
        issues = [
            ValidationIssue(
                IssueSeverity.ERROR,
                "Required field 'project_description' is missing",
                "project_description",
            ),
        ]
        validation_result = ValidationResult(issues=issues, completeness_score=0.3)

        self.llm_client.complete = AsyncMock(side_effect=LLMError("API Error"))

        with pytest.raises(
            ClarificationError,
            match="Failed to generate clarification questions: API Error",
        ):
            await self.clarifier.generate_questions(validation_result)

    @pytest.mark.asyncio
    async def test_generate_questions_invalid_json_response(self):
        """Test handling of invalid JSON responses from LLM."""
        issues = [
            ValidationIssue(
                IssueSeverity.ERROR, "Required field 'agents' is missing", "agents"
            ),
        ]
        validation_result = ValidationResult(issues=issues, completeness_score=0.4)

        self.llm_client.complete = AsyncMock(return_value="Invalid JSON response")

        with pytest.raises(
            ClarificationError, match="Invalid JSON response from question generation"
        ):
            await self.clarifier.generate_questions(validation_result)

    @pytest.mark.asyncio
    async def test_generate_questions_filters_by_severity(self):
        """Test that only error and warning issues generate questions."""
        issues = [
            ValidationIssue(IssueSeverity.ERROR, "Required field missing", "field1"),
            ValidationIssue(IssueSeverity.WARNING, "Vague content", "field2"),
            ValidationIssue(
                IssueSeverity.INFO, "Info message", "field3"
            ),  # Should be filtered out
        ]
        validation_result = ValidationResult(issues=issues, completeness_score=0.5)

        self.llm_client.complete = AsyncMock(
            return_value="""
        [
            {
                "type": "missing_field",
                "field": "field1",
                "question": "Question 1?",
                "context": "Context 1",
                "suggestions": ["A", "B"]
            },
            {
                "type": "vague_content", 
                "field": "field2",
                "question": "Question 2?",
                "context": "Context 2",
                "suggestions": ["C", "D"]
            }
        ]
        """
        )

        questions = await self.clarifier.generate_questions(validation_result)

        assert len(questions) == 2
        # Should not include INFO level issues
        assert not any(q.field == "field3" for q in questions)

    def test_conversation_history_tracking(self):
        """Test that conversation history is tracked properly."""
        assert self.clarifier.conversation_history == []

        # Add some history
        self.clarifier.add_to_history("user", "What agents do I need?")
        self.clarifier.add_to_history(
            "assistant", "You might need a Research Agent and Writer Agent."
        )

        assert len(self.clarifier.conversation_history) == 2
        assert self.clarifier.conversation_history[0]["role"] == "user"
        assert self.clarifier.conversation_history[1]["role"] == "assistant"

    def test_clear_conversation_history(self):
        """Test clearing conversation history."""
        self.clarifier.add_to_history("user", "Test message")
        assert len(self.clarifier.conversation_history) == 1

        self.clarifier.clear_history()
        assert self.clarifier.conversation_history == []


class TestQuestion:
    """Test the Question data class."""

    def test_question_creation(self):
        """Test creating a Question object."""
        question = Question(
            question_type=QuestionType.MISSING_FIELD,
            field="project_description",
            question="What should your project do?",
            context="Project descriptions help define scope.",
            suggestions=["Example 1", "Example 2"],
        )

        assert question.question_type == QuestionType.MISSING_FIELD
        assert question.field == "project_description"
        assert question.question == "What should your project do?"
        assert question.context == "Project descriptions help define scope."
        assert question.suggestions == ["Example 1", "Example 2"]

    def test_question_with_no_suggestions(self):
        """Test creating a Question without suggestions."""
        question = Question(
            question_type=QuestionType.VAGUE_CONTENT,
            field="agents[0].role",
            question="What specific role?",
            context="Be more specific.",
            suggestions=[],
        )

        assert question.suggestions == []


class TestQuestionType:
    """Test the QuestionType enum."""

    def test_question_type_values(self):
        """Test that all expected question types exist."""
        assert QuestionType.MISSING_FIELD.value == "missing_field"
        assert QuestionType.VAGUE_CONTENT.value == "vague_content"
        assert QuestionType.TOOL_MISMATCH.value == "tool_mismatch"
        assert QuestionType.INCOMPLETE_SPEC.value == "incomplete_spec"


class TestIntegrationWithValidator:
    """Test integration between InteractiveClarifier and SpecificationValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SpecificationValidator()
        self.llm_client = Mock(spec=LLMClient)
        self.clarifier = InteractiveClarifier(llm_client=self.llm_client)

    @pytest.mark.asyncio
    async def test_end_to_end_validation_and_clarification(self):
        """Test complete workflow from validation to question generation."""
        # Create an incomplete specification
        incomplete_spec = {
            "project_name": "test-project",
            "agents": [
                {
                    "role": "agent",  # Vague role
                    "goal": "help",  # Vague goal
                    "backstory": "",  # Empty backstory
                    "tools": [],  # No tools
                }
            ],
            "tasks": [],  # No tasks
            "dependencies": [],  # No dependencies
        }

        # Validate the specification
        validation_result = self.validator.validate(incomplete_spec)

        # Should have issues
        assert len(validation_result.issues) > 0
        assert validation_result.completeness_score < 0.8

        # Mock LLM response for question generation
        self.llm_client.complete = AsyncMock(
            return_value="""
        [
            {
                "type": "missing_field",
                "field": "project_description",
                "question": "What should your project accomplish?",
                "context": "A clear description helps define the project scope.",
                "suggestions": ["Analyze data", "Create content", "Process information"]
            },
            {
                "type": "vague_content",
                "field": "agents[0].role",
                "question": "What specific role should this agent have instead of 'agent'?",
                "context": "Specific roles like 'Research Analyst' work better than generic names.",
                "suggestions": ["Research Analyst", "Content Writer", "Data Processor"]
            }
        ]
        """
        )

        # Generate questions based on validation issues
        questions = await self.clarifier.generate_questions(validation_result)

        assert len(questions) >= 1
        assert any(q.question_type == QuestionType.MISSING_FIELD for q in questions)
        assert any(q.question_type == QuestionType.VAGUE_CONTENT for q in questions)

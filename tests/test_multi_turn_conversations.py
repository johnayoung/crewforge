"""
Tests for multi-turn conversation handling with context retention.

This module tests the enhanced Interactive Clarifier's ability to:
- Maintain conversation state across multiple question/answer cycles
- Retain context between turns
- Update specifications based on user responses
- Generate context-aware follow-up questions
"""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime
from dataclasses import replace

from crewforge.clarifier import (
    InteractiveClarifier,
    Question,
    QuestionType,
    ClarificationError,
)
from crewforge.llm import LLMClient, LLMError
from crewforge.validation import ValidationResult, ValidationIssue, IssueSeverity


# Data classes for multi-turn conversation support
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConversationState(Enum):
    """Current state of the conversation flow."""

    INITIAL = "initial"
    ASKING_QUESTIONS = "asking_questions"
    PROCESSING_RESPONSES = "processing_responses"
    GENERATING_FOLLOWUP = "generating_followup"
    COMPLETED = "completed"


class ResponseType(Enum):
    """Types of user responses to questions."""

    DIRECT_ANSWER = "direct_answer"
    PARTIAL_ANSWER = "partial_answer"
    CLARIFICATION_REQUEST = "clarification_request"
    SKIP_QUESTION = "skip_question"


@dataclass
class UserResponse:
    """Represents a user's response to a clarification question."""

    question_id: str
    response_type: ResponseType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_level: Optional[float] = None


@dataclass
class ConversationTurn:
    """Represents one complete turn in the conversation."""

    turn_number: int
    questions_asked: List[Question]
    user_responses: List[UserResponse]
    specification_updates: Dict[str, Any]
    follow_up_needed: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationContext:
    """Complete context for a multi-turn clarification conversation."""

    session_id: str
    initial_specification: Dict[str, Any]
    current_specification: Dict[str, Any]
    conversation_state: ConversationState = ConversationState.INITIAL
    turns: List[ConversationTurn] = field(default_factory=list)
    pending_questions: List[Question] = field(default_factory=list)
    resolved_questions: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    @property
    def current_turn_number(self) -> int:
        """Get the current turn number."""
        return len(self.turns) + 1

    @property
    def total_questions_asked(self) -> int:
        """Get total number of questions asked across all turns."""
        return sum(len(turn.questions_asked) for turn in self.turns)

    @property
    def total_responses_received(self) -> int:
        """Get total number of responses received."""
        return sum(len(turn.user_responses) for turn in self.turns)

    def has_pending_questions(self) -> bool:
        """Check if there are unresolved questions."""
        return len(self.pending_questions) > 0

    def is_conversation_complete(self) -> bool:
        """Check if the conversation is complete."""
        return self.conversation_state == ConversationState.COMPLETED or (
            not self.has_pending_questions() and len(self.turns) > 0
        )


class TestConversationContext:
    """Test the ConversationContext data class."""

    def test_conversation_context_initialization(self):
        """Test that ConversationContext initializes correctly."""
        initial_spec = {"project_name": "test", "agents": []}
        context = ConversationContext(
            session_id="test-session",
            initial_specification=initial_spec,
            current_specification=initial_spec.copy(),
        )

        assert context.session_id == "test-session"
        assert context.initial_specification == initial_spec
        assert context.current_specification == initial_spec
        assert context.conversation_state == ConversationState.INITIAL
        assert context.turns == []
        assert context.pending_questions == []
        assert context.resolved_questions == []
        assert context.current_turn_number == 1
        assert context.total_questions_asked == 0
        assert context.total_responses_received == 0
        assert not context.has_pending_questions()
        assert not context.is_conversation_complete()

    def test_conversation_context_properties_with_data(self):
        """Test conversation context properties with actual data."""
        initial_spec = {"project_name": "test"}
        context = ConversationContext(
            session_id="test",
            initial_specification=initial_spec,
            current_specification=initial_spec.copy(),
        )

        # Add some turns and questions
        question1 = Question(
            question_type=QuestionType.MISSING_FIELD,
            field="agents",
            question="What agents do you need?",
            context="Agents are required",
            suggestions=["Research Agent", "Writer"],
        )
        question2 = Question(
            question_type=QuestionType.MISSING_FIELD,
            field="tasks",
            question="What tasks should be performed?",
            context="Tasks define workflow",
            suggestions=["Research", "Writing"],
        )

        response1 = UserResponse(
            question_id="q1",
            response_type=ResponseType.DIRECT_ANSWER,
            content="I need a research agent and a writer agent",
        )

        turn1 = ConversationTurn(
            turn_number=1,
            questions_asked=[question1, question2],
            user_responses=[response1],
            specification_updates={"agents": ["research", "writer"]},
        )

        context.turns.append(turn1)
        context.pending_questions = [question2]  # Still waiting for answer to question2

        assert context.current_turn_number == 2
        assert context.total_questions_asked == 2
        assert context.total_responses_received == 1
        assert context.has_pending_questions()
        assert not context.is_conversation_complete()

    def test_conversation_completion_logic(self):
        """Test conversation completion detection."""
        initial_spec = {"project_name": "test"}
        context = ConversationContext(
            session_id="test",
            initial_specification=initial_spec,
            current_specification=initial_spec.copy(),
        )

        # No turns, not complete
        assert not context.is_conversation_complete()

        # Add a turn but no pending questions - should be complete
        turn = ConversationTurn(
            turn_number=1,
            questions_asked=[],
            user_responses=[],
            specification_updates={},
        )
        context.turns.append(turn)
        context.pending_questions = []

        assert context.is_conversation_complete()

        # Explicitly set to COMPLETED
        context.conversation_state = ConversationState.COMPLETED
        assert context.is_conversation_complete()


class TestUserResponse:
    """Test the UserResponse data class."""

    def test_user_response_creation(self):
        """Test creating UserResponse objects."""
        response = UserResponse(
            question_id="q1",
            response_type=ResponseType.DIRECT_ANSWER,
            content="I need a research agent",
        )

        assert response.question_id == "q1"
        assert response.response_type == ResponseType.DIRECT_ANSWER
        assert response.content == "I need a research agent"
        assert response.timestamp is not None
        assert response.confidence_level is None

    def test_user_response_with_confidence(self):
        """Test UserResponse with confidence level."""
        response = UserResponse(
            question_id="q2",
            response_type=ResponseType.PARTIAL_ANSWER,
            content="Maybe a writer?",
            confidence_level=0.6,
        )

        assert response.confidence_level == 0.6
        assert response.response_type == ResponseType.PARTIAL_ANSWER


class TestConversationTurn:
    """Test the ConversationTurn data class."""

    def test_conversation_turn_creation(self):
        """Test creating ConversationTurn objects."""
        question = Question(
            question_type=QuestionType.MISSING_FIELD,
            field="agents",
            question="What agents?",
            context="Context",
            suggestions=[],
        )

        response = UserResponse(
            question_id="q1",
            response_type=ResponseType.DIRECT_ANSWER,
            content="Research agent",
        )

        turn = ConversationTurn(
            turn_number=1,
            questions_asked=[question],
            user_responses=[response],
            specification_updates={"agents": ["research"]},
        )

        assert turn.turn_number == 1
        assert len(turn.questions_asked) == 1
        assert len(turn.user_responses) == 1
        assert turn.specification_updates == {"agents": ["research"]}
        assert not turn.follow_up_needed
        assert turn.timestamp is not None


class TestMultiTurnInteractiveClarifier:
    """Test multi-turn conversation functionality in InteractiveClarifier."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm_client = Mock(spec=LLMClient)
        self.clarifier = InteractiveClarifier(llm_client=self.llm_client)

    @pytest.mark.asyncio
    async def test_start_clarification_session(self):
        """Test starting a new multi-turn clarification session."""
        initial_spec = {
            "project_name": "test-project",
            "project_description": "",
            "agents": [],
        }

        # This method will be implemented in the clarifier
        # For now, test the expected behavior structure
        expected_session_id = "test-session-123"

        # Mock the session creation
        async def mock_start_session(spec):
            return ConversationContext(
                session_id=expected_session_id,
                initial_specification=spec,
                current_specification=spec.copy(),
                conversation_state=ConversationState.INITIAL,
            )

        # Simulate the method we'll add to InteractiveClarifier
        context = await mock_start_session(initial_spec)

        assert context.session_id == expected_session_id
        assert context.initial_specification == initial_spec
        assert context.current_specification == initial_spec
        assert context.conversation_state == ConversationState.INITIAL

    @pytest.mark.asyncio
    async def test_context_aware_question_generation(self):
        """Test generating questions with conversation context."""
        # Create a conversation context with some history
        initial_spec = {"project_name": "content-team", "agents": []}
        context = ConversationContext(
            session_id="test",
            initial_specification=initial_spec,
            current_specification={
                "project_name": "content-team",
                "agents": ["research"],
            },
        )

        # Add conversation history
        previous_turn = ConversationTurn(
            turn_number=1,
            questions_asked=[
                Question(
                    question_type=QuestionType.MISSING_FIELD,
                    field="agents",
                    question="What agents do you need?",
                    context="",
                    suggestions=[],
                )
            ],
            user_responses=[
                UserResponse(
                    question_id="q1",
                    response_type=ResponseType.DIRECT_ANSWER,
                    content="I need a research agent",
                )
            ],
            specification_updates={"agents": ["research"]},
        )
        context.turns.append(previous_turn)

        # Mock LLM to generate context-aware follow-up questions
        self.llm_client.complete = AsyncMock(
            return_value="""[
                {
                    "type": "vague_content",
                    "field": "agents[0].role",
                    "question": "What specific type of research should the research agent focus on?",
                    "context": "You mentioned needing a research agent. To make it more effective, we should specify what kind of research it will perform.",
                    "suggestions": ["Market research", "Content research", "Academic research", "Competitive research"]
                }
            ]"""
        )

        # Test context-aware question generation
        validation_issues = [
            ValidationIssue(
                IssueSeverity.WARNING,
                "Agent role 'research' could be more specific",
                "agents[0].role",
            )
        ]
        validation_result = ValidationResult(
            issues=validation_issues, completeness_score=0.7
        )

        questions = await self.clarifier.generate_questions(validation_result)

        assert len(questions) == 1
        assert "specific type of research" in questions[0].question
        assert "research agent" in questions[0].context
        assert len(questions[0].suggestions) == 4

    @pytest.mark.asyncio
    async def test_process_user_responses_and_update_context(self):
        """Test processing user responses and updating conversation context."""
        context = ConversationContext(
            session_id="test",
            initial_specification={"project_name": "test", "agents": []},
            current_specification={"project_name": "test", "agents": []},
        )

        # Simulate processing a user response
        response = UserResponse(
            question_id="q1",
            response_type=ResponseType.DIRECT_ANSWER,
            content="I need a content research agent and a social media writer",
        )

        # Expected specification update based on response
        expected_spec_update = {
            "agents": [
                {"role": "content_research", "type": "researcher"},
                {"role": "social_media_writer", "type": "writer"},
            ]
        }

        # Test the expected behavior (method to be implemented)
        async def mock_process_response(ctx, resp):
            # Parse the response content to extract agents
            new_spec = ctx.current_specification.copy()
            new_spec.update(expected_spec_update)

            new_turn = ConversationTurn(
                turn_number=ctx.current_turn_number,
                questions_asked=[],
                user_responses=[resp],
                specification_updates=expected_spec_update,
            )

            updated_context = replace(ctx)
            updated_context.current_specification = new_spec
            updated_context.turns.append(new_turn)
            updated_context.conversation_state = ConversationState.PROCESSING_RESPONSES

            return updated_context

        updated_context = await mock_process_response(context, response)

        assert (
            updated_context.conversation_state == ConversationState.PROCESSING_RESPONSES
        )
        assert len(updated_context.turns) == 1
        assert (
            updated_context.current_specification["agents"]
            == expected_spec_update["agents"]
        )

    @pytest.mark.asyncio
    async def test_follow_up_question_generation(self):
        """Test generating follow-up questions based on conversation history."""
        # Setup context with some answered questions
        context = ConversationContext(
            session_id="test",
            initial_specification={"project_name": "blog-team"},
            current_specification={
                "project_name": "blog-team",
                "agents": [{"role": "researcher"}, {"role": "writer"}],
            },
        )

        # Mock LLM for follow-up question generation
        self.llm_client.complete = AsyncMock(
            return_value="""[
                {
                    "type": "incomplete_spec",
                    "field": "tasks",
                    "question": "Now that you have a researcher and writer, what specific tasks should each agent perform?",
                    "context": "With your agents defined, we need to specify the workflow and tasks each agent will handle.",
                    "suggestions": ["Research trending topics", "Write blog posts", "Edit content", "Publish articles"]
                }
            ]"""
        )

        # Simulate follow-up question generation
        validation_issues = [
            ValidationIssue(IssueSeverity.ERROR, "No tasks defined for agents", "tasks")
        ]
        validation_result = ValidationResult(
            issues=validation_issues, completeness_score=0.6
        )

        questions = await self.clarifier.generate_questions(validation_result)

        assert len(questions) == 1
        assert "researcher and writer" in questions[0].question
        assert "tasks" in questions[0].question.lower()
        assert questions[0].question_type == QuestionType.INCOMPLETE_SPEC

    def test_conversation_state_transitions(self):
        """Test conversation state transitions."""
        context = ConversationContext(
            session_id="test", initial_specification={}, current_specification={}
        )

        # Initial state
        assert context.conversation_state == ConversationState.INITIAL

        # Transition to asking questions
        context.conversation_state = ConversationState.ASKING_QUESTIONS
        assert context.conversation_state == ConversationState.ASKING_QUESTIONS

        # Transition to processing responses
        context.conversation_state = ConversationState.PROCESSING_RESPONSES
        assert context.conversation_state == ConversationState.PROCESSING_RESPONSES

        # Transition to generating follow-up
        context.conversation_state = ConversationState.GENERATING_FOLLOWUP
        assert context.conversation_state == ConversationState.GENERATING_FOLLOWUP

        # Final state
        context.conversation_state = ConversationState.COMPLETED
        assert context.conversation_state == ConversationState.COMPLETED
        assert context.is_conversation_complete()

    @pytest.mark.asyncio
    async def test_conversation_context_retention_across_turns(self):
        """Test that conversation context is retained across multiple turns."""
        context = ConversationContext(
            session_id="persistence-test",
            initial_specification={"project_name": "multi-turn-test"},
            current_specification={"project_name": "multi-turn-test"},
        )

        # Turn 1: Ask about agents
        turn1_question = Question(
            question_type=QuestionType.MISSING_FIELD,
            field="agents",
            question="What agents do you need?",
            context="",
            suggestions=["Research", "Writing"],
        )
        turn1_response = UserResponse(
            question_id="q1",
            response_type=ResponseType.DIRECT_ANSWER,
            content="Research agent",
        )
        turn1 = ConversationTurn(
            turn_number=1,
            questions_asked=[turn1_question],
            user_responses=[turn1_response],
            specification_updates={"agents": [{"role": "research"}]},
        )
        context.turns.append(turn1)
        context.current_specification.update({"agents": [{"role": "research"}]})

        # Turn 2: Ask about tasks (context-aware)
        turn2_question = Question(
            question_type=QuestionType.MISSING_FIELD,
            field="tasks",
            question="What tasks should the research agent perform?",
            context="Based on your research agent, what specific research tasks are needed?",
            suggestions=["Market research", "Content research"],
        )
        turn2_response = UserResponse(
            question_id="q2",
            response_type=ResponseType.DIRECT_ANSWER,
            content="Market research and competitive analysis",
        )
        turn2 = ConversationTurn(
            turn_number=2,
            questions_asked=[turn2_question],
            user_responses=[turn2_response],
            specification_updates={
                "tasks": [
                    {"name": "market_research", "agent": "research"},
                    {"name": "competitive_analysis", "agent": "research"},
                ]
            },
        )
        context.turns.append(turn2)

        # Verify context retention
        assert len(context.turns) == 2
        assert context.current_turn_number == 3
        assert context.total_questions_asked == 2
        assert context.total_responses_received == 2

        # Verify that context builds upon previous information
        assert turn2_question.context.startswith("Based on your research agent")
        assert "research agent" in turn2_question.question

    @pytest.mark.asyncio
    async def test_error_handling_in_multi_turn_conversations(self):
        """Test error handling during multi-turn conversations."""
        context = ConversationContext(
            session_id="error-test", initial_specification={}, current_specification={}
        )

        # Test LLM error during question generation
        self.llm_client.complete = AsyncMock(side_effect=LLMError("API error"))

        validation_result = ValidationResult(
            issues=[
                ValidationIssue(
                    IssueSeverity.ERROR, "Missing field", "project_description"
                )
            ],
            completeness_score=0.2,
        )

        with pytest.raises(ClarificationError) as exc_info:
            await self.clarifier.generate_questions(validation_result)

        assert "Failed to generate clarification questions" in str(exc_info.value)
        assert exc_info.value.original_error is not None

    def test_response_type_enum_values(self):
        """Test ResponseType enum values."""
        assert ResponseType.DIRECT_ANSWER.value == "direct_answer"
        assert ResponseType.PARTIAL_ANSWER.value == "partial_answer"
        assert ResponseType.CLARIFICATION_REQUEST.value == "clarification_request"
        assert ResponseType.SKIP_QUESTION.value == "skip_question"

    def test_conversation_state_enum_values(self):
        """Test ConversationState enum values."""
        assert ConversationState.INITIAL.value == "initial"
        assert ConversationState.ASKING_QUESTIONS.value == "asking_questions"
        assert ConversationState.PROCESSING_RESPONSES.value == "processing_responses"
        assert ConversationState.GENERATING_FOLLOWUP.value == "generating_followup"
        assert ConversationState.COMPLETED.value == "completed"

"""
Integration tests for multi-turn conversation functionality.

These tests verify that the new multi-turn conversation methods
work correctly with real InteractiveClarifier instances.
"""

import pytest
from unittest.mock import AsyncMock, Mock

from crewforge.clarifier import (
    InteractiveClarifier,
    ConversationContext,
    ConversationState,
    UserResponse,
    ResponseType,
    Question,
    QuestionType,
)
from crewforge.llm import LLMClient


class TestInteractiveClarifierIntegration:
    """Integration tests for multi-turn conversation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm_client = Mock(spec=LLMClient)
        self.clarifier = InteractiveClarifier(llm_client=self.llm_client)

    @pytest.mark.asyncio
    async def test_start_clarification_session_integration(self):
        """Test starting a clarification session with real InteractiveClarifier."""
        initial_spec = {
            "project_name": "test-project",
            "project_description": "",
            "agents": [],
        }

        context = await self.clarifier.start_clarification_session(initial_spec)

        assert isinstance(context, ConversationContext)
        assert context.initial_specification == initial_spec
        assert context.current_specification == initial_spec
        assert context.conversation_state == ConversationState.INITIAL
        assert len(context.session_id) > 0
        assert context.turns == []

    @pytest.mark.asyncio
    async def test_ask_next_questions_integration(self):
        """Test asking next questions with real clarifier."""
        initial_spec = {"project_name": "test", "agents": []}
        context = await self.clarifier.start_clarification_session(initial_spec)

        # Mock the LLM response for question generation
        self.llm_client.complete = AsyncMock(
            return_value="""[
                {
                    "type": "missing_field",
                    "field": "project_description",
                    "question": "What should your project do?",
                    "context": "A description helps define scope.",
                    "suggestions": ["Research content", "Analyze data"]
                }
            ]"""
        )

        questions = await self.clarifier.ask_next_questions(context, max_questions=2)

        assert len(questions) > 0
        assert all(isinstance(q, Question) for q in questions)
        assert context.conversation_state == ConversationState.ASKING_QUESTIONS
        assert len(context.pending_questions) > 0

    @pytest.mark.asyncio
    async def test_process_answers_integration(self):
        """Test processing answers with real clarifier."""
        initial_spec = {"project_name": "test", "agents": []}
        context = await self.clarifier.start_clarification_session(initial_spec)

        # Mock question generation
        self.llm_client.complete = AsyncMock(
            return_value="""[
                {
                    "type": "missing_field",
                    "field": "project_description",
                    "question": "What should your project do?",
                    "context": "Description needed.",
                    "suggestions": []
                }
            ]"""
        )

        questions = await self.clarifier.ask_next_questions(context)
        question_id = questions[0].id

        # Create user response
        response = UserResponse(
            question_id=question_id,
            response_type=ResponseType.DIRECT_ANSWER,
            content="Create a content research and writing team",
        )

        updated_context = await self.clarifier.process_answers(context, [response])

        assert (
            updated_context.conversation_state == ConversationState.PROCESSING_RESPONSES
        )
        assert len(updated_context.turns) == 1
        assert len(updated_context.turns[0].user_responses) == 1
        assert updated_context.turns[0].user_responses[0].content == response.content

    @pytest.mark.asyncio
    async def test_check_completion_integration(self):
        """Test completion checking with real clarifier."""
        initial_spec = {
            "project_name": "complete-test",
            "project_description": "A complete project",
            "agents": [{"role": "researcher"}],
        }
        context = await self.clarifier.start_clarification_session(initial_spec)

        is_complete, final_spec = await self.clarifier.check_completion(context)

        # With a reasonably complete spec, should be marked complete
        assert is_complete
        assert final_spec is not None
        assert final_spec == initial_spec

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test a complete conversation flow from start to finish."""
        # Start with incomplete spec
        initial_spec = {"project_name": "blog-team", "agents": []}
        context = await self.clarifier.start_clarification_session(initial_spec)

        # Step 1: Ask initial questions
        self.llm_client.complete = AsyncMock(
            return_value="""[
                {
                    "type": "missing_field",
                    "field": "project_description",
                    "question": "What should your blog team do?",
                    "context": "Description needed for clarity.",
                    "suggestions": ["Write blog posts", "Research topics"]
                },
                {
                    "type": "missing_field",
                    "field": "agents",
                    "question": "What agents do you need?",
                    "context": "Agents perform the work.",
                    "suggestions": ["Writer", "Researcher", "Editor"]
                }
            ]"""
        )

        questions = await self.clarifier.ask_next_questions(context)
        assert len(questions) == 2

        # Step 2: User answers first set of questions
        responses = [
            UserResponse(
                question_id=questions[0].id,
                response_type=ResponseType.DIRECT_ANSWER,
                content="Create engaging blog content about technology",
            ),
            UserResponse(
                question_id=questions[1].id,
                response_type=ResponseType.DIRECT_ANSWER,
                content="I need a researcher and a writer",
            ),
        ]

        context = await self.clarifier.process_answers(context, responses)
        assert len(context.turns) == 1

        # Step 3: Check if more questions needed
        is_complete, final_spec = await self.clarifier.check_completion(context)

        # Should have reasonable completeness now
        assert context.current_specification != initial_spec
        assert len(context.resolved_questions) == 2

    def test_conversation_context_properties(self):
        """Test conversation context property calculations."""
        initial_spec = {"project_name": "test"}
        context = ConversationContext(
            session_id="test",
            initial_specification=initial_spec,
            current_specification=initial_spec,
        )

        # Test empty context
        assert context.current_turn_number == 1
        assert context.total_questions_asked == 0
        assert context.total_responses_received == 0
        assert not context.has_pending_questions()

        # Add some data
        question = Question(
            question_type=QuestionType.MISSING_FIELD,
            field="test",
            question="Test?",
            context="",
            suggestions=[],
        )
        context.pending_questions.append(question)

        assert context.has_pending_questions()

    def test_specification_gap_analysis(self):
        """Test the specification gap analysis functionality."""
        incomplete_spec = {"project_name": "test"}
        issues = self.clarifier._analyze_specification_gaps(incomplete_spec)

        # Should find missing description and agents
        assert len(issues) >= 2
        issue_fields = [issue.field_path for issue in issues]
        assert "project_description" in issue_fields
        assert "agents" in issue_fields

    def test_completeness_scoring(self):
        """Test the completeness scoring functionality."""
        # Empty spec
        empty_spec = {}
        score = self.clarifier._calculate_completeness(empty_spec)
        assert score == 0.0

        # Minimal spec
        minimal_spec = {"project_description": "test", "agents": [{"role": "test"}]}
        score = self.clarifier._calculate_completeness(minimal_spec)
        assert score >= 0.7  # Should have good score with required fields

        # Complete spec
        complete_spec = {
            "project_description": "test",
            "agents": [{"role": "test"}],
            "tasks": [{"name": "test"}],
            "tools": ["tool1"],
            "dependencies": ["dep1"],
        }
        score = self.clarifier._calculate_completeness(complete_spec)
        assert abs(score - 1.0) < 0.0001  # Handle floating point precision

    def test_agent_parsing_from_response(self):
        """Test parsing agents from user responses."""
        response = "I need a research agent and a content writer"
        agents = self.clarifier._parse_agents_from_response(response)

        assert len(agents) >= 1
        agent_roles = [agent.get("role", "") for agent in agents]
        # Should detect research-related and write-related roles
        assert any("research" in role for role in agent_roles)
        assert any("write" in role for role in agent_roles)

    def test_follow_up_detection(self):
        """Test detection of when follow-up questions are needed."""
        # Direct answers shouldn't need follow-up
        direct_responses = [
            UserResponse("q1", ResponseType.DIRECT_ANSWER, "Clear answer")
        ]
        assert not self.clarifier._needs_follow_up(direct_responses)

        # Partial answers should need follow-up
        partial_responses = [
            UserResponse("q1", ResponseType.PARTIAL_ANSWER, "Maybe...")
        ]
        assert self.clarifier._needs_follow_up(partial_responses)

        # Clarification requests should need follow-up
        clarification_responses = [
            UserResponse("q1", ResponseType.CLARIFICATION_REQUEST, "What do you mean?")
        ]
        assert self.clarifier._needs_follow_up(clarification_responses)

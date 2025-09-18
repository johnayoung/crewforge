"""
Integration tests for preference collection with multi-turn conversation system.

This module tests the integration between user preference collection and the
existing clarification conversation workflow.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock

from crewforge.clarifier import (
    InteractiveClarifier,
    ConversationContext,
    ConversationState,
    UserPreferences,
    LLMProvider,
    ProjectStructure,
    PreferenceCategory,
    QuestionType,
)
from crewforge.llm import LLMError


class TestPreferenceCollectionIntegration:
    """Test integration of preference collection with conversation system."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client for testing."""
        mock_client = Mock()
        mock_client.complete = AsyncMock()
        return mock_client

    @pytest.fixture
    def sample_context(self):
        """Create a sample conversation context for testing."""
        return ConversationContext(
            session_id="test-session",
            initial_specification={
                "project_name": "content-team",
                "project_description": "AI content creation team",
                "agents": [
                    {"role": "researcher", "goal": "Research topics"},
                    {"role": "writer", "goal": "Write articles"},
                ],
                "tasks": [
                    {"description": "Research trending topics", "agent": "researcher"},
                    {"description": "Write blog article", "agent": "writer"},
                ],
            },
            current_specification={
                "project_name": "content-team",
                "project_description": "AI content creation team",
                "agents": [
                    {"role": "researcher", "goal": "Research topics"},
                    {"role": "writer", "goal": "Write articles"},
                ],
                "tasks": [
                    {"description": "Research trending topics", "agent": "researcher"},
                    {"description": "Write blog article", "agent": "writer"},
                ],
            },
        )

    @pytest.mark.asyncio
    async def test_start_preference_collection_full_workflow(
        self, mock_llm_client, sample_context
    ):
        """Test complete preference collection workflow integration."""
        # Mock LLM responses for different categories
        llm_responses = [
            '{"preferred_llm_provider": "openai", "model_temperature": 0.3}',  # LLM_SETUP
            '{"project_structure": "standard", "naming_convention": "descriptive"}',  # PROJECT_STRUCTURE
            '{"max_agents_per_crew": 4, "prefer_sequential_execution": true}',  # CREW_CONFIG
            '{"include_web_search": true, "include_file_operations": true}',  # TOOL_INTEGRATION
            '{"verbose_output": false, "generate_readme": true}',  # OUTPUT_OPTIONS
            '{"interactive_mode": true, "auto_install_dependencies": true}',  # USER_EXPERIENCE
        ]
        mock_llm_client.complete.side_effect = llm_responses

        clarifier = InteractiveClarifier(mock_llm_client)
        updated_context = await clarifier.start_preference_collection(sample_context)

        # Verify context was updated with preferences
        assert isinstance(updated_context.user_preferences, UserPreferences)
        assert (
            updated_context.user_preferences.preferred_llm_provider
            == LLMProvider.OPENAI
        )
        assert (
            updated_context.user_preferences.project_structure
            == ProjectStructure.STANDARD
        )
        assert updated_context.user_preferences.max_agents_per_crew == 4

        # Verify conversation state was updated
        assert updated_context.conversation_state == ConversationState.ASKING_QUESTIONS

        # Verify preference confirmation questions were added
        assert len(updated_context.pending_questions) > 0
        preference_questions = [
            q for q in updated_context.pending_questions if "preferences." in q.field
        ]
        assert len(preference_questions) > 0

    @pytest.mark.asyncio
    async def test_start_preference_collection_specific_categories(
        self, mock_llm_client, sample_context
    ):
        """Test preference collection for specific categories only."""
        mock_llm_client.complete.return_value = (
            '{"preferred_llm_provider": "anthropic"}'
        )

        clarifier = InteractiveClarifier(mock_llm_client)
        categories = [PreferenceCategory.LLM_SETUP]
        updated_context = await clarifier.start_preference_collection(
            sample_context, categories
        )

        # Should only have collected preferences for specified categories
        assert (
            updated_context.user_preferences.preferred_llm_provider
            == LLMProvider.ANTHROPIC
        )
        # Other preferences should remain None/default
        assert updated_context.user_preferences.project_structure is None

        # Should only have questions for the specified category
        preference_questions = [
            q for q in updated_context.pending_questions if "preferences." in q.field
        ]
        assert len(preference_questions) == 1

    @pytest.mark.asyncio
    async def test_preference_collection_with_llm_failure(
        self, mock_llm_client, sample_context
    ):
        """Test preference collection gracefully handles LLM failures."""
        mock_llm_client.complete.side_effect = LLMError("API unavailable")

        clarifier = InteractiveClarifier(mock_llm_client)
        updated_context = await clarifier.start_preference_collection(sample_context)

        # Should still work with fallback defaults
        assert isinstance(updated_context.user_preferences, UserPreferences)
        # Should have some default preferences set
        assert (
            updated_context.user_preferences.preferred_llm_provider
            == LLMProvider.OPENAI
        )  # default

        # Should still create conversation flow
        assert updated_context.conversation_state == ConversationState.ASKING_QUESTIONS

    def test_generate_preference_confirmation_questions(self, mock_llm_client):
        """Test generation of preference confirmation questions."""
        clarifier = InteractiveClarifier(mock_llm_client)
        preferences = UserPreferences(
            preferred_llm_provider=LLMProvider.OPENAI,
            project_structure=ProjectStructure.STANDARD,
            max_agents_per_crew=4,
        )
        categories = [
            PreferenceCategory.LLM_SETUP,
            PreferenceCategory.PROJECT_STRUCTURE,
        ]

        questions = clarifier._generate_preference_confirmation_questions(
            preferences, categories
        )

        # Should generate questions for categories with preferences
        assert len(questions) == 2

        # All questions should be about preferences
        for question in questions:
            assert "preferences." in question.field
            assert question.question_type == QuestionType.INCOMPLETE_SPEC

    def test_extract_category_preferences(self, mock_llm_client):
        """Test extraction of category-specific preferences."""
        clarifier = InteractiveClarifier(mock_llm_client)
        preferences = UserPreferences(
            preferred_llm_provider=LLMProvider.ANTHROPIC,
            model_temperature=0.5,
            project_structure=ProjectStructure.DETAILED,
            max_agents_per_crew=6,
        )

        # Extract LLM setup preferences
        llm_prefs = clarifier._extract_category_preferences(
            preferences, PreferenceCategory.LLM_SETUP
        )
        assert "preferred_llm_provider" in llm_prefs
        assert "model_temperature" in llm_prefs
        assert "project_structure" not in llm_prefs  # Different category

        # Extract project structure preferences
        project_prefs = clarifier._extract_category_preferences(
            preferences, PreferenceCategory.PROJECT_STRUCTURE
        )
        assert "project_structure" in project_prefs
        assert "preferred_llm_provider" not in project_prefs  # Different category

    def test_format_preferences_summary(self, mock_llm_client):
        """Test formatting of preferences for user display."""
        clarifier = InteractiveClarifier(mock_llm_client)

        # Test with various data types
        prefs = {
            "preferred_llm_provider": LLMProvider.OPENAI,
            "max_agents_per_crew": 5,
            "include_examples": True,
            "verbose_output": False,
        }

        summary = clarifier._format_preferences_summary(prefs)

        # Should be human-readable
        assert "Preferred Llm Provider: Openai" in summary
        assert "Max Agents Per Crew: 5" in summary
        assert "Include Examples: Yes" in summary
        assert "Verbose Output: No" in summary

        # Test with empty preferences
        empty_summary = clarifier._format_preferences_summary({})
        assert empty_summary == "none"

    @pytest.mark.asyncio
    async def test_preference_collection_preserves_existing_context(
        self, mock_llm_client, sample_context
    ):
        """Test that preference collection preserves existing conversation context."""
        # Add some existing conversation history
        sample_context.conversation_state = ConversationState.PROCESSING_RESPONSES
        original_session_id = sample_context.session_id
        original_spec = sample_context.initial_specification.copy()

        mock_llm_client.complete.return_value = '{"preferred_llm_provider": "google"}'

        clarifier = InteractiveClarifier(mock_llm_client)
        updated_context = await clarifier.start_preference_collection(sample_context)

        # Should preserve original context data
        assert updated_context.session_id == original_session_id
        assert updated_context.initial_specification == original_spec

        # Should have added preferences without affecting other data
        assert (
            updated_context.user_preferences.preferred_llm_provider
            == LLMProvider.GOOGLE
        )

        # Should update conversation state for preference questions
        assert updated_context.conversation_state == ConversationState.ASKING_QUESTIONS

    @pytest.mark.asyncio
    async def test_preference_collection_creates_actionable_questions(
        self, mock_llm_client, sample_context
    ):
        """Test that preference collection creates clear, actionable questions."""
        mock_llm_client.complete.return_value = (
            '{"preferred_llm_provider": "openai", "max_agents_per_crew": 3}'
        )

        clarifier = InteractiveClarifier(mock_llm_client)
        categories = [PreferenceCategory.LLM_SETUP]
        updated_context = await clarifier.start_preference_collection(
            sample_context, categories
        )

        questions = updated_context.pending_questions

        # Should have at least one preference question
        pref_questions = [q for q in questions if "preferences." in q.field]
        assert len(pref_questions) >= 1

        for question in pref_questions:
            # Questions should be clear and actionable
            assert len(question.question) > 20  # Reasonably detailed
            assert "?" in question.question or question.question.endswith(":")
            assert len(question.suggestions) > 0  # Should have suggestions
            assert question.context  # Should have context explanation


if __name__ == "__main__":
    pytest.main([__file__])

"""
Tests for user preference collection functionality.

This module tests the UserPreferences data model, preference collection workflow,
and integration with the clarification system.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock
from dataclasses import asdict

from crewforge.clarifier import (
    UserPreferences,
    LLMProvider,
    ProjectStructure,
    NamingConvention,
    PreferenceCategory,
    InteractiveClarifier,
    ConversationContext,
    ConversationState,
)
from crewforge.llm import LLMError


class TestUserPreferences:
    """Test the UserPreferences data model."""

    def test_empty_preferences_creation(self):
        """Test creating empty preferences."""
        prefs = UserPreferences()

        assert prefs.preferred_llm_provider is None
        assert prefs.project_structure is None
        assert prefs.naming_convention is None
        assert prefs.max_agents_per_crew is None

    def test_preferences_with_values(self):
        """Test creating preferences with specific values."""
        prefs = UserPreferences(
            preferred_llm_provider=LLMProvider.OPENAI,
            project_structure=ProjectStructure.STANDARD,
            naming_convention=NamingConvention.DESCRIPTIVE,
            max_agents_per_crew=5,
            include_examples=True,
            verbose_output=False,
        )

        assert prefs.preferred_llm_provider == LLMProvider.OPENAI
        assert prefs.project_structure == ProjectStructure.STANDARD
        assert prefs.naming_convention == NamingConvention.DESCRIPTIVE
        assert prefs.max_agents_per_crew == 5
        assert prefs.include_examples is True
        assert prefs.verbose_output is False

    def test_to_dict_empty(self):
        """Test converting empty preferences to dictionary."""
        prefs = UserPreferences()
        result = prefs.to_dict()

        assert result == {}

    def test_to_dict_with_values(self):
        """Test converting preferences with values to dictionary."""
        prefs = UserPreferences(
            preferred_llm_provider=LLMProvider.ANTHROPIC,
            max_agents_per_crew=3,
            include_examples=True,
        )
        result = prefs.to_dict()

        expected = {
            "preferred_llm_provider": "anthropic",
            "max_agents_per_crew": 3,
            "include_examples": True,
        }
        assert result == expected

    def test_from_dict_empty(self):
        """Test creating preferences from empty dictionary."""
        prefs = UserPreferences.from_dict({})

        assert prefs.preferred_llm_provider is None
        assert prefs.project_structure is None
        assert prefs.max_agents_per_crew is None

    def test_from_dict_with_values(self):
        """Test creating preferences from dictionary with values."""
        data = {
            "preferred_llm_provider": "openai",
            "project_structure": "detailed",
            "naming_convention": "role_based",
            "max_agents_per_crew": 4,
            "include_examples": False,
        }
        prefs = UserPreferences.from_dict(data)

        assert prefs.preferred_llm_provider == LLMProvider.OPENAI
        assert prefs.project_structure == ProjectStructure.DETAILED
        assert prefs.naming_convention == NamingConvention.ROLE_BASED
        assert prefs.max_agents_per_crew == 4
        assert prefs.include_examples is False

    def test_from_dict_invalid_enum_values(self):
        """Test handling invalid enum values gracefully."""
        data = {
            "preferred_llm_provider": "invalid_provider",
            "project_structure": "invalid_structure",
            "max_agents_per_crew": 5,  # Valid non-enum field
        }
        prefs = UserPreferences.from_dict(data)

        # Invalid enum values should be ignored
        assert prefs.preferred_llm_provider is None
        assert prefs.project_structure is None
        # Valid field should be set
        assert prefs.max_agents_per_crew == 5

    def test_from_dict_unknown_fields(self):
        """Test handling unknown fields gracefully."""
        data = {"unknown_field": "value", "max_agents_per_crew": 3}
        prefs = UserPreferences.from_dict(data)

        # Unknown field should be ignored
        assert not hasattr(prefs, "unknown_field")
        # Known field should be set
        assert prefs.max_agents_per_crew == 3

    def test_merge_preferences_empty(self):
        """Test merging with empty preferences."""
        prefs1 = UserPreferences(max_agents_per_crew=5, include_examples=True)
        prefs2 = UserPreferences()

        merged = prefs1.merge(prefs2)

        assert merged.max_agents_per_crew == 5
        assert merged.include_examples is True

    def test_merge_preferences_override(self):
        """Test merging preferences with override."""
        prefs1 = UserPreferences(
            max_agents_per_crew=5,
            include_examples=True,
            preferred_llm_provider=LLMProvider.OPENAI,
        )
        prefs2 = UserPreferences(max_agents_per_crew=3, verbose_output=False)

        merged = prefs1.merge(prefs2)

        # Overridden value
        assert merged.max_agents_per_crew == 3
        # Preserved from prefs1
        assert merged.include_examples is True
        assert merged.preferred_llm_provider == LLMProvider.OPENAI
        # New from prefs2
        assert merged.verbose_output is False

    def test_roundtrip_dict_conversion(self):
        """Test that to_dict/from_dict preserve all data."""
        original = UserPreferences(
            preferred_llm_provider=LLMProvider.GOOGLE,
            project_structure=ProjectStructure.FLAT,
            naming_convention=NamingConvention.FUNCTIONAL,
            max_agents_per_crew=2,
            include_examples=True,
            verbose_output=False,
            interactive_mode=True,
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = UserPreferences.from_dict(data)

        # All fields should match
        assert restored.preferred_llm_provider == original.preferred_llm_provider
        assert restored.project_structure == original.project_structure
        assert restored.naming_convention == original.naming_convention
        assert restored.max_agents_per_crew == original.max_agents_per_crew
        assert restored.include_examples == original.include_examples
        assert restored.verbose_output == original.verbose_output
        assert restored.interactive_mode == original.interactive_mode


class TestEnumValues:
    """Test enum value definitions and usage."""

    def test_llm_provider_values(self):
        """Test LLMProvider enum values."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GOOGLE.value == "google"
        assert LLMProvider.GROQ.value == "groq"
        assert LLMProvider.SAMBA_NOVA.value == "sambanova"

    def test_project_structure_values(self):
        """Test ProjectStructure enum values."""
        assert ProjectStructure.STANDARD.value == "standard"
        assert ProjectStructure.FLAT.value == "flat"
        assert ProjectStructure.DETAILED.value == "detailed"

    def test_naming_convention_values(self):
        """Test NamingConvention enum values."""
        assert NamingConvention.DESCRIPTIVE.value == "descriptive"
        assert NamingConvention.ROLE_BASED.value == "role_based"
        assert NamingConvention.FUNCTIONAL.value == "functional"

    def test_preference_category_values(self):
        """Test PreferenceCategory enum values."""
        assert PreferenceCategory.LLM_SETUP.value == "llm_setup"
        assert PreferenceCategory.PROJECT_STRUCTURE.value == "project_structure"
        assert PreferenceCategory.CREW_CONFIGURATION.value == "crew_configuration"
        assert PreferenceCategory.TOOL_INTEGRATION.value == "tool_integration"
        assert PreferenceCategory.OUTPUT_OPTIONS.value == "output_options"
        assert PreferenceCategory.USER_EXPERIENCE.value == "user_experience"


class TestConversationContextWithPreferences:
    """Test ConversationContext with user preferences integration."""

    def test_conversation_context_includes_preferences(self):
        """Test that ConversationContext includes user preferences."""
        context = ConversationContext(
            session_id="test-session",
            initial_specification={"project_name": "test"},
            current_specification={"project_name": "test"},
        )

        # Should have default empty preferences
        assert isinstance(context.user_preferences, UserPreferences)
        assert context.user_preferences.preferred_llm_provider is None

    def test_conversation_context_with_custom_preferences(self):
        """Test ConversationContext with custom preferences."""
        prefs = UserPreferences(
            preferred_llm_provider=LLMProvider.ANTHROPIC, max_agents_per_crew=3
        )

        context = ConversationContext(
            session_id="test-session",
            initial_specification={"project_name": "test"},
            current_specification={"project_name": "test"},
            user_preferences=prefs,
        )

        assert context.user_preferences.preferred_llm_provider == LLMProvider.ANTHROPIC
        assert context.user_preferences.max_agents_per_crew == 3


class TestPreferenceCollectionPrompts:
    """Test preference collection prompt templates."""

    def test_preference_collection_prompts_exist(self):
        """Test that all preference collection prompts are defined."""
        clarifier = InteractiveClarifier(Mock())

        assert hasattr(clarifier, "PREFERENCE_COLLECTION_PROMPTS")
        prompts = clarifier.PREFERENCE_COLLECTION_PROMPTS

        # Check all categories have prompts
        for category in PreferenceCategory:
            assert category in prompts
            assert isinstance(prompts[category], str)
            assert len(prompts[category].strip()) > 0

    def test_llm_setup_prompt_content(self):
        """Test LLM setup prompt contains expected content."""
        clarifier = InteractiveClarifier(Mock())
        prompt = clarifier.PREFERENCE_COLLECTION_PROMPTS[PreferenceCategory.LLM_SETUP]

        # Should mention all supported providers
        assert "OpenAI" in prompt
        assert "Anthropic" in prompt
        assert "Google" in prompt
        assert "Groq" in prompt
        assert "SambaNova" in prompt

        # Should have placeholder for tasks
        assert "{tasks_summary}" in prompt

    def test_project_structure_prompt_content(self):
        """Test project structure prompt contains expected content."""
        clarifier = InteractiveClarifier(Mock())
        prompt = clarifier.PREFERENCE_COLLECTION_PROMPTS[
            PreferenceCategory.PROJECT_STRUCTURE
        ]

        # Should mention structure options
        assert "Standard" in prompt
        assert "Flat" in prompt
        assert "Detailed" in prompt

        # Should mention naming conventions
        assert "Descriptive" in prompt
        assert "Role-based" in prompt
        assert "Functional" in prompt

    def test_crew_configuration_prompt_content(self):
        """Test crew configuration prompt has placeholders."""
        clarifier = InteractiveClarifier(Mock())
        prompt = clarifier.PREFERENCE_COLLECTION_PROMPTS[
            PreferenceCategory.CREW_CONFIGURATION
        ]

        # Should have placeholders for dynamic content
        assert "{agent_count}" in prompt
        assert "{task_count}" in prompt
        assert "{max_agents_suggestion}" in prompt
        assert "{max_tasks_suggestion}" in prompt

    def test_tool_integration_prompt_categories(self):
        """Test tool integration prompt covers main categories."""
        clarifier = InteractiveClarifier(Mock())
        prompt = clarifier.PREFERENCE_COLLECTION_PROMPTS[
            PreferenceCategory.TOOL_INTEGRATION
        ]

        # Should cover main tool categories
        assert "Web Search" in prompt
        assert "File" in prompt
        assert "Data" in prompt or "data" in prompt
        assert "Communication" in prompt

    def test_all_prompts_are_questions(self):
        """Test that all prompts end with question or input request."""
        clarifier = InteractiveClarifier(Mock())
        prompts = clarifier.PREFERENCE_COLLECTION_PROMPTS

        for category, prompt in prompts.items():
            # Should end with question mark, colon, or request for input
            assert (
                prompt.strip().endswith("?")
                or prompt.strip().endswith(":")
                or "preferences" in prompt.lower()
            ), f"Prompt for {category} doesn't end with clear request for input"


if __name__ == "__main__":
    pytest.main([__file__])


class TestPreferenceCollection:
    """Test preference collection methods in InteractiveClarifier."""

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
                "project_name": "test-project",
                "project_description": "A test project for content creation",
                "agents": [
                    {"role": "researcher", "goal": "Research topics"},
                    {"role": "writer", "goal": "Write content"},
                ],
                "tasks": [
                    {"description": "Research the topic", "agent": "researcher"},
                    {"description": "Write an article", "agent": "writer"},
                ],
            },
            current_specification={
                "project_name": "test-project",
                "project_description": "A test project for content creation",
                "agents": [
                    {"role": "researcher", "goal": "Research topics"},
                    {"role": "writer", "goal": "Write content"},
                ],
                "tasks": [
                    {"description": "Research the topic", "agent": "researcher"},
                    {"description": "Write an article", "agent": "writer"},
                ],
            },
        )

    @pytest.mark.asyncio
    async def test_collect_default_preferences_categories(
        self, mock_llm_client, sample_context
    ):
        """Test collecting preferences for default categories."""
        # Mock LLM response with valid JSON
        mock_llm_client.complete.return_value = (
            '{"preferred_llm_provider": "openai", "max_agents_per_crew": 3}'
        )

        clarifier = InteractiveClarifier(mock_llm_client)
        preferences = await clarifier.collect_user_preferences(sample_context)

        # Should have returned some preferences
        assert isinstance(preferences, UserPreferences)

        # LLM should have been called for each category
        assert mock_llm_client.complete.call_count == len(PreferenceCategory)

    @pytest.mark.asyncio
    async def test_collect_specific_categories(self, mock_llm_client, sample_context):
        """Test collecting preferences for specific categories."""
        mock_llm_client.complete.return_value = (
            '{"preferred_llm_provider": "anthropic"}'
        )

        clarifier = InteractiveClarifier(mock_llm_client)
        categories = [
            PreferenceCategory.LLM_SETUP,
            PreferenceCategory.PROJECT_STRUCTURE,
        ]
        preferences = await clarifier.collect_user_preferences(
            sample_context, categories
        )

        assert isinstance(preferences, UserPreferences)
        # Should only call LLM for the specified categories
        assert mock_llm_client.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_collect_category_preferences_with_llm_suggestions(
        self, mock_llm_client, sample_context
    ):
        """Test collecting preferences with LLM-generated suggestions."""
        # Mock valid JSON response
        mock_response = {
            "preferred_llm_provider": "openai",
            "model_temperature": 0.3,
            "max_tokens_per_response": 2000,
        }
        mock_llm_client.complete.return_value = json.dumps(mock_response)

        clarifier = InteractiveClarifier(mock_llm_client)
        preferences = await clarifier._collect_category_preferences(
            sample_context, PreferenceCategory.LLM_SETUP
        )

        assert preferences.preferred_llm_provider == LLMProvider.OPENAI
        assert preferences.model_temperature == 0.3
        assert preferences.max_tokens_per_response == 2000

    @pytest.mark.asyncio
    async def test_collect_category_preferences_fallback_on_json_error(
        self, mock_llm_client, sample_context
    ):
        """Test fallback to defaults when LLM returns invalid JSON."""
        # Mock invalid JSON response
        mock_llm_client.complete.return_value = "This is not valid JSON"

        clarifier = InteractiveClarifier(mock_llm_client)
        preferences = await clarifier._collect_category_preferences(
            sample_context, PreferenceCategory.LLM_SETUP
        )

        # Should fall back to defaults
        assert isinstance(preferences, UserPreferences)
        # Default for LLM_SETUP category is OpenAI
        assert preferences.preferred_llm_provider == LLMProvider.OPENAI

    @pytest.mark.asyncio
    async def test_collect_category_preferences_fallback_on_llm_error(
        self, mock_llm_client, sample_context
    ):
        """Test fallback to defaults when LLM call fails."""
        # Mock LLM error
        mock_llm_client.complete.side_effect = LLMError("API error")

        clarifier = InteractiveClarifier(mock_llm_client)
        preferences = await clarifier._collect_category_preferences(
            sample_context, PreferenceCategory.PROJECT_STRUCTURE
        )

        # Should fall back to defaults
        assert isinstance(preferences, UserPreferences)
        # Default for PROJECT_STRUCTURE category
        assert preferences.project_structure == ProjectStructure.STANDARD

    def test_customize_preference_prompt(self, mock_llm_client, sample_context):
        """Test prompt customization with context information."""
        clarifier = InteractiveClarifier(mock_llm_client)
        base_prompt = "Project has {agent_count} agents and {task_count} tasks. Summary: {tasks_summary}"

        customized = clarifier._customize_preference_prompt(
            base_prompt, sample_context, PreferenceCategory.CREW_CONFIGURATION
        )

        # Should have replaced placeholders
        assert "{agent_count}" not in customized
        assert "{task_count}" not in customized
        assert "{tasks_summary}" not in customized
        assert "2 agents" in customized
        assert "2 tasks" in customized

    def test_summarize_tasks(self, mock_llm_client):
        """Test task summarization for prompt customization."""
        clarifier = InteractiveClarifier(mock_llm_client)

        # Test empty tasks
        assert clarifier._summarize_tasks([]) == "no specific tasks defined"

        # Test single task type
        tasks = [{"description": "Research the market"}]
        assert "research" in clarifier._summarize_tasks(tasks)

        # Test multiple task types
        tasks = [
            {"description": "Research the market"},
            {"description": "Write content about findings"},
            {"description": "Analyze the data"},
        ]
        summary = clarifier._summarize_tasks(tasks)
        assert "research" in summary
        assert "content creation" in summary
        assert "analysis" in summary

    def test_get_default_preferences_for_category(self, mock_llm_client):
        """Test default preference generation for each category."""
        clarifier = InteractiveClarifier(mock_llm_client)

        # Test LLM_SETUP defaults
        defaults = clarifier._get_default_preferences_for_category(
            PreferenceCategory.LLM_SETUP
        )
        assert defaults.preferred_llm_provider == LLMProvider.OPENAI
        assert defaults.model_temperature == 0.3

        # Test PROJECT_STRUCTURE defaults
        defaults = clarifier._get_default_preferences_for_category(
            PreferenceCategory.PROJECT_STRUCTURE
        )
        assert defaults.project_structure == ProjectStructure.STANDARD
        assert defaults.naming_convention == NamingConvention.DESCRIPTIVE

        # Test CREW_CONFIGURATION defaults
        defaults = clarifier._get_default_preferences_for_category(
            PreferenceCategory.CREW_CONFIGURATION
        )
        assert defaults.max_agents_per_crew == 5
        assert defaults.prefer_sequential_execution is True

    def test_update_context_preferences(self, mock_llm_client, sample_context):
        """Test updating conversation context with new preferences."""
        clarifier = InteractiveClarifier(mock_llm_client)

        # Create new preferences
        new_prefs = UserPreferences(
            preferred_llm_provider=LLMProvider.ANTHROPIC, max_agents_per_crew=3
        )

        # Update context
        updated_context = clarifier.update_context_preferences(
            sample_context, new_prefs
        )

        # Should have merged preferences
        assert (
            updated_context.user_preferences.preferred_llm_provider
            == LLMProvider.ANTHROPIC
        )
        assert updated_context.user_preferences.max_agents_per_crew == 3

        # Should have updated timestamp
        assert updated_context.last_activity > sample_context.last_activity

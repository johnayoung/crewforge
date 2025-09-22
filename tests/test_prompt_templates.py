"""Test the new Jinja2 template-based prompt system."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from crewforge.core.llm import LLMClient, LLMError


class TestPromptTemplates:
    """Test cases for Jinja2-based prompt templates."""

    def test_template_based_system_prompt(self):
        """Test that system prompt can be loaded from Jinja2 template."""
        client = LLMClient()
        system_prompt = client.get_system_prompt()

        assert "CrewAI project architect" in system_prompt
        assert "roles (job titles)" in system_prompt
        assert "JSON" in system_prompt
        assert len(system_prompt) > 100

    def test_template_based_user_prompt(self):
        """Test that user prompt can be formatted using Jinja2 template."""
        client = LLMClient()
        user_requirements = "Create a content research crew"

        user_prompt = client.format_user_prompt(user_requirements)

        assert user_requirements in user_prompt
        assert "requirements" in user_prompt.lower()
        assert "configuration" in user_prompt.lower()
        assert len(user_prompt) > len(user_requirements)

    def test_template_required_behavior(self):
        """Test that system properly fails when templates are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a client that will fail to find templates
            with patch.object(Path, "exists", return_value=False):
                client = LLMClient()

                # Should raise LLMError when templates are missing
                with pytest.raises(LLMError):
                    client.get_system_prompt()

                with pytest.raises(LLMError):
                    client.format_user_prompt("test requirements")

    def test_template_variable_substitution(self):
        """Test that template variables are properly substituted."""
        client = LLMClient()
        test_requirements = "Build a data analysis team for financial reporting"

        user_prompt = client.format_user_prompt(test_requirements)

        # Verify the exact requirements text appears in the prompt
        assert test_requirements in user_prompt
        # Verify template structure is preserved
        assert "Requirements:" in user_prompt
        assert "configuration" in user_prompt

    def test_backwards_compatibility_functions(self):
        """Test that standalone functions still work for backwards compatibility."""
        from crewforge.core.llm import format_user_prompt, get_crewai_system_prompt

        client = LLMClient()

        # Test standalone functions
        system_prompt = get_crewai_system_prompt(client)
        user_prompt = format_user_prompt("test", client)

        assert len(system_prompt) > 0
        assert "test" in user_prompt
        assert len(user_prompt) > 4  # More than just "test"

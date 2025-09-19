"""Test agent models for CrewForge using TDD approach."""

import pytest
from pydantic import ValidationError


class TestAgentConfig:
    """Test AgentConfig model for CrewAI agent definition."""

    def test_agent_config_valid_creation(self):
        """Test AgentConfig can be created with valid data."""
        # Import here to avoid import errors during development
        from crewforge.models.agent import AgentConfig

        # Test successful creation with minimum required fields
        agent = AgentConfig(
            role="Content Researcher",
            goal="Find and analyze relevant articles on specified topics",
            backstory="An experienced researcher with expertise in finding high-quality content online",
        )

        assert agent.role == "Content Researcher"
        assert agent.goal == "Find and analyze relevant articles on specified topics"
        assert (
            agent.backstory
            == "An experienced researcher with expertise in finding high-quality content online"
        )
        assert agent.verbose is False  # Default value
        assert agent.allow_delegation is True  # Default value

    def test_agent_config_required_fields(self):
        """Test AgentConfig requires role, goal, and backstory."""
        from crewforge.models.agent import AgentConfig

        # Missing role should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(
                goal="Find relevant articles", backstory="An experienced researcher"
            )
        assert "role" in str(exc_info.value)

        # Missing goal should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(role="Researcher", backstory="An experienced researcher")
        assert "goal" in str(exc_info.value)

        # Missing backstory should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(role="Researcher", goal="Find relevant articles")
        assert "backstory" in str(exc_info.value)

    def test_agent_config_field_types(self):
        """Test AgentConfig validates field types correctly."""
        from crewforge.models.agent import AgentConfig

        # Non-string role should fail
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(
                role=123,
                goal="Find relevant articles",
                backstory="An experienced researcher",
            )
        assert "str_type" in str(exc_info.value) or "string" in str(exc_info.value)

        # Non-string goal should fail
        with pytest.raises(ValidationError):
            AgentConfig(
                role="Researcher", goal=123, backstory="An experienced researcher"
            )

        # Non-string backstory should fail
        with pytest.raises(ValidationError):
            AgentConfig(role="Researcher", goal="Find relevant articles", backstory=123)

    def test_agent_config_optional_fields(self):
        """Test AgentConfig optional fields work correctly."""
        from crewforge.models.agent import AgentConfig

        agent = AgentConfig(
            role="Content Researcher",
            goal="Find and analyze relevant articles",
            backstory="An experienced researcher",
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            max_execution_time=300,
        )

        assert agent.verbose is True
        assert agent.allow_delegation is False
        assert agent.max_iter == 5
        assert agent.max_execution_time == 300

    def test_agent_config_role_validation(self):
        """Test role field validation for CrewAI compliance."""
        from crewforge.models.agent import AgentConfig

        # Empty role should fail
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(
                role="",
                goal="Find relevant articles",
                backstory="An experienced researcher",
            )
        assert (
            "character" in str(exc_info.value).lower()
            or "short" in str(exc_info.value).lower()
        )

        # Role too long should fail (over 100 characters)
        long_role = "A" * 101
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(
                role=long_role,
                goal="Find relevant articles",
                backstory="An experienced researcher",
            )
        assert (
            "length" in str(exc_info.value)
            or "long" in str(exc_info.value)
            or "too long" in str(exc_info.value)
        )

    def test_agent_config_goal_validation(self):
        """Test goal field validation for CrewAI compliance."""
        from crewforge.models.agent import AgentConfig

        # Empty goal should fail
        with pytest.raises(ValidationError):
            AgentConfig(
                role="Researcher", goal="", backstory="An experienced researcher"
            )

        # Goal too long should fail (over 500 characters)
        long_goal = "A" * 501
        with pytest.raises(ValidationError):
            AgentConfig(
                role="Researcher", goal=long_goal, backstory="An experienced researcher"
            )

    def test_agent_config_backstory_validation(self):
        """Test backstory field validation for CrewAI compliance."""
        from crewforge.models.agent import AgentConfig

        # Empty backstory should fail
        with pytest.raises(ValidationError):
            AgentConfig(role="Researcher", goal="Find relevant articles", backstory="")

        # Backstory too long should fail (over 1000 characters)
        long_backstory = "A" * 1001
        with pytest.raises(ValidationError):
            AgentConfig(
                role="Researcher",
                goal="Find relevant articles",
                backstory=long_backstory,
            )


class TestAgentTemplate:
    """Test AgentTemplate model for generation patterns."""

    def test_agent_template_creation(self):
        """Test AgentTemplate can be created with valid data."""
        from crewforge.models.agent import AgentTemplate

        template = AgentTemplate(
            name="researcher_template",
            description="Template for research agents",
            role_pattern="{{domain}} Researcher",
            goal_pattern="Research and analyze {{topic}} for {{purpose}}",
            backstory_pattern="An expert in {{domain}} with experience in {{specific_area}}",
        )

        assert template.name == "researcher_template"
        assert template.description == "Template for research agents"
        assert "{{domain}}" in template.role_pattern
        assert "{{topic}}" in template.goal_pattern
        assert "{{specific_area}}" in template.backstory_pattern

    def test_agent_template_required_fields(self):
        """Test AgentTemplate requires name and pattern fields."""
        from crewforge.models.agent import AgentTemplate

        # Missing name should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            AgentTemplate(
                description="Template for research agents",
                role_pattern="{{domain}} Researcher",
                goal_pattern="Research {{topic}}",
                backstory_pattern="Expert in {{domain}}",
            )
        assert "name" in str(exc_info.value)

        # Missing role_pattern should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            AgentTemplate(
                name="researcher_template",
                description="Template for research agents",
                goal_pattern="Research {{topic}}",
                backstory_pattern="Expert in {{domain}}",
            )
        assert "role_pattern" in str(exc_info.value)

    def test_agent_template_pattern_validation(self):
        """Test template patterns contain required Jinja2 variables."""
        from crewforge.models.agent import AgentTemplate

        # Valid patterns should work
        template = AgentTemplate(
            name="valid_template",
            role_pattern="{{role_type}} Agent",
            goal_pattern="Accomplish {{objective}}",
            backstory_pattern="Expert with {{expertise}}",
        )
        assert template is not None

        # Invalid pattern syntax should be caught (if we implement Jinja2 validation)
        # This test may need updating based on actual validation implementation
        template = AgentTemplate(
            name="template_with_typo",
            role_pattern="{{invalid_syntax",  # Missing closing brace
            goal_pattern="Accomplish {{objective}}",
            backstory_pattern="Expert with {{expertise}}",
        )
        # For now, just ensure it creates (validation might be added later)
        assert template is not None

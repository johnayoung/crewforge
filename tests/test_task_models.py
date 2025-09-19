"""Test task models for CrewForge using TDD approach."""

import pytest
from pydantic import ValidationError


class TestTaskConfig:
    """Test TaskConfig model for CrewAI task definition."""

    def test_task_config_valid_creation(self):
        """Test TaskConfig can be created with valid data."""
        from crewforge.models.task import TaskConfig

        # Test successful creation with minimum required fields
        task = TaskConfig(
            description="Research and analyze competitor pricing strategies",
            expected_output="A comprehensive report comparing pricing models with recommendations",
            agent="content_researcher",
        )

        assert task.description == "Research and analyze competitor pricing strategies"
        assert (
            task.expected_output
            == "A comprehensive report comparing pricing models with recommendations"
        )
        assert task.agent == "content_researcher"
        assert task.async_execution is False  # Default value
        assert task.context is None  # Default value

    def test_task_config_required_fields(self):
        """Test TaskConfig requires description, expected_output, and agent."""
        from crewforge.models.task import TaskConfig

        # Missing description should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            TaskConfig(expected_output="A comprehensive report", agent="researcher")
        assert "description" in str(exc_info.value)

        # Missing expected_output should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            TaskConfig(description="Research competitors", agent="researcher")
        assert "expected_output" in str(exc_info.value)

        # Missing agent should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            TaskConfig(
                description="Research competitors",
                expected_output="A comprehensive report",
            )
        assert "agent" in str(exc_info.value)

    def test_task_config_field_types(self):
        """Test TaskConfig validates field types correctly."""
        from crewforge.models.task import TaskConfig

        # Non-string description should fail
        with pytest.raises(ValidationError) as exc_info:
            TaskConfig(
                description=123,
                expected_output="A comprehensive report",
                agent="researcher",
            )
        assert "str_type" in str(exc_info.value) or "string" in str(exc_info.value)

        # Non-string expected_output should fail
        with pytest.raises(ValidationError):
            TaskConfig(
                description="Research competitors",
                expected_output=123,
                agent="researcher",
            )

        # Non-string agent should fail
        with pytest.raises(ValidationError):
            TaskConfig(
                description="Research competitors",
                expected_output="A comprehensive report",
                agent=123,
            )

    def test_task_config_optional_fields(self):
        """Test TaskConfig optional fields work correctly."""
        from crewforge.models.task import TaskConfig

        task = TaskConfig(
            description="Research competitor pricing strategies",
            expected_output="A comprehensive pricing analysis report",
            agent="content_researcher",
            async_execution=True,
            context=["previous_market_analysis"],
            tools=["web_search", "pdf_analyzer"],
            output_file="pricing_report.md",
        )

        assert task.async_execution is True
        assert task.context == ["previous_market_analysis"]
        assert task.tools == ["web_search", "pdf_analyzer"]
        assert task.output_file == "pricing_report.md"

    def test_task_config_description_validation(self):
        """Test description field validation for CrewAI compliance."""
        from crewforge.models.task import TaskConfig

        # Empty description should fail
        with pytest.raises(ValidationError) as exc_info:
            TaskConfig(
                description="",
                expected_output="A comprehensive report",
                agent="researcher",
            )
        assert (
            "character" in str(exc_info.value).lower()
            or "short" in str(exc_info.value).lower()
        )

        # Description too long should fail (over 1000 characters)
        long_description = "A" * 1001
        with pytest.raises(ValidationError) as exc_info:
            TaskConfig(
                description=long_description,
                expected_output="A comprehensive report",
                agent="researcher",
            )
        assert (
            "length" in str(exc_info.value)
            or "long" in str(exc_info.value)
            or "too long" in str(exc_info.value)
        )

    def test_task_config_expected_output_validation(self):
        """Test expected_output field validation for CrewAI compliance."""
        from crewforge.models.task import TaskConfig

        # Empty expected_output should fail
        with pytest.raises(ValidationError):
            TaskConfig(
                description="Research competitors",
                expected_output="",
                agent="researcher",
            )

        # Expected_output too long should fail (over 1000 characters)
        long_output = "A" * 1001
        with pytest.raises(ValidationError):
            TaskConfig(
                description="Research competitors",
                expected_output=long_output,
                agent="researcher",
            )

    def test_task_config_agent_validation(self):
        """Test agent field validation for CrewAI compliance."""
        from crewforge.models.task import TaskConfig

        # Empty agent should fail
        with pytest.raises(ValidationError):
            TaskConfig(
                description="Research competitors",
                expected_output="A comprehensive report",
                agent="",
            )


class TestTaskDependency:
    """Test TaskDependency model for task sequencing."""

    def test_task_dependency_creation(self):
        """Test TaskDependency can be created with valid data."""
        from crewforge.models.task import TaskDependency

        dependency = TaskDependency(
            task_id="analysis_task",
            depends_on=["research_task", "data_collection_task"],
            dependency_type="sequential",
        )

        assert dependency.task_id == "analysis_task"
        assert dependency.depends_on == ["research_task", "data_collection_task"]
        assert dependency.dependency_type == "sequential"

    def test_task_dependency_required_fields(self):
        """Test TaskDependency requires task_id and depends_on."""
        from crewforge.models.task import TaskDependency

        # Missing task_id should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            TaskDependency(depends_on=["research_task"], dependency_type="sequential")
        assert "task_id" in str(exc_info.value)

        # Missing depends_on should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            TaskDependency(task_id="analysis_task", dependency_type="sequential")
        assert "depends_on" in str(exc_info.value)

    def test_task_dependency_types(self):
        """Test valid dependency types."""
        from crewforge.models.task import TaskDependency

        # Test sequential dependency
        dep_sequential = TaskDependency(
            task_id="task2", depends_on=["task1"], dependency_type="sequential"
        )
        assert dep_sequential.dependency_type == "sequential"

        # Test parallel dependency
        dep_parallel = TaskDependency(
            task_id="task3", depends_on=["task1", "task2"], dependency_type="parallel"
        )
        assert dep_parallel.dependency_type == "parallel"

        # Test conditional dependency
        dep_conditional = TaskDependency(
            task_id="task4", depends_on=["task3"], dependency_type="conditional"
        )
        assert dep_conditional.dependency_type == "conditional"

    def test_task_dependency_invalid_type(self):
        """Test invalid dependency types are rejected."""
        from crewforge.models.task import TaskDependency

        with pytest.raises(ValidationError) as exc_info:
            TaskDependency(
                task_id="task2", depends_on=["task1"], dependency_type="invalid_type"
            )
        assert "dependency_type" in str(exc_info.value) or "literal_error" in str(
            exc_info.value
        )


class TestTaskAgentCompatibility:
    """Test agent-task compatibility validation."""

    def test_validate_agent_task_compatibility(self):
        """Test agent-task compatibility validation function."""
        from crewforge.models.task import validate_agent_task_compatibility
        from crewforge.models.agent import AgentConfig
        from crewforge.models.task import TaskConfig

        # Compatible agent and task
        agent = AgentConfig(
            role="Content Researcher",
            goal="Find and analyze relevant articles on specified topics",
            backstory="An experienced researcher with expertise in finding high-quality content",
            tools=["web_search", "pdf_reader"],
        )

        task = TaskConfig(
            description="Research latest developments in AI technology",
            expected_output="A comprehensive research report with key findings",
            agent="content_researcher",
            tools=["web_search"],  # Subset of agent tools
        )

        # Should return True for compatible combinations
        is_compatible = validate_agent_task_compatibility(agent, task)
        assert is_compatible is True

    def test_incompatible_agent_task_tools(self):
        """Test agent-task incompatibility when task requires tools agent doesn't have."""
        from crewforge.models.task import validate_agent_task_compatibility
        from crewforge.models.agent import AgentConfig
        from crewforge.models.task import TaskConfig

        # Agent with limited tools
        agent = AgentConfig(
            role="Basic Researcher",
            goal="Find basic information",
            backstory="A basic researcher",
            tools=["web_search"],
        )

        # Task requiring tools agent doesn't have
        task = TaskConfig(
            description="Analyze PDF documents and generate summary",
            expected_output="Document analysis report",
            agent="basic_researcher",
            tools=[
                "web_search",
                "pdf_analyzer",
                "image_processor",
            ],  # Agent missing these tools
        )

        # Should return False for incompatible combinations
        is_compatible = validate_agent_task_compatibility(agent, task)
        assert is_compatible is False

    def test_agent_task_compatibility_no_tools(self):
        """Test compatibility when agent or task have no specific tools."""
        from crewforge.models.task import validate_agent_task_compatibility
        from crewforge.models.agent import AgentConfig
        from crewforge.models.task import TaskConfig

        # Agent with no specific tools
        agent = AgentConfig(
            role="General Assistant",
            goal="Help with various tasks",
            backstory="A versatile assistant",
        )

        # Task with no specific tools
        task = TaskConfig(
            description="Write a summary of given information",
            expected_output="A well-structured summary",
            agent="general_assistant",
        )

        # Should be compatible when no specific tools are required
        is_compatible = validate_agent_task_compatibility(agent, task)
        assert is_compatible is True

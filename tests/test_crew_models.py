"""Test crew models for CrewForge using TDD approach."""

import pytest
from pydantic import ValidationError


class TestCrewConfig:
    """Test CrewConfig model for CrewAI crew definition."""

    def test_crew_config_valid_creation(self):
        """Test CrewConfig can be created with valid data."""
        from crewforge.models.crew import CrewConfig
        from crewforge.models.agent import AgentConfig
        from crewforge.models.task import TaskConfig

        # Create test agents and tasks
        agent1 = AgentConfig(
            role="Content Researcher",
            goal="Find and analyze relevant articles",
            backstory="An experienced researcher",
        )
        agent2 = AgentConfig(
            role="Content Writer",
            goal="Create engaging content",
            backstory="A skilled writer",
        )

        task1 = TaskConfig(
            description="Research latest AI developments",
            expected_output="Research report with key findings",
            agent="content_researcher",
        )
        task2 = TaskConfig(
            description="Write an article based on research",
            expected_output="Well-structured article",
            agent="content_writer",
        )

        # Test successful creation
        crew = CrewConfig(
            name="content_creation_crew",
            description="A crew for researching and creating content",
            agents=[agent1, agent2],
            tasks=[task1, task2],
        )

        assert crew.name == "content_creation_crew"
        assert crew.description == "A crew for researching and creating content"
        assert len(crew.agents) == 2
        assert len(crew.tasks) == 2
        assert crew.verbose is False  # Default value
        assert crew.process == "sequential"  # Default value

    def test_crew_config_required_fields(self):
        """Test CrewConfig requires name, agents, and tasks."""
        from crewforge.models.crew import CrewConfig
        from crewforge.models.agent import AgentConfig
        from crewforge.models.task import TaskConfig

        agent = AgentConfig(
            role="Test Agent", goal="Test goal", backstory="Test backstory"
        )
        task = TaskConfig(
            description="Test task", expected_output="Test output", agent="test_agent"
        )

        # Missing name should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            CrewConfig(description="Test crew", agents=[agent], tasks=[task])
        assert "name" in str(exc_info.value)

        # Missing agents should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            CrewConfig(name="test_crew", description="Test crew", tasks=[task])
        assert "agents" in str(exc_info.value)

        # Missing tasks should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            CrewConfig(name="test_crew", description="Test crew", agents=[agent])
        assert "tasks" in str(exc_info.value)

    def test_crew_config_validation(self):
        """Test CrewConfig validation rules."""
        from crewforge.models.crew import CrewConfig
        from crewforge.models.agent import AgentConfig
        from crewforge.models.task import TaskConfig

        agent = AgentConfig(
            role="Test Agent", goal="Test goal", backstory="Test backstory"
        )
        task = TaskConfig(
            description="Test task", expected_output="Test output", agent="test_agent"
        )

        # Empty name should fail
        with pytest.raises(ValidationError):
            CrewConfig(name="", description="Test crew", agents=[agent], tasks=[task])

        # Empty agents list should fail
        with pytest.raises(ValidationError):
            CrewConfig(
                name="test_crew", description="Test crew", agents=[], tasks=[task]
            )

        # Empty tasks list should fail
        with pytest.raises(ValidationError):
            CrewConfig(
                name="test_crew", description="Test crew", agents=[agent], tasks=[]
            )

    def test_crew_config_optional_fields(self):
        """Test CrewConfig optional fields work correctly."""
        from crewforge.models.crew import CrewConfig
        from crewforge.models.agent import AgentConfig
        from crewforge.models.task import TaskConfig

        agent = AgentConfig(
            role="Test Agent", goal="Test goal", backstory="Test backstory"
        )
        task = TaskConfig(
            description="Test task", expected_output="Test output", agent="test_agent"
        )

        crew = CrewConfig(
            name="advanced_crew",
            description="An advanced crew configuration",
            agents=[agent],
            tasks=[task],
            verbose=True,
            process="hierarchical",
            max_rpm=100,
            language="en",
            full_output=True,
        )

        assert crew.verbose is True
        assert crew.process == "hierarchical"
        assert crew.max_rpm == 100
        assert crew.language == "en"
        assert crew.full_output is True


class TestGenerationRequest:
    """Test GenerationRequest model for user prompts."""

    def test_generation_request_valid_creation(self):
        """Test GenerationRequest can be created with valid data."""
        from crewforge.models.crew import GenerationRequest

        request = GenerationRequest(
            prompt="Create a content research crew that finds and analyzes articles",
            project_name="content-research-crew",
            description="A crew for content research and analysis",
        )

        assert (
            request.prompt
            == "Create a content research crew that finds and analyzes articles"
        )
        assert request.project_name == "content-research-crew"
        assert request.description == "A crew for content research and analysis"
        assert request.requirements == []  # Default value
        assert request.domain is None  # Default value

    def test_generation_request_required_fields(self):
        """Test GenerationRequest requires prompt."""
        from crewforge.models.crew import GenerationRequest

        # Missing prompt should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(project_name="test-crew", description="Test description")
        assert "prompt" in str(exc_info.value)

    def test_generation_request_validation(self):
        """Test GenerationRequest validation rules."""
        from crewforge.models.crew import GenerationRequest

        # Empty prompt should fail
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="", project_name="test-crew")

        # Prompt too short should fail
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="Hi", project_name="test-crew")  # Too short

        # Prompt too long should fail (over 2000 characters)
        long_prompt = "A" * 2001
        with pytest.raises(ValidationError):
            GenerationRequest(prompt=long_prompt, project_name="test-crew")

    def test_generation_request_project_name_validation(self):
        """Test project name validation and cleaning."""
        from crewforge.models.crew import GenerationRequest

        # Valid project name should work
        request = GenerationRequest(
            prompt="Create a test crew", project_name="valid-project-name"
        )
        assert request.project_name == "valid-project-name"

        # Project name with spaces should be converted to hyphens
        request = GenerationRequest(
            prompt="Create a test crew", project_name="My Test Crew"
        )
        assert request.project_name == "my-test-crew"

        # Project name with special characters should be cleaned
        request = GenerationRequest(
            prompt="Create a test crew", project_name="Test@Crew#123!"
        )
        assert request.project_name == "test-crew-123"

    def test_generation_request_optional_fields(self):
        """Test GenerationRequest optional fields work correctly."""
        from crewforge.models.crew import GenerationRequest

        request = GenerationRequest(
            prompt="Create an advanced AI research crew",
            project_name="ai-research-crew",
            description="Advanced crew for AI research",
            requirements=["web_search", "data_analysis", "report_generation"],
            domain="artificial_intelligence",
            complexity="advanced",
            output_format="markdown",
        )

        assert request.requirements == [
            "web_search",
            "data_analysis",
            "report_generation",
        ]
        assert request.domain == "artificial_intelligence"
        assert request.complexity == "advanced"
        assert request.output_format == "markdown"


class TestValidationResult:
    """Test ValidationResult model for project checking."""

    def test_validation_result_valid_creation(self):
        """Test ValidationResult can be created with valid data."""
        from crewforge.models.crew import ValidationResult

        result = ValidationResult(
            is_valid=True,
            project_path="/path/to/project",
            syntax_valid=True,
            crewai_compliant=True,
            functional=True,
        )

        assert result.is_valid is True
        assert result.project_path == "/path/to/project"
        assert result.syntax_valid is True
        assert result.crewai_compliant is True
        assert result.functional is True
        assert result.errors == []  # Default value
        assert result.warnings == []  # Default value

    def test_validation_result_required_fields(self):
        """Test ValidationResult requires is_valid and project_path."""
        from crewforge.models.crew import ValidationResult

        # Missing is_valid should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            ValidationResult(project_path="/path/to/project", syntax_valid=True)
        assert "is_valid" in str(exc_info.value)

        # Missing project_path should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            ValidationResult(is_valid=True, syntax_valid=True)
        assert "project_path" in str(exc_info.value)

    def test_validation_result_with_errors(self):
        """Test ValidationResult with errors and warnings."""
        from crewforge.models.crew import ValidationResult

        result = ValidationResult(
            is_valid=False,
            project_path="/path/to/project",
            syntax_valid=False,
            crewai_compliant=True,
            functional=False,
            errors=["Syntax error in agents.py", "Missing import in tasks.py"],
            warnings=["Deprecated function used", "Large file detected"],
            suggestions=[
                "Fix import statement",
                "Consider breaking down large functions",
            ],
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 2
        assert len(result.suggestions) == 2
        assert result.syntax_valid is False
        assert result.functional is False

    def test_validation_result_summary_property(self):
        """Test ValidationResult summary property provides useful overview."""
        from crewforge.models.crew import ValidationResult

        # Valid result summary
        valid_result = ValidationResult(
            is_valid=True,
            project_path="/path/to/valid/project",
            syntax_valid=True,
            crewai_compliant=True,
            functional=True,
        )

        summary = valid_result.summary
        assert "valid" in summary.lower()
        assert "success" in summary.lower() or "passed" in summary.lower()

        # Invalid result summary
        invalid_result = ValidationResult(
            is_valid=False,
            project_path="/path/to/invalid/project",
            syntax_valid=False,
            crewai_compliant=True,
            functional=False,
            errors=["Syntax error", "Runtime error"],
        )

        summary = invalid_result.summary
        assert "invalid" in summary.lower() or "failed" in summary.lower()
        assert "2" in summary  # Should mention number of errors


class TestModelIntegration:
    """Test integration between different models."""

    def test_crew_config_with_real_models(self):
        """Test CrewConfig works with actual agent and task models."""
        from crewforge.models.crew import CrewConfig
        from crewforge.models.agent import AgentConfig
        from crewforge.models.task import TaskConfig

        # Create realistic agents
        researcher = AgentConfig(
            role="Senior Research Analyst",
            goal="Conduct comprehensive research and analysis on given topics",
            backstory="Expert researcher with 10+ years experience in data analysis and market research",
            tools=["web_search", "pdf_analyzer", "data_extractor"],
            verbose=True,
        )

        writer = AgentConfig(
            role="Content Writer",
            goal="Create engaging and informative content based on research",
            backstory="Professional writer specializing in technical and business content",
            tools=["text_editor", "grammar_checker"],
            allow_delegation=False,
        )

        # Create realistic tasks
        research_task = TaskConfig(
            description="Research the latest trends in artificial intelligence and machine learning",
            expected_output="Comprehensive research report with key trends, statistics, and insights",
            agent="senior_research_analyst",
            tools=["web_search", "pdf_analyzer"],
        )

        writing_task = TaskConfig(
            description="Write an engaging article based on the research findings",
            expected_output="Well-structured 1500-word article with clear sections and engaging narrative",
            agent="content_writer",
            context=["research_report"],
            tools=["text_editor"],
        )

        # Create crew with realistic configuration
        crew = CrewConfig(
            name="ai_content_creation_crew",
            description="Professional crew for AI content research and creation",
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process="sequential",
            verbose=True,
            max_rpm=60,
        )

        # Validate the integration works
        assert len(crew.agents) == 2
        assert len(crew.tasks) == 2
        assert crew.agents[0].role == "Senior Research Analyst"
        assert crew.tasks[0].description.startswith("Research the latest trends")
        assert crew.process == "sequential"

    def test_generation_request_to_crew_config_flow(self):
        """Test the flow from GenerationRequest to CrewConfig."""
        from crewforge.models.crew import GenerationRequest, CrewConfig
        from crewforge.models.agent import AgentConfig
        from crewforge.models.task import TaskConfig

        # User request
        request = GenerationRequest(
            prompt="Create a crew that researches competitors and writes analysis reports",
            project_name="competitor-analysis-crew",
            description="Automated competitor research and analysis",
            requirements=["web_search", "data_analysis", "report_writing"],
            domain="business_intelligence",
        )

        # Simulated generation result (what the AI would create)
        generated_crew = CrewConfig(
            name=request.project_name,
            description=request.description,
            agents=[
                AgentConfig(
                    role="Competitive Intelligence Analyst",
                    goal="Research and analyze competitor strategies and performance",
                    backstory="Experienced business analyst specializing in competitive intelligence",
                    tools=request.requirements[:2],  # web_search, data_analysis
                ),
                AgentConfig(
                    role="Business Report Writer",
                    goal="Create comprehensive analysis reports",
                    backstory="Professional business writer with expertise in analytical reporting",
                    tools=request.requirements[2:],  # report_writing
                ),
            ],
            tasks=[
                TaskConfig(
                    description="Research competitor products, pricing, and market positioning",
                    expected_output="Detailed competitor analysis with strengths, weaknesses, and opportunities",
                    agent="competitive_intelligence_analyst",
                    tools=["web_search", "data_analysis"],
                ),
                TaskConfig(
                    description="Write comprehensive analysis report based on research findings",
                    expected_output="Professional business report with executive summary, analysis, and recommendations",
                    agent="business_report_writer",
                    tools=["report_writing"],
                ),
            ],
        )

        # Validate the flow worked correctly
        assert generated_crew.name == request.project_name
        assert generated_crew.description == request.description
        assert len(generated_crew.agents) == 2
        assert len(generated_crew.tasks) == 2

        # Check that requirements were properly distributed
        all_agent_tools = []
        for agent in generated_crew.agents:
            if agent.tools:
                all_agent_tools.extend(agent.tools)

        # All requested tools should be available across agents
        for requirement in request.requirements:
            assert requirement in all_agent_tools

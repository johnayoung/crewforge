"""Tests for CrewAI project scaffolding functionality."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, call, patch
import pytest
import tempfile
import shutil

from crewforge.core.scaffolding import (
    ProjectScaffolder,
    ScaffoldingError,
    CrewAICommandError,
    ProjectStructureError,
    FileSystemError,
)
from crewforge.models import AgentConfig, TaskConfig, GenerationRequest


class TestProjectScaffolder:
    """Test suite for ProjectScaffolder class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_generation_engine(self):
        """Mock generation engine with sample data."""
        engine = Mock()
        engine.analyze_prompt.return_value = {
            "business_context": "Content research and analysis",
            "required_roles": ["Content Researcher", "Content Analyst"],
            "objectives": ["Find articles", "Analyze content"],
            "tools_needed": ["web_search", "file_writing"],
        }

        engine.generate_agents.return_value = [
            AgentConfig(
                role="Content Researcher",
                goal="Find and gather relevant articles",
                backstory="Expert researcher with years of experience",
            ),
            AgentConfig(
                role="Content Analyst",
                goal="Analyze and summarize content",
                backstory="Skilled analyst with domain expertise",
            ),
        ]

        engine.generate_tasks.return_value = [
            TaskConfig(
                description="Research articles on the given topic",
                expected_output="List of relevant article URLs with summaries",
                agent="Content Researcher",
            ),
            TaskConfig(
                description="Analyze the gathered articles",
                expected_output="Comprehensive analysis report",
                agent="Content Analyst",
            ),
        ]

        engine.select_tools.return_value = {
            "selected_tools": [
                {"name": "SerperDevTool", "reason": "For web search capabilities"},
                {"name": "FileWriterTool", "reason": "For writing analysis reports"},
            ],
            "unavailable_tools": [],
        }

        return engine

    @pytest.fixture
    def mock_template_engine(self):
        """Mock template engine."""
        engine = Mock()
        engine.populate_template = Mock()
        return engine

    @pytest.fixture
    def scaffolder(self, mock_generation_engine, mock_template_engine):
        """Create a ProjectScaffolder instance with mocked dependencies."""
        return ProjectScaffolder(
            generation_engine=mock_generation_engine,
            template_engine=mock_template_engine,
        )

    @pytest.fixture
    def sample_request(self):
        """Sample generation request."""
        return GenerationRequest(
            prompt="Create a content research crew", project_name="test-crew"
        )

    def test_init_with_default_engines(self):
        """Test initialization with default engines."""
        scaffolder = ProjectScaffolder()
        assert scaffolder.generation_engine is not None
        assert scaffolder.template_engine is not None

    def test_init_with_custom_engines(
        self, mock_generation_engine, mock_template_engine
    ):
        """Test initialization with custom engines."""
        scaffolder = ProjectScaffolder(
            generation_engine=mock_generation_engine,
            template_engine=mock_template_engine,
        )
        assert scaffolder.generation_engine == mock_generation_engine
        assert scaffolder.template_engine == mock_template_engine

    @patch("subprocess.run")
    def test_create_crewai_project_success(self, mock_run, scaffolder, temp_dir):
        """Test successful CrewAI project creation."""
        project_path = temp_dir / "test-crew"

        # Mock subprocess to simulate successful CrewAI command
        def mock_subprocess_side_effect(*args, **kwargs):
            # First call is --version check
            if args[0] == ["crewai", "--version"]:
                return Mock(returncode=0, stdout="CrewAI 0.1.0", stderr="")
            # Second call is create command - simulate CrewAI creating the directory
            elif args[0] == ["crewai", "create", "crew", "test-crew"]:
                if not project_path.exists():
                    project_path.mkdir()
                return Mock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = mock_subprocess_side_effect

        result = scaffolder.create_crewai_project("test-crew", temp_dir)

        assert result == project_path
        # Check that both calls were made
        assert mock_run.call_count == 2
        mock_run.assert_any_call(
            ["crewai", "--version"], capture_output=True, text=True, timeout=10
        )
        mock_run.assert_any_call(
            ["crewai", "create", "crew", "test-crew"],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    @patch("subprocess.run")
    def test_create_crewai_project_command_failure(
        self, mock_run, scaffolder, temp_dir
    ):
        """Test CrewAI project creation command failure."""

        def mock_subprocess_side_effect(*args, **kwargs):
            # First call is --version check (should succeed)
            if args[0] == ["crewai", "--version"]:
                return Mock(returncode=0, stdout="CrewAI 0.1.0", stderr="")
            # Second call is create command (should fail)
            elif args[0] == ["crewai", "create", "crew", "test-crew"]:
                return Mock(
                    returncode=1, stdout="", stderr="Error: CrewAI command failed"
                )

        mock_run.side_effect = mock_subprocess_side_effect

        with pytest.raises(ScaffoldingError, match="CrewAI project creation failed"):
            scaffolder.create_crewai_project("test-crew", temp_dir)

    @patch("subprocess.run")
    def test_create_crewai_project_subprocess_exception(
        self, mock_run, scaffolder, temp_dir
    ):
        """Test CrewAI project creation with subprocess exception."""
        mock_run.side_effect = subprocess.SubprocessError("Command not found")

        with pytest.raises(
            CrewAICommandError, match="Failed to execute CrewAI command"
        ):
            scaffolder.create_crewai_project("test-crew", temp_dir)

    def test_populate_project_files_success(self, scaffolder, temp_dir):
        """Test successful project file population."""
        project_path = temp_dir / "test-crew"
        project_path.mkdir()
        (project_path / "src").mkdir()
        (project_path / "src" / "test_crew").mkdir()

        agents = [
            AgentConfig(role="Test Agent", goal="Test goal", backstory="Test backstory")
        ]

        tasks = [
            TaskConfig(
                description="Test task",
                expected_output="Test output",
                agent="Test Agent",
            )
        ]

        tools = {
            "selected_tools": [{"name": "TestTool", "reason": "For testing"}],
            "unavailable_tools": [],
        }

        scaffolder.populate_project_files(project_path, agents, tasks, tools)

        # Verify template engine was called for each file type
        expected_calls = [
            call(
                "agents.py.j2",
                project_path / "src" / "test_crew" / "agents.py",
                agents=agents,
            ),
            call(
                "tasks.py.j2",
                project_path / "src" / "test_crew" / "tasks.py",
                tasks=tasks,
                agents=agents,
            ),
            call(
                "tools.py.j2",
                project_path / "src" / "test_crew" / "tools.py",
                tools=tools["selected_tools"],
            ),
            call(
                "crew.py.j2",
                project_path / "src" / "test_crew" / "crew.py",
                agents=agents,
                tasks=tasks,
                tools=tools["selected_tools"],
            ),
        ]

        scaffolder.template_engine.populate_template.assert_has_calls(expected_calls)

    def test_populate_project_files_invalid_project_path(self, scaffolder, temp_dir):
        """Test project file population with invalid project path."""
        invalid_path = temp_dir / "nonexistent"

        with pytest.raises(ScaffoldingError, match="Project directory does not exist"):
            scaffolder.populate_project_files(invalid_path, [], [], {})

    def test_populate_project_files_missing_src_structure(self, scaffolder, temp_dir):
        """Test project file population with missing src structure."""
        project_path = temp_dir / "test-crew"
        project_path.mkdir()

        with pytest.raises(
            ProjectStructureError,
            match="Invalid CrewAI project structure: missing src directory",
        ):
            scaffolder.populate_project_files(project_path, [], [], {})

    def test_populate_project_files_template_error(self, scaffolder, temp_dir):
        """Test project file population with template error."""
        project_path = temp_dir / "test-crew"
        project_path.mkdir()
        (project_path / "src").mkdir()
        (project_path / "src" / "test_crew").mkdir()

        scaffolder.template_engine.populate_template.side_effect = Exception(
            "Template error"
        )

        # Provide proper tools structure to get past the KeyError
        tools = {"selected_tools": []}

        with pytest.raises(FileSystemError, match="Failed to populate agents.py"):
            scaffolder.populate_project_files(project_path, [], [], tools)

    def test_generate_project_success(self, scaffolder, sample_request, temp_dir):
        """Test complete project generation success."""
        with (
            patch.object(scaffolder, "create_crewai_project") as mock_create,
            patch.object(scaffolder, "populate_project_files") as mock_populate,
        ):

            project_path = temp_dir / "test-crew"
            mock_create.return_value = project_path

            result = scaffolder.generate_project(sample_request, temp_dir)

            assert result == project_path

            # Verify the generation pipeline was executed
            scaffolder.generation_engine.analyze_prompt.assert_called_once_with(
                sample_request.prompt, None
            )
            scaffolder.generation_engine.generate_agents.assert_called_once()
            scaffolder.generation_engine.generate_tasks.assert_called_once()
            scaffolder.generation_engine.select_tools.assert_called_once()

            mock_create.assert_called_once_with(sample_request.project_name, temp_dir)
            mock_populate.assert_called_once()

    def test_generate_project_analysis_failure(
        self, scaffolder, sample_request, temp_dir
    ):
        """Test project generation with prompt analysis failure."""
        scaffolder.generation_engine.analyze_prompt.side_effect = Exception(
            "Analysis failed"
        )

        with pytest.raises(ScaffoldingError, match="Failed to analyze prompt"):
            scaffolder.generate_project(sample_request, temp_dir)

    def test_generate_project_agent_generation_failure(
        self, scaffolder, sample_request, temp_dir
    ):
        """Test project generation with agent generation failure."""
        scaffolder.generation_engine.generate_agents.side_effect = Exception(
            "Agent generation failed"
        )

        with pytest.raises(ScaffoldingError, match="Failed to generate agents"):
            scaffolder.generate_project(sample_request, temp_dir)

    def test_generate_project_task_generation_failure(
        self, scaffolder, sample_request, temp_dir
    ):
        """Test project generation with task generation failure."""
        scaffolder.generation_engine.generate_tasks.side_effect = Exception(
            "Task generation failed"
        )

        with pytest.raises(ScaffoldingError, match="Failed to generate tasks"):
            scaffolder.generate_project(sample_request, temp_dir)

    def test_generate_project_tool_selection_failure(
        self, scaffolder, sample_request, temp_dir
    ):
        """Test project generation with tool selection failure."""
        scaffolder.generation_engine.select_tools.side_effect = Exception(
            "Tool selection failed"
        )

        with pytest.raises(ScaffoldingError, match="Failed to select tools"):
            scaffolder.generate_project(sample_request, temp_dir)

    def test_generate_project_crewai_creation_failure(
        self, scaffolder, sample_request, temp_dir
    ):
        """Test project generation with CrewAI project creation failure."""
        with patch.object(scaffolder, "create_crewai_project") as mock_create:
            mock_create.side_effect = ScaffoldingError("CrewAI creation failed")

            with pytest.raises(ScaffoldingError, match="CrewAI creation failed"):
                scaffolder.generate_project(sample_request, temp_dir)

    def test_generate_project_file_population_failure(
        self, scaffolder, sample_request, temp_dir
    ):
        """Test project generation with file population failure."""
        with (
            patch.object(scaffolder, "create_crewai_project") as mock_create,
            patch.object(scaffolder, "populate_project_files") as mock_populate,
        ):

            project_path = temp_dir / "test-crew"
            mock_create.return_value = project_path
            mock_populate.side_effect = ScaffoldingError("File population failed")

            with pytest.raises(ScaffoldingError, match="File population failed"):
                scaffolder.generate_project(sample_request, temp_dir)

    def test_cleanup_on_failure(self, scaffolder, sample_request, temp_dir):
        """Test that cleanup occurs when generation fails."""
        with (
            patch.object(scaffolder, "create_crewai_project") as mock_create,
            patch.object(scaffolder, "populate_project_files") as mock_populate,
            patch("shutil.rmtree") as mock_rmtree,
        ):

            project_path = temp_dir / "test-crew"
            project_path.mkdir()  # Simulate project directory exists
            mock_create.return_value = project_path
            mock_populate.side_effect = ScaffoldingError("Population failed")

            with pytest.raises(ScaffoldingError):
                scaffolder.generate_project(sample_request, temp_dir)

            # Verify cleanup was attempted
            mock_rmtree.assert_called_once_with(project_path, ignore_errors=True)

    def test_validate_project_structure_valid(self, scaffolder, temp_dir):
        """Test project structure validation with valid structure."""
        project_path = temp_dir / "test-crew"
        project_path.mkdir()
        (project_path / "src").mkdir()
        (project_path / "src" / "test_crew").mkdir()

        # Should not raise exception for valid structure
        scaffolder._validate_project_structure(project_path)

    def test_validate_project_structure_invalid(self, scaffolder, temp_dir):
        """Test project structure validation with invalid structure."""
        project_path = temp_dir / "test-crew"
        project_path.mkdir()

        with pytest.raises(
            ProjectStructureError,
            match="Invalid CrewAI project structure: missing src directory",
        ):
            scaffolder._validate_project_structure(project_path)

    def test_get_project_module_path(self, scaffolder, temp_dir):
        """Test getting the correct module path from project name."""
        project_path = temp_dir / "test-crew"
        project_path.mkdir()
        (project_path / "src").mkdir()
        (project_path / "src" / "test_crew").mkdir()

        module_path = scaffolder._get_project_module_path(project_path)
        expected_path = project_path / "src" / "test_crew"
        assert module_path == expected_path

    def test_get_project_module_path_complex_name(self, scaffolder, temp_dir):
        """Test getting module path with complex project name."""
        project_path = temp_dir / "my-awesome-crew"
        project_path.mkdir()
        (project_path / "src").mkdir()
        (project_path / "src" / "my_awesome_crew").mkdir()

        module_path = scaffolder._get_project_module_path(project_path)
        expected_path = project_path / "src" / "my_awesome_crew"
        assert module_path == expected_path

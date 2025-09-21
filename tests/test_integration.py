"""Integration tests for CrewForge end-to-end functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from click.testing import CliRunner

from crewforge.cli.main import cli, generate
from crewforge.models import GenerationRequest, AgentConfig, TaskConfig, CrewConfig
from crewforge.core.generator import GenerationEngine
from crewforge.core.scaffolding import ProjectScaffolder
from crewforge.core.validator import ValidationSuite
from tests.fixtures.sample_prompts import (
    SAMPLE_PROMPTS,
    MOCK_LLM_RESPONSES,
    get_prompt_by_name,
    get_mock_llm_response,
)


class TestEndToEndIntegration:
    """Test complete generation pipeline integration."""

    def test_generation_request_model_validation(self):
        """Test GenerationRequest model accepts valid inputs."""
        # Test valid request
        request = GenerationRequest(
            prompt="Create a content research crew", project_name="test-crew"
        )
        assert request.prompt == "Create a content research crew"
        assert request.project_name == "test-crew"

        # Test validation with invalid data
        with pytest.raises(Exception):  # Pydantic validation error
            GenerationRequest(prompt="", project_name="test-crew")

    def test_cli_command_orchestration_with_mocks(self):
        """Test CLI command orchestrates all components correctly with mocked LLM."""
        runner = CliRunner()

        # Mock all external dependencies
        with (
            patch("crewforge.cli.main.ProjectScaffolder") as mock_scaffolder_class,
            tempfile.TemporaryDirectory() as temp_dir,
        ):

            # Setup mock scaffolder
            mock_scaffolder = Mock()
            mock_project_path = Path(temp_dir) / "test-crew"
            mock_scaffolder.generate_project.return_value = mock_project_path
            mock_scaffolder_class.return_value = mock_scaffolder

            # Run CLI command
            with runner.isolated_filesystem():
                result = runner.invoke(
                    cli,
                    [
                        "generate",
                        "Create a content research crew that finds and summarizes articles",
                    ],
                )

            # Verify command execution
            assert result.exit_code == 0, f"Command failed with output: {result.output}"

            # Verify success messages appear
            assert "✅ Prompt validated" in result.output
            assert "✅ Project name:" in result.output
            assert "✅ Directory is available" in result.output
            assert "✅ Generation request created" in result.output
            assert "✅ Scaffolder ready" in result.output
            assert "🤖 Analyzing prompt for crew requirements" in result.output
            assert "🎉 CrewAI project generated successfully!" in result.output

            # Verify scaffolder was called
            mock_scaffolder_class.assert_called_once()
            mock_scaffolder.generate_project.assert_called_once()

    def test_generation_engine_pipeline_with_mocked_llm(self):
        """Test GenerationEngine processes prompts correctly with mocked LLM responses."""
        # Create mock LLM client with expected responses
        mock_llm = Mock()

        # Setup mock responses for simple research prompt
        sample_prompt = get_prompt_by_name("simple_research")
        mock_analyze_response = get_mock_llm_response(
            "analyze_prompt", "simple_research"
        )
        mock_agents_response = get_mock_llm_response(
            "generate_agents", "simple_research"
        )
        mock_tasks_response = get_mock_llm_response("generate_tasks", "simple_research")

        # Mock tools response
        mock_tools_response = {
            "selected_tools": [
                {"name": "SerperDevTool", "reason": "For web search capabilities"},
                {"name": "FileWriterTool", "reason": "For text analysis and writing"},
            ],
            "unavailable_tools": [],
        }

        # Mock the LLM generate method to return proper responses in sequence
        mock_llm.generate.side_effect = [
            mock_analyze_response,  # For analyze_prompt
            mock_agents_response,  # For generate_agents
            mock_tasks_response,  # For generate_tasks
            mock_tools_response,  # For select_tools
        ]

        # Test generation engine
        engine = GenerationEngine(llm_client=mock_llm)

        # Test analyze_prompt
        analysis = engine.analyze_prompt(sample_prompt["prompt"])
        assert analysis["business_context"] == mock_analyze_response["business_context"]
        assert len(analysis["required_roles"]) >= 2

        # Test generate_agents
        agents = engine.generate_agents(analysis)
        assert len(agents) == sample_prompt["expected_agents"]
        assert all(isinstance(agent, AgentConfig) for agent in agents)

        # Test generate_tasks - note the correct parameter order
        tasks = engine.generate_tasks(agents, analysis)
        assert len(tasks) == sample_prompt["expected_tasks"]
        assert all(isinstance(task, TaskConfig) for task in tasks)

        # Test select_tools
        tools = engine.select_tools(analysis.get("tools_needed", []))
        assert isinstance(tools, dict)  # select_tools returns a dict
        assert "selected_tools" in tools

    @patch("crewforge.core.scaffolding.subprocess.run")
    def test_scaffolding_integration(self, mock_subprocess_run):
        """Test project scaffolding integration with mocked CrewAI command."""
        # Mock successful CrewAI scaffolding command
        mock_subprocess_run.return_value = MagicMock(
            returncode=0, stdout="✅ Project created successfully", stderr=""
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test generation request
            request = GenerationRequest(
                prompt="Create a simple test crew", project_name="test-crew"
            )

            # Test scaffolder
            scaffolder = ProjectScaffolder()

            # Create the directory structure that would be created by crewai command
            project_path = temp_path / "test-crew"
            project_path.mkdir()  # Simulate crewai command creating the directory
            (project_path / "src").mkdir()  # Create src directory structure
            (
                project_path / "src" / "test_crew"
            ).mkdir()  # Create project module directory

            # Mock the generation components
            with (
                patch.object(scaffolder, "generation_engine") as mock_generator,
                patch.object(scaffolder, "template_engine") as mock_template,
            ):

                # Setup mock generation results
                mock_generator.analyze_prompt.return_value = {
                    "business_context": "test"
                }
                mock_generator.generate_agents.return_value = [
                    AgentConfig(
                        role="Test Agent", goal="Test goal", backstory="Test backstory"
                    )
                ]
                mock_generator.generate_tasks.return_value = [
                    TaskConfig(
                        description="Test task",
                        expected_output="Test output",
                        agent="Test Agent",
                    )
                ]
                mock_generator.select_tools.return_value = {
                    "selected_tools": [{"name": "test_tool", "description": "test"}]
                }

                # Setup mock template engine
                mock_template.populate_agents_file.return_value = True
                mock_template.populate_tasks_file.return_value = True
                mock_template.populate_tools_file.return_value = True
                mock_template.populate_crew_file.return_value = True

                # Test project generation
                result_path = scaffolder.generate_project(request, temp_path)

                # Verify project path
                expected_path = temp_path / "test-crew"
                assert result_path == expected_path

    def test_validation_suite_integration(self):
        """Test validation suite can validate generated projects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test-crew"
            project_path.mkdir()

            # Create minimal valid Python files for testing
            (project_path / "agents.py").write_text(
                """
from crewai import Agent

class TestAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Test Agent",
            goal="Test goal", 
            backstory="Test backstory"
        )
"""
            )

            (project_path / "tasks.py").write_text(
                """
from crewai import Task

class TestTask(Task):
    def __init__(self):
        super().__init__(
            description="Test task",
            expected_output="Test output"
        )
"""
            )

            (project_path / "crew.py").write_text(
                """
from crewai import Crew
from .agents import TestAgent
from .tasks import TestTask

class TestCrew(Crew):
    def __init__(self):
        agent = TestAgent()
        task = TestTask()
        super().__init__(
            agents=[agent],
            tasks=[task]
        )
"""
            )

            # Test validation
            validator = ValidationSuite()

            # Test syntax validation
            syntax_result = validator.validate_syntax(str(project_path))
            assert syntax_result.is_valid is True

            # Test basic project structure
            assert (project_path / "agents.py").exists()
            assert (project_path / "tasks.py").exists()
            assert (project_path / "crew.py").exists()


class TestErrorHandling:
    """Test error handling scenarios in integration."""

    def test_cli_handles_invalid_prompts(self):
        """Test CLI handles invalid prompts gracefully."""
        runner = CliRunner()

        # Test empty prompt
        result = runner.invoke(cli, ["generate", ""])
        assert result.exit_code != 0
        assert "cannot be empty" in result.output.lower()

        # Test too short prompt
        result = runner.invoke(cli, ["generate", "hi"])
        assert result.exit_code != 0
        assert "too short" in result.output.lower()

    def test_cli_handles_directory_conflicts(self):
        """Test CLI handles existing directory conflicts."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create conflicting directory
            Path("test-crew").mkdir()

            # Try to generate with same name
            result = runner.invoke(
                cli, ["generate", "--name", "test-crew", "Create a test crew"]
            )

            assert result.exit_code != 0
            assert "already exists" in result.output

    @patch("crewforge.core.generator.GenerationEngine.analyze_prompt")
    def test_cli_handles_llm_failures(self, mock_analyze):
        """Test CLI handles LLM failures gracefully."""
        # Mock LLM failure
        mock_analyze.side_effect = Exception("LLM API failed")

        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["generate", "Create a test crew"])

            assert result.exit_code != 0
            assert "generation failed" in result.output.lower()


class TestPerformance:
    """Test performance requirements from ROADMAP."""

    @patch("crewforge.core.generator.GenerationEngine")
    @patch("crewforge.core.scaffolding.ProjectScaffolder.create_crewai_project")
    @patch("crewforge.cli.main.ProjectScaffolder")
    def test_generation_completes_quickly(
        self, mock_scaffolder_class, mock_create_project, mock_engine_class
    ):
        """Test generation completes within reasonable time limits."""
        import time

        # Setup mocks for fast execution
        mock_engine = Mock()
        mock_engine.analyze_prompt.return_value = {"business_context": "test"}
        mock_engine.generate_agents.return_value = [
            AgentConfig(role="Test Agent", goal="Test goal", backstory="Test backstory")
        ]
        mock_engine.generate_tasks.return_value = [
            TaskConfig(
                description="Test task",
                expected_output="Test output",
                agent="Test Agent",
            )
        ]
        mock_engine.select_tools.return_value = {"selected_tools": []}
        mock_engine_class.return_value = mock_engine

        # Mock scaffolder
        mock_scaffolder = Mock()
        mock_scaffolder.generate_project.return_value = Path("/tmp/test-project")
        mock_scaffolder_class.return_value = mock_scaffolder

        mock_create_project.return_value = Path("/tmp/test-crew")

        runner = CliRunner()

        with runner.isolated_filesystem():
            start_time = time.time()

            result = runner.invoke(
                cli,
                [
                    "generate",
                    "Create a content research crew that finds and summarizes articles",
                ],
            )

            end_time = time.time()
            duration = end_time - start_time

            # Should complete within 30 seconds (much less than 10 minute requirement)
            assert (
                duration < 30
            ), f"Generation took {duration:.2f} seconds, too slow for tests"
            assert result.exit_code == 0


class TestMultiplePromptTypes:
    """Test generation with various prompt types from fixtures."""

    @pytest.mark.parametrize(
        "prompt_name", ["simple_research", "customer_service", "minimal"]
    )
    @patch("crewforge.cli.main.ProjectScaffolder")
    def test_different_prompt_types(self, mock_scaffolder_class, prompt_name):
        """Test CLI handles different types of prompts correctly."""
        # Setup mock scaffolder
        mock_scaffolder = Mock()
        mock_scaffolder.generate_project.return_value = Path("/tmp/test-project")
        mock_scaffolder_class.return_value = mock_scaffolder

        # Get sample prompt
        sample = get_prompt_by_name(prompt_name)

        runner = CliRunner()

        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["generate", sample["prompt"]])

            # Should succeed for all prompt types
            assert (
                result.exit_code == 0
            ), f"Failed for prompt type '{prompt_name}': {result.output}"
            assert "🎉 CrewAI project generated successfully!" in result.output

            # Verify scaffolder was called
            mock_scaffolder.generate_project.assert_called_once()

            # Verify the generation request was created properly
            call_args = mock_scaffolder.generate_project.call_args[0]
            generation_request = call_args[0]
            assert generation_request.prompt == sample["prompt"]

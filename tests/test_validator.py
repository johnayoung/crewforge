"""Tests for the validation suite module."""

import ast
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from crewforge.models.crew import ValidationResult


class TestValidationSuite:
    """Test the ValidationSuite class for project validation."""

    def test_validation_suite_imports(self):
        """Test that ValidationSuite can be imported."""
        from crewforge.core.validator import ValidationSuite

        assert ValidationSuite is not None

    def test_validation_suite_initialization(self):
        """Test ValidationSuite can be initialized."""
        from crewforge.core.validator import ValidationSuite

        validator = ValidationSuite()
        assert validator is not None

    def test_validate_syntax_with_valid_python_file(self, tmp_path):
        """Test validate_syntax with a valid Python file."""
        from crewforge.core.validator import ValidationSuite

        # Create valid Python file
        valid_file = tmp_path / "valid.py"
        valid_file.write_text(
            """
def hello_world():
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello_world()
"""
        )

        validator = ValidationSuite()
        result = validator.validate_syntax(str(tmp_path))

        assert isinstance(result, ValidationResult)
        assert result.syntax_valid is True
        assert len(result.errors) == 0
        assert str(tmp_path) in result.project_path

    def test_validate_syntax_with_invalid_python_file(self, tmp_path):
        """Test validate_syntax with invalid Python syntax."""
        from crewforge.core.validator import ValidationSuite

        # Create invalid Python file with syntax error
        invalid_file = tmp_path / "invalid.py"
        invalid_file.write_text(
            """
def hello_world(
    print("Missing closing parenthesis")
    return True
"""
        )

        validator = ValidationSuite()
        result = validator.validate_syntax(str(tmp_path))

        assert isinstance(result, ValidationResult)
        assert result.syntax_valid is False
        assert len(result.errors) > 0
        assert any("syntax" in error.lower() for error in result.errors)
        assert str(tmp_path) in result.project_path

    def test_validate_syntax_with_multiple_files(self, tmp_path):
        """Test validate_syntax with multiple Python files."""
        from crewforge.core.validator import ValidationSuite

        # Create multiple valid Python files
        (tmp_path / "agents.py").write_text("class Agent: pass")
        (tmp_path / "tasks.py").write_text("class Task: pass")
        (tmp_path / "crew.py").write_text("from agents import Agent")

        validator = ValidationSuite()
        result = validator.validate_syntax(str(tmp_path))

        assert isinstance(result, ValidationResult)
        assert result.syntax_valid is True
        assert len(result.files_checked) == 3
        assert any("agents.py" in f for f in result.files_checked)
        assert any("tasks.py" in f for f in result.files_checked)
        assert any("crew.py" in f for f in result.files_checked)

    def test_validate_syntax_ignores_non_python_files(self, tmp_path):
        """Test validate_syntax ignores non-Python files."""
        from crewforge.core.validator import ValidationSuite

        # Create Python and non-Python files
        (tmp_path / "valid.py").write_text("print('hello')")
        (tmp_path / "readme.txt").write_text("This is not Python")
        (tmp_path / "config.json").write_text('{"key": "value"}')

        validator = ValidationSuite()
        result = validator.validate_syntax(str(tmp_path))

        assert isinstance(result, ValidationResult)
        assert result.syntax_valid is True
        assert len(result.files_checked) == 1
        assert "valid.py" in str(result.files_checked[0])

    def test_validate_crewai_compliance_with_valid_structure(self, tmp_path):
        """Test validate_crewai_compliance with valid CrewAI structure."""
        from crewforge.core.validator import ValidationSuite

        # Create typical CrewAI project structure
        (tmp_path / "agents.py").write_text(
            """
from crewai import Agent

class MyAgent:
    def __init__(self):
        self.agent = Agent(
            role="Test Role",
            goal="Test Goal",
            backstory="Test Backstory"
        )
"""
        )

        (tmp_path / "tasks.py").write_text(
            """
from crewai import Task

class MyTask:
    def __init__(self):
        self.task = Task(
            description="Test description",
            expected_output="Test output"
        )
"""
        )

        (tmp_path / "crew.py").write_text(
            """
from crewai import Crew
from agents import MyAgent
from tasks import MyTask

def create_crew():
    return Crew(
        agents=[MyAgent().agent],
        tasks=[MyTask().task]
    )
"""
        )

        validator = ValidationSuite()
        result = validator.validate_crewai_compliance(str(tmp_path))

        assert isinstance(result, ValidationResult)
        assert result.crewai_compliant is True
        assert len(result.errors) == 0

    def test_validate_crewai_compliance_with_missing_imports(self, tmp_path):
        """Test validate_crewai_compliance with missing CrewAI imports."""
        from crewforge.core.validator import ValidationSuite

        # Create files without proper CrewAI imports
        (tmp_path / "agents.py").write_text(
            """
class MyAgent:
    def __init__(self):
        pass  # Missing CrewAI Agent import and usage
"""
        )

        validator = ValidationSuite()
        result = validator.validate_crewai_compliance(str(tmp_path))

        assert isinstance(result, ValidationResult)
        assert result.crewai_compliant is False
        assert len(result.errors) > 0
        assert any("crewai" in error.lower() for error in result.errors)

    def test_validate_functionality_with_working_crew(self, tmp_path):
        """Test validate_functionality with a working crew."""
        from crewforge.core.validator import ValidationSuite

        # Create minimal working crew structure
        (tmp_path / "agents.py").write_text(
            """
from crewai import Agent

def get_agent():
    return Agent(
        role="Test Role",
        goal="Test Goal",
        backstory="Test Backstory"
    )
"""
        )

        (tmp_path / "tasks.py").write_text(
            """
from crewai import Task

def get_task():
    return Task(
        description="Test description",
        expected_output="Test output"
    )
"""
        )

        (tmp_path / "crew.py").write_text(
            """
from crewai import Crew
from agents import get_agent
from tasks import get_task

def create_crew():
    return Crew(
        agents=[get_agent()],
        tasks=[get_task()]
    )

crew = create_crew()
"""
        )

        validator = ValidationSuite()

        # Mock the actual crew instantiation since we don't have CrewAI in test env
        with (
            patch("importlib.util.spec_from_file_location") as mock_spec,
            patch("importlib.util.module_from_spec") as mock_module,
        ):
            mock_module.return_value = Mock()
            mock_spec.return_value = Mock()
            mock_spec.return_value.loader = Mock()

            result = validator.validate_functionality(str(tmp_path))

            assert isinstance(result, ValidationResult)
            # For functionality test, we expect it to try to import
            # The actual result depends on our mock implementation

    def test_validate_functionality_with_import_error(self, tmp_path):
        """Test validate_functionality with files that can't be imported."""
        from crewforge.core.validator import ValidationSuite

        # Create file with import error
        (tmp_path / "crew.py").write_text(
            """
from non_existent_module import something

def create_crew():
    return something()
"""
        )

        validator = ValidationSuite()
        result = validator.validate_functionality(str(tmp_path))

        assert isinstance(result, ValidationResult)
        assert result.functional is False
        assert len(result.errors) > 0

    def test_full_validation_pipeline(self, tmp_path):
        """Test complete validation pipeline with all checks."""
        from crewforge.core.validator import ValidationSuite

        # Create a complete but minimal project
        (tmp_path / "agents.py").write_text(
            """
from crewai import Agent

class ResearchAgent:
    def __init__(self):
        self.agent = Agent(
            role="Researcher",
            goal="Research topics thoroughly",
            backstory="An experienced researcher"
        )
"""
        )

        (tmp_path / "tasks.py").write_text(
            """
from crewai import Task

class ResearchTask:
    def __init__(self):
        self.task = Task(
            description="Research the given topic",
            expected_output="A comprehensive research report"
        )
"""
        )

        (tmp_path / "crew.py").write_text(
            """
from crewai import Crew
from agents import ResearchAgent
from tasks import ResearchTask

def create_crew():
    agent = ResearchAgent()
    task = ResearchTask()
    return Crew(
        agents=[agent.agent],
        tasks=[task.task]
    )
"""
        )

        validator = ValidationSuite()

        # Mock CrewAI imports for functionality test
        with (
            patch("sys.path"),
            patch("importlib.util.spec_from_file_location"),
            patch("importlib.util.module_from_spec"),
        ):

            result = validator.validate_project(str(tmp_path))

            assert isinstance(result, ValidationResult)
            assert result.project_path == str(tmp_path)
            # At minimum, syntax should be valid
            assert result.syntax_valid is True

    def test_validation_error_reporting(self, tmp_path):
        """Test that validation provides detailed error reporting."""
        from crewforge.core.validator import ValidationSuite

        # Create file with multiple issues
        (tmp_path / "bad_file.py").write_text(
            """
# Missing imports
def bad_function(
    print("Syntax error - missing closing paren")
    return None

# This will cause import issues
from non_existent import something
"""
        )

        validator = ValidationSuite()
        result = validator.validate_project(str(tmp_path))

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert result.syntax_valid is False

        # Check that error messages contain useful information
        error_text = " ".join(result.errors).lower()
        assert "syntax" in error_text

    def test_validation_with_empty_directory(self, tmp_path):
        """Test validation with empty project directory."""
        from crewforge.core.validator import ValidationSuite

        validator = ValidationSuite()
        result = validator.validate_project(str(tmp_path))

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any(
            "no python files" in error.lower() or "empty" in error.lower()
            for error in result.errors
        )

    def test_validation_with_nonexistent_directory(self):
        """Test validation with nonexistent directory."""
        from crewforge.core.validator import ValidationSuite

        validator = ValidationSuite()
        result = validator.validate_project("/nonexistent/path")

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any(
            "not found" in error.lower() or "does not exist" in error.lower()
            for error in result.errors
        )

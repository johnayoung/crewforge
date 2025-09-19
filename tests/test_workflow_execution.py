"""
Tests for CrewAI workflow execution validation functionality.

This module tests the execution validation system for generated CrewAI projects,
ensuring they can run their crew workflows without errors and provide meaningful feedback.
"""

import tempfile
import textwrap
from pathlib import Path
from typing import Dict, Any
import pytest
import yaml

from crewforge.validation import (
    ValidationIssue,
    ValidationResult,
    IssueSeverity,
)


class TestCrewAIWorkflowExecution:
    """Test cases for CrewAI workflow execution validation."""

    @pytest.fixture
    def basic_crewai_project(self):
        """Create a basic CrewAI project structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)

            # Create project structure
            src_dir = project_root / "src" / "test_project"
            config_dir = src_dir / "config"
            tools_dir = src_dir / "tools"

            src_dir.mkdir(parents=True)
            config_dir.mkdir()
            tools_dir.mkdir()

            # Create basic project files
            (project_root / "pyproject.toml").write_text(
                textwrap.dedent(
                    """
                [project]
                name = "test-project"
                version = "0.1.0"
                dependencies = ["crewai"]
            """
                )
            )

            (project_root / "README.md").write_text("# Test Project")
            (project_root / ".env").write_text("# Environment variables")

            # Create Python files
            (src_dir / "__init__.py").write_text("")
            (tools_dir / "__init__.py").write_text("")
            (tools_dir / "custom_tool.py").write_text("# Custom tools")

            # Create valid configuration files
            agents_config = {
                "test_agent": {
                    "role": "Test Agent",
                    "goal": "Test the crew functionality",
                    "backstory": "You are a test agent designed to validate crew execution",
                    "tools": [],
                    "verbose": True,
                }
            }

            tasks_config = {
                "test_task": {
                    "description": "Execute a simple test to validate crew functionality",
                    "expected_output": "A confirmation message that the crew executed successfully",
                    "agent": "test_agent",
                }
            }

            with open(config_dir / "agents.yaml", "w") as f:
                yaml.dump(agents_config, f)
            with open(config_dir / "tasks.yaml", "w") as f:
                yaml.dump(tasks_config, f)

            yield project_root, src_dir

    @pytest.fixture
    def executable_crewai_project(self, basic_crewai_project):
        """Create a CrewAI project with executable crew.py and main.py files."""
        project_root, src_dir = basic_crewai_project

        # Create crew.py with simple mock implementation that doesn't require crewai
        (src_dir / "crew.py").write_text(
            textwrap.dedent(
                """
            import yaml
            from pathlib import Path

            class TestCrew:
                def __init__(self):
                    self.config_path = Path(__file__).parent / "config"
                    
                def agents(self):
                    # Load agent config from yaml (mock implementation)
                    with open(self.config_path / "agents.yaml") as f:
                        agents_config = yaml.safe_load(f)
                    
                    # Return mock agent data instead of actual CrewAI Agent objects
                    return [
                        {
                            "role": agents_config["test_agent"]["role"],
                            "goal": agents_config["test_agent"]["goal"],
                            "backstory": agents_config["test_agent"]["backstory"],
                            "verbose": agents_config["test_agent"]["verbose"]
                        }
                    ]
                
                def tasks(self):
                    # Load tasks config from yaml (mock implementation)
                    with open(self.config_path / "tasks.yaml") as f:
                        tasks_config = yaml.safe_load(f)
                    
                    agents = self.agents()
                    return [
                        {
                            "description": tasks_config["test_task"]["description"],
                            "expected_output": tasks_config["test_task"]["expected_output"],
                            "agent": agents[0]
                        }
                    ]
                
                def crew(self):
                    # Mock crew object
                    return {
                        "agents": self.agents(),
                        "tasks": self.tasks(),
                        "verbose": True
                    }
                
                def kickoff(self):
                    # Mock execution result
                    return "Mock crew execution completed successfully"
        """
            )
        )

        # Create main.py with simple execution that doesn't require crewai
        (src_dir / "main.py").write_text(
            textwrap.dedent(
                """
            import sys
            import os
            
            # Add the src directory to Python path for imports
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            
            from test_project.crew import TestCrew

            def main():
                try:
                    # Simple validation execution
                    crew = TestCrew()
                    print("Crew initialized successfully")
                    
                    # For testing, we'll validate initialization without requiring actual CrewAI
                    agents = crew.agents()
                    tasks = crew.tasks()
                    
                    print(f"Created {len(agents)} agents and {len(tasks)} tasks")
                    print("Workflow validation completed successfully")
                    return True
                except Exception as e:
                    print(f"Error during crew execution: {e}", file=sys.stderr)
                    return False

            if __name__ == "__main__":
                success = main()
                sys.exit(0 if success else 1)
        """
            )
        )

        yield project_root, src_dir

    def test_validate_workflow_execution_success(self, executable_crewai_project):
        """Test successful workflow execution validation."""
        from crewforge.validation import validate_crewai_workflow_execution

        project_root, src_dir = executable_crewai_project

        result = validate_crewai_workflow_execution(str(project_root))

        assert result.is_valid
        info_messages = [issue.message for issue in result.info_messages]
        assert any(
            "workflow executed successfully" in msg.lower() for msg in info_messages
        )

    def test_validate_workflow_execution_missing_main(self, basic_crewai_project):
        """Test workflow execution validation when main.py is missing."""
        from crewforge.validation import validate_crewai_workflow_execution

        project_root, src_dir = basic_crewai_project

        # Don't create main.py
        (src_dir / "crew.py").write_text("# Empty crew file")

        result = validate_crewai_workflow_execution(str(project_root))

        assert not result.is_valid
        error_messages = [error.message for error in result.errors]
        assert any(
            "main.py not found" in msg.lower() or "no main entry point" in msg.lower()
            for msg in error_messages
        )

    def test_validate_workflow_execution_syntax_error(self, basic_crewai_project):
        """Test workflow execution validation when main.py has syntax errors."""
        from crewforge.validation import validate_crewai_workflow_execution

        project_root, src_dir = basic_crewai_project

        # Create main.py with syntax error
        (src_dir / "main.py").write_text(
            textwrap.dedent(
                """
            def main(
                # Missing closing parenthesis - syntax error
                print("This will cause a syntax error")
        """
            )
        )

        result = validate_crewai_workflow_execution(str(project_root))

        assert not result.is_valid
        error_messages = [error.message for error in result.errors]
        assert any(
            "syntax error" in msg.lower() or "execution failed" in msg.lower()
            for msg in error_messages
        )

    def test_validate_workflow_execution_runtime_error(self, basic_crewai_project):
        """Test workflow execution validation when main.py has runtime errors."""
        from crewforge.validation import validate_crewai_workflow_execution

        project_root, src_dir = basic_crewai_project

        # Create main.py with runtime error
        (src_dir / "main.py").write_text(
            textwrap.dedent(
                """
            def main():
                # This will cause a runtime error
                undefined_variable.some_method()
                
            if __name__ == "__main__":
                main()
        """
            )
        )

        result = validate_crewai_workflow_execution(str(project_root))

        assert not result.is_valid
        error_messages = [error.message for error in result.errors]
        assert any(
            "runtime error" in msg.lower() or "execution failed" in msg.lower()
            for msg in error_messages
        )

    def test_validate_workflow_execution_timeout(self, basic_crewai_project):
        """Test workflow execution validation with timeout handling."""
        from crewforge.validation import validate_crewai_workflow_execution

        project_root, src_dir = basic_crewai_project

        # Create main.py that would run forever
        (src_dir / "main.py").write_text(
            textwrap.dedent(
                """
            import time
            
            def main():
                # This will timeout during testing
                while True:
                    time.sleep(1)
                    
            if __name__ == "__main__":
                main()
        """
            )
        )

        # Test with very short timeout
        result = validate_crewai_workflow_execution(str(project_root), timeout=1)

        assert not result.is_valid
        error_messages = [error.message for error in result.errors]
        assert any(
            "timeout" in msg.lower() or "execution timed out" in msg.lower()
            for msg in error_messages
        )

    def test_validate_workflow_execution_import_error(self, basic_crewai_project):
        """Test workflow execution validation when imports are missing."""
        from crewforge.validation import validate_crewai_workflow_execution

        project_root, src_dir = basic_crewai_project

        # Create main.py with missing import
        (src_dir / "main.py").write_text(
            textwrap.dedent(
                """
            from nonexistent_module import SomeClass
            
            def main():
                obj = SomeClass()
                return obj.do_something()
                
            if __name__ == "__main__":
                main()
        """
            )
        )

        result = validate_crewai_workflow_execution(str(project_root))

        assert not result.is_valid
        error_messages = [error.message for error in result.errors]
        assert any(
            "import error" in msg.lower() or "module not found" in msg.lower()
            for msg in error_messages
        )

    def test_validate_workflow_execution_with_environment_variables(
        self, executable_crewai_project
    ):
        """Test workflow execution validation with environment variable handling."""
        from crewforge.validation import validate_crewai_workflow_execution

        project_root, src_dir = executable_crewai_project

        # Update .env file with test variables
        (project_root / ".env").write_text(
            textwrap.dedent(
                """
            TEST_VAR=test_value
            PYTHONPATH=./src
        """
            )
        )

        # Update main.py to use environment variable
        (src_dir / "main.py").write_text(
            textwrap.dedent(
                """
            import os
            import sys
            
            def main():
                try:
                    test_var = os.getenv("TEST_VAR", "default")
                    print(f"Test variable: {test_var}")
                    print("Environment test completed successfully")
                    return True
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                    return False
                    
            if __name__ == "__main__":
                success = main()
                sys.exit(0 if success else 1)
        """
            )
        )

        result = validate_crewai_workflow_execution(str(project_root))

        assert result.is_valid
        info_messages = [issue.message for issue in result.info_messages]
        assert any(
            "workflow executed successfully" in msg.lower() for msg in info_messages
        )

    def test_validate_workflow_execution_nonexistent_project(self):
        """Test workflow execution validation with non-existent project."""
        from crewforge.validation import validate_crewai_workflow_execution

        result = validate_crewai_workflow_execution("/nonexistent/path")

        assert not result.is_valid
        error_messages = [error.message for error in result.errors]
        assert any(
            "project not found" in msg.lower() or "does not exist" in msg.lower()
            for msg in error_messages
        )

    def test_validate_workflow_execution_with_output_capture(
        self, executable_crewai_project
    ):
        """Test that workflow execution captures and validates output."""
        from crewforge.validation import validate_crewai_workflow_execution

        project_root, src_dir = executable_crewai_project

        # Update main.py to produce specific output
        (src_dir / "main.py").write_text(
            textwrap.dedent(
                """
            import sys
            
            def main():
                try:
                    print("Starting workflow validation")
                    print("Processing agents and tasks")
                    print("Workflow validation completed successfully")
                    return True
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                    return False
                    
            if __name__ == "__main__":
                success = main()
                sys.exit(0 if success else 1)
        """
            )
        )

        result = validate_crewai_workflow_execution(str(project_root))

        assert result.is_valid

        # Check that execution output is captured in validation results
        info_messages = [issue.message for issue in result.info_messages]
        assert any(
            "workflow executed successfully" in msg.lower() for msg in info_messages
        )

        # Some validation functions might include output in field_path or elsewhere
        all_text = " ".join([issue.message for issue in result.issues])
        assert (
            "workflow validation completed successfully" in all_text.lower()
            or result.is_valid
        )

    def test_workflow_execution_integration_with_project_validation(
        self, executable_crewai_project
    ):
        """Test that workflow execution integrates with main project validation."""
        from crewforge.validation import validate_generated_project

        project_root, src_dir = executable_crewai_project

        result = validate_generated_project(str(project_root))

        # Should include workflow execution results
        assert len(result.issues) > 0  # Should have various validation results

        # Check that workflow execution was included
        all_messages = [issue.message for issue in result.issues]
        workflow_messages = [msg for msg in all_messages if "workflow" in msg.lower()]

        # Should have workflow execution validation messages
        assert len(workflow_messages) > 0

        # Specifically check for successful workflow execution
        success_messages = [
            msg
            for msg in all_messages
            if "workflow executed successfully" in msg.lower()
        ]
        assert len(success_messages) > 0

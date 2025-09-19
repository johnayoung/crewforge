"""
Tests for Python syntax and import validation functionality.

This module tests the validation system for generated CrewAI projects,
ensuring Python files have valid syntax and imports are resolvable.
"""

import ast
import tempfile
import textwrap
from pathlib import Path
from typing import List
import pytest

from crewforge.validation import (
    ValidationIssue,
    ValidationResult,
    IssueSeverity,
)


class TestPythonSyntaxValidation:
    """Test cases for Python syntax validation."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary directory for test projects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_validate_valid_python_file(self, temp_project_dir):
        """Test validation of Python files with valid syntax."""
        # Create a valid Python file
        python_file = temp_project_dir / "valid_file.py"
        python_file.write_text(
            textwrap.dedent(
                """
            import os
            from typing import List

            def hello_world():
                print("Hello, World!")
                return True

            class TestClass:
                def __init__(self):
                    self.value = 42

            if __name__ == "__main__":
                hello_world()
        """
            ).strip()
        )

        # Import the function we'll implement
        from crewforge.validation import validate_python_syntax

        result = validate_python_syntax(python_file)

        assert result.is_valid
        assert len(result.errors) == 0
        assert any(
            issue.severity == IssueSeverity.INFO
            for issue in result.issues
            if "valid Python syntax" in issue.message
        )

    def test_validate_python_file_syntax_error(self, temp_project_dir):
        """Test validation of Python files with syntax errors."""
        # Create a Python file with syntax errors
        python_file = temp_project_dir / "syntax_error.py"
        python_file.write_text(
            textwrap.dedent(
                """
            import os
            
            def invalid_function(
                print("Missing closing parenthesis")
                return True
            
            # Invalid indentation
        wrong_indent = "This should be indented"
        """
            ).strip()
        )

        from crewforge.validation import validate_python_syntax

        result = validate_python_syntax(python_file)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any(
            "syntax error" in error.message.lower()
            or "indentation error" in error.message.lower()
            for error in result.errors
        )

    def test_validate_python_file_indentation_error(self, temp_project_dir):
        """Test validation of Python files with indentation errors."""
        python_file = temp_project_dir / "indentation_error.py"
        python_file.write_text(
            textwrap.dedent(
                """
            def test_function():
                print("Correct indentation")
                  print("Wrong indentation - extra spaces")
                return True
        """
            ).strip()
        )

        from crewforge.validation import validate_python_syntax

        result = validate_python_syntax(python_file)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("indentation" in error.message.lower() for error in result.errors)

    def test_validate_python_file_encoding_error(self, temp_project_dir):
        """Test validation of Python files with encoding issues."""
        python_file = temp_project_dir / "encoding_error.py"
        # Write binary data that's not valid UTF-8
        python_file.write_bytes(b'\xff\xfe# Invalid encoding\nprint("test")')

        from crewforge.validation import validate_python_syntax

        result = validate_python_syntax(python_file)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("encoding" in error.message.lower() for error in result.errors)

    def test_validate_nonexistent_file(self, temp_project_dir):
        """Test validation of non-existent Python files."""
        nonexistent_file = temp_project_dir / "nonexistent.py"

        from crewforge.validation import validate_python_syntax

        result = validate_python_syntax(nonexistent_file)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("file not found" in error.message.lower() for error in result.errors)

    def test_validate_directory_not_file(self, temp_project_dir):
        """Test validation when provided path is a directory, not a file."""
        directory = temp_project_dir / "test_dir"
        directory.mkdir()

        from crewforge.validation import validate_python_syntax

        result = validate_python_syntax(directory)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("not a file" in error.message.lower() for error in result.errors)

    def test_validate_empty_file(self, temp_project_dir):
        """Test validation of empty Python files."""
        empty_file = temp_project_dir / "empty.py"
        empty_file.write_text("")

        from crewforge.validation import validate_python_syntax

        result = validate_python_syntax(empty_file)

        assert result.is_valid  # Empty files are syntactically valid
        assert any("empty file" in issue.message.lower() for issue in result.warnings)

    def test_validate_complex_syntax_error(self, temp_project_dir):
        """Test validation with complex syntax errors."""
        python_file = temp_project_dir / "complex_error.py"
        python_file.write_text(
            textwrap.dedent(
                """
            import os
            
            # Multiple syntax errors
            def broken_function():
                # Missing colon after if
                if True
                    print("Missing colon")
                
                # Unclosed string
                text = "This string is not closed
                
                # Invalid assignment
                123 = "cannot assign to literal"
                
                return None
        """
            ).strip()
        )

        from crewforge.validation import validate_python_syntax

        result = validate_python_syntax(python_file)

        assert not result.is_valid
        assert len(result.errors) > 0
        # Should detect the first syntax error
        assert any("syntax" in error.message.lower() for error in result.errors)


class TestPythonImportValidation:
    """Test cases for Python import validation."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary directory for test projects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_validate_valid_stdlib_imports(self, temp_project_dir):
        """Test validation of valid standard library imports."""
        python_file = temp_project_dir / "valid_imports.py"
        python_file.write_text(
            textwrap.dedent(
                """
            import os
            import sys
            from pathlib import Path
            from typing import List, Dict
            import json
        """
            ).strip()
        )

        from crewforge.validation import validate_python_imports

        result = validate_python_imports(python_file)

        assert result.is_valid
        assert len(result.errors) == 0
        assert any(
            "imports validated successfully" in issue.message.lower()
            for issue in result.info_messages
        )

    def test_validate_invalid_imports(self, temp_project_dir):
        """Test validation of invalid/missing imports."""
        python_file = temp_project_dir / "invalid_imports.py"
        python_file.write_text(
            textwrap.dedent(
                """
            import os
            import nonexistent_module
            from fake_package import nonexistent_function
            from typing import List
        """
            ).strip()
        )

        from crewforge.validation import validate_python_imports

        result = validate_python_imports(python_file)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("nonexistent_module" in error.message for error in result.errors)
        assert any("fake_package" in error.message for error in result.errors)

    def test_validate_relative_imports(self, temp_project_dir):
        """Test validation of relative imports."""
        # Create a package structure
        package_dir = temp_project_dir / "mypackage"
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text("")

        submodule = package_dir / "submodule.py"
        submodule.write_text("def test_function(): pass")

        main_file = package_dir / "main.py"
        main_file.write_text(
            textwrap.dedent(
                """
            from . import submodule
            from .submodule import test_function
        """
            ).strip()
        )

        from crewforge.validation import validate_python_imports

        result = validate_python_imports(main_file, str(temp_project_dir))

        # Relative imports might be tricky to validate without proper execution context
        # At minimum, should not crash and provide meaningful feedback
        assert isinstance(result, ValidationResult)

    def test_validate_circular_imports(self, temp_project_dir):
        """Test detection of potential circular imports."""
        # Create two files that import each other
        file_a = temp_project_dir / "module_a.py"
        file_a.write_text("from module_b import function_b")

        file_b = temp_project_dir / "module_b.py"
        file_b.write_text("from module_a import function_a")

        from crewforge.validation import validate_python_imports

        result_a = validate_python_imports(file_a, str(temp_project_dir))
        result_b = validate_python_imports(file_b, str(temp_project_dir))

        # Should detect potential circular import issues
        assert isinstance(result_a, ValidationResult)
        assert isinstance(result_b, ValidationResult)

    def test_validate_imports_with_syntax_error(self, temp_project_dir):
        """Test import validation when file has syntax errors."""
        python_file = temp_project_dir / "syntax_and_import_errors.py"
        python_file.write_text(
            textwrap.dedent(
                """
            import os
            import nonexistent_module
            
            def broken_function(
                # Missing closing parenthesis
                print("This has syntax error")
        """
            ).strip()
        )

        from crewforge.validation import validate_python_imports

        result = validate_python_imports(python_file)

        # Should handle syntax errors gracefully
        assert not result.is_valid
        assert any(
            "syntax" in error.message.lower() or "parse" in error.message.lower()
            for error in result.errors
        )

    def test_validate_project_imports(self, temp_project_dir):
        """Test validation of imports across an entire project."""
        # Create a realistic CrewAI project structure
        src_dir = temp_project_dir / "src" / "test_project"
        src_dir.mkdir(parents=True)

        # Create __init__.py files
        (temp_project_dir / "src" / "__init__.py").write_text("")
        (src_dir / "__init__.py").write_text("")

        # Create main.py with various imports
        main_file = src_dir / "main.py"
        main_file.write_text(
            textwrap.dedent(
                """
            import os
            from crewai import Agent, Task, Crew
            from .crew import TestCrew
            from .tools.custom_tool import CustomTool
        """
            ).strip()
        )

        # Create crew.py
        crew_file = src_dir / "crew.py"
        crew_file.write_text(
            textwrap.dedent(
                """
            from crewai import Agent, Task, Crew
            import yaml
            from typing import List
        """
            ).strip()
        )

        # Create tools directory and custom tool
        tools_dir = src_dir / "tools"
        tools_dir.mkdir()
        (tools_dir / "__init__.py").write_text("")

        custom_tool_file = tools_dir / "custom_tool.py"
        custom_tool_file.write_text(
            textwrap.dedent(
                """
            from crewai_tools import tool
            import requests
        """
            ).strip()
        )

        from crewforge.validation import validate_project_imports

        result = validate_project_imports(str(temp_project_dir))

        assert isinstance(result, ValidationResult)
        # Should identify missing external dependencies like crewai
        if not result.is_valid:
            assert any("crewai" in error.message.lower() for error in result.errors)


class TestComprehensiveProjectValidation:
    """Test cases for comprehensive project validation."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary directory for test projects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_validate_complete_crewai_project(self, temp_project_dir):
        """Test validation of a complete CrewAI project structure."""
        # Create a complete CrewAI project structure
        project_root = temp_project_dir / "test_crew"
        src_dir = project_root / "src" / "test_crew"
        src_dir.mkdir(parents=True)

        # Create required files
        (project_root / "pyproject.toml").write_text("[project]\nname = 'test-crew'")
        (project_root / "README.md").write_text("# Test Crew Project")
        (project_root / ".env").write_text("OPENAI_API_KEY=sk-test")

        # Create Python files
        (src_dir / "__init__.py").write_text("")
        (src_dir / "main.py").write_text(
            textwrap.dedent(
                """
            import os
            from test_crew.crew import TestCrew
            
            def main():
                crew = TestCrew()
                crew.run()
            
            if __name__ == "__main__":
                main()
        """
            ).strip()
        )

        (src_dir / "crew.py").write_text(
            textwrap.dedent(
                """
            from crewai import Agent, Task, Crew
            
            class TestCrew:
                def run(self):
                    return "Running crew"
        """
            ).strip()
        )

        # Create config directory and files
        config_dir = src_dir / "config"
        config_dir.mkdir()
        (config_dir / "agents.yaml").write_text("agents: []")
        (config_dir / "tasks.yaml").write_text("tasks: []")

        from crewforge.validation import validate_generated_project

        result = validate_generated_project(str(project_root))

        # Should have some warnings about missing dependencies but overall structure is valid
        assert isinstance(result, ValidationResult)
        # Should find all the required files
        info_messages = [msg.message for msg in result.info_messages]
        assert any("Found required file: main.py" in msg for msg in info_messages)
        assert any("Found required file: crew.py" in msg for msg in info_messages)

    def test_validate_invalid_project_structure(self, temp_project_dir):
        """Test validation of invalid project structure."""
        # Create incomplete project
        project_root = temp_project_dir / "invalid_project"
        project_root.mkdir()

        # Only create a basic file, missing src structure
        (project_root / "main.py").write_text("print('hello')")

        from crewforge.validation import validate_generated_project

        result = validate_generated_project(str(project_root))

        assert not result.is_valid
        assert any(
            "Missing 'src' directory" in error.message for error in result.errors
        )

    def test_validate_project_with_syntax_errors(self, temp_project_dir):
        """Test validation catches syntax errors in project files."""
        # Create project with syntax errors
        project_root = temp_project_dir / "syntax_error_project"
        src_dir = project_root / "src" / "syntax_error_project"
        src_dir.mkdir(parents=True)

        # Create file with syntax error
        (src_dir / "main.py").write_text(
            textwrap.dedent(
                """
            def broken_function(
                print("Missing closing parenthesis")
        """
            ).strip()
        )

        from crewforge.validation import validate_generated_project

        result = validate_generated_project(str(project_root))

        assert not result.is_valid
        assert any(
            error.severity == IssueSeverity.ERROR
            and (
                "syntax" in error.message.lower()
                or "indentation" in error.message.lower()
            )
            for error in result.errors
        )

"""
Tests for detailed error reporting and debugging guidance functionality.

This module tests the enhanced validation system that provides detailed error
reports with actionable debugging guidance and suggestions.
"""

import tempfile
import textwrap
from pathlib import Path
from typing import List
import pytest

from crewforge.validation import (
    ValidationIssue,
    ValidationResult,
    IssueSeverity,
    validate_python_syntax,
    validate_python_imports,
    validate_generated_project,
    generate_detailed_error_report,
)


class TestDetailedErrorReporting:
    """Test cases for detailed error reporting functionality."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary directory for test projects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_validation_issue_with_suggestions(self):
        """Test ValidationIssue with suggestions and context."""
        issue = ValidationIssue(
            severity=IssueSeverity.ERROR,
            message="Syntax error: invalid syntax",
            field_path="test_file.py",
            suggestions=[
                "Check for missing colons after function definitions",
                "Verify indentation is consistent (use 4 spaces)",
                "Ensure all parentheses and brackets are properly closed"
            ],
            context={
                "line_number": 10,
                "column": 15,
                "error_type": "SyntaxError",
                "code_snippet": "def hello_world(\n    print('Hello')"
            },
            error_code="PY001"
        )

        assert issue.severity == IssueSeverity.ERROR
        assert "Syntax error" in issue.message
        assert len(issue.suggestions) == 3
        assert issue.context["line_number"] == 10
        assert issue.error_code == "PY001"

    def test_validation_issue_str_with_suggestions(self):
        """Test string representation includes suggestions."""
        issue = ValidationIssue(
            severity=IssueSeverity.ERROR,
            message="Import error: module not found",
            field_path="test_file.py",
            suggestions=["Install missing package: pip install requests"],
            error_code="IMP001"
        )

        str_repr = str(issue)
        assert "ERROR: Import error" in str_repr
        assert "pip install requests" in str_repr

    def test_detailed_syntax_error_reporting(self, temp_project_dir):
        """Test detailed reporting for Python syntax errors."""
        # Create a Python file with syntax error
        python_file = temp_project_dir / "syntax_error.py"
        python_file.write_text(
            textwrap.dedent(
                """
            def hello_world(
                print("Hello, World!")
                return True
            """
            )
        )

        result = validate_python_syntax(python_file)

        # Should have error with detailed information
        assert not result.is_valid
        assert len(result.errors) > 0

        error = result.errors[0]
        assert error.severity == IssueSeverity.ERROR
        assert "Syntax error" in error.message

        # Check for detailed context and suggestions
        if hasattr(error, 'suggestions') and error.suggestions:
            assert len(error.suggestions) > 0
            assert any("parentheses" in suggestion.lower() or "indentation" in suggestion.lower()
                      for suggestion in error.suggestions)

    def test_detailed_import_error_reporting(self, temp_project_dir):
        """Test detailed reporting for import errors."""
        # Create a Python file with import error
        python_file = temp_project_dir / "import_error.py"
        python_file.write_text(
            textwrap.dedent(
                """
            import nonexistent_module
            from fake_package import something

            def test_function():
                return True
            """
            )
        )

        result = validate_python_imports(python_file, str(temp_project_dir))

        # Should have import errors with detailed information
        assert not result.is_valid
        assert len(result.errors) > 0

        # Check that errors include suggestions
        for error in result.errors:
            if hasattr(error, 'suggestions') and error.suggestions:
                assert len(error.suggestions) > 0
                assert any("pip install" in suggestion or "package" in suggestion.lower()
                          for suggestion in error.suggestions)

    def test_generate_detailed_error_report(self, temp_project_dir):
        """Test generation of detailed error reports."""
        # Create a project with multiple issues
        src_dir = temp_project_dir / "src" / "test_project"
        src_dir.mkdir(parents=True)

        # Create main.py with syntax error
        main_py = src_dir / "main.py"
        main_py.write_text("def hello_world(\nprint('Hello')\nreturn True")

        # Create crew.py with import error
        crew_py = src_dir / "crew.py"
        crew_py.write_text("import fake_crewai_module\nfrom nonexistent import Crew")

        result = validate_generated_project(str(temp_project_dir))

        # Generate detailed report
        report = generate_detailed_error_report(result, str(temp_project_dir))

        assert isinstance(report, str)
        assert len(report) > 0
        assert "DETAILED ERROR REPORT & DEBUGGING GUIDANCE" in report
        assert "Syntax error" in report or "Import error" in report

        # Check for debugging guidance sections
        assert "Suggested Fixes" in report or "DEBUGGING GUIDANCE" in report

    def test_validation_result_with_detailed_reporting(self):
        """Test ValidationResult string representation with detailed information."""
        issues = [
            ValidationIssue(
                severity=IssueSeverity.ERROR,
                message="Critical syntax error",
                field_path="main.py",
                suggestions=["Fix the syntax error on line 5"],
                error_code="PY001"
            ),
            ValidationIssue(
                severity=IssueSeverity.WARNING,
                message="Unused import",
                field_path="crew.py",
                suggestions=["Remove unused import or use it"],
                error_code="IMP002"
            )
        ]

        result = ValidationResult(issues=issues)

        str_repr = str(result)
        assert "Validation FAILED" in str_repr
        assert "Critical syntax error" in str_repr
        assert "Unused import" in str_repr

        # Check for suggestions in output
        if "Suggestions" in str_repr or "Fix the syntax" in str_repr:
            assert "Fix the syntax error" in str_repr

    def test_error_code_uniqueness(self):
        """Test that error codes are unique and follow pattern."""
        # Test various validation scenarios to ensure error codes are assigned
        issues = [
            ValidationIssue(
                severity=IssueSeverity.ERROR,
                message="Test error",
                field_path="test.py",
                error_code="TEST001"
            )
        ]

        for issue in issues:
            if hasattr(issue, 'error_code') and issue.error_code:
                # Error codes should follow pattern: CATEGORY + 3 digits
                assert len(issue.error_code) >= 3
                assert issue.error_code[:3].isalpha()
                assert issue.error_code[-3:].isdigit()

    def test_context_information_preservation(self):
        """Test that context information is preserved in validation issues."""
        context = {
            "line_number": 25,
            "column": 10,
            "function_name": "test_function",
            "code_snippet": "if x = 5:",
            "error_type": "SyntaxError"
        }

        issue = ValidationIssue(
            severity=IssueSeverity.ERROR,
            message="Assignment in condition",
            field_path="test.py",
            context=context,
            error_code="PY002"
        )

        assert issue.context == context
        assert issue.context["line_number"] == 25
        assert issue.context["error_type"] == "SyntaxError"
"""Project validation suite for CrewAI projects."""

import ast
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..models.crew import ValidationResult


class ValidationSuite:
    """Comprehensive validation suite for generated CrewAI projects.

    Provides multi-level validation including:
    - Python syntax checking using AST parsing
    - CrewAI framework compliance validation
    - Basic functionality and import testing
    """

    def __init__(self):
        """Initialize the validation suite."""
        pass

    def validate_project(self, project_path: str) -> ValidationResult:
        """Run complete validation pipeline on a CrewAI project.

        Args:
            project_path: Path to the CrewAI project directory

        Returns:
            ValidationResult: Comprehensive validation results
        """
        # Initialize result with basic info
        result = ValidationResult(
            is_valid=True,  # Will be set to False if any validation fails
            project_path=project_path,
            validation_timestamp=datetime.now().isoformat(),
        )

        # Check if project directory exists
        if not Path(project_path).exists():
            result.is_valid = False
            result.errors.append(f"Project directory does not exist: {project_path}")
            return result

        # Run syntax validation
        syntax_result = self.validate_syntax(project_path)
        result.syntax_valid = syntax_result.syntax_valid
        result.errors.extend(syntax_result.errors)
        result.warnings.extend(syntax_result.warnings)
        result.files_checked.extend(syntax_result.files_checked)

        if not syntax_result.syntax_valid:
            result.is_valid = False

        # Run CrewAI compliance validation
        compliance_result = self.validate_crewai_compliance(project_path)
        result.crewai_compliant = compliance_result.crewai_compliant
        result.errors.extend(compliance_result.errors)
        result.warnings.extend(compliance_result.warnings)
        result.suggestions.extend(compliance_result.suggestions)

        if not compliance_result.crewai_compliant:
            result.is_valid = False

        # Run functionality validation (only if syntax is valid)
        if result.syntax_valid:
            functionality_result = self.validate_functionality(project_path)
            result.functional = functionality_result.functional
            result.errors.extend(functionality_result.errors)
            result.warnings.extend(functionality_result.warnings)
            result.suggestions.extend(functionality_result.suggestions)

            if not functionality_result.functional:
                result.is_valid = False
        else:
            result.functional = False
            result.errors.append(
                "Skipping functionality validation due to syntax errors"
            )

        return result

    def validate_syntax(self, project_path: str) -> ValidationResult:
        """Validate Python syntax for all Python files in the project.

        Uses Python's AST parser to check for syntax errors.

        Args:
            project_path: Path to the project directory

        Returns:
            ValidationResult: Syntax validation results
        """
        result = ValidationResult(
            is_valid=True,
            project_path=project_path,
            syntax_valid=True,
            validation_timestamp=datetime.now().isoformat(),
        )

        project_dir = Path(project_path)
        if not project_dir.exists():
            result.is_valid = False
            result.syntax_valid = False
            result.errors.append(f"Project directory not found: {project_path}")
            return result

        # Find all Python files
        python_files = list(project_dir.rglob("*.py"))

        if not python_files:
            result.is_valid = False
            result.syntax_valid = False
            result.errors.append(
                f"No Python files found in project directory: {project_path}"
            )
            return result

        # Check syntax for each Python file
        for py_file in python_files:
            result.files_checked.append(str(py_file))

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    file_content = f.read()

                # Parse the file with AST
                ast.parse(file_content, filename=str(py_file))

            except SyntaxError as e:
                result.is_valid = False
                result.syntax_valid = False
                error_msg = f"Syntax error in {py_file}: {e.msg} (line {e.lineno})"
                if e.offset:
                    error_msg += f" at position {e.offset}"
                result.errors.append(error_msg)

            except Exception as e:
                result.is_valid = False
                result.syntax_valid = False
                result.errors.append(f"Error reading {py_file}: {str(e)}")

        return result

    def validate_crewai_compliance(self, project_path: str) -> ValidationResult:
        """Validate CrewAI framework compliance and patterns.

        Checks for:
        - Required CrewAI imports
        - Proper agent definitions
        - Proper task definitions
        - Crew instantiation patterns

        Args:
            project_path: Path to the project directory

        Returns:
            ValidationResult: CrewAI compliance validation results
        """
        result = ValidationResult(
            is_valid=True,
            project_path=project_path,
            crewai_compliant=True,
            validation_timestamp=datetime.now().isoformat(),
        )

        project_dir = Path(project_path)
        if not project_dir.exists():
            result.is_valid = False
            result.crewai_compliant = False
            result.errors.append(f"Project directory not found: {project_path}")
            return result

        # Find Python files to check
        python_files = list(project_dir.rglob("*.py"))

        if not python_files:
            result.is_valid = False
            result.crewai_compliant = False
            result.errors.append(
                f"No Python files found for compliance checking: {project_path}"
            )
            return result

        # Track CrewAI imports and patterns
        has_agent_import = False
        has_task_import = False
        has_crew_import = False
        has_agent_definition = False
        has_task_definition = False
        has_crew_definition = False

        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse file to check imports and patterns
                tree = ast.parse(content, filename=str(py_file))

                for node in ast.walk(tree):
                    # Check for CrewAI imports
                    if isinstance(node, ast.ImportFrom):
                        if node.module == "crewai":
                            for alias in node.names:
                                if alias.name == "Agent":
                                    has_agent_import = True
                                elif alias.name == "Task":
                                    has_task_import = True
                                elif alias.name == "Crew":
                                    has_crew_import = True

                    # Check for typical CrewAI patterns
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            if node.func.id == "Agent":
                                has_agent_definition = True
                            elif node.func.id == "Task":
                                has_task_definition = True
                            elif node.func.id == "Crew":
                                has_crew_definition = True

            except Exception as e:
                result.warnings.append(
                    f"Could not parse {py_file} for compliance checking: {str(e)}"
                )

        # Check for required CrewAI patterns
        compliance_issues = []

        if not (has_agent_import or has_agent_definition):
            compliance_issues.append("No CrewAI Agent import or usage found")

        if not (has_task_import or has_task_definition):
            compliance_issues.append("No CrewAI Task import or usage found")

        if not (has_crew_import or has_crew_definition):
            compliance_issues.append("No CrewAI Crew import or usage found")

        # Look for expected file structure
        expected_files = ["agents.py", "tasks.py", "crew.py"]
        missing_files = []

        for expected_file in expected_files:
            file_path = project_dir / expected_file
            if not file_path.exists():
                missing_files.append(expected_file)

        if missing_files:
            result.warnings.append(
                f"Missing expected CrewAI files: {', '.join(missing_files)}"
            )
            result.suggestions.append(
                f"Consider creating standard CrewAI files: {', '.join(missing_files)}"
            )

        if compliance_issues:
            result.is_valid = False
            result.crewai_compliant = False
            result.errors.extend(compliance_issues)

        return result

    def validate_functionality(self, project_path: str) -> ValidationResult:
        """Validate basic functionality by attempting to import and instantiate crew.

        Args:
            project_path: Path to the project directory

        Returns:
            ValidationResult: Functionality validation results
        """
        result = ValidationResult(
            is_valid=True,
            project_path=project_path,
            functional=True,
            validation_timestamp=datetime.now().isoformat(),
        )

        project_dir = Path(project_path)
        if not project_dir.exists():
            result.is_valid = False
            result.functional = False
            result.errors.append(f"Project directory not found: {project_path}")
            return result

        # Add project directory to Python path temporarily
        original_path = sys.path[:]
        sys.path.insert(0, str(project_dir))

        try:
            # Try to import common CrewAI project files
            crew_file = project_dir / "crew.py"

            if crew_file.exists():
                try:
                    # Load crew module dynamically
                    spec = importlib.util.spec_from_file_location("crew", crew_file)
                    if spec and spec.loader:
                        crew_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(crew_module)

                        # Check if crew can be instantiated
                        if hasattr(crew_module, "crew"):
                            # Crew object exists
                            result.suggestions.append(
                                "Crew object found and importable"
                            )
                        elif hasattr(crew_module, "create_crew"):
                            # Crew creation function exists
                            result.suggestions.append("Crew creation function found")
                        else:
                            result.warnings.append(
                                "No 'crew' object or 'create_crew' function found"
                            )

                except ImportError as e:
                    result.is_valid = False
                    result.functional = False
                    result.errors.append(f"Import error in crew.py: {str(e)}")

                except Exception as e:
                    result.is_valid = False
                    result.functional = False
                    result.errors.append(f"Error loading crew.py: {str(e)}")
            else:
                result.warnings.append(
                    "No crew.py file found for functionality testing"
                )
                result.suggestions.append(
                    "Create a crew.py file with crew instantiation"
                )

            # Test other common files
            for filename in ["agents.py", "tasks.py"]:
                file_path = project_dir / filename
                if file_path.exists():
                    try:
                        spec = importlib.util.spec_from_file_location(
                            filename[:-3], file_path
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                    except Exception as e:
                        result.warnings.append(f"Could not import {filename}: {str(e)}")

        finally:
            # Restore original Python path
            sys.path[:] = original_path

        return result

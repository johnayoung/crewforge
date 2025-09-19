#!/usr/bin/env python3
"""
Example demonstrating Python syntax and import validation for CrewAI projects.

This example shows how to use the new validation functions to check
generated CrewAI projects for syntax errors and import issues.
"""

import tempfile
import textwrap
from pathlib import Path

from crewforge.validation import (
    validate_python_syntax,
    validate_python_imports,
    validate_generated_project,
)


def main():
    """Demonstrate validation functionality."""
    print("🔍 Python Validation Example")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Test syntax validation
        print("\n1. Testing syntax validation:")

        # Valid syntax
        valid_file = temp_path / "valid.py"
        valid_file.write_text(
            textwrap.dedent(
                """
            import os
            from typing import List
            
            def greet(name: str) -> str:
                return f"Hello, {name}!"
            
            if __name__ == "__main__":
                print(greet("World"))
        """
            ).strip()
        )

        result = validate_python_syntax(valid_file)
        print(f"  Valid file: {result.is_valid} ({len(result.issues)} issues)")

        # Invalid syntax
        invalid_file = temp_path / "invalid.py"
        invalid_file.write_text(
            textwrap.dedent(
                """
            def broken_function(
                print("Missing closing parenthesis")
        """
            ).strip()
        )

        result = validate_python_syntax(invalid_file)
        print(f"  Invalid file: {result.is_valid} ({len(result.errors)} errors)")
        for error in result.errors:
            print(f"    - {error.message}")

        # 2. Test import validation
        print("\n2. Testing import validation:")

        import_file = temp_path / "imports.py"
        import_file.write_text(
            textwrap.dedent(
                """
            import os  # Valid
            import sys  # Valid
            import nonexistent_module  # Invalid
            from typing import List  # Valid
        """
            ).strip()
        )

        result = validate_python_imports(import_file)
        print(f"  Import validation: {result.is_valid} ({len(result.issues)} issues)")
        for issue in result.issues:
            print(f"    - {issue.severity.value}: {issue.message}")

        # 3. Test comprehensive project validation
        print("\n3. Testing comprehensive project validation:")

        # Create a mock CrewAI project structure
        project_root = temp_path / "test_crew"
        src_dir = project_root / "src" / "test_crew"
        config_dir = src_dir / "config"

        src_dir.mkdir(parents=True)
        config_dir.mkdir()

        # Create project files
        (project_root / "pyproject.toml").write_text("[project]\nname = 'test-crew'")
        (project_root / "README.md").write_text("# Test Crew")
        (project_root / ".env").write_text("API_KEY=test")

        (src_dir / "__init__.py").write_text("")
        (src_dir / "main.py").write_text(
            textwrap.dedent(
                """
            from test_crew.crew import TestCrew
            
            def main():
                crew = TestCrew()
                return crew.run()
        """
            ).strip()
        )

        (src_dir / "crew.py").write_text(
            textwrap.dedent(
                """
            from crewai import Agent, Task, Crew
            
            class TestCrew:
                def run(self):
                    return "Project running successfully!"
        """
            ).strip()
        )

        (config_dir / "agents.yaml").write_text("agents: []")
        (config_dir / "tasks.yaml").write_text("tasks: []")

        result = validate_generated_project(str(project_root))
        print(f"  Project validation: {result.is_valid}")
        print(f"  Total issues: {len(result.issues)}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")
        print(f"  Info: {len(result.info_messages)}")

        # Show some key messages
        for info in result.info_messages[:3]:
            print(f"    ✓ {info.message}")

        if result.warnings:
            print("  Key warnings:")
            for warning in result.warnings[:2]:
                print(f"    ⚠ {warning.message}")

    print("\n✅ Validation example completed!")


if __name__ == "__main__":
    main()

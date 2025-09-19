#!/usr/bin/env python3
"""Debug script to check what error message is generated."""

import tempfile
import textwrap
from pathlib import Path

from crewforge.validation import validate_python_syntax

# Create a temporary file with syntax error
with tempfile.TemporaryDirectory() as temp_dir:
    temp_project_dir = Path(temp_dir)
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

    result = validate_python_syntax(python_file)

    print("Is valid:", result.is_valid)
    print("Errors:")
    for error in result.errors:
        print(f"  - {error.message}")
    print("Warnings:")
    for warning in result.warnings:
        print(f"  - {warning.message}")
    print("Info:")
    for info in result.info_messages:
        print(f"  - {info.message}")

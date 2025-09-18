"""
Test the basic project structure and CLI entry point.
"""

import os
import sys
import subprocess
from pathlib import Path
import pytest

# Add src to path so we can import crewforge
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_package_structure_exists():
    """Test that the basic package structure exists."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    package_dir = src_dir / "crewforge"

    assert src_dir.exists(), "src directory should exist"
    assert package_dir.exists(), "crewforge package directory should exist"
    assert (
        package_dir / "__init__.py"
    ).exists(), "crewforge package should have __init__.py"


def test_cli_module_exists():
    """Test that the CLI module exists and can be imported."""
    try:
        import crewforge.cli

        assert hasattr(crewforge.cli, "main"), "CLI module should have a main function"
    except ImportError:
        pytest.fail("Could not import crewforge.cli module")


def test_package_can_be_imported():
    """Test that the main package can be imported."""
    try:
        import crewforge

        assert crewforge.__version__ is not None, "Package should have a version"
    except ImportError:
        pytest.fail("Could not import crewforge package")


def test_cli_entry_point_exists():
    """Test that the CLI can be executed via python -m."""
    project_root = Path(__file__).parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "crewforge", "--help"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    # Should not return error code and should contain help text
    assert result.returncode == 0, f"CLI execution failed: {result.stderr}"
    assert "crewforge" in result.stdout.lower(), "Help text should mention crewforge"


def test_pyproject_toml_exists():
    """Test that pyproject.toml exists with proper configuration."""
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"

    assert pyproject_path.exists(), "pyproject.toml should exist"

    # Read and validate basic content
    content = pyproject_path.read_text()
    assert "[tool.uv]" in content, "Should use uv for package management"
    assert 'name = "crewforge"' in content, "Should have correct package name"
    assert (
        "click" in content or "typer" in content
    ), "Should have CLI framework dependency"

"""Test package structure and basic imports for CrewForge."""

import sys
from pathlib import Path

import pytest


class TestPackageStructure:
    """Test that the package structure matches SPEC.md requirements."""

    def test_package_directory_structure(self):
        """Test that all required directories exist."""
        project_root = Path(__file__).parent.parent

        # Required directories from SPEC.md
        expected_dirs = [
            "src/crewforge",
            "src/crewforge/cli",
            "src/crewforge/core",
            "src/crewforge/models",
            "src/crewforge/templates",
            "src/crewforge/storage",
            "tests",
            "tests/fixtures",
            "examples/generated_crews",
        ]

        for dir_path in expected_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Required directory {dir_path} does not exist"
            assert full_path.is_dir(), f"{dir_path} exists but is not a directory"

    def test_init_files_exist(self):
        """Test that all required __init__.py files exist."""
        project_root = Path(__file__).parent.parent

        # Required __init__.py files for proper package structure
        expected_inits = [
            "src/crewforge/__init__.py",
            "src/crewforge/cli/__init__.py",
            "src/crewforge/core/__init__.py",
            "src/crewforge/models/__init__.py",
            "src/crewforge/storage/__init__.py",
            "tests/__init__.py",
        ]

        for init_path in expected_inits:
            full_path = project_root / init_path
            assert (
                full_path.exists()
            ), f"Required __init__.py file {init_path} does not exist"
            assert full_path.is_file(), f"{init_path} exists but is not a file"

    def test_package_metadata(self):
        """Test that package has proper metadata."""
        # Add src to path to allow imports
        project_root = Path(__file__).parent.parent
        src_path = str(project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        import crewforge

        # Test required metadata attributes
        assert hasattr(crewforge, "__version__")
        assert hasattr(crewforge, "__author__")
        assert hasattr(crewforge, "__description__")

        # Test values are not empty
        assert crewforge.__version__ != ""
        assert crewforge.__author__ != ""
        assert crewforge.__description__ != ""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists and contains required fields."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        assert pyproject_path.exists(), "pyproject.toml does not exist"
        assert pyproject_path.is_file(), "pyproject.toml exists but is not a file"

        # Read and parse basic structure
        content = pyproject_path.read_text()

        # Test required sections exist
        assert "[project]" in content, "Missing [project] section in pyproject.toml"
        assert (
            "[build-system]" in content
        ), "Missing [build-system] section in pyproject.toml"
        assert (
            "[tool.setuptools]" in content
        ), "Missing [tool.setuptools] section in pyproject.toml"

        # Test Python version requirement
        assert 'requires-python = ">=' in content, "Missing Python version requirement"
        assert "3.11" in content, "Python 3.11+ requirement not found"

        # Test entry point for CLI
        assert "crewforge" in content, "CLI entry point not found"

        # Test required dependencies
        required_deps = ["click", "litellm", "pydantic", "jinja2", "pyyaml", "crewai"]
        for dep in required_deps:
            assert dep in content, f"Required dependency {dep} not found"

        # Test dev dependencies
        dev_deps = ["pytest", "mypy", "ruff"]
        for dep in dev_deps:
            assert dep in content, f"Required dev dependency {dep} not found"


class TestImportFunctionality:
    """Test that package modules can be imported properly."""

    def test_base_package_import(self):
        """Test that base crewforge package can be imported."""
        project_root = Path(__file__).parent.parent
        src_path = str(project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            import crewforge

            assert crewforge is not None
        except ImportError as e:
            pytest.fail(f"Failed to import crewforge package: {e}")

    def test_submodule_imports(self):
        """Test that submodules can be imported without errors."""
        project_root = Path(__file__).parent.parent
        src_path = str(project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Test submodule imports
        modules_to_test = [
            "crewforge.cli",
            "crewforge.core",
            "crewforge.models",
            "crewforge.storage",
        ]

        for module in modules_to_test:
            try:
                __import__(module)
            except ImportError as e:
                pytest.fail(f"Failed to import {module}: {e}")

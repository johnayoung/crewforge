import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from crewforge.executor import ProjectExecutor


class TestProjectExecutor:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_dir = Path(self.temp_dir) / "test_project"
        self.project_dir.mkdir()
        # Create minimal project structure
        (self.project_dir / "pyproject.toml").write_text(
            """
[project]
name = "test_project"
version = "0.1.0"
dependencies = ["crewai"]
"""
        )
        (self.project_dir / "src").mkdir()
        (self.project_dir / "src" / "test_project").mkdir()
        (self.project_dir / "src" / "test_project" / "__init__.py").write_text("")
        (self.project_dir / "src" / "test_project" / "main.py").write_text(
            """
def main():
    print("Hello from test project")
if __name__ == "__main__":
    main()
"""
        )

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("subprocess.run")
    def test_create_isolated_env(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(returncode=0)
        executor = ProjectExecutor(self.project_dir)
        result = executor.create_isolated_env()
        assert result is True
        mock_subprocess.assert_called_with(
            ["uv", "venv", str(self.project_dir / ".venv")],
            cwd=self.project_dir,
            check=True,
            capture_output=True,
            text=True,
        )

    @patch("subprocess.run")
    def test_install_dependencies(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(returncode=0)
        executor = ProjectExecutor(self.project_dir)
        result = executor.install_dependencies()
        assert result is True
        mock_subprocess.assert_called_with(
            ["uv", "pip", "install", "-e", "."],
            cwd=self.project_dir,
            check=True,
            capture_output=True,
            text=True,
        )

    @patch("subprocess.run")
    def test_execute_project_success(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(
            returncode=0, stdout="Hello from test project\n", stderr=""
        )
        executor = ProjectExecutor(self.project_dir)
        result = executor.execute_project()
        assert result["success"] is True
        assert "Hello from test project" in result["output"]
        assert result["execution_time"] > 0
        mock_subprocess.assert_called_with(
            ["uv", "run", "python", "-m", "test_project"],
            cwd=self.project_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )

    @patch("subprocess.run")
    def test_execute_project_failure(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(
            returncode=1, stdout="", stderr="Error: Module not found\n"
        )
        executor = ProjectExecutor(self.project_dir)
        result = executor.execute_project()
        assert result["success"] is False
        assert "Error: Module not found" in result["error"]
        assert result["execution_time"] > 0

    @patch("subprocess.run")
    def test_execute_project_timeout(self, mock_subprocess):
        from subprocess import TimeoutExpired

        mock_subprocess.side_effect = TimeoutExpired(["uv", "run"], 300)
        executor = ProjectExecutor(self.project_dir)
        result = executor.execute_project()
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

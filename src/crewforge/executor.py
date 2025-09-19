import subprocess
import time
from pathlib import Path
from typing import Dict, Optional


class ProjectExecutor:
    """Executes generated CrewAI projects in isolated environments using uv."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.project_name = project_dir.name
        self.venv_dir = project_dir / ".venv"

    def create_isolated_env(self) -> bool:
        """Create isolated virtual environment using uv."""
        try:
            result = subprocess.run(
                ["uv", "venv", str(self.venv_dir)],
                cwd=self.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

    def install_dependencies(self) -> bool:
        """Install project dependencies in the isolated environment."""
        try:
            result = subprocess.run(
                ["uv", "pip", "install", "-e", "."],
                cwd=self.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

    def execute_project(self, params: Optional[Dict] = None) -> Dict:
        """Execute the project and return results with metrics."""
        start_time = time.time()
        try:
            result = subprocess.run(
                ["uv", "run", "python", "-m", self.project_name],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )
            execution_time = time.time() - start_time
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "execution_time": execution_time,
            }
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "output": "",
                "error": "Execution timed out after 300 seconds",
                "execution_time": execution_time,
            }
        except subprocess.CalledProcessError as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "output": e.stdout or "",
                "error": e.stderr or str(e),
                "execution_time": execution_time,
            }

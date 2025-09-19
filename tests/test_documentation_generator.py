import pytest
import tempfile
from pathlib import Path
from crewforge.documentation_generator import DocumentationGenerator


class TestDocumentationGenerator:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_dir = Path(self.temp_dir) / "test_project"
        self.project_dir.mkdir()
        self.templates_dir = Path(self.temp_dir) / "templates"
        self.templates_dir.mkdir()
        # Create a simple README template
        (self.templates_dir / "docs").mkdir()
        (self.templates_dir / "docs" / "README.md.jinja").write_text(
            """
# {{ project_name }}

{{ description }}

## Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage
{{ usage_instructions }}
"""
        )

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_generate_readme(self):
        generator = DocumentationGenerator(self.project_dir, self.templates_dir)
        project_info = {
            "project_name": "Test Project",
            "description": "A test CrewAI project",
            "usage_instructions": "Run python -m test_project",
        }
        result = generator.generate_readme(project_info)
        assert result is True
        readme_path = self.project_dir / "README.md"
        assert readme_path.exists()
        content = readme_path.read_text()
        assert "# Test Project" in content
        assert "A test CrewAI project" in content
        assert "Run python -m test_project" in content

    def test_generate_readme_missing_template(self):
        # Remove template
        (self.templates_dir / "docs" / "README.md.jinja").unlink()
        generator = DocumentationGenerator(self.project_dir, self.templates_dir)
        project_info = {"project_name": "Test"}
        result = generator.generate_readme(project_info)
        assert result is False

    def test_generate_readme_invalid_template(self):
        # Create invalid template
        (self.templates_dir / "docs" / "README.md.jinja").write_text(
            "{{ invalid_var }}"
        )
        generator = DocumentationGenerator(self.project_dir, self.templates_dir)
        project_info = {"project_name": "Test"}
        result = generator.generate_readme(project_info)
        assert result is False

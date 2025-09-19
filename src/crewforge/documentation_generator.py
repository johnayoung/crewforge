from pathlib import Path
from typing import Dict, Optional
from jinja2 import Environment, FileSystemLoader, TemplateError, StrictUndefined


class DocumentationGenerator:
    """Generates documentation for created CrewAI projects using Jinja2 templates."""

    def __init__(self, project_dir: Path, templates_dir: Path):
        self.project_dir = project_dir
        self.templates_dir = templates_dir
        self.env = Environment(
            loader=FileSystemLoader(templates_dir), undefined=StrictUndefined
        )

    def generate_readme(self, project_info: Dict) -> bool:
        """Generate README.md from template."""
        try:
            template = self.env.get_template("docs/README.md.jinja")
            content = template.render(**project_info)
            readme_path = self.project_dir / "README.md"
            readme_path.write_text(content)
            return True
        except (TemplateError, FileNotFoundError):
            return False

    def generate_documentation(self, project_info: Dict) -> Dict[str, bool]:
        """Generate all documentation files."""
        results = {}
        results["readme"] = self.generate_readme(project_info)
        # Could add more docs here like API docs, etc.
        return results

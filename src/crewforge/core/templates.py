"""Jinja2 template engine for generating CrewAI project files.

This module provides the TemplateEngine class that handles loading, rendering,
and populating Jinja2 templates for CrewAI project generation.
"""

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import (
    Environment,
    FileSystemLoader,
    Template,
    TemplateNotFound,
    TemplateError as Jinja2TemplateError,
)


class TemplateError(Exception):
    """Exception raised for template-related errors."""

    pass


class TemplateEngine:
    """Jinja2-based template engine for generating CrewAI project files.

    This class provides methods to load, render, and populate Jinja2 templates
    with CrewAI configuration data to generate functional Python files.
    """

    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize the template engine.

        Args:
            template_dir: Path to the templates directory. If None, uses the
                         default templates directory within the package.
        """
        if template_dir is None:
            # Default to templates directory within the package
            self.template_dir = Path(__file__).parent.parent / "templates"
        else:
            self.template_dir = Path(template_dir)

        # Ensure template directory exists
        if not self.template_dir.exists():
            raise TemplateError(f"Template directory not found: {self.template_dir}")

        # Initialize Jinja2 environment with custom filters
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

        # Add custom filters for template processing
        self.env.filters["to_python_identifier"] = self._to_python_identifier
        self.env.filters["to_class_name"] = self._to_class_name
        self.env.filters["truncate_identifier"] = self._truncate_identifier

    def _to_python_identifier(self, text: str, prefix: str = "") -> str:
        """Convert any string to a valid Python identifier.

        Args:
            text: Input text to convert
            prefix: Optional prefix to add if the identifier starts with a number

        Returns:
            Valid Python identifier

        Examples:
            "Content Research Expert!" -> "content_research_expert"
            "API-Integration Manager" -> "api_integration_manager"
            "123 Numbers First" -> "numbers_first" (or "prefix_123_numbers_first" with prefix)
        """
        if not text:
            return prefix or "unnamed"

        # Convert to lowercase and normalize unicode
        text = unicodedata.normalize("NFKD", text.lower())

        # Replace common separators with underscores
        text = text.replace("-", "_").replace(" ", "_").replace(".", "_")

        # Remove all non-alphanumeric characters except underscores
        text = "".join(c if c.isalnum() or c == "_" else "" for c in text)

        # Remove consecutive underscores
        text = re.sub(r"_+", "_", text)

        # Remove leading/trailing underscores
        text = text.strip("_")

        # Handle edge cases
        if not text:
            return prefix or "unnamed"

        # Ensure it doesn't start with a number
        if text[0].isdigit():
            text = f"{prefix}_{text}" if prefix else f"item_{text}"

        # Ensure it's not a Python keyword
        import keyword

        if keyword.iskeyword(text):
            text = f"{text}_var"

        return text

    def _to_class_name(self, text: str) -> str:
        """Convert string to PascalCase class name.

        Args:
            text: Input text to convert

        Returns:
            PascalCase class name

        Examples:
            "content-research-crew" -> "ContentResearchCrew"
            "my_project_name" -> "MyProjectName"
        """
        if not text:
            return "UnnamedClass"

        # First convert to identifier
        identifier = self._to_python_identifier(text)

        # Split by underscores and capitalize each part
        parts = identifier.split("_")
        return "".join(word.capitalize() for word in parts if word)

    def _truncate_identifier(self, text: str, max_length: int = 30) -> str:
        """Truncate an identifier to a maximum length while keeping it valid.

        Args:
            text: Identifier to truncate
            max_length: Maximum length (default 30)

        Returns:
            Truncated identifier
        """
        if len(text) <= max_length:
            return text

        # Truncate and ensure we don't end with an underscore
        truncated = text[:max_length].rstrip("_")

        # If we truncated to nothing, return a default
        return truncated if truncated else "truncated"

    def get_template(self, template_name: str) -> Template:
        """Load and return a Jinja2 template by name.

        Args:
            template_name: Name of the template file (e.g., 'agents.py.j2')

        Returns:
            Loaded Jinja2 template object

        Raises:
            TemplateError: If template is not found or cannot be loaded
        """
        try:
            return self.env.get_template(template_name)
        except TemplateNotFound as e:
            raise TemplateError(f"Template not found: {template_name}") from e
        except Exception as e:
            raise TemplateError(
                f"Failed to load template {template_name}: {str(e)}"
            ) from e

    def render_template(self, template_name: str, **context: Any) -> str:
        """Render a template with the provided context.

        Args:
            template_name: Name of the template file to render
            **context: Template context variables

        Returns:
            Rendered template as a string

        Raises:
            TemplateError: If template rendering fails
        """
        try:
            template = self.get_template(template_name)
            return template.render(**context)
        except TemplateError:
            # Re-raise TemplateError as-is
            raise
        except Jinja2TemplateError as e:
            raise TemplateError(
                f"Template rendering failed for {template_name}: {str(e)}"
            ) from e
        except Exception as e:
            raise TemplateError(
                f"Unexpected error rendering {template_name}: {str(e)}"
            ) from e

    def populate_template(
        self, template_name: str, output_path: Path, **context: Any
    ) -> None:
        """Render a template and write it to a file.

        Args:
            template_name: Name of the template file to render
            output_path: Path where the rendered content should be written
            **context: Template context variables

        Raises:
            TemplateError: If template rendering or file writing fails
        """
        try:
            # Render the template
            content = self.render_template(template_name, **context)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the rendered content to file
            output_path.write_text(content, encoding="utf-8")

        except TemplateError:
            # Re-raise TemplateError as-is
            raise
        except (OSError, IOError) as e:
            raise TemplateError(
                f"Failed to write template to {output_path}: {str(e)}"
            ) from e
        except Exception as e:
            raise TemplateError(
                f"Unexpected error populating template {template_name}: {str(e)}"
            ) from e

    def validate_template(self, template_name: str) -> bool:
        """Validate that a template exists and can be loaded.

        Args:
            template_name: Name of the template file to validate

        Returns:
            True if template is valid, False otherwise
        """
        try:
            self.get_template(template_name)
            return True
        except TemplateError:
            return False

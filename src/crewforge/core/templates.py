"""Jinja2 template engine for generating CrewAI project files.

This module provides the TemplateEngine class that handles loading, rendering,
and populating Jinja2 templates for CrewAI project generation.
"""

import re
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
        self.env.filters["regex_replace"] = self._regex_replace_filter

    def _regex_replace_filter(
        self, text: str, pattern: str, replacement: str = ""
    ) -> str:
        """Custom Jinja2 filter for regex-based string replacement.

        Args:
            text: Input text to process
            pattern: Regular expression pattern to match
            replacement: Replacement string

        Returns:
            Text with regex replacements applied
        """
        return re.sub(pattern, replacement, text)

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

"""Main CLI entry point for CrewForge."""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
from pydantic import ValidationError

from crewforge.models import GenerationRequest
from crewforge.core.progress import (
    ProgressTracker,
    ProgressStep,
    ProgressEvent,
    ProgressStatus,
    StreamingCallbacks,
    get_standard_generation_steps,
)
from crewforge.core.scaffolding import (
    ProjectScaffolder,
    ScaffoldingError,
    CrewAICommandError,
    FileSystemError,
    ProjectStructureError,
)
from crewforge.core.llm import (
    LLMError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMNetworkError,
)


@click.group()
@click.version_option()
def cli():
    """CrewForge - Intelligent CrewAI Project Generator

    Generate complete CrewAI projects from natural language prompts using agentic AI.
    """
    pass


def create_progress_callback():
    """Create a callback function for progress tracking display."""

    def progress_callback(event: ProgressEvent) -> None:
        """Handle progress events and display real-time feedback."""
        if event.status == ProgressStatus.IN_PROGRESS:
            click.echo(
                f"⏳ Step {event.step_id}: {event.description}... ({event.progress_percentage:.1f}% complete)"
            )
        elif event.status == ProgressStatus.COMPLETED:
            click.echo(
                f"✅ {event.description} - Complete ({event.progress_percentage:.1f}%)"
            )
        elif event.status == ProgressStatus.FAILED:
            click.echo(f"❌ {event.description} - Failed: {event.error_message}")

    return progress_callback


def create_streaming_callbacks():
    """Create streaming callbacks for LLM token display."""

    def on_token(token: str) -> None:
        """Handle individual tokens from LLM streaming."""
        click.echo(f"   📝 {token}", nl=False)

    def on_completion(response: str) -> None:
        """Handle completion of LLM streaming response."""
        click.echo("")  # New line after streaming

    return StreamingCallbacks(on_token=on_token, on_completion=on_completion)


@cli.command()
@click.argument("prompt")
@click.option("--name", help="Name for the generated project")
def generate(prompt: str, name: Optional[str] = None):
    """Generate a CrewAI project from a natural language prompt.

    PROMPT: Natural language description of the crew you want to create.

    Examples:
      crewforge generate "Create a content research crew that finds and analyzes articles"
      crewforge generate --name my-crew "Build agents for customer service automation"
    """
    try:
        # Create simple project name from input or use default
        project_name = name or "generated-crew"

        # Step 1: Create generation request using our models
        click.echo("📝 Creating generation request...")
        try:
            generation_request = GenerationRequest(
                prompt=prompt, project_name=project_name
            )
            click.echo("✅ Generation request created")
        except ValidationError as e:
            raise click.ClickException(f"Validation error: {e}")

        # Step 2: Create progress tracker and initialize project scaffolder
        click.echo("🔧 Initializing project scaffolder with progress tracking...")
        try:
            # Create progress tracker with standard generation steps
            progress_tracker = ProgressTracker(get_standard_generation_steps())
            progress_callback = create_progress_callback()
            progress_tracker.add_callback(progress_callback)

            # Initialize scaffolder with progress tracker
            scaffolder = ProjectScaffolder(progress_tracker=progress_tracker)
            click.echo("✅ Scaffolder ready with progress tracking")
        except Exception as e:
            raise click.ClickException(f"Failed to initialize scaffolder: {e}")

        # Step 5: Generate complete CrewAI project with progress tracking
        click.echo("")
        click.echo("🤖 Starting CrewAI project generation...")

        # Check if API key is available before proceeding
        if not os.getenv("OPENAI_API_KEY"):
            raise click.ClickException(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set your OpenAI API key:\n"
                "  export OPENAI_API_KEY='your-api-key-here'"
            )

        try:
            # Create streaming callbacks for real-time LLM response display
            streaming_callbacks = create_streaming_callbacks()

            # Generate the complete project using scaffolding
            current_dir = Path.cwd()

            click.echo(
                "🔄 Generation pipeline started with real-time progress tracking..."
            )
            click.echo("")

            project_path = scaffolder.generate_project(
                generation_request,
                current_dir,
                streaming_callbacks=streaming_callbacks,
            )

            click.echo("")
            click.echo("✅ Project generation completed successfully!")

        except LLMAuthenticationError as e:
            raise click.ClickException(
                f"Authentication failed: {e}\n"
                "Please check your OPENAI_API_KEY and try again."
            )
        except LLMRateLimitError as e:
            raise click.ClickException(
                f"Rate limit exceeded: {e}\n"
                "Please wait a moment and try again, or check your API quota."
            )
        except LLMNetworkError as e:
            raise click.ClickException(
                f"Network error: {e}\n"
                "Please check your internet connection and try again."
            )
        except LLMError as e:
            raise click.ClickException(
                f"LLM API error: {e}\n"
                "This may be a temporary issue. Please try again."
            )
        except ScaffoldingError as e:
            raise click.ClickException(f"Project generation failed: {e}")
        except Exception as e:
            raise click.ClickException(f"Unexpected error during generation: {e}")

        # Success confirmation
        click.echo("")
        click.echo("🎉 CrewAI project generated successfully!")
        click.echo(f"📍 Location: {project_path}")
        click.echo(f"📛 Project name: {project_name}")
        click.echo("")
        click.echo("Next steps:")
        click.echo(f"  cd {project_name}")
        click.echo("  # Install dependencies (if needed)")
        click.echo("  # Review and customize the generated agents and tasks")
        click.echo("  # Run your CrewAI project")
        click.echo("")
        click.echo("Your CrewAI project is ready to use!")

    except click.ClickException:
        # Re-raise Click exceptions as-is
        raise
    except Exception as e:
        # Handle unexpected errors gracefully
        click.echo(f"❌ Unexpected error: {e}", err=True)
        click.echo("Please report this issue if it persists.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()

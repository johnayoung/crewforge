"""Main CLI entry point for CrewForge."""

import logging
import os
import re
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


def validate_prompt(prompt: str) -> str:
    """Validate and clean prompt input.

    Args:
        prompt: User-provided prompt string

    Returns:
        Cleaned prompt string

    Raises:
        click.BadParameter: If prompt is invalid
    """
    if not prompt or not prompt.strip():
        raise click.BadParameter(
            "Prompt cannot be empty. Please provide a description of the crew you want to create."
        )

    cleaned_prompt = prompt.strip()

    # Check minimum length (10 characters)
    if len(cleaned_prompt) < 10:
        raise click.BadParameter(
            f"Prompt is too short ({len(cleaned_prompt)} characters). "
            "Please provide at least 10 characters describing your crew requirements. "
            "Example: 'Create a content research crew that finds and analyzes articles'"
        )

    # Check maximum length (2000 characters)
    if len(cleaned_prompt) > 2000:
        raise click.BadParameter(
            f"Prompt is too long ({len(cleaned_prompt)} characters). "
            "Please keep your description under 2000 characters for optimal processing."
        )

    # Check for meaningful content (not just numbers, special chars, or gibberish)
    # Must contain at least some alphabetic characters and reasonable word structure
    word_chars = re.findall(r"[a-zA-Z]+", cleaned_prompt)
    if not word_chars or len(" ".join(word_chars)) < 5:
        raise click.BadParameter(
            "Prompt must contain meaningful text describing your crew requirements. "
            "Please use clear English to describe what you want your crew to do."
        )

    # Check for reasonable word distribution - at least 40% of content should be letters
    total_chars = len(
        re.sub(r"\s+", "", cleaned_prompt)
    )  # Remove whitespace for calculation
    letter_chars = len(re.sub(r"[^a-zA-Z]", "", cleaned_prompt))
    if total_chars > 0 and (letter_chars / total_chars) < 0.4:
        raise click.BadParameter(
            "Prompt must contain meaningful text with sufficient letters. "
            "Please describe your crew requirements using clear English words."
        )

    # Check for reasonable word patterns (words shouldn't be too long without vowels)
    words = re.findall(r"[a-zA-Z]+", cleaned_prompt)
    suspicious_word_count = 0
    for word in words:
        if len(word) > 4 and not re.search(r"[aeiouAEIOU]", word):
            suspicious_word_count += 1
        # Check for words that look like random sequences (too many consonants in a row)
        if len(word) > 5 and re.search(
            r"[bcdfgjklmnpqrstvwxyzBCDFGJKLMNPQRSTVWXYZ]{5,}", word
        ):
            suspicious_word_count += 1
        # Check for repeating patterns that might indicate key mashing
        if len(word) > 6 and any(
            word.lower().count(char) > len(word) * 0.3 for char in set(word.lower())
        ):
            suspicious_word_count += 1

    # If most words look suspicious, reject the prompt
    if len(words) > 0 and suspicious_word_count / len(words) > 0.7:
        raise click.BadParameter(
            "Prompt appears to contain gibberish or random characters. "
            "Please use meaningful English words to describe your crew requirements."
        )

    # Check for obviously invalid characters
    if "\x00" in cleaned_prompt or any(
        ord(c) < 32 and c not in "\n\t" for c in cleaned_prompt
    ):
        raise click.BadParameter(
            "Prompt contains invalid characters. Please use only standard text characters."
        )

    return cleaned_prompt


def validate_project_name(name: Optional[str], prompt: str) -> str:
    """Validate and clean project name.

    Args:
        name: User-provided project name (optional)
        prompt: User prompt to generate name from if needed

    Returns:
        Clean project name

    Raises:
        click.BadParameter: If name is invalid
    """
    if name is None:
        # Auto-generate from prompt
        # Extract key words and create a reasonable name
        words = re.findall(r"[a-zA-Z]+", prompt.lower())
        if len(words) >= 2:
            # Take first few meaningful words
            name_words = [
                w
                for w in words[:4]
                if w
                not in {
                    "a",
                    "an",
                    "the",
                    "and",
                    "or",
                    "but",
                    "create",
                    "build",
                    "make",
                    "generate",
                }
            ][:3]
            if name_words:
                generated_name = "-".join(name_words)
            else:
                generated_name = "generated-crew"
        else:
            generated_name = "generated-crew"
        return generated_name

    # Validate provided name
    if not name or not name.strip():
        raise click.BadParameter("Project name cannot be empty.")

    # Clean the name
    cleaned_name = name.strip().lower()

    # Replace spaces and special characters with hyphens
    cleaned_name = re.sub(r"[^a-z0-9]+", "-", cleaned_name)

    # Remove leading/trailing hyphens and multiple consecutive hyphens
    cleaned_name = re.sub(r"^-+|-+$", "", cleaned_name)
    cleaned_name = re.sub(r"-+", "-", cleaned_name)

    # Check length
    if len(cleaned_name) < 2:
        raise click.BadParameter(
            f"Project name '{name}' is too short after cleaning. "
            "Please provide a name with at least 2 alphanumeric characters."
        )

    if len(cleaned_name) > 50:
        raise click.BadParameter(
            f"Project name is too long ({len(cleaned_name)} characters). "
            "Please keep it under 50 characters."
        )

    return cleaned_name


def check_directory_conflicts(project_name: str) -> None:
    """Check for directory conflicts and permissions.

    Args:
        project_name: Clean project name

    Raises:
        click.ClickException: If there are directory issues
    """
    current_dir = Path.cwd()
    # CrewAI normalizes hyphens to underscores in directory names
    normalized_name = project_name.replace("-", "_")
    project_path = current_dir / normalized_name

    # Check if directory already exists
    if project_path.exists():
        raise click.ClickException(
            f"Directory '{normalized_name}' already exists in {current_dir}. "
            f"Please choose a different project name or remove the existing directory."
        )

    # Check write permissions
    try:
        # Test if we can create directories in current location
        test_path = current_dir / f".crewforge_test_{os.getpid()}"
        test_path.mkdir(exist_ok=True)
        test_path.rmdir()
    except (OSError, PermissionError) as e:
        raise click.ClickException(
            f"Cannot create project in {current_dir}. Permission denied. "
            f"Please check directory permissions or choose a different location."
        ) from e


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
        # Step 1: Validate and clean inputs
        click.echo("🔍 Validating inputs...")

        validated_prompt = validate_prompt(prompt)
        validated_name = validate_project_name(name, validated_prompt)

        click.echo(f"✅ Prompt validated ({len(validated_prompt)} characters)")
        click.echo(f"✅ Project name: {validated_name}")

        # Step 2: Check directory conflicts
        click.echo("📁 Checking directory...")
        check_directory_conflicts(validated_name)
        click.echo("✅ Directory is available")

        # Step 3: Create generation request using our models
        click.echo("📝 Creating generation request...")
        try:
            generation_request = GenerationRequest(
                prompt=validated_prompt, project_name=validated_name
            )
            click.echo("✅ Generation request created")
        except ValidationError as e:
            raise click.ClickException(f"Validation error: {e}")

        # Step 4: Create progress tracker and initialize project scaffolder
        click.echo("🔧 Initializing project scaffolder with progress tracking...")
        try:
            # Create progress tracker with initial steps
            initial_steps = [
                ProgressStep(
                    "analyze_prompt", "Analyzing prompt for requirements", 5.0
                ),
                ProgressStep(
                    "generate_agents", "Generating agent configurations", 15.0
                ),
                ProgressStep("generate_tasks", "Creating task definitions", 10.0),
                ProgressStep("select_tools", "Selecting appropriate tools", 3.0),
                ProgressStep(
                    "create_scaffold", "Creating CrewAI project structure", 8.0
                ),
                ProgressStep("populate_files", "Populating project files", 4.0),
            ]

            progress_tracker = ProgressTracker(initial_steps)
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
            # Create progress tracker with initial steps
            initial_steps = [
                ProgressStep(
                    "analyze_prompt", "Analyzing prompt for requirements", 5.0
                ),
                ProgressStep(
                    "generate_agents", "Generating agent configurations", 15.0
                ),
                ProgressStep("generate_tasks", "Creating task definitions", 10.0),
                ProgressStep("select_tools", "Selecting appropriate tools", 3.0),
                ProgressStep(
                    "create_scaffold", "Creating CrewAI project structure", 8.0
                ),
                ProgressStep("populate_files", "Populating project files", 4.0),
            ]

            progress_tracker = ProgressTracker(initial_steps)
            progress_callback = create_progress_callback()
            progress_tracker.add_callback(progress_callback)

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
        click.echo(f"📛 Project name: {validated_name}")
        click.echo("")
        click.echo("Next steps:")
        click.echo(f"  cd {validated_name}")
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

"""
CrewForge CLI - Command Line Interface

Entry point for the CrewForge CLI tool that generates CrewAI projects
from natural language prompts.
"""

import asyncio
import signal
import sys
from functools import wraps
from pathlib import Path
import click
from crewforge import __version__
from crewforge.progress import ProgressIndicator, StatusDisplay
from crewforge.llm import LLMClient, LLMError
from crewforge.prompt_templates import PromptTemplates, PromptTemplateError
from crewforge.scaffolder import CrewAIScaffolder, CrewAIError
from crewforge.orchestrator import WorkflowOrchestrator, WorkflowContext, WorkflowError


def handle_keyboard_interrupt(signum, frame):
    """Handle KeyboardInterrupt (Ctrl+C) gracefully."""
    click.echo("\n\n🛑 Operation cancelled by user", err=True)
    click.echo("💡 You can restart the project creation at any time.", err=True)
    sys.exit(130)  # Standard exit code for SIGINT


def handle_errors(func):
    """Decorator to handle common errors gracefully."""

    @wraps(func)  # This preserves the original function's metadata
    def wrapper(*args, **kwargs):
        try:
            # Set up keyboard interrupt handler
            signal.signal(signal.SIGINT, handle_keyboard_interrupt)
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            handle_keyboard_interrupt(None, None)
        except PermissionError as e:
            click.echo(f"\n❌ Permission error: {str(e)}", err=True)
            click.echo(
                "💡 Try running with appropriate permissions or check directory access rights.",
                err=True,
            )
            sys.exit(1)
        except FileNotFoundError as e:
            click.echo(f"\n❌ File not found: {str(e)}", err=True)
            click.echo(
                "💡 Verify the file path and ensure all required files exist.", err=True
            )
            sys.exit(1)
        except OSError as e:
            error_msg = str(e).lower()
            click.echo(f"\n❌ System error: {str(e)}", err=True)
            if "space" in error_msg or "disk" in error_msg:
                click.echo("💡 Free up disk space and try again.", err=True)
            elif "permission" in error_msg or "access" in error_msg:
                click.echo("💡 Check file and directory permissions.", err=True)
            else:
                click.echo("💡 Please try again or check system resources.", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"\n💥 Unexpected error occurred: {str(e)}", err=True)
            click.echo(
                "💡 Please try again. If the problem persists, check your project description and target directory.",
                err=True,
            )
            click.echo(
                "🐛 For support, please report this error with the project details you provided.",
                err=True,
            )
            sys.exit(1)

    return wrapper


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="crewforge")
@click.pass_context
def main(ctx):
    """
    CrewForge CLI - Generate CrewAI projects from natural language prompts.

    CrewForge is a command-line tool that converts natural language descriptions
    into fully functional CrewAI projects with intelligent agent configurations.

    Transform your ideas into working CrewAI projects in minutes:

    \b
    Examples:
      crewforge create my-research-team "Build a content research team"
      crewforge create sales-analyzer "Analyze sales data and generate reports"
      crewforge create --interactive customer-support

    Get started by running:
      crewforge create --help
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("project_name")
@click.argument("prompt", required=False)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Enable interactive mode for additional clarification and project customization",
)
@click.option(
    "--output-dir",
    "-o",
    default=".",
    help="Directory where the project will be created (default: current directory)",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=str),
)
@handle_errors
def create(project_name, prompt, interactive, output_dir):
    """
    Create a new CrewAI project from a natural language prompt.

    PROJECT_NAME: Name of the CrewAI project to create (must be valid directory name)

    PROMPT: Natural language description of the project you want to build.
            If not provided, you must use --interactive mode.

    \b
    Examples:
      crewforge create my-blog-team "Create a content creation team with writer and editor agents"
      crewforge create data-pipeline "Build agents to collect, process and analyze sales data"
      crewforge create --interactive customer-service
      crewforge create research-crew "Scientific research team" --output-dir ~/projects

    The tool will generate a complete CrewAI project structure with:
    - Intelligent agent configurations based on your description
    - Task definitions optimized for your use case
    - Ready-to-run Python code with proper dependencies
    - Documentation and setup instructions
    """
    # Initialize progress tracking
    project_steps = [
        "Parsing natural language prompt",
        "Validating project requirements",
        "Planning project structure",
        "Preparing CrewAI configuration",
        "Setting up project files",
    ]

    progress = ProgressIndicator(steps=project_steps)
    status = StatusDisplay()

    # Display version and project info
    click.echo(f"🔥 CrewForge v{__version__}")
    click.echo(
        f"📁 Creating CrewAI project: {click.style(project_name, fg='blue', bold=True)}"
    )

    # Validate project name
    # Check for empty or whitespace-only names
    if not project_name or not project_name.strip():
        status.error("Project name cannot be empty")
        raise click.Abort()

    # Check that name contains at least one alphanumeric character
    alphanumeric_chars = project_name.replace("-", "").replace("_", "").replace(".", "")
    if not alphanumeric_chars or not alphanumeric_chars.isalnum():
        status.error(
            "Project name must contain at least one alphanumeric character (letters or numbers)"
        )
        raise click.Abort()

    # Check for invalid characters (only allow alphanumeric, hyphens, underscores, dots)
    import re

    if not re.match(r"^[a-zA-Z0-9._-]+$", project_name):
        status.error(
            "Project name must be a valid directory name (alphanumeric, hyphens, underscores, dots only)"
        )
        raise click.Abort()

    # Handle prompt input
    if not prompt:
        if interactive:
            status.info("Interactive Mode: Let's build your CrewAI project!")
            try:
                prompt = click.prompt(
                    "Describe the CrewAI project you want to create",
                    type=str,
                    default="",  # Allow empty input
                    show_default=False,
                )
            except KeyboardInterrupt:
                handle_keyboard_interrupt(None, None)
            if not prompt.strip():
                status.error("Project description cannot be empty")
                raise click.Abort()
        else:
            status.error("Please provide a prompt or use --interactive mode")
            status.info("Try: crewforge create --help")
            raise click.Abort()

    # Display configuration
    click.echo(f"📝 Project prompt: {click.style(prompt, fg='green')}")
    click.echo(f"📂 Output directory: {click.style(output_dir, fg='cyan')}")

    # Create workflow context
    context = WorkflowContext(
        user_prompt=prompt,
        project_name=project_name,
        output_dir=Path(output_dir),
        llm_provider="openai",  # Default provider
        llm_model="gpt-3.5-turbo"  # Default model
    )

    # Initialize orchestrator
    orchestrator = WorkflowOrchestrator()

    try:
        # Execute the complete workflow
        result_context = asyncio.run(orchestrator.execute_workflow(context))

        # Display workflow summary
        summary = orchestrator.get_workflow_summary(result_context)
        click.echo(f"\n� Workflow Summary:")
        click.echo(f"  • Total steps: {summary['total_steps']}")
        click.echo(f"  • Completed: {summary['completed_steps']}")
        click.echo(f"  • Failed: {summary['failed_steps']}")
        click.echo(f"  • Duration: {summary['total_time_seconds']:.2f}s")

        if summary['success']:
            status.success("CrewAI project created successfully!")

            if result_context.scaffold_result and result_context.scaffold_result["success"]:
                project_path = result_context.scaffold_result["project_path"]
                click.echo(
                    f"\n📁 Project created at: {click.style(str(project_path), fg='cyan', bold=True)}"
                )

                # Show what was created
                project_files = list(project_path.rglob("*"))
                click.echo(
                    f"📝 Generated {len([f for f in project_files if f.is_file()])} files:"
                )

                # Show key structure files
                key_files = [
                    "pyproject.toml",
                    "README.md",
                    ".env",
                    f"src/{project_name}/main.py",
                    f"src/{project_name}/crew.py",
                    f"src/{project_name}/config/agents.yaml",
                    f"src/{project_name}/config/tasks.yaml",
                ]

                for file_path in key_files:
                    full_path = project_path / file_path
                    if full_path.exists():
                        click.echo(f"  ✅ {file_path}")
                    else:
                        click.echo(f"  📄 {file_path}")

        else:
            status.error("Workflow completed with errors")
            # Show failed steps
            for step in result_context.steps:
                if step.status == "failed":
                    click.echo(f"  ❌ {step.name}: {step.error}")

        return 0

    except WorkflowError as e:
        status.error(f"Workflow failed: {str(e)}")
        if e.context:
            # Show which step failed
            failed_steps = [s for s in e.context.steps if s.status == "failed"]
            if failed_steps:
                click.echo(f"Failed at step: {failed_steps[-1].name}")
        raise click.Abort()

    except Exception as e:
        status.error(f"Unexpected error: {str(e)}")
        raise click.Abort()


if __name__ == "__main__":
    main()

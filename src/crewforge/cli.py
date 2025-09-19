"""
CrewForge CLI - Command Line Interface

Entry point for the CrewForge CLI tool that generates CrewAI projects
from natural language prompts.
"""

import asyncio
import logging
import signal
import sys
from functools import wraps
from pathlib import Path
from typing import Callable
import click
from crewforge import __version__
from crewforge.progress import ProgressIndicator, StatusDisplay
from crewforge.llm import LLMClient, LLMError
from crewforge.prompt_templates import PromptTemplates, PromptTemplateError
from crewforge.scaffolder import CrewAIScaffolder, CrewAIError
from crewforge.orchestrator import WorkflowOrchestrator, WorkflowContext, WorkflowError


def setup_logging(
    log_level: int = logging.WARNING, log_file: str | None = None
) -> logging.Logger:
    """
    Configure structured logging for the CLI application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs to
    """
    # Create logger
    logger = logging.getLogger("crewforge")
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def handle_keyboard_interrupt(signum: int | None, frame: object | None) -> None:
    """Handle KeyboardInterrupt (Ctrl+C) gracefully."""
    logger = logging.getLogger("crewforge")
    logger.info("Operation cancelled by user")
    click.echo("\n\n🛑 Operation cancelled by user", err=True)
    click.echo("💡 You can restart the project creation at any time.", err=True)
    sys.exit(130)  # Standard exit code for SIGINT


def handle_errors(func: Callable[..., object]) -> Callable[..., object]:
    """Decorator to handle common errors gracefully."""

    @wraps(func)  # This preserves the original function's metadata
    def wrapper(*args, **kwargs) -> object:
        try:
            # Set up keyboard interrupt handler
            signal.signal(signal.SIGINT, handle_keyboard_interrupt)
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            handle_keyboard_interrupt(None, None)
        except PermissionError as e:
            logger = logging.getLogger("crewforge")
            logger.error(f"Permission error: {str(e)}", exc_info=True)
            click.echo(f"\n❌ Permission error: {str(e)}", err=True)
            click.echo(
                "💡 Try running with appropriate permissions or check directory access rights.",
                err=True,
            )
            sys.exit(1)
        except FileNotFoundError as e:
            logger = logging.getLogger("crewforge")
            logger.error(f"File not found: {str(e)}", exc_info=True)
            click.echo(f"\n❌ File not found: {str(e)}", err=True)
            click.echo(
                "💡 Verify the file path and ensure all required files exist.", err=True
            )
            sys.exit(1)
        except OSError as e:
            logger = logging.getLogger("crewforge")
            logger.error(f"OS error: {str(e)}", exc_info=True)
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
            logger = logging.getLogger("crewforge")
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
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
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for detailed troubleshooting information",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging for additional information",
)
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, writable=True),
    help="Write logs to specified file in addition to console",
)
@click.pass_context
def main(ctx: click.Context, debug: bool, verbose: bool, log_file: str | None) -> None:
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
    # Set up logging based on options
    log_level = logging.WARNING
    if verbose:
        log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG

    logger = setup_logging(log_level=log_level, log_file=log_file)
    logger.debug("CrewForge CLI started")
    logger.debug(f"Log level: {logging.getLevelName(log_level)}")
    if log_file:
        logger.debug(f"Log file: {log_file}")

    # Store logger in context for use in subcommands
    ctx.ensure_object(dict)
    ctx.obj["logger"] = logger

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
@click.pass_context
@handle_errors
def create(
    ctx: click.Context,
    project_name: str,
    prompt: str | None,
    interactive: bool,
    output_dir: str,
) -> None:
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
    logger = ctx.obj["logger"]
    logger.info(f"Starting project creation: {project_name}")
    logger.debug(f"Output directory: {output_dir}")
    logger.debug(f"Interactive mode: {interactive}")
    logger.debug(f"Prompt provided: {bool(prompt)}")

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

    logger.info(f"CrewForge v{__version__} - Creating project: {project_name}")

    # Validate project name
    logger.debug("Validating project name")
    # Check for empty or whitespace-only names
    if not project_name or not project_name.strip():
        logger.error("Project name validation failed: empty name")
        status.error("Project name cannot be empty")
        raise click.Abort()

    # Check that name contains at least one alphanumeric character
    alphanumeric_chars = project_name.replace("-", "").replace("_", "").replace(".", "")
    if not alphanumeric_chars or not alphanumeric_chars.isalnum():
        logger.error("Project name validation failed: no alphanumeric characters")
        status.error(
            "Project name must contain at least one alphanumeric character (letters or numbers)"
        )
        raise click.Abort()

    # Check for invalid characters (only allow alphanumeric, hyphens, underscores, dots)
    import re

    if not re.match(r"^[a-zA-Z0-9._-]+$", project_name):
        logger.error("Project name validation failed: invalid characters")
        status.error(
            "Project name must be a valid directory name (alphanumeric, hyphens, underscores, dots only)"
        )
        raise click.Abort()

    logger.debug("Project name validation passed")

    # Handle prompt input
    if not prompt:
        if interactive:
            logger.info("Entering interactive mode for prompt input")
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
            if not prompt or not prompt.strip():
                logger.error("Interactive prompt input failed: empty prompt")
                status.error("Project description cannot be empty")
                raise click.Abort()
        else:
            logger.error("No prompt provided and interactive mode not enabled")
            status.error("Please provide a prompt or use --interactive mode")
            status.info("Try: crewforge create --help")
            raise click.Abort()

    logger.info("Prompt processing completed")
    assert prompt is not None  # Should be validated above
    logger.debug(
        f"Final prompt: {prompt[:100]}..."
    )  # Log first 100 chars for debugging

    # Display configuration
    click.echo(f"📝 Project prompt: {click.style(prompt, fg='green')}")
    click.echo(f"📂 Output directory: {click.style(output_dir, fg='cyan')}")

    # Create workflow context
    context = WorkflowContext(
        user_prompt=prompt,
        project_name=project_name,
        output_dir=Path(output_dir),
        llm_provider="openai",  # Default provider
        llm_model="gpt-3.5-turbo",  # Default model
    )

    logger.debug(
        f"Workflow context created: provider={context.llm_provider}, model={context.llm_model}"
    )

    # Initialize orchestrator
    orchestrator = WorkflowOrchestrator()
    logger.debug("Workflow orchestrator initialized")

    try:
        # Execute the complete workflow
        logger.info("Starting workflow execution")
        result_context = asyncio.run(orchestrator.execute_workflow(context))
        logger.info("Workflow execution completed")

        # Display workflow summary
        summary = orchestrator.get_workflow_summary(result_context)
        logger.info(
            f"Workflow summary: {summary['completed_steps']}/{summary['total_steps']} steps completed"
        )
        click.echo(f"\n� Workflow Summary:")
        click.echo(f"  • Total steps: {summary['total_steps']}")
        click.echo(f"  • Completed: {summary['completed_steps']}")
        click.echo(f"  • Failed: {summary['failed_steps']}")
        click.echo(f"  • Duration: {summary['total_time_seconds']:.2f}s")

        if summary["success"]:
            logger.info("Project creation successful")
            status.success("CrewAI project created successfully!")

            if (
                result_context.scaffold_result
                and result_context.scaffold_result["success"]
            ):
                project_path = result_context.scaffold_result["project_path"]
                logger.info(f"Project created at: {project_path}")
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
            logger.warning("Workflow completed with errors")
            status.error("Workflow completed with errors")
            # Show failed steps
            for step in result_context.steps:
                if step.status == "failed":
                    logger.error(f"Failed step: {step.name} - {step.error}")
                    click.echo(f"  ❌ {step.name}: {step.error}")

    except WorkflowError as e:
        logger.error(f"Workflow error: {str(e)}", exc_info=True)
        status.error(f"Workflow failed: {str(e)}")
        if e.context:
            # Show which step failed
            failed_steps = [s for s in e.context.steps if s.status == "failed"]
            if failed_steps:
                logger.error(f"Failed at step: {failed_steps[-1].name}")
                click.echo(f"Failed at step: {failed_steps[-1].name}")
        raise click.Abort()

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        status.error(f"Unexpected error: {str(e)}")
        raise click.Abort()


if __name__ == "__main__":
    main()

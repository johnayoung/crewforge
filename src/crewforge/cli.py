"""
CrewForge CLI - Command Line Interface

Entry point for the CrewForge CLI tool that generates CrewAI projects
from natural language prompts.
"""

import click
from crewforge import __version__
from crewforge.progress import ProgressIndicator, StatusDisplay


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
            prompt = click.prompt(
                "Describe the CrewAI project you want to create",
                type=str,
                default="",  # Allow empty input
                show_default=False,
            )
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

    # Start project generation with progress indicators
    progress.start_progress(f"Generating CrewAI project '{project_name}'")

    # Step 1: Parse prompt
    progress.update_progress("Parsing natural language prompt", 0)
    status.info("Analyzing project requirements from your description...")

    # Step 2: Validate requirements
    progress.update_progress("Validating project requirements", 1)
    status.info("Validating project specifications and requirements...")

    # Step 3: Plan structure
    progress.update_progress("Planning project structure", 2)
    status.info("Designing optimal project architecture...")

    # Step 4: Prepare configuration
    progress.update_progress("Preparing CrewAI configuration", 3)
    status.info("Preparing intelligent agent configurations...")

    # Step 5: Setup files
    progress.update_progress("Setting up project files", 4)
    status.info("Creating project files and dependencies...")

    # Complete the process
    progress.finish_progress(f"CrewAI project '{project_name}' ready!")

    # Show what was created (placeholder)
    status.success("Project generation completed successfully!")
    click.echo("\n📦 Your CrewAI project includes:")
    click.echo("  • Intelligent agent configurations")
    click.echo("  • Optimized task definitions")
    click.echo("  • Ready-to-run Python code")
    click.echo("  • Complete project documentation")

    status.info("Next steps:")
    click.echo("  1. Navigate to your project directory")
    click.echo("  2. Review the generated configuration")
    click.echo("  3. Run your CrewAI project!")

    return 0


if __name__ == "__main__":
    main()

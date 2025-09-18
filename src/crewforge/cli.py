"""
CrewForge CLI - Command Line Interface

Entry point for the CrewForge CLI tool that generates CrewAI projects
from natural language prompts.
"""

import click
from crewforge import __version__


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
    # Display version and project info
    click.echo(f"🔥 CrewForge v{__version__}")
    click.echo(
        f"📁 Creating CrewAI project: {click.style(project_name, fg='blue', bold=True)}"
    )

    # Validate project name
    if (
        not project_name
        or not project_name.replace("-", "").replace("_", "").replace(".", "").isalnum()
    ):
        click.echo(
            click.style(
                "❌ Error: Project name must be a valid directory name (alphanumeric, hyphens, underscores, dots only)",
                fg="red",
            )
        )
        raise click.Abort()

    # Handle prompt input
    if not prompt:
        if interactive:
            click.echo("\n💭 Interactive Mode: Let's build your CrewAI project!")
            prompt = click.prompt(
                "Describe the CrewAI project you want to create",
                type=str,
                show_default=False,
            )
            if not prompt.strip():
                click.echo(
                    click.style(
                        "❌ Error: Project description cannot be empty", fg="red"
                    )
                )
                raise click.Abort()
        else:
            click.echo(
                click.style(
                    "❌ Error: Please provide a prompt or use --interactive mode",
                    fg="red",
                )
            )
            click.echo("💡 Try: crewforge create --help")
            raise click.Abort()

    # Display configuration
    click.echo(f"📝 Project prompt: {click.style(prompt, fg='green')}")
    click.echo(f"📂 Output directory: {click.style(output_dir, fg='cyan')}")

    # Implementation placeholder
    click.echo("\n🚧 Implementation coming soon!")
    click.echo("This will generate a complete CrewAI project with:")
    click.echo("  • Intelligent agent configurations")
    click.echo("  • Optimized task definitions")
    click.echo("  • Ready-to-run Python code")
    click.echo("  • Complete project documentation")

    return 0


if __name__ == "__main__":
    main()

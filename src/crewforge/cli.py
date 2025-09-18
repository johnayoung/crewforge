"""
CrewForge CLI - Command Line Interface

Entry point for the CrewForge CLI tool that generates CrewAI projects
from natural language prompts.
"""

import click
from crewforge import __version__


@click.group()
@click.version_option(version=__version__, prog_name="crewforge")
def main():
    """
    CrewForge CLI - Generate CrewAI projects from natural language prompts.

    CrewForge is a command-line tool that converts natural language descriptions
    into fully functional CrewAI projects with intelligent agent configurations.
    """
    pass


@main.command()
@click.argument("project_name")
@click.argument("prompt", required=False)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Enable interactive mode for additional clarification",
)
@click.option(
    "--output-dir",
    "-o",
    default=".",
    help="Directory where the project will be created",
)
def create(project_name, prompt, interactive, output_dir):
    """
    Create a new CrewAI project from a natural language prompt.

    PROJECT_NAME: Name of the CrewAI project to create
    PROMPT: Natural language description of the project (optional, can be provided interactively)
    """
    click.echo(f"CrewForge v{__version__}")
    click.echo(f"Creating CrewAI project: {project_name}")

    if not prompt:
        if interactive:
            prompt = click.prompt("Describe the CrewAI project you want to create")
        else:
            click.echo("Error: Please provide a prompt or use --interactive mode")
            return

    click.echo(f"Project prompt: {prompt}")
    click.echo(f"Output directory: {output_dir}")
    click.echo("🚧 Implementation coming soon!")


if __name__ == "__main__":
    main()

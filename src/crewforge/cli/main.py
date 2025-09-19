"""Main CLI entry point for CrewForge."""

import click


@click.group()
@click.version_option()
def cli():
    """CrewForge - Intelligent CrewAI Project Generator

    Generate complete CrewAI projects from natural language prompts using agentic AI.
    """
    pass


@cli.command()
@click.argument("prompt")
@click.option("--name", help="Name for the generated project")
def generate(prompt: str, name: str | None = None):
    """Generate a CrewAI project from a natural language prompt."""
    click.echo(f"Generating CrewAI project from prompt: {prompt}")
    if name:
        click.echo(f"Project name: {name}")
    click.echo("CrewForge CLI is ready! (Implementation in progress)")


if __name__ == "__main__":
    cli()

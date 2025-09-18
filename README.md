# CrewForge CLI Tool

CrewForge is a command-line tool that generates functional CrewAI projects from natural language prompts.

## Installation

```bash
pip install crewforge
```

## Usage

```bash
# Create a new CrewAI project
crewforge create "my-project" "Build a content research team that analyzes market trends"

# Interactive mode
crewforge create "my-project" --interactive
```

## Features

- Natural language prompt parsing
- Intelligent CrewAI project scaffolding
- Interactive clarification for ambiguous prompts
- Complete setup-to-running cycle in under 10 minutes

## Development

This project uses uv for dependency management:

```bash
# Install development dependencies
uv sync

# Run tests
uv run pytest

# Run CLI in development
uv run python -m crewforge --help
```
# CrewForge

Intelligent CrewAI Project Generator - A CLI tool that leverages agentic AI to generate complete CrewAI projects from natural language prompts.

## Installation

```bash
# Install with uv (recommended)
uv sync

# Install for development
uv sync --dev
```

## Usage

```bash
# Generate a CrewAI project from a natural language prompt
crewforge generate "Create a content research crew that finds and summarizes articles"

# Specify project name
crewforge generate --name research-crew "Build agents that research competitors"
```

## Development

This project uses:
- **Python 3.11+** for modern language features
- **uv** for fast package management
- **LiteLLM** for multi-provider LLM access
- **CrewAI** as the target framework
- **Click** for CLI interface
- **Pydantic** for data validation

## Status

This is an MVP in development. See `docs/ROADMAP.md` for implementation progress.
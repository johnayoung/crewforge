# CrewForge

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Intelligent CrewAI Project Generator** - A CLI tool that leverages agentic AI to generate complete, production-ready CrewAI projects from natural language prompts.

## Features

🤖 **Agentic AI Generation** - Uses LLM-powered analysis to create intelligent agent configurations  
📋 **Complete Project Scaffolding** - Generates full CrewAI project structure with all necessary files  
🔧 **Multi-Provider LLM Support** - Works with GPT-4, Claude, and other providers via LiteLLM  
✅ **Robust Validation** - Pydantic v2 models ensure data integrity and type safety  
🎯 **Smart Template Engine** - Jinja2-powered templates for clean, maintainable code generation  
📊 **Learning System** - Stores successful patterns to improve future generations  
⚡ **Fast Development** - Generate working CrewAI projects in seconds

## Quick Start

### Installation

```bash
# Install with uv (recommended)
uv add crewforge

# Or with pip
pip install crewforge
```

### Basic Usage

```bash
# Generate a CrewAI project from a natural language prompt
crewforge generate "Create a content research crew that finds and analyzes competitor articles"

# Specify a custom project name
crewforge generate --name research-team "Build agents for automated social media analysis"

# View help for all available commands
crewforge --help
```

### Example Output

The generated project will include:
- Complete agent definitions with roles, goals, and backstories
- Task configurations with clear expected outputs
- Tool integrations for web search, file operations, and data analysis
- Proper CrewAI project structure and entry points
- Ready-to-run Python files

## Architecture Overview

CrewForge follows a modular architecture designed for reliability and extensibility:

```
src/crewforge/
├── cli/           # Click-based command-line interface
├── core/          # Core generation and orchestration logic
│   ├── generator.py    # LLM-powered configuration generation
│   ├── llm.py         # Multi-provider LLM client with retry logic
│   ├── scaffolding.py # Project structure creation and file population
│   ├── templates.py   # Jinja2 template engine
│   └── validator.py   # Configuration validation and quality checks
├── models/        # Pydantic v2 data models
│   ├── agent.py       # Agent configuration models
│   ├── crew.py        # Crew and project models
│   └── task.py        # Task definition models
├── storage/       # Learning and persistence layer
│   └── learning.py    # Pattern storage and retrieval
└── templates/     # Jinja2 templates for code generation
    ├── agents.py.j2   # Agent class templates
    ├── crew.py.j2     # Crew configuration templates
    ├── tasks.py.j2    # Task definition templates
    └── tools.py.j2    # Tool integration templates
```

### Key Components

1. **GenerationEngine** - Orchestrates the AI-powered analysis and generation process
2. **LLMClient** - Handles multi-provider LLM access with exponential backoff retry logic
3. **ProjectScaffolder** - Creates project structure and populates files using templates
4. **Learning System** - Stores successful configurations for pattern recognition
5. **Validation Layer** - Ensures generated configurations meet CrewAI requirements

## Development

### Prerequisites

- Python 3.11+ (for improved error messages and modern typing syntax)
- `uv` package manager (recommended) or `pip`
- An LLM API key (OpenAI, Anthropic, etc.)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-org/crewforge.git
cd crewforge

# Install dependencies with uv (recommended)
uv sync --dev

# Or with pip
pip install -e ".[dev]"

# Set up your LLM API key
export OPENAI_API_KEY="your-api-key-here"
# or
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Code Quality Standards

This project follows strict development standards:

#### Python Best Practices
- **Python 3.11+** features like improved error messages and modern type hints
- **PEP 8** conventions with 88-character line limits (Black formatter standard)  
- **Explicit imports** over wildcard imports for namespace clarity
- **Consistent type hints** with Pydantic models for data validation

#### Framework-Specific Patterns
- **Click Framework**: Command groups for organization, built-in validation, proper error handling
- **LiteLLM Integration**: Explicit model provider specification, retry logic with exponential backoff
- **Pydantic v2**: Modern syntax, Field validators, custom serializers, ConfigDict usage
- **Jinja2 Templates**: Descriptive variables, template inheritance, safety filters

### Development Workflow

```bash
# Run type checking
mypy src/

# Run linting and formatting  
ruff check src/ tests/
ruff format src/ tests/

# Run tests with coverage
pytest --cov=crewforge --cov-report=html

# Run integration tests
pytest tests/test_integration.py -v
```

### Testing

The project includes comprehensive test coverage:

- **Unit Tests** - Individual component testing with fixtures
- **Integration Tests** - End-to-end CLI and generation workflow testing  
- **Model Tests** - Pydantic validation and edge case testing
- **Error Handling Tests** - Exception handling and recovery testing

```bash
# Run all tests
pytest

# Run with coverage reporting
pytest --cov=crewforge --cov-report=term-missing

# Run specific test categories
pytest tests/test_models/ -v
pytest tests/test_cli.py::test_generate_command -v
```

### Configuration

CrewForge supports configuration via environment variables:

```bash
# LLM Configuration
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export CREWFORGE_DEFAULT_MODEL="gpt-4"  # Default: gpt-4

# Generation Settings  
export CREWFORGE_TEMPERATURE="0.1"      # LLM temperature (0.0-1.0)
export CREWFORGE_MAX_RETRIES="3"        # Maximum API retry attempts
export CREWFORGE_TIMEOUT="30"           # Request timeout in seconds

# Storage and Learning
export CREWFORGE_STORAGE_DIR="~/.crewforge"  # Local storage directory
export CREWFORGE_ENABLE_LEARNING="true"      # Enable pattern learning
```

## Project Structure

Generated CrewAI projects follow this structure:

```
my-crew/
├── pyproject.toml      # Project configuration and dependencies
├── README.md           # Usage instructions and documentation
├── src/
│   └── my_crew/
│       ├── __init__.py
│       ├── main.py     # Entry point and crew execution
│       ├── agents.py   # Agent definitions
│       ├── tasks.py    # Task configurations
│       └── tools.py    # Custom tools and integrations
└── tests/
    ├── __init__.py
    └── test_crew.py    # Basic test structure
```

## Advanced Usage

### Custom Templates

You can provide custom Jinja2 templates for specialized use cases:

```bash
# Use custom template directory
crewforge generate --template-dir ./my-templates "Custom crew requirements"
```

### Configuration Validation

CrewForge includes built-in validation for generated configurations:

```python
from crewforge.core.validator import ConfigValidator

validator = ConfigValidator()
issues = validator.validate_crew_config(crew_config)
```

### Learning System Integration

The learning system automatically improves generation quality:

```python
from crewforge.storage.learning import LearningSystem

learning = LearningSystem()
learning.store_successful_config(crew_config, metrics)
similar_patterns = learning.find_similar_patterns(prompt)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Guidelines

1. **Follow PEP 8** and use `black` for formatting
2. **Write comprehensive tests** for new features
3. **Use type hints** consistently throughout the codebase  
4. **Document public APIs** with docstrings
5. **Validate with mypy** before submitting PRs

## Troubleshooting

### Common Issues

**LLM API Errors**
```bash
# Check API key configuration
echo $OPENAI_API_KEY

# Test with different model
crewforge generate --model claude-3-sonnet "test prompt"
```

**Project Generation Failures**
```bash
# Enable verbose logging
export CREWFORGE_LOG_LEVEL=DEBUG
crewforge generate "your prompt"
```

**Template Errors**
```bash
# Validate template syntax
crewforge validate-templates
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Status

CrewForge is actively developed and maintained. See [docs/ROADMAP.md](docs/ROADMAP.md) for upcoming features and improvements.

---

**Need Help?** 
- 📚 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/your-org/crewforge/issues)  
- 💬 [Discussions](https://github.com/your-org/crewforge/discussions)
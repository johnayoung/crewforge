## Development Standards

### Python Best Practices
- Use type hints for all function parameters and return values (Python 3.11+ features)
- Leverage `pathlib.Path` over string paths for cross-platform compatibility
- Implement proper exception handling with specific exception types rather than bare `except` clauses
- Use dataclasses or Pydantic models for structured data to avoid dict-based configurations

### CLI Framework Patterns  
- Structure Click/Typer commands with clear parameter validation and help text
- Use Click's context passing for shared state between command groups rather than global variables
- Implement proper exit codes (0 for success, non-zero for errors) and rich error messages for user experience

### LLM Integration Guidelines
- Always implement retry logic with exponential backoff for liteLLM API calls
- Use structured output parsing with validation rather than raw string manipulation of LLM responses
- Track token usage and implement rate limiting to prevent API quota exhaustion
- Handle provider-specific errors gracefully with fallback mechanisms

### Template Engine Guidelines
- Use Jinja2 autoescaping for security and validate template inputs before rendering
- Structure templates with inheritance and macros for maintainable and reusable code generation
- Implement template validation to catch syntax errors before project generation

### Package Management Guidelines
- Use uv for all dependency management operations instead of pip for faster resolution
- Pin dependencies with version ranges in pyproject.toml for reproducible builds
- Implement proper virtual environment isolation for generated projects

### Code Quality Standards
- Run `mypy --strict` for comprehensive type checking before commits
- Use `ruff` for linting and formatting to catch common Python pitfalls early
- Implement comprehensive error handling for subprocess calls (CrewAI CLI integration)
- Use structured logging with appropriate levels rather than print statements for debugging

### Project Conventions
- Follow CrewAI naming conventions for agent roles and task names in generated projects
- Organize modules by domain (parsing, scaffolding, validation) rather than by layer
- Use absolute imports with clear module paths for better IDE support and maintainability
- Store templates and configuration patterns in version-controlled directories for consistency
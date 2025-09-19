# Copilot Instructions

## Development Standards

### Python Best Practices
- Use Python 3.11+ features like improved error messages and typing syntax.
- Follow PEP 8 conventions with 88-character line limits (Black formatter standard).
- Prefer explicit imports over wildcard imports to maintain namespace clarity.
- Use type hints consistently with Pydantic models for data validation.

### Click Framework Patterns
- Group related commands using Click command groups for better CLI organization.
- Use Click's built-in validation for parameters rather than manual validation.
- Implement proper error handling with Click's `ClickException` for user-friendly messages.
- Leverage Click's context system for sharing state between commands and subcommands.

### LiteLLM Integration Guidelines
- Always specify model provider explicitly to avoid ambiguous routing.
- Implement retry logic with exponential backoff for LLM API calls.
- Use structured outputs with JSON mode when possible for reliable parsing.
- Cache LLM responses where appropriate to reduce API costs and latency.

### Pydantic Data Modeling
- Use Pydantic v2 syntax for model definitions and validation.
- Leverage `Field` validators for complex validation logic beyond basic types.
- Implement custom serializers for complex data transformations.
- Use Pydantic's `ConfigDict` for model configuration instead of inner Config classes.

### CrewAI Project Generation
- Validate agent role compatibility before assigning tasks to prevent execution failures.
- Ensure task expected_output formats match downstream agent input requirements.
- Test tool availability and permissions before including in generated configurations.
- Maintain consistent naming patterns across agents, tasks, and tools for clarity.

### Template Engine (Jinja2) Standards
- Use descriptive variable names in templates to improve maintainability.
- Implement template inheritance for common patterns across different file types.
- Add safety filters like `|safe` only when HTML content is intentionally unescaped.
- Validate template syntax with Jinja2's compile-time checking before runtime usage.

### Code Quality Standards
- Run `mypy` for static type checking before commits to catch type-related issues.
- Use `ruff` for fast linting and formatting to maintain code consistency.
- Implement comprehensive exception handling for external dependencies (LLM APIs, file I/O).
- Add structured logging with appropriate levels (DEBUG for development, INFO for user feedback).
- Write unit tests for core generation logic and integration tests for CLI commands.

### Project Conventions
- Use snake_case for Python modules, functions, and variables following PEP 8.
- Organize imports in three groups: standard library, third-party, local imports.
- Place Pydantic models in dedicated modules under `models/` directory.
- Use absolute imports within the package to avoid relative import confusion.
- Name template files with `.j2` extension to clearly identify Jinja2 templates.
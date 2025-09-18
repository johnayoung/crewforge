# PROJECT: [Project Name]

## 📂 PROJECT FILES
Key project documents (use @ references in prompts):
- `@docs/BRIEF.md` - Project requirements and success metrics
- `@docs/SPEC.md` - Technical architecture and implementation details
- `@docs/ROADMAP.md` - Current milestones, tasks, and progress tracking
- `@.github/copilot-instructions.md` - This file (project constitution)

## 🎯 CURRENT STATUS
**Phase:** [Brief/Spec/Plan/Implementation]
**Focus:** [Current area of work - get details from ROADMAP.md]

## � DEVELOPMENT PRINCIPLES
- Test-driven development always
- One task at a time with explicit completion
- Update ROADMAP.md for all progress tracking
- No automatic commits - manual control only
- Stop after each task for review

## 📝 NOTES
[Any important architectural decisions, constraints, or reminders]

## Development Standards

### Python 3.11+ Best Practices
- Use pathlib for all file operations instead of os.path for better cross-platform compatibility
- Leverage type hints with Pydantic models for data validation and serialization
- Use async/await patterns sparingly - prefer synchronous code for CLI tools unless I/O bound
- Apply dataclasses or Pydantic BaseModel for structured data over plain dictionaries
- Import subprocess for external command execution, avoid os.system()

### Click/Typer CLI Framework Patterns
- Use Click decorators for command structure, Typer for modern type-hinted alternatives
- Group related commands under command groups for better organization
- Implement progressive disclosure - start with simple commands, add complexity through options
- Use click.echo() instead of print() for consistent output formatting
- Handle exceptions gracefully with try-catch and user-friendly error messages

### OpenAI/Anthropic LLM Guidelines
- Structure prompts with clear system/user message separation for consistent responses
- Implement token counting and management to avoid API limits during generation
- Use streaming responses for long-running operations to provide user feedback
- Cache LLM responses when possible to reduce API costs and improve performance
- Add retry logic with exponential backoff for API reliability

### uv Package Management Patterns
- Use `uv init` for project scaffolding instead of manual setup.py/pyproject.toml creation
- Leverage `uv add` for dependency management in generated projects
- Validate uv installation before project generation attempts
- Use `uv run` for executing generated project commands in isolated environments
- Include uv lock files in generated project templates for reproducible builds

### Jinja2 Templating Guidelines
- Separate template logic from generation logic - keep templates focused on structure
- Use template inheritance for common project scaffolding patterns
- Escape user input in templates to prevent injection vulnerabilities
- Organize templates by project type for maintainable template libraries
- Test templates with edge case data to ensure robust generation

### Code Quality Standards
- Wrap external command execution in try-catch with specific error handling per tool (uv, crewai)
- Use structured logging with different levels for user feedback vs debugging information
- Implement validation at multiple stages: input parsing, generation, and final execution
- Create custom exception classes for different failure modes (validation, generation, execution)
- Add timeout handling for external command execution and LLM API calls

### Project-Specific Conventions
- Name generated agents with descriptive, action-oriented names (ResearchAgent, not agent1)
- Follow CrewAI naming patterns for consistency with framework expectations
- Use kebab-case for generated project directory names, snake_case for Python modules
- Structure generated projects with clear src/ separation and tests/ directory
- Include comprehensive README.md in generated projects with setup and usage instructions

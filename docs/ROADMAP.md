# Implementation Roadmap

## Progress Checklist
- [x] **Commit 1**: Project Foundation & Package Configuration ✅
- [x] **Commit 2**: Pydantic Data Models & Type System ✅
- [x] **Commit 3**: CLI Framework & Command Structure ✅
- [x] **Commit 4**: LiteLLM Integration & Prompt Processing ✅
- [x] **Commit 5**: Jinja2 Template Engine & CrewAI Templates ✅
- [x] **Commit 6**: Agentic AI Generation Engine ✅
- [x] **Commit 7**: CrewAI Project Scaffolding & File Population ✅
- [x] **Commit 8**: Project Validation Suite ✅
- [x] **Commit 9**: Configuration Learning & Storage System ✅ [2025-09-21 14:24]
- [x] **Commit 10**: End-to-End Integration & Testing ✅ [2025-09-21 14:40]
- [x] **Commit 11**: Error Handling & User Experience Polish ✅ [2025-09-21 15:45]

## Post-MVP Enhancement: Progress Tracking System

- [x] **Commit 12**: Progress Tracking Foundation & Models ✅ [2025-09-22 12:45]
- [x] **Commit 13**: Integration with ProjectScaffolder & Real-time Progress Display ✅ [2025-09-22 15:30]

## Post-MVP Enhancement: Template-Based Prompt Management

- [x] **Commit 14**: Jinja2 Template-Based Prompt System ✅ [2025-09-22 16:15]

## Implementation Sequence

### Commit 1: Project Foundation & Package Configuration

**Goal**: Establish Python project structure with uv dependency management and development environment.

**Requirements**:
1. Create `pyproject.toml` with:
   - Python 3.11+ requirement
   - uv package manager configuration
   - Dependencies: click, litellm, pydantic, jinja2, pyyaml, crewai
   - Development dependencies: pytest, mypy, ruff
   - Entry point: `crewforge = crewforge.cli.main:cli`
2. Create project directory structure from SPEC.md:
   - `src/crewforge/` with all module directories
   - `tests/` with test structure
   - `examples/generated_crews/` directory
3. Initialize `src/crewforge/__init__.py` with package metadata

**Validation**: `uv sync && uv run python -c "import crewforge; print('Package imports successfully')"` → Package loads without errors

### Commit 2: Pydantic Data Models & Type System

**Goal**: Define data structures for agents, tasks, crews, and validation using Pydantic v2 syntax.

**Requirements**:
1. Create `src/crewforge/models/agent.py` with:
   - `AgentConfig` model with role, goal, backstory fields
   - `AgentTemplate` model for generation patterns
   - Field validators for CrewAI compliance
2. Create `src/crewforge/models/task.py` with:
   - `TaskConfig` model with description, expected_output fields
   - `TaskDependency` model for task sequencing
   - Validation for agent-task compatibility
3. Create `src/crewforge/models/crew.py` with:
   - `CrewConfig` model combining agents and tasks
   - `GenerationRequest` model for user prompts
   - `ValidationResult` model for project checking
4. Implement `src/crewforge/models/__init__.py` with exports

**Validation**: `uv run python -c "from crewforge.models import AgentConfig, TaskConfig, CrewConfig; print('Models import successfully')"` → All models accessible

### Commit 3: CLI Framework & Command Structure

**Goal**: Implement Click-based CLI interface with command structure from SPEC.md integration patterns.

**Requirements**:
1. Create `src/crewforge/cli/main.py` with:
   - `@click.group()` for main CLI entry point
   - `@click.command()` for `generate` command
   - Prompt parameter with validation
   - Optional `--name` flag for project naming
2. Implement command validation:
   - Prompt length and content checks
   - Project name format validation
   - Directory existence checking
3. Add progress feedback system:
   - Step-by-step generation progress
   - Error messaging with Click exceptions
   - Success confirmation with project path

**Validation**: `uv run crewforge --help` → Shows command help with generate command listed; `uv run crewforge generate --help` → Shows generate command options with prompt and --name parameters

### Commit 4: LiteLLM Integration & Prompt Processing

**Goal**: Establish LiteLLM connection for multi-provider LLM access and implement prompt processing.

**Requirements**:
1. Create `src/crewforge/core/llm.py` with:
   - `LLMClient` class wrapping LiteLLM
   - Model provider configuration (GPT-4, Claude support)
   - Retry logic with exponential backoff
   - Structured output support with JSON mode
2. Implement prompt engineering:
   - System prompt for CrewAI generation context
   - User prompt template for requirements extraction
   - Response parsing for agent/task configurations
3. Add configuration management:
   - API key handling from environment variables
   - Model selection defaults and overrides
   - Rate limiting and cost tracking

**Validation**: `uv run python -c "from crewforge.core.llm import LLMClient; print('LLM client ready')"` → Imports successfully without API calls

### Commit 5: Jinja2 Template Engine & CrewAI Templates

**Goal**: Create template system for generating CrewAI project files from validated configurations.

**Requirements**:
1. Create `src/crewforge/templates/agents.py.j2` with:
   - CrewAI agent class template
   - Role, goal, backstory injection points
   - Tool integration patterns
2. Create `src/crewforge/templates/tasks.py.j2` with:
   - CrewAI task class template
   - Description and expected_output templating
   - Agent assignment patterns
3. Create `src/crewforge/templates/tools.py.j2` with:
   - Tool import and configuration templates
   - Dynamic tool selection patterns
4. Create `src/crewforge/templates/crew.py.j2` with:
   - Crew instantiation template
   - Agent-task wiring logic
5. Implement `src/crewforge/core/templates.py` with:
   - `TemplateEngine` class using Jinja2
   - `populate_template()` method for file generation
   - Template validation and error handling

**Validation**: `uv run python -c "from crewforge.core.templates import TemplateEngine; te = TemplateEngine(); print('Template engine ready')"` → Template system initializes without errors

### Commit 6: Agentic AI Generation Engine

**Goal**: Implement core generation logic that analyzes prompts and creates CrewAI configurations using LLM.

**Requirements**:
1. Create `src/crewforge/core/generator.py` with:
   - `GenerationEngine` class coordinating LLM calls
   - `analyze_prompt()` method extracting crew requirements
   - `generate_agents()` method creating agent configurations
   - `generate_tasks()` method creating task definitions
   - `select_tools()` method choosing appropriate CrewAI tools
2. Implement generation pipeline:
   - Prompt analysis for business context
   - Agent role generation with appropriate goals/backstories
   - Task creation with clear expected outputs
   - Tool selection from CrewAI library
   - Configuration compatibility validation
3. Add generation quality controls:
   - Agent-task alignment checking
   - Tool availability validation
   - Output format consistency

**Validation**: `uv run python -c "from crewforge.core.generator import GenerationEngine; ge = GenerationEngine(); print('Generator ready')"` → Generation engine initializes successfully

### Commit 7: CrewAI Project Scaffolding & File Population

### Commit 6: Agentic AI Generation Engine

**Goal**: Implement core generation logic that analyzes prompts and creates CrewAI configurations using LLM.

**Requirements**:
1. Create `src/crewforge/core/generator.py` with:
   - `GenerationEngine` class coordinating LLM calls
   - `analyze_prompt()` method extracting crew requirements
   - `generate_agents()` method creating agent configurations
   - `generate_tasks()` method creating task definitions
   - `select_tools()` method choosing appropriate CrewAI tools
2. Implement generation pipeline:
   - Prompt analysis for business context
   - Agent role generation with appropriate goals/backstories
   - Task creation with clear expected outputs
   - Tool selection from CrewAI library
   - Configuration compatibility validation
3. Add generation quality controls:
   - Agent-task alignment checking
   - Tool availability validation
   - Output format consistency

**Validation**: `uv run python -c "from crewforge.core.generator import GenerationEngine; ge = GenerationEngine(); print('Generator ready')"` → Generation engine initializes

### Commit 7: CrewAI Project Scaffolding Integration

**Goal**: Integrate CrewAI scaffolding with complete file population using generated configurations.

**Requirements**:
1. Create `src/crewforge/core/scaffolding.py` with:
   - `ProjectScaffolder` class managing project creation
   - `create_crewai_project()` method calling `crewai create crew <name>`
   - `populate_project_files()` method injecting generated content
   - Directory management and cleanup on failures
2. Integrate with existing components:
   - GenerationEngine integration for getting configurations
   - TemplateEngine integration for file population
   - Model validation before file creation
3. Implement complete file population:
   - agents.py with generated agent definitions
   - tasks.py with generated task definitions
   - tools.py with selected tool configurations
   - crew.py with complete crew wiring

**Validation**: `uv run crewforge generate --name test-crew "Test scaffolding system"` → Creates functional CrewAI project directory with populated files that can be executed

### Commit 8: Project Validation Suite

**Goal**: Implement comprehensive validation for generated CrewAI projects ensuring syntax and functionality.

**Requirements**:
1. Create `src/crewforge/core/validator.py` with:
   - `ValidationSuite` class for multi-level checking
   - `validate_syntax()` method using Python AST parsing
   - `validate_crewai_compliance()` method for framework patterns
   - `validate_functionality()` method for basic crew instantiation
2. Implement validation pipeline:
   - Python syntax checking for all generated files
   - CrewAI configuration compliance verification
   - Import resolution and dependency checking
   - Basic crew execution testing
3. Add validation reporting:
   - Detailed error messages with file/line references
   - Success confirmation with project summary
   - Suggestions for common issues

**Validation**: Generated project validates successfully with: `cd <project_directory> && python -m pytest --collect-only` → Shows collected test cases; and `python -c "from src.crew import crew; print('Crew instantiated successfully')"` → Crew objects load without errors

### Commit 9: Configuration Learning & Storage System

**Goal**: Implement learning system to save successful configurations and improve future generations.

**Requirements**:
1. Create `src/crewforge/storage/learning.py` with:
   - `LearningStore` class for configuration persistence
   - `save_successful_config()` method for pattern storage
   - `retrieve_patterns()` method for historical data
   - JSON/YAML serialization for configurations
2. Implement pattern recognition:
   - Successful agent-task-tool combinations tracking
   - Generation success rate metrics
   - Configuration effectiveness scoring
3. Add storage management:
   - Local file system storage with organization
   - Configuration deduplication and merging
   - Storage cleanup and maintenance

**Validation**: `uv run python -c "from crewforge.storage.learning import LearningStore; ls = LearningStore(); ls.save_successful_config({'test': 'config'}); patterns = ls.retrieve_patterns(); print(f'Learning store working: {len(patterns)} patterns')"` → Shows learning store saves and retrieves configurations

### Commit 10: End-to-End Integration & Testing

**Goal**: Complete full generation pipeline integration with comprehensive testing coverage.

**Requirements**:
1. Update `src/crewforge/cli/main.py` with:
   - Complete pipeline orchestration in `generate` command
   - All component integration (Generator → Templates → Validation → Learning)
   - Progress feedback throughout generation process
   - Error handling with graceful degradation
2. Create comprehensive test suite:
   - `tests/test_cli.py` for command interface testing
   - `tests/test_generator.py` for generation logic testing
   - `tests/test_validator.py` for validation suite testing
   - `tests/fixtures/sample_prompts.py` for test data
3. Add integration testing:
   - Full generation cycle testing
   - Generated project execution testing
   - Error scenario handling

**Validation**: `uv run crewforge generate "Create a content research crew that finds and summarizes articles"` → Complete functional CrewAI MVP generated in <10 minutes, passes syntax validation, and crew can be instantiated successfully

### Commit 11: Error Handling & User Experience Polish

**Goal**: Enhance error handling, user feedback, and overall CLI experience for production readiness.

**Requirements**:
1. Implement comprehensive error handling:
   - LLM API failure recovery with retry logic
   - CrewAI command execution error handling
   - File system permission and space checking
   - Network connectivity validation
2. Enhance user experience:
   - Improved progress indicators with time estimates
   - Helpful error messages with suggested solutions
   - Success feedback with next steps guidance
   - Optional verbose mode for debugging
3. Add production safeguards:
   - Input sanitization and validation
   - Resource usage monitoring and limits
   - Graceful shutdown handling
   - Logging system for debugging

**Validation**: All error scenarios handled gracefully with helpful user feedback, achieving 95% syntax success rate target from BRIEF.md ✅ [2025-09-21 15:45]

**Implementation Details**:
- Enhanced LLM error handling with categorized exceptions, retry logic, and exponential backoff
- Added comprehensive scaffolding error handling for CrewAI CLI failures, file system issues, and project structure validation
- Implemented improved CLI with verbose mode, colored output, progress indicators, and specific error messages for different failure types
- Created comprehensive test suite for all error scenarios with integration testing
- Added production safeguards including input sanitization, resource monitoring, and structured logging

## Dependency Analysis

**Foundation Layer** (Commits 1-2):
- Project configuration enables all development
- Data models provide type safety for all components

**Interface Layer** (Commits 3-4):
- CLI provides user interface
- LLM integration enables AI generation

**Core Services** (Commits 5-6):
- Templates enable file generation
- Generation engine coordinates all AI operations

**Integration Layer** (Commits 7-9):
- CrewAI scaffolding creates target projects
- Validation ensures quality and functionality
- Learning store improves future generations

**Production Ready** (Commits 10-11):
- End-to-end integration completes MVP
- Error handling ensures production stability

**Critical Path**: Each commit builds essential functionality for the next, with correct dependency ordering: Config → Models → CLI → LLM → Templates → Generator → Scaffolding → Validation → Learning → Integration → Polish.

## Post-MVP Enhancement Implementation

### Commit 12: Progress Tracking Foundation & Models

**Goal**: Implement composable progress tracking system with step tracking, percentage calculation, and LLM streaming support.

**Requirements**:
1. Create `src/crewforge/core/progress.py` with:
   - `ProgressTracker` class for step and percentage tracking
   - `ProgressStep` dataclass for step definitions
   - `StreamingCallbacks` for LLM response streaming
   - Event-driven progress reporting with callbacks
2. Implement progress tracking models:
   - Step definitions with descriptions and estimated durations
   - Progress calculation based on completed vs total steps
   - Status tracking (not-started, in-progress, completed, failed)
3. Add streaming support for LLM responses:
   - Token-by-token streaming callbacks
   - Real-time response display integration
   - Buffering and error handling for streaming

**Validation**: `uv run python -c "from crewforge.core.progress import ProgressTracker, ProgressStep; pt = ProgressTracker([ProgressStep('test', 'Test step')]); print('Progress tracking ready')"` → Progress system initializes and tracks steps

### Commit 13: Integration with ProjectScaffolder & Real-time Progress Display

**Goal**: Integrate progress tracking into ProjectScaffolder.generate_project() with real-time CLI feedback.

**Requirements**:
1. Update `src/crewforge/core/scaffolding.py` with:
   - Progress tracking integration in `generate_project()` method
   - Step-by-step progress callbacks during generation pipeline
   - LLM streaming integration for real-time AI response display
2. Enhance CLI with progress display:
   - Real-time step progress indicators ("Step 3/7: Generating agents...")
   - Percentage completion display
   - LLM response streaming to show active generation
3. Add comprehensive testing:
   - Progress tracking unit tests
   - Integration tests with ProjectScaffolder
   - Mock streaming and callback verification

**Validation**: `uv run crewforge generate "Create test crew"` → Shows real-time progress with step tracking, percentage completion, and streaming LLM responses during generation ✅ [2025-09-22 15:30]

**Implementation Details**:
- Enhanced ProjectScaffolder.generate_project() with optional progress_tracker and streaming_callbacks parameters
- Added step definitions for complete generation pipeline: analyze_prompt, generate_agents, generate_tasks, select_tools, create_scaffold, populate_files
- Integrated progress event emission throughout generation process with proper error handling and step completion tracking
- Updated GenerationEngine methods to support streaming callbacks for real-time LLM response display
- Enhanced LLMClient with generate_streaming() method and _execute_with_retry_streaming() for token-by-token streaming support

### Commit 14: Jinja2 Template-Based Prompt System

**Goal**: Refactor LLM prompt management from hardcoded strings to maintainable Jinja2 templates.

**Requirements**:
1. Create template infrastructure:
   - Add `src/crewforge/templates/prompts/` directory structure
   - Create `crewai_system_prompt.j2` template for system prompt
   - Create `user_requirements_prompt.j2` template with variable substitution
2. Update `src/crewforge/core/llm.py` with:
   - Jinja2 Environment initialization for prompt templates
   - Template loading and rendering methods
   - Backwards compatibility with fallback to hardcoded prompts
   - Integration with existing LLMClient API
3. Update documentation:
   - Add requirement to BRIEF.md Core Requirements
   - Update SPEC.md file structure with prompts directory
   - Add commit tracking to ROADMAP.md

**Validation**: Templates load successfully, prompts render with variables, existing functionality maintained with fallback support ✅ [2025-09-22 16:15]

**Implementation Details**:
- Added Jinja2 environment setup in LLMClient constructor with FileSystemLoader for prompts directory
- Created template rendering methods with error handling and fallback to hardcoded prompts
- Maintained backwards compatibility by keeping original function signatures and constant exports
- Added convenient instance methods for system and user prompt generation on LLMClient
- Enhanced error handling with warning logs when templates fail to load, ensuring robust operation
- Added CLI progress callback functions for real-time step tracking with progress percentages and status indicators
- Created streaming callback system for live LLM response display during AI generation phases
- Implemented comprehensive test suite covering progress integration, CLI display, and streaming functionality
- Added error-resilient callback handling to prevent interruption of generation pipeline
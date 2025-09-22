# MVP Technical Specification: CrewForge CLI Tool

## Core Requirements (from Brief)

**MVP Requirements:**
- Parse natural language prompts
- Leverage `crewai create crew <name>` for project scaffolding
- Using agentic AI, Generate agent definitions (roles, goals, backstories) from prompt context
- Using agentic AI, Generate task definitions (descriptions, expected outputs) from prompt context
- Select and configure tools from CrewAI's library based on requirements
- Wire agents, tasks, and tools into functional crew configuration
- Complete full generation cycle in <10 minutes
- Validate generated projects execute without syntax errors
- Ensure generated crews perform specified business functions
- Save configured agents, tasks, and tools to refine process over time

**Post-MVP Items:**
- Interactive clarification for ambiguous prompts
- Suggest optimal configurations based on detected project patterns
- Handle edge cases through guided specification prompts
- Enable post-generation refinement of agents and tasks

## Technology Stack

**Core Technologies:**
- **Python 3.11+** - Primary development language
- **uv** - Fast Python package manager and dependency resolution
- **LiteLLM** - Unified interface for LLM providers (GPT-4, Claude, etc.)
- **CrewAI** - Target framework for generated projects
- **Click** - CLI framework for command-line interface
- **Pydantic** - Data validation and settings management
- **YAML** - Configuration file format for agent/task templates

**Justification for Non-Standard Choices:**
- **uv over pip**: Significantly faster dependency resolution and installation
- **LiteLLM over direct OpenAI**: Provider flexibility and unified API interface
- **YAML over JSON**: Human-readable configuration files for templates

## Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   CLI Interface     │    │   Generation Engine  │    │   CrewAI Project    │
│   (Click Commands)  │───▶│   (LiteLLM + AI)    │───▶│   (Generated MVP)   │  
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Template Engine   │    │   Validation Suite   │    │   Learning Store    │
│   (Jinja2 + YAML)  │    │   (Syntax + Logic)   │    │   (Saved Configs)   │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

## Data Flow

**Simplest End-to-End User Flow for MVP:**
1. User runs `crewforge generate "Build a content research crew"`
2. Generation Engine (LLM) analyzes prompt and generates agent/task/tool configurations
3. Template Engine populates CrewAI project files with generated configurations
4. CrewAI scaffolding creates project structure via `crewai create crew`
5. Validation Suite checks syntax and basic functionality
6. Learning Store saves successful configuration patterns
7. User receives ready-to-run CrewAI MVP

## System Components

### CLI Interface
**Purpose:** Primary user interaction point for the MVP
**Inputs:** Natural language prompts, command flags (--name, --output)
**Outputs:** Status messages, generated project paths
**Dependencies:** Click framework
**Key Responsibilities:**
- Accept and validate user prompts
- Handle optional project name specification via --name flag
- Support custom output directory specification via --output flag
- Orchestrate generation pipeline
- Provide progress feedback to user
- Handle basic error scenarios

### Generation Engine
**Purpose:** Core agentic AI that creates CrewAI configurations
**Inputs:** Natural language prompt
**Outputs:** Agent definitions, task definitions, tool configurations
**Dependencies:** LiteLLM, CrewAI knowledge patterns, Pydantic models
**Key Responsibilities:**
- Parse natural language prompts for crew requirements
- Generate contextually appropriate agent roles, goals, and backstories
- Create task descriptions with clear expected outputs
- Select optimal tools from CrewAI library
- Ensure agent-task-tool compatibility

### Template Engine
**Purpose:** Transform AI-generated configurations into CrewAI project files
**Inputs:** Generated configurations, CrewAI project template
**Outputs:** Populated Python files (agents.py, tasks.py, tools.py, crew.py)
**Dependencies:** Jinja2 templating, YAML configuration files
**Key Responsibilities:**
- Populate CrewAI project templates with generated content
- Maintain proper Python syntax and CrewAI patterns
- Handle configuration dependencies and imports

### Validation Suite
**Purpose:** Ensure generated projects are syntactically correct and functional
**Inputs:** Generated CrewAI project directory
**Outputs:** Validation results, error reports
**Dependencies:** Python AST parsing, CrewAI runtime
**Key Responsibilities:**
- Check Python syntax validity
- Verify CrewAI configuration compliance
- Test basic crew instantiation

### Learning Store
**Purpose:** Persist successful configurations to improve future generations
**Inputs:** Validated project configurations
**Outputs:** Configuration patterns, success metrics
**Dependencies:** Local file system, JSON/YAML serialization
**Key Responsibilities:**
- Save working agent/task/tool combinations
- Track generation success patterns

### Progress Tracking System
**Purpose:** Provide real-time visibility into long-running generation processes
**Inputs:** Generation step events, LLM token streams, process status updates
**Outputs:** Progress indicators, step tracking, completion percentages, streaming content
**Dependencies:** Event-driven architecture, async streaming capabilities
**Key Responsibilities:**
- Track current generation step and total progress percentage
- Stream LLM responses token-by-token for real-time feedback
- Display step descriptions and time estimation
- Handle progress callback integration with existing components

## File Structure

```
crewforge/
├── pyproject.toml              # uv dependency management
├── README.md                   # Installation and usage
├── src/
│   └── crewforge/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py         # Click CLI commands
│       ├── core/
│       │   ├── __init__.py
│       │   ├── generator.py        # Agentic AI generation
│       │   ├── templates.py        # Template population
│       │   └── validator.py        # Project validation
│       ├── models/
│       │   ├── __init__.py
│       │   ├── agent.py            # Agent configuration models
│       │   ├── task.py             # Task configuration models
│       │   └── crew.py             # Crew configuration models
│       ├── templates/
│       │   ├── agents.py.j2        # Jinja2 agent template
│       │   ├── tasks.py.j2         # Jinja2 task template
│       │   ├── tools.py.j2         # Jinja2 tools template
│       │   ├── crew.py.j2          # Jinja2 crew template
│       │   └── prompts/
│       │       ├── crewai_system_prompt.j2    # System prompt template
│       │       └── user_requirements_prompt.j2 # User prompt template
│       └── storage/
│           ├── __init__.py
│           └── learning.py         # Configuration persistence
├── tests/
│   ├── __init__.py
│   ├── test_cli.py
│   ├── test_generator.py
│   ├── test_validator.py
│   └── fixtures/
│       └── sample_prompts.py
└── examples/
    └── generated_crews/           # Sample outputs
```

## Integration Patterns

**MVP Usage Pattern:**
```bash
# Basic generation
crewforge generate "Create a content research crew that finds and summarizes articles"

# With project name  
crewforge generate --name research-crew "Build agents that research competitors"

# With custom output directory
crewforge generate --output ./projects/research-crew "Build agents that research competitors"

# With both name and output directory
crewforge generate --name research-crew --output ./my-projects "Build agents that research competitors"
```

**Core Integration Flow:**
1. CLI accepts prompt → Generation Engine (LLM) → Template Engine → CrewAI Project
2. Validation Suite ensures quality → Learning Store captures patterns  
3. User receives functional CrewAI MVP ready for customization

**Post-MVP Extensions (Future Capabilities):**
- Interactive clarification for ambiguous prompts
- Optimal configuration suggestions based on patterns
- Edge case handling through guided specification
- Post-generation refinement of agents and tasks

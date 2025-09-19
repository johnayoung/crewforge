# CrewForge Examples

This directory contains example scripts and demonstrations of CrewForge features.

## Agent Generation Demo

**File:** `agent_generation_demo.py`

Demonstrates the intelligent agent role and backstory generation feature that uses liteLLM to create contextual, domain-specific agent configurations.

### Features Shown:
- Generation of intelligent agent roles based on project context
- Domain-specific customization (content marketing, e-commerce)
- Contextual goal and backstory creation
- Integration with the Enhancement Engine

### Running the Demo:

```bash
cd /path/to/crewforge
python -m examples.agent_generation_demo
```

### What It Does:

1. Creates sample project specifications for different domains
2. Uses the Enhancement Engine to generate intelligent agent roles
3. Shows how agents are customized based on:
   - Project domain (content marketing, e-commerce)
   - Agent role (researcher, analyst, optimizer)
   - Project context and requirements

### Example Output:

The demo generates professional agent configurations like:

- **Senior Content Research Specialist** for content marketing projects
- **E-commerce Data Analyst** for e-commerce optimization 
- **Conversion Rate Optimization Specialist** for UX improvement

Each agent includes:
- Professional role title
- Clear, actionable goal statement  
- Compelling backstory with relevant experience

### Integration with CrewForge:

This feature is integrated into the main CrewForge CLI workflow and enhances generated CrewAI projects with intelligent, context-aware agent configurations instead of generic templates.
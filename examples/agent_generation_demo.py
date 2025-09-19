"""
Demo script for intelligent agent role and backstory generation.

This script demonstrates how the new feature generates intelligent
agent roles and backstories using liteLLM based on project specifications.
"""

import asyncio
import json
import logging
from pathlib import Path

from crewforge.enhancement import EnhancementEngine
from crewforge.llm import LLMClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_agent_generation():
    """Demonstrate intelligent agent generation."""
    print("🤖 CrewForge Intelligent Agent Generation Demo")
    print("=" * 50)

    # Sample project specifications
    project_specs = [
        {
            "project_name": "content_research_team",
            "description": "A team that researches and analyzes content trends",
            "project_type": "research",
            "domain": "content marketing",
            "agents": [
                {
                    "role": "researcher",
                    "description": "Finds and gathers information about trending content",
                },
                {
                    "role": "analyst",
                    "description": "Analyzes collected data and identifies patterns",
                },
            ],
        },
        {
            "project_name": "ecommerce_optimizer",
            "description": "Optimize e-commerce store performance and conversion rates",
            "project_type": "optimization",
            "domain": "e-commerce",
            "agents": [
                {
                    "role": "data_analyst",
                    "description": "Analyzes e-commerce metrics and KPIs",
                },
                {
                    "role": "conversion_optimizer",
                    "description": "Optimizes conversion funnels and user experience",
                },
            ],
        },
    ]

    # Create enhancement engine
    try:
        enhancement_engine = EnhancementEngine()
        print(f"✅ Enhancement Engine initialized")

        # Create mock LLM client (in real usage, this would connect to actual LLM)
        llm_client = MockLLMClient()  # type: ignore
        print(f"✅ LLM Client created (mock)")

        # Generate agents for each project
        for i, project_spec in enumerate(project_specs, 1):
            print(f"\n📋 Project {i}: {project_spec['project_name']}")
            print(f"   Domain: {project_spec['domain']}")
            print(f"   Type: {project_spec['project_type']}")

            try:
                generated_agents = await enhancement_engine.generate_agent_roles(
                    project_spec=project_spec, llm_client=llm_client
                )

                print(f"✅ Generated {len(generated_agents)} agent roles:")
                for agent in generated_agents:
                    print(f"\n   🎭 {agent['role']}")
                    print(f"      Goal: {agent['goal']}")
                    print(f"      Backstory: {agent['backstory'][:100]}...")

            except Exception as e:
                print(f"❌ Failed to generate agents: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

    print(f"\n🎉 Demo completed!")


class MockLLMClient:
    """Mock LLM client for demonstration purposes."""

    def __init__(self):
        # Pre-defined responses for different agent roles
        self.responses = {
            "researcher": {
                "content marketing": {
                    "role": "Senior Content Research Specialist",
                    "goal": "Identify emerging content trends, viral patterns, and audience engagement opportunities in the content marketing landscape",
                    "backstory": "You are a seasoned content researcher with 8+ years of experience analyzing digital content performance. You have worked with Fortune 500 companies to identify viral content patterns and predict trending topics. Your data-driven approach has helped brands increase engagement by 300% through strategic content discovery.",
                },
                "e-commerce": {
                    "role": "E-commerce Market Research Analyst",
                    "goal": "Research product trends, competitor strategies, and customer behavior patterns to inform e-commerce optimization",
                    "backstory": "You are an experienced market research professional specializing in e-commerce analytics. With a background in retail analytics and consumer behavior, you excel at identifying market opportunities and competitive advantages.",
                },
            },
            "analyst": {
                "content marketing": {
                    "role": "Content Performance Analyst",
                    "goal": "Analyze content metrics, engagement patterns, and performance data to provide actionable insights for content strategy optimization",
                    "backstory": "You are a detail-oriented data analyst with expertise in content marketing analytics. Your background in statistics and digital marketing allows you to transform raw engagement data into strategic recommendations. You have helped over 50 brands optimize their content strategies through data-driven insights.",
                }
            },
            "data_analyst": {
                "e-commerce": {
                    "role": "E-commerce Data Analyst",
                    "goal": "Analyze e-commerce metrics, conversion funnels, and customer behavior data to identify optimization opportunities",
                    "backstory": "You are a skilled data analyst with 6+ years of experience in e-commerce analytics. You specialize in conversion rate optimization and have helped online stores increase revenue by 40% through data-driven insights and A/B testing strategies.",
                }
            },
            "conversion_optimizer": {
                "e-commerce": {
                    "role": "Conversion Rate Optimization Specialist",
                    "goal": "Optimize e-commerce conversion funnels, user experience, and checkout processes to maximize sales and customer satisfaction",
                    "backstory": "You are a CRO expert with proven expertise in e-commerce optimization. With a psychology background and deep understanding of user behavior, you have increased conversion rates for over 100 e-commerce sites through strategic UX improvements and testing methodologies.",
                }
            },
        }

    async def complete(self, prompt, **kwargs):
        """Mock completion that returns pre-defined responses based on prompt content."""
        prompt_lower = prompt.lower()

        # Extract role and domain from prompt
        role = None
        domain = None

        if "researcher" in prompt_lower:
            role = "researcher"
        elif "analyst" in prompt_lower and "data" not in prompt_lower:
            role = "analyst"
        elif "data_analyst" in prompt_lower or "data analyst" in prompt_lower:
            role = "data_analyst"
        elif "conversion_optimizer" in prompt_lower or "conversion" in prompt_lower:
            role = "conversion_optimizer"

        if "content marketing" in prompt_lower:
            domain = "content marketing"
        elif "e-commerce" in prompt_lower or "ecommerce" in prompt_lower:
            domain = "e-commerce"

        # Get appropriate response
        if (
            role
            and domain
            and role in self.responses
            and domain in self.responses[role]
        ):
            response_data = self.responses[role][domain]
        else:
            # Fallback response
            role_name = role.replace("_", " ").title() if role else "Professional"
            domain_name = domain if domain else "general"
            response_data = {
                "role": f"Professional {role_name}",
                "goal": f"Execute {role_name.lower()} responsibilities effectively in the {domain_name} domain",
                "backstory": f"You are an experienced professional specializing in {role_name.lower()} with expertise in {domain_name}.",
            }

        return json.dumps(response_data)


if __name__ == "__main__":
    asyncio.run(demo_agent_generation())

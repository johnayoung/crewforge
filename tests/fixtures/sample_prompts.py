"""Sample prompts and configurations for testing CrewForge generation pipeline."""

from typing import Dict, Any, List

# Sample prompts for testing various crew generation scenarios
SAMPLE_PROMPTS = {
    "simple_research": {
        "prompt": "Create a content research crew that finds and summarizes articles",
        "expected_agents": 2,
        "expected_tasks": 2,
        "expected_roles": ["Content Researcher", "Content Summarizer"],
    },
    "complex_marketing": {
        "prompt": "Build a comprehensive marketing crew that researches competitors, creates content strategies, writes marketing copy, and analyzes campaign performance metrics",
        "expected_agents": 4,
        "expected_tasks": 4,
        "expected_roles": [
            "Market Researcher",
            "Strategy Analyst",
            "Content Writer",
            "Performance Analyst",
        ],
    },
    "customer_service": {
        "prompt": "Create a customer service automation crew with agents for handling inquiries, escalating issues, and generating satisfaction reports",
        "expected_agents": 3,
        "expected_tasks": 3,
        "expected_roles": [
            "Customer Support Agent",
            "Escalation Handler",
            "Satisfaction Analyst",
        ],
    },
    "data_analysis": {
        "prompt": "Build a data analysis crew that collects data from various sources, cleans and processes it, performs statistical analysis, and creates visualization reports",
        "expected_agents": 4,
        "expected_tasks": 4,
        "expected_roles": [
            "Data Collector",
            "Data Processor",
            "Statistical Analyst",
            "Report Generator",
        ],
    },
    "minimal": {
        "prompt": "Simple task automation crew",
        "expected_agents": 1,
        "expected_tasks": 1,
        "expected_roles": ["Task Automator"],
    },
}

# Sample LLM responses for mocking generation engine
MOCK_LLM_RESPONSES = {
    "analyze_prompt": {
        "simple_research": {
            "business_context": "content research and summarization for information gathering and analysis",
            "required_roles": ["researcher", "content_analyst"],
            "objectives": [
                "find relevant articles",
                "extract key information",
                "create summaries",
            ],
            "tools_needed": ["web_search", "text_analysis"],
            "complexity_level": "moderate",
            "estimated_agents": 2,
            "workflow_pattern": "sequential",
        },
        "complex_marketing": {
            "business_context": "comprehensive marketing strategy development and execution",
            "required_roles": [
                "market_researcher",
                "strategist",
                "content_creator",
                "analyst",
            ],
            "objectives": [
                "competitor analysis",
                "strategy development",
                "content creation",
                "performance tracking",
            ],
            "tools_needed": [
                "web_search",
                "data_analysis",
                "content_generation",
                "reporting",
            ],
            "complexity_level": "high",
            "estimated_agents": 4,
            "workflow_pattern": "parallel_with_dependencies",
        },
        "customer_service": {
            "business_context": "automated customer service and support operations",
            "required_roles": ["support_agent", "escalation_handler", "analyst"],
            "objectives": [
                "handle inquiries",
                "escalate complex issues",
                "track satisfaction",
            ],
            "tools_needed": ["communication", "ticketing", "reporting"],
            "complexity_level": "moderate",
            "estimated_agents": 3,
            "workflow_pattern": "event_driven",
        },
    },
    "generate_agents": {
        "simple_research": {
            "agents": [
                {
                    "role": "Content Researcher",
                    "goal": "Find and collect relevant articles and information sources on specified topics using web search and research methodologies",
                    "backstory": "You are an experienced digital researcher with expertise in finding high-quality, credible sources across the internet. You have a keen eye for identifying reliable information and can quickly assess the relevance and quality of content sources.",
                    "tools": ["web_search", "source_validator"],
                },
                {
                    "role": "Content Summarizer",
                    "goal": "Analyze collected content and create concise, accurate summaries that highlight key insights and findings",
                    "backstory": "You are a skilled content analyst and writer who excels at distilling complex information into clear, actionable summaries. You understand how to identify the most important points and present them in an organized, easy-to-understand format.",
                    "tools": ["text_analysis", "summary_generator"],
                },
            ]
        },
        "complex_marketing": {
            "agents": [
                {
                    "role": "Market Researcher",
                    "goal": "Conduct comprehensive competitor analysis and market research to identify opportunities and threats in the target market",
                    "backstory": "You are a seasoned market research professional with deep expertise in competitive analysis, market trends, and consumer behavior. You excel at gathering and synthesizing market intelligence from multiple sources.",
                    "tools": ["web_search", "data_collector", "competitive_analysis"],
                },
                {
                    "role": "Strategy Analyst",
                    "goal": "Develop data-driven marketing strategies based on research findings and market analysis",
                    "backstory": "You are a strategic marketing consultant with extensive experience in developing comprehensive marketing strategies. You excel at translating market research into actionable strategic recommendations.",
                    "tools": [
                        "strategic_planner",
                        "data_analysis",
                        "presentation_builder",
                    ],
                },
                {
                    "role": "Content Writer",
                    "goal": "Create compelling marketing copy and content that aligns with strategic objectives and resonates with target audiences",
                    "backstory": "You are a creative marketing writer with expertise in crafting persuasive copy across multiple channels and formats. You understand how to adapt messaging for different audiences and platforms.",
                    "tools": [
                        "content_generator",
                        "copy_editor",
                        "brand_voice_analyzer",
                    ],
                },
                {
                    "role": "Performance Analyst",
                    "goal": "Monitor and analyze campaign performance metrics to provide insights for optimization and reporting",
                    "backstory": "You are a marketing analytics expert who specializes in measuring campaign effectiveness and ROI. You excel at interpreting data to provide actionable insights for campaign optimization.",
                    "tools": [
                        "analytics_tracker",
                        "reporting_tool",
                        "performance_dashboard",
                    ],
                },
            ]
        },
    },
    "generate_tasks": {
        "simple_research": {
            "tasks": [
                {
                    "description": "Search for and collect relevant articles and information sources on the specified research topic",
                    "expected_output": "A curated list of 5-10 high-quality articles with URLs, titles, authors, and brief relevance descriptions, formatted as a structured report",
                    "agent": "Content Researcher",
                    "tools": ["web_search", "source_validator"],
                },
                {
                    "description": "Analyze the collected articles and create a comprehensive summary highlighting key findings and insights",
                    "expected_output": "A detailed summary report of 500-800 words that synthesizes the main points, identifies key themes, and provides actionable insights from the research",
                    "agent": "Content Summarizer",
                    "tools": ["text_analysis", "summary_generator"],
                    "context": [
                        "Search for and collect relevant articles and information sources on the specified research topic"
                    ],
                },
            ]
        },
        "complex_marketing": {
            "tasks": [
                {
                    "description": "Conduct comprehensive competitor analysis and market research for the specified industry and target market",
                    "expected_output": "A detailed competitive landscape report including competitor profiles, market positioning, pricing strategies, and identified opportunities",
                    "agent": "Market Researcher",
                    "tools": ["web_search", "data_collector", "competitive_analysis"],
                },
                {
                    "description": "Develop comprehensive marketing strategy based on market research findings",
                    "expected_output": "A complete marketing strategy document with target audience definitions, positioning statements, channel recommendations, and tactical plans",
                    "agent": "Strategy Analyst",
                    "tools": [
                        "strategic_planner",
                        "data_analysis",
                        "presentation_builder",
                    ],
                    "context": [
                        "Conduct comprehensive competitor analysis and market research for the specified industry and target market"
                    ],
                },
                {
                    "description": "Create marketing copy and content assets aligned with the developed strategy",
                    "expected_output": "A portfolio of marketing materials including email copy, social media content, blog posts, and advertising copy tailored to the target audience",
                    "agent": "Content Writer",
                    "tools": [
                        "content_generator",
                        "copy_editor",
                        "brand_voice_analyzer",
                    ],
                    "context": [
                        "Develop comprehensive marketing strategy based on market research findings"
                    ],
                },
                {
                    "description": "Analyze campaign performance metrics and provide optimization recommendations",
                    "expected_output": "A performance analysis report with key metrics, trend analysis, and specific recommendations for campaign optimization",
                    "agent": "Performance Analyst",
                    "tools": [
                        "analytics_tracker",
                        "reporting_tool",
                        "performance_dashboard",
                    ],
                    "context": [
                        "Create marketing copy and content assets aligned with the developed strategy"
                    ],
                },
            ]
        },
    },
}

# Sample generated CrewAI project configurations
SAMPLE_CREW_CONFIGS = {
    "simple_research": {
        "name": "content-research-crew",
        "agents": [
            {
                "role": "Content Researcher",
                "goal": "Find and collect relevant articles and information sources on specified topics",
                "backstory": "You are an experienced digital researcher with expertise in finding high-quality sources.",
                "tools": ["web_search"],
            },
            {
                "role": "Content Summarizer",
                "goal": "Create concise and accurate summaries of research findings",
                "backstory": "You are a skilled writer who excels at distilling complex information.",
                "tools": ["text_analysis"],
            },
        ],
        "tasks": [
            {
                "description": "Search for and collect relevant articles on the specified topic",
                "expected_output": "A list of high-quality articles with URLs and descriptions",
                "agent": "Content Researcher",
            },
            {
                "description": "Analyze and summarize the collected articles",
                "expected_output": "A comprehensive summary report with key insights",
                "agent": "Content Summarizer",
                "context": [
                    "Search for and collect relevant articles on the specified topic"
                ],
            },
        ],
        "tools": ["web_search", "text_analysis"],
    }
}

# Test validation criteria
VALIDATION_CRITERIA = {
    "syntax_requirements": [
        "valid_python_syntax",
        "proper_imports",
        "class_definitions_present",
        "method_definitions_complete",
    ],
    "crewai_requirements": [
        "agent_class_inheritance",
        "task_class_inheritance",
        "crew_instantiation",
        "proper_agent_task_wiring",
    ],
    "functionality_requirements": [
        "crew_can_be_imported",
        "agents_can_be_instantiated",
        "tasks_can_be_instantiated",
        "crew_can_be_executed_without_error",
    ],
}


def get_prompt_by_name(name: str) -> Dict[str, Any]:
    """Get sample prompt configuration by name.

    Args:
        name: Name of the sample prompt

    Returns:
        Dictionary with prompt and expected results

    Raises:
        KeyError: If prompt name not found
    """
    if name not in SAMPLE_PROMPTS:
        available = ", ".join(SAMPLE_PROMPTS.keys())
        raise KeyError(f"Prompt '{name}' not found. Available: {available}")

    return SAMPLE_PROMPTS[name]


def get_mock_llm_response(operation: str, prompt_name: str) -> Dict[str, Any]:
    """Get mock LLM response for testing.

    Args:
        operation: LLM operation ('analyze_prompt', 'generate_agents', 'generate_tasks')
        prompt_name: Name of the sample prompt

    Returns:
        Mock LLM response data

    Raises:
        KeyError: If operation or prompt name not found
    """
    if operation not in MOCK_LLM_RESPONSES:
        available_ops = ", ".join(MOCK_LLM_RESPONSES.keys())
        raise KeyError(f"Operation '{operation}' not found. Available: {available_ops}")

    if prompt_name not in MOCK_LLM_RESPONSES[operation]:
        available_prompts = ", ".join(MOCK_LLM_RESPONSES[operation].keys())
        raise KeyError(
            f"Prompt '{prompt_name}' not found for operation '{operation}'. Available: {available_prompts}"
        )

    return MOCK_LLM_RESPONSES[operation][prompt_name]


def get_crew_config(name: str) -> Dict[str, Any]:
    """Get sample crew configuration by name.

    Args:
        name: Name of the sample configuration

    Returns:
        Sample crew configuration

    Raises:
        KeyError: If configuration name not found
    """
    if name not in SAMPLE_CREW_CONFIGS:
        available = ", ".join(SAMPLE_CREW_CONFIGS.keys())
        raise KeyError(f"Configuration '{name}' not found. Available: {available}")

    return SAMPLE_CREW_CONFIGS[name]

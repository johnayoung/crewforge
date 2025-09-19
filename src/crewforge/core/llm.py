"""LiteLLM integration for multi-provider LLM access with retry logic and structured outputs."""

import json
import time
from typing import Any

import litellm
from pydantic import BaseModel, Field, validator


class LLMError(Exception):
    """Custom exception for LLM-related errors."""

    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


class RetryConfig(BaseModel):
    """Configuration for retry logic with exponential backoff."""

    max_attempts: int = Field(default=3, ge=1)
    base_delay: float = Field(default=1.0, ge=0.0)
    max_delay: float = Field(default=60.0, gt=0.0)
    exponential_base: float = Field(default=2.0, gt=1.0)

    @validator("max_delay")
    def max_delay_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_delay must be positive")
        return v

    @validator("base_delay")
    def base_delay_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("base_delay must be non-negative")
        return v


class LLMClient:
    """Wrapper around LiteLLM for multi-provider LLM access with enhanced features."""

    def __init__(self, model: str = "gpt-4", retry_config: RetryConfig | None = None):
        """Initialize LLM client.

        Args:
            model: LLM model identifier (e.g., 'gpt-4', 'claude-3-sonnet-20240229')
            retry_config: Retry configuration for handling API failures
        """
        self.model = model
        self.retry_config = retry_config or RetryConfig()

        # Configure LiteLLM
        self._configure_litellm()

    def _configure_litellm(self) -> None:
        """Configure LiteLLM with appropriate settings."""
        # Set up model routing and API key detection
        # LiteLLM automatically detects API keys from environment variables
        litellm.drop_params = True  # Drop unsupported parameters
        # Note: litellm.set_verbose does not exist, using litellm logging configuration instead

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        use_json_mode: bool = False,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str | dict[str, Any]:
        """Generate response using LLM with retry logic.

        Args:
            system_prompt: System prompt to set context
            user_prompt: User prompt with requirements
            use_json_mode: Whether to request structured JSON output
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response

        Returns:
            String response or parsed JSON dict if use_json_mode=True

        Raises:
            LLMError: If generation fails after all retries
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Build completion arguments
        completion_args = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if use_json_mode:
            completion_args["response_format"] = {"type": "json_object"}

        if max_tokens:
            completion_args["max_tokens"] = max_tokens

        # Execute with retry logic
        return self._execute_with_retry(completion_args, use_json_mode)

    def _execute_with_retry(
        self, completion_args: dict[str, Any], parse_json: bool
    ) -> str | dict[str, Any]:
        """Execute LLM call with exponential backoff retry logic."""
        last_exception = None

        for attempt in range(self.retry_config.max_attempts):
            try:
                response = litellm.completion(**completion_args)

                # Handle response structure safely - type ignore needed for LiteLLM response types
                if not hasattr(response, "choices") or not response.choices:  # type: ignore
                    raise LLMError("Invalid response structure: no choices found")

                choice = response.choices[0]  # type: ignore
                if not hasattr(choice, "message") or not hasattr(choice.message, "content"):  # type: ignore
                    raise LLMError(
                        "Invalid response structure: no message content found"
                    )

                content = choice.message.content  # type: ignore
                if content is None:
                    raise LLMError("Empty response content received")

                if parse_json:
                    try:
                        parsed_result: dict[str, Any] = json.loads(content)
                        return parsed_result
                    except json.JSONDecodeError as e:
                        raise LLMError(
                            f"Failed to parse JSON response: {content}",
                            original_exception=e,
                        ) from e

                return content

            except Exception as e:
                last_exception = e

                # Don't retry on JSON parsing errors
                if isinstance(e, LLMError):
                    raise

                # Calculate delay with exponential backoff
                if (
                    attempt < self.retry_config.max_attempts - 1
                ):  # Don't sleep on last attempt
                    delay = min(
                        self.retry_config.base_delay
                        * (self.retry_config.exponential_base**attempt),
                        self.retry_config.max_delay,
                    )
                    time.sleep(delay)

        # All retries exhausted
        raise LLMError(
            f"Failed after {self.retry_config.max_attempts} attempts: {str(last_exception)}",
            original_exception=last_exception,
        )


# Prompt engineering templates and utilities

CREWAI_SYSTEM_PROMPT = """You are an expert CrewAI project architect. Your role is to analyze natural language requirements and generate comprehensive agent, task, and tool configurations for CrewAI projects.

CrewAI Framework Knowledge:
- Agents have roles (job titles), goals (objectives), and backstories (context/expertise)
- Tasks have descriptions (what to do) and expected_output (format/content of results)
- Tools extend agent capabilities (web search, file operations, APIs, etc.)
- Agents are assigned to tasks based on role compatibility
- Tasks can depend on outputs from other tasks

Your responses must be valid JSON with this structure:
{
  "agents": [
    {
      "role": "Agent Job Title",
      "goal": "Clear objective the agent should achieve",
      "backstory": "Professional background and expertise context"
    }
  ],
  "tasks": [
    {
      "description": "Detailed description of what needs to be done",
      "expected_output": "Specific format and content expected as output"
    }
  ],
  "tools": ["tool_name_1", "tool_name_2"]
}

Generate configurations that are:
1. Aligned with the business requirements
2. Professionally realistic and practical
3. Compatible with CrewAI framework patterns
4. Optimized for the specified use case"""


def format_user_prompt(user_requirements: str) -> str:
    """Format user requirements into structured prompt for LLM.

    Args:
        user_requirements: Natural language description of crew requirements

    Returns:
        Formatted prompt with context and structure
    """
    return f"""Analyze the following requirements and generate a comprehensive CrewAI project configuration:

Requirements: {user_requirements}

Please provide a complete configuration that includes:
1. Appropriate agents with distinct roles that cover all aspects of the requirements
2. Well-defined tasks that utilize the agents effectively
3. Relevant tools from the CrewAI ecosystem that enhance agent capabilities
4. Logical task dependencies and workflow organization

Focus on creating a practical, production-ready crew that can accomplish the specified business objectives."""


def parse_generation_response(response_data: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate LLM response for configuration completeness.

    Args:
        response_data: Parsed JSON response from LLM

    Returns:
        Validated configuration dict

    Raises:
        LLMError: If response is missing required components
    """
    required_keys = ["agents", "tasks", "tools"]

    for key in required_keys:
        if key not in response_data:
            raise LLMError(f"Missing required component in response: {key}")

    # Validate agents structure
    if not isinstance(response_data["agents"], list) or not response_data["agents"]:
        raise LLMError("Agents must be a non-empty list")

    for agent in response_data["agents"]:
        required_agent_keys = ["role", "goal", "backstory"]
        for agent_key in required_agent_keys:
            if agent_key not in agent:
                raise LLMError(f"Agent missing required field: {agent_key}")

    # Validate tasks structure
    if not isinstance(response_data["tasks"], list) or not response_data["tasks"]:
        raise LLMError("Tasks must be a non-empty list")

    for task in response_data["tasks"]:
        required_task_keys = ["description", "expected_output"]
        for task_key in required_task_keys:
            if task_key not in task:
                raise LLMError(f"Task missing required field: {task_key}")

    # Validate tools structure
    if not isinstance(response_data["tools"], list):
        raise LLMError("Tools must be a list")

    return response_data

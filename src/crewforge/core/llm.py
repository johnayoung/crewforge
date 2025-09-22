"""LiteLLM integration for multi-provider LLM access with retry logic and structured outputs."""

import json
import logging
import signal
import time
from typing import Any, Optional

import litellm
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom exception for LLM-related errors."""

    def __init__(
        self,
        message: str,
        original_exception: Exception | None = None,
        error_type: str = "general",
        retry_exhausted: bool = False,
    ):
        super().__init__(message)
        self.original_exception = original_exception
        self.error_type = error_type
        self.retry_exhausted = retry_exhausted


class LLMRateLimitError(LLMError):
    """Raised when API rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: int | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(message, original_exception, "rate_limit")
        self.retry_after = retry_after


class LLMAuthenticationError(LLMError):
    """Raised when API authentication fails."""

    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message, original_exception, "authentication")


class LLMNetworkError(LLMError):
    """Raised when network-related issues occur."""

    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message, original_exception, "network")


class LLMResponseError(LLMError):
    """Raised when response parsing or validation fails."""

    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message, original_exception, "response")


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
        # Enable debug mode for troubleshooting
        # litellm._turn_on_debug()  # Uncomment for debugging

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
            "timeout": 30,  # 30 second timeout to prevent hanging
        }

        if use_json_mode:
            completion_args["response_format"] = {"type": "json_object"}

        if max_tokens:
            completion_args["max_tokens"] = max_tokens

        # Execute with retry logic
        return self._execute_with_retry(completion_args, use_json_mode)

    def generate_streaming(
        self,
        system_prompt: str,
        user_prompt: str,
        streaming_callbacks: Optional[Any] = None,
        use_json_mode: bool = False,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str | dict[str, Any]:
        """Generate response using LLM with streaming support.

        Args:
            system_prompt: System prompt to set context
            user_prompt: User prompt with requirements
            streaming_callbacks: StreamingCallbacks instance for token streaming
            use_json_mode: Whether to request structured JSON output
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response

        Returns:
            String response or parsed JSON dict if use_json_mode=True

        Raises:
            LLMError: If generation fails after all retries
        """
        # For now, if streaming_callbacks is provided, we collect tokens
        # and call the callbacks, but still return the full response
        # In a full implementation, we would use LiteLLM's streaming support

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Build completion arguments
        completion_args = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "timeout": 30,
            "stream": streaming_callbacks
            is not None,  # Enable streaming if callbacks provided
        }

        if use_json_mode:
            completion_args["response_format"] = {"type": "json_object"}

        if max_tokens:
            completion_args["max_tokens"] = max_tokens

        # Execute with retry logic, handling streaming if enabled
        return self._execute_with_retry_streaming(
            completion_args, use_json_mode, streaming_callbacks
        )

    def _execute_with_retry(
        self, completion_args: dict[str, Any], parse_json: bool
    ) -> str | dict[str, Any]:
        """Execute LLM call with exponential backoff retry logic and timeout."""
        last_exception = None

        for attempt in range(self.retry_config.max_attempts):
            try:
                logger.debug(
                    f"LLM API call attempt {attempt + 1}/{self.retry_config.max_attempts}"
                )

                # Use signal-based timeout for more aggressive timeout handling
                def timeout_handler(signum, frame):
                    raise TimeoutError("LLM API call timed out after 30 seconds")

                # Set up signal handler (Unix only)
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)  # 30 second timeout
                except (AttributeError, OSError):
                    # signal.SIGALRM not available on Windows or in some environments
                    pass

                try:
                    response = litellm.completion(**completion_args)
                finally:
                    # Clear the alarm
                    try:
                        signal.alarm(0)
                    except (AttributeError, OSError):
                        pass

                # Handle response structure safely - type ignore needed for LiteLLM response types
                if not hasattr(response, "choices") or not response.choices:  # type: ignore
                    raise LLMResponseError(
                        "Invalid response structure: no choices found"
                    )

                choice = response.choices[0]  # type: ignore
                if not hasattr(choice, "message") or not hasattr(choice.message, "content"):  # type: ignore
                    raise LLMResponseError(
                        "Invalid response structure: no message content found"
                    )

                content = choice.message.content  # type: ignore
                if content is None:
                    raise LLMResponseError("Empty response content received")

                if parse_json:
                    try:
                        parsed_result: dict[str, Any] = json.loads(content)
                        logger.debug("Successfully parsed JSON response")
                        return parsed_result
                    except json.JSONDecodeError as e:
                        raise LLMResponseError(
                            f"Failed to parse JSON response: {content[:200]}...",
                            original_exception=e,
                        ) from e

                logger.debug("Successfully received text response")
                return content

            except Exception as e:
                last_exception = e

                # Categorize errors for better handling
                categorized_error = self._categorize_error(e)

                # Log the error with appropriate level
                log_level = (
                    logging.WARNING
                    if attempt < self.retry_config.max_attempts - 1
                    else logging.ERROR
                )
                logger.log(
                    log_level, f"LLM API error on attempt {attempt + 1}: {str(e)}"
                )

                # Don't retry on certain error types
                if isinstance(
                    categorized_error, (LLMAuthenticationError, LLMResponseError)
                ):
                    logger.error(
                        f"Non-retryable error encountered: {categorized_error}"
                    )
                    raise categorized_error

                # Calculate delay with exponential backoff
                if (
                    attempt < self.retry_config.max_attempts - 1
                ):  # Don't sleep on last attempt
                    delay = min(
                        self.retry_config.base_delay
                        * (self.retry_config.exponential_base**attempt),
                        self.retry_config.max_delay,
                    )

                    # Handle rate limiting with special backoff
                    if (
                        isinstance(categorized_error, LLMRateLimitError)
                        and categorized_error.retry_after
                    ):
                        delay = max(delay, categorized_error.retry_after)

                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)

        # All retries exhausted
        final_error = (
            self._categorize_error(last_exception)
            if last_exception
            else LLMError("Unknown error")
        )
        final_error.retry_exhausted = True

        logger.error(f"LLM API failed after {self.retry_config.max_attempts} attempts")
        raise final_error

    def _execute_with_retry_streaming(
        self,
        completion_args: dict[str, Any],
        parse_json: bool,
        streaming_callbacks: Optional[Any],
    ) -> str | dict[str, Any]:
        """Execute LLM call with streaming support and retry logic."""
        if not streaming_callbacks or not completion_args.get("stream", False):
            # Fall back to regular non-streaming execution
            completion_args["stream"] = False
            return self._execute_with_retry(completion_args, parse_json)

        # For now, simulate streaming by calling the regular method
        # and then calling streaming callbacks with the full response
        # In a real implementation, we would handle actual streaming
        completion_args["stream"] = False  # Disable streaming for now
        result = self._execute_with_retry(completion_args, parse_json)

        # Simulate token streaming by sending the result to callbacks
        if streaming_callbacks and hasattr(streaming_callbacks, "handle_completion"):
            response_text = result if isinstance(result, str) else json.dumps(result)
            streaming_callbacks.handle_completion(response_text)

        return result

    def _categorize_error(self, error: Exception) -> LLMError:
        """Categorize exceptions into appropriate LLMError types."""
        error_str = str(error).lower()

        # Timeout errors
        if isinstance(error, TimeoutError) or any(
            keyword in error_str for keyword in ["timeout", "timed out", "time out"]
        ):
            return LLMNetworkError(
                f"Request timed out: {str(error)}", original_exception=error
            )

        # Authentication errors
        if any(
            keyword in error_str
            for keyword in ["auth", "unauthorized", "invalid key", "api key"]
        ):
            return LLMAuthenticationError(
                f"Authentication failed: {str(error)}", original_exception=error
            )

        # Rate limiting errors
        if any(
            keyword in error_str
            for keyword in ["rate limit", "quota", "too many requests"]
        ):
            # Try to extract retry-after from error message
            retry_after = None
            import re

            match = re.search(r"retry.*?(\d+)", error_str)
            if match:
                retry_after = int(match.group(1))

            return LLMRateLimitError(
                f"Rate limit exceeded: {str(error)}",
                retry_after=retry_after,
                original_exception=error,
            )

        # Network errors
        if any(
            keyword in error_str for keyword in ["connection", "network", "unreachable"]
        ):
            return LLMNetworkError(
                f"Network error: {str(error)}", original_exception=error
            )

        # Response errors (already handled above but for completeness)
        if isinstance(error, LLMResponseError):
            return error

        # Generic LLM error
        return LLMError(f"LLM API error: {str(error)}", original_exception=error)


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

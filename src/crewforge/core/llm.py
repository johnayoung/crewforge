"""LiteLLM integration for multi-provider LLM access with retry logic and structured outputs."""

import json
import logging
import signal
import time
from pathlib import Path
from typing import Any, Optional

import click
import litellm
from jinja2 import Environment, FileSystemLoader, Template
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

    def __init__(
        self,
        model: str = "gpt-4o",
        retry_config: RetryConfig | None = None,
        verbose: bool = False,
    ):
        """Initialize LLM client.

        Args:
            model: LLM model identifier (e.g., 'gpt-4', 'claude-3-sonnet-20240229')
            retry_config: Retry configuration for handling API failures
            verbose: Enable verbose logging of requests and responses
        """
        self.model = model
        self.retry_config = retry_config or RetryConfig()
        self.verbose = verbose

        # Configure LiteLLM
        self._configure_litellm()

        # Initialize Jinja2 environment for prompt templates
        self._setup_prompt_templates()

    def _setup_prompt_templates(self) -> None:
        """Initialize Jinja2 environment for loading prompt templates."""
        # Path to prompt templates directory
        templates_dir = Path(__file__).parent.parent / "templates" / "prompts"

        if not templates_dir.exists():
            logger.warning(f"Prompt templates directory not found: {templates_dir}")
            self.prompt_env = None
            return

        self.prompt_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def _configure_litellm(self) -> None:
        """Configure LiteLLM with appropriate settings."""
        # Set up model routing and API key detection
        # LiteLLM automatically detects API keys from environment variables
        litellm.drop_params = True  # Drop unsupported parameters

    def _log_verbose(self, message_type: str, content: str) -> None:
        """Log verbose output with formatting and truncation.

        Args:
            message_type: Type of message (e.g., 'MODEL', 'SYSTEM_PROMPT', 'USER_PROMPT', 'RESPONSE', 'ERROR')
            content: Content to log (will be truncated and formatted appropriately)
        """
        if not self.verbose:
            return

        # Color coding for different message types
        colors = {
            "MODEL": "blue",
            "SYSTEM_PROMPT": "cyan",
            "USER_PROMPT": "green",
            "RESPONSE": "yellow",
            "TOKEN_COUNT": "magenta",
            "DURATION": "white",
            "ERROR": "red",
        }

        color = colors.get(message_type, "white")

        # Format content based on type
        if message_type in ["SYSTEM_PROMPT", "USER_PROMPT"]:
            # Truncate long prompts intelligently
            formatted_content = self._truncate_text(content, max_length=200)
        elif message_type == "RESPONSE":
            # Try to format as JSON if possible, otherwise truncate
            try:
                parsed = json.loads(content)
                formatted_content = json.dumps(parsed, indent=2)
                # Truncate formatted JSON if too long
                if len(formatted_content) > 500:
                    formatted_content = self._truncate_text(
                        formatted_content, max_length=500
                    )
            except (json.JSONDecodeError, TypeError):
                formatted_content = self._truncate_text(content, max_length=300)
        else:
            formatted_content = str(content)

        # Output with color styling
        try:
            click.echo(
                click.style(f"[LLM {message_type}] {formatted_content}", fg=color)
            )
        except Exception:
            # Fallback to plain text if click styling fails
            print(f"[LLM {message_type}] {formatted_content}")

    def _truncate_text(self, text: str, max_length: int = 200) -> str:
        """Intelligently truncate text showing beginning and end.

        Args:
            text: Text to truncate
            max_length: Maximum length of output

        Returns:
            Truncated text with ellipsis indicator
        """
        if len(text) <= max_length:
            return text

        # Show first and last portions with ellipsis in middle
        part_length = (max_length - 5) // 2  # Account for " ... " in middle
        start = text[:part_length].strip()
        end = text[-part_length:].strip()
        return f"{start} ... {end}"

    def _render_prompt_template(self, template_name: str, **context: Any) -> str:
        """Load and render a prompt template with the given context.

        Args:
            template_name: Name of the template file (e.g., 'system_prompt.j2')
            **context: Template context variables

        Returns:
            Rendered prompt string

        Raises:
            LLMError: If template loading or rendering fails
        """
        if self.prompt_env is None:
            raise LLMError(
                "Prompt templates not available - environment not initialized"
            )

        try:
            template = self.prompt_env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            raise LLMError(
                f"Failed to render prompt template {template_name}: {str(e)}"
            ) from e

    def format_user_prompt(self, user_requirements: str) -> str:
        """Format user requirements using template.

        Args:
            user_requirements: Natural language description of crew requirements

        Returns:
            Formatted user prompt string

        Raises:
            LLMError: If template loading fails
        """
        return self._render_prompt_template(
            "user_requirements_prompt.j2", user_requirements=user_requirements
        )

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

        # Log verbose information before API call
        if self.verbose:
            self._log_verbose("MODEL", f"Using model: {self.model}")
            self._log_verbose("SYSTEM_PROMPT", system_prompt)
            self._log_verbose("USER_PROMPT", user_prompt)

        # Track timing
        start_time = time.time()

        try:
            # Execute with retry logic
            result = self._execute_with_retry(completion_args, use_json_mode)

            # Log successful response
            if self.verbose:
                duration = time.time() - start_time
                self._log_verbose("DURATION", f"{duration:.2f} seconds")

                # Log the response content
                if isinstance(result, dict):
                    self._log_verbose("RESPONSE", json.dumps(result))
                else:
                    self._log_verbose("RESPONSE", str(result))

                # Log token count if available (note: LiteLLM response structure varies by provider)
                # We'll attempt to extract it but won't fail if not available
                self._log_verbose(
                    "TOKEN_COUNT", "Token count not available in current implementation"
                )

            return result

        except Exception as e:
            # Log error in verbose mode
            if self.verbose:
                duration = time.time() - start_time
                self._log_verbose("DURATION", f"{duration:.2f} seconds (failed)")
                self._log_verbose("ERROR", f"API call failed: {str(e)}")
            raise

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

        # Log verbose information before API call
        if self.verbose:
            self._log_verbose(
                "MODEL",
                f"Using model: {self.model} (streaming: {streaming_callbacks is not None})",
            )
            self._log_verbose("SYSTEM_PROMPT", system_prompt)
            self._log_verbose("USER_PROMPT", user_prompt)

        # Track timing
        start_time = time.time()

        try:
            # Execute with retry logic, handling streaming if enabled
            result = self._execute_with_retry_streaming(
                completion_args, use_json_mode, streaming_callbacks
            )

            # Log successful response
            if self.verbose:
                duration = time.time() - start_time
                self._log_verbose("DURATION", f"{duration:.2f} seconds")

                # Log the response content
                if isinstance(result, dict):
                    self._log_verbose("RESPONSE", json.dumps(result))
                else:
                    self._log_verbose("RESPONSE", str(result))

                # Log token count if available
                self._log_verbose(
                    "TOKEN_COUNT", "Token count not available in current implementation"
                )

            return result

        except Exception as e:
            # Log error in verbose mode
            if self.verbose:
                duration = time.time() - start_time
                self._log_verbose("DURATION", f"{duration:.2f} seconds (failed)")
                self._log_verbose("ERROR", f"API call failed: {str(e)}")
            raise

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


def format_user_prompt(
    user_requirements: str, llm_client: LLMClient | None = None
) -> str:
    """Format user requirements into structured prompt for LLM.

    Args:
        user_requirements: Natural language description of crew requirements
        llm_client: Optional LLM client instance for template rendering

    Returns:
        Formatted prompt with context and structure

    Raises:
        LLMError: If template loading fails and no client provided
    """
    if llm_client and hasattr(llm_client, "prompt_env") and llm_client.prompt_env:
        return llm_client._render_prompt_template(
            "user_requirements_prompt.j2", user_requirements=user_requirements
        )
    else:
        raise LLMError(
            "LLM client with prompt templates is required for prompt formatting"
        )


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

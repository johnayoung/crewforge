"""
LLM client module for CrewForge.

Provides unified API access to multiple LLM providers through liteLLM,
with token management, rate limiting, and error handling.
"""

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Union

import litellm
import tiktoken
from jsonschema import validate, ValidationError


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class RateLimitError(LLMError):
    """Raised when rate limits are exceeded."""

    pass


class TokenLimitError(LLMError):
    """Raised when token limits are exceeded."""

    pass


class LLMClient:
    """
    Unified LLM client supporting multiple providers through liteLLM.

    Supports OpenAI, Anthropic, Google, Groq, and SambaNova providers
    with token management, rate limiting, and error handling.
    """

    # Default models for each provider
    DEFAULT_MODELS = {
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-sonnet-20240229",
        "google": "gemini-pro",
        "groq": "mixtral-8x7b-32768",
        "sambanova": "Meta-Llama-3.1-8B-Instruct",
    }

    # Supported providers
    SUPPORTED_PROVIDERS = list(DEFAULT_MODELS.keys())

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        timeout: int = 60,
        max_retries: int = 3,
        requests_per_minute: int = 60,
    ):
        """
        Initialize the LLM client.

        Args:
            provider: LLM provider name (openai, anthropic, google, etc.)
            model: Model name (uses provider default if not specified)
            api_key: API key (uses environment variable if not provided)
            max_tokens: Maximum tokens for responses
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            requests_per_minute: Rate limit for requests per minute
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            raise LLMError(
                f"Unsupported provider: {provider}. Supported: {self.SUPPORTED_PROVIDERS}"
            )

        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries
        self.requests_per_minute = requests_per_minute

        # Set up API key
        self.api_key = api_key or self._get_api_key_from_env()
        if self.api_key:
            self._set_api_key_for_provider()

        # Rate limiting
        self._request_history: List[float] = []

        # Configure liteLLM
        litellm.set_verbose = False

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables."""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "sambanova": "SAMBANOVA_API_KEY",
        }

        env_var = env_vars.get(self.provider)
        if env_var:
            return os.getenv(env_var)
        return None

    def _set_api_key_for_provider(self) -> None:
        """Set the API key for the current provider in environment."""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "sambanova": "SAMBANOVA_API_KEY",
        }

        env_var = env_vars.get(self.provider)
        if env_var and self.api_key:
            os.environ[env_var] = self.api_key

    def count_tokens(self, text: Union[str, List[Dict[str, str]]]) -> int:
        """
        Count tokens in text or messages.

        Args:
            text: String or list of message dictionaries

        Returns:
            Number of tokens
        """
        try:
            # Get appropriate encoding for the model
            if self.provider == "openai":
                encoding = tiktoken.encoding_for_model(self.model)
            else:
                # Use cl100k_base as default for non-OpenAI models
                encoding = tiktoken.get_encoding("cl100k_base")

            if isinstance(text, str):
                return len(encoding.encode(text))
            elif isinstance(text, list):
                # Count tokens for message format
                total_tokens = 0
                for message in text:
                    # Add tokens for message content
                    total_tokens += len(encoding.encode(message.get("content", "")))
                    # Add tokens for role and formatting (approximately 4 per message)
                    total_tokens += 4
                # Add tokens for conversation start/end
                total_tokens += 2
                return total_tokens
            else:
                raise ValueError("Text must be string or list of message dictionaries")

        except Exception as e:
            # Fallback: rough estimation of 4 characters per token
            if isinstance(text, str):
                return len(text) // 4
            elif isinstance(text, list):
                total_chars = sum(len(msg.get("content", "")) for msg in text)
                return total_chars // 4 + len(text) * 6  # Add formatting tokens
            else:
                raise LLMError(f"Failed to count tokens: {str(e)}")

    def check_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> None:
        """
        Check if prompt exceeds token limits.

        Args:
            prompt: The prompt to check

        Raises:
            TokenLimitError: If prompt exceeds maximum tokens
        """
        token_count = self.count_tokens(prompt)
        if token_count > self.max_tokens:
            raise TokenLimitError(
                f"Token limit exceeded: {token_count} tokens > {self.max_tokens} limit"
            )

    def check_rate_limit(self) -> None:
        """
        Check and enforce rate limits.

        Raises:
            RateLimitError: If rate limit is exceeded
        """
        current_time = time.time()
        cutoff_time = current_time - 60  # 1 minute ago

        # Remove old requests from history
        self._request_history = [t for t in self._request_history if t > cutoff_time]

        # Check if we're within rate limits (before adding current request)
        if len(self._request_history) >= self.requests_per_minute:
            raise RateLimitError(
                f"Rate limit exceeded: {len(self._request_history)} requests in last minute"
            )

        # Add current request to history
        self._request_history.append(current_time)

    async def complete(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """
        Complete a prompt using the configured LLM.

        Args:
            prompt: Text prompt or list of messages
            **kwargs: Additional parameters for liteLLM

        Returns:
            Generated text completion

        Raises:
            LLMError: For various LLM-related errors
            RateLimitError: If rate limits exceeded
            TokenLimitError: If token limits exceeded
        """
        # Pre-flight checks
        self.check_rate_limit()
        self.check_token_limit(prompt)

        # Prepare messages format
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        # Prepare parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "timeout": kwargs.get("timeout", self.timeout),
            **kwargs,
        }

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.to_thread(litellm.completion, **params)
                return response.choices[0].message.content

            except Exception as e:
                last_error = e

                # Handle specific liteLLM exceptions
                error_message = str(e)

                if "rate limit" in error_message.lower():
                    raise RateLimitError(
                        f"Provider rate limit exceeded: {error_message}"
                    )
                elif (
                    "context" in error_message.lower()
                    and "exceeded" in error_message.lower()
                ):
                    raise TokenLimitError(f"Context length exceeded: {error_message}")
                elif (
                    "api" in error_message.lower() and "error" in error_message.lower()
                ):
                    raise LLMError(f"API request failed: {error_message}")

                # Check for retryable errors (server errors, network issues)
                retryable_errors = [
                    "internal server error",
                    "service unavailable",
                    "server error",
                    "timeout",
                    "connection",
                    "network",
                    "temporary",
                    "throttling",
                ]

                is_retryable = isinstance(e, (ConnectionError, TimeoutError)) or any(
                    error_term in error_message.lower()
                    for error_term in retryable_errors
                )

                # For retryable errors, retry up to max_retries
                if is_retryable and attempt < self.max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    continue

                # Check for specific API errors that should not retry
                non_retryable_errors = [
                    "invalid api key",
                    "authentication failed",
                    "unauthorized",
                    "quota exceeded",
                    "model not found",
                    "invalid model",
                    "permission denied",
                ]

                if any(
                    error_term in error_message.lower()
                    for error_term in non_retryable_errors
                ):
                    raise LLMError(f"API request failed: {error_message}")

                # For other errors, raise immediately on first attempt
                if attempt == 0:
                    raise LLMError(f"LLM request failed: {error_message}")

        # If we've exhausted all retries
        raise LLMError(f"Max retries exceeded. Last error: {str(last_error)}")

    async def complete_structured(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Complete a prompt and return structured JSON output.

        Args:
            prompt: Text prompt or list of messages
            schema: Optional JSON schema for validation
            **kwargs: Additional parameters for liteLLM

        Returns:
            Parsed JSON response as dictionary

        Raises:
            LLMError: If JSON parsing fails or schema validation fails
        """
        # Add JSON instruction to prompt if it's a string
        if isinstance(prompt, str):
            json_prompt = (
                f"{prompt}\n\nRespond with valid JSON only, no additional text."
            )
            messages = [{"role": "user", "content": json_prompt}]
        else:
            messages = prompt.copy()
            # Add system message for JSON formatting
            system_msg = {
                "role": "system",
                "content": "Respond with valid JSON only, no additional text.",
            }
            messages.insert(0, system_msg)

        # Get completion
        response = await self.complete(messages, **kwargs)

        # Parse JSON
        try:
            parsed_response = json.loads(response.strip())
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse JSON response: {str(e)}")

        # Validate against schema if provided
        if schema:
            try:
                validate(instance=parsed_response, schema=schema)
            except ValidationError as e:
                raise LLMError(f"Response doesn't match schema: {str(e)}")

        return parsed_response

    def update_config(self, **kwargs) -> None:
        """
        Update client configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Handle provider change
        if "provider" in kwargs:
            if kwargs["provider"] not in self.SUPPORTED_PROVIDERS:
                raise LLMError(f"Unsupported provider: {kwargs['provider']}")

            # Update model if not explicitly set
            if "model" not in kwargs:
                self.model = self.DEFAULT_MODELS[self.provider]

            # Update API key configuration
            if self.api_key:
                self._set_api_key_for_provider()

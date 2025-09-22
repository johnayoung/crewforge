"""Common test configuration and fixtures."""

import os
import pytest


def requires_openai_api_key():
    """Skip marker for tests that require a real OpenAI API key."""
    return pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "",
        reason="Test requires OPENAI_API_KEY environment variable",
    )


def requires_crewai_cli():
    """Skip marker for tests that require CrewAI CLI to be installed."""
    import shutil

    return pytest.mark.skipif(
        not shutil.which("crewai"), reason="Test requires crewai CLI to be installed"
    )


def integration_test():
    """Skip marker for integration tests that may be slow or require external dependencies."""
    return pytest.mark.skipif(
        os.getenv("SKIP_INTEGRATION_TESTS", "").lower() in ("1", "true", "yes"),
        reason="Integration tests skipped (set SKIP_INTEGRATION_TESTS=0 to enable)",
    )

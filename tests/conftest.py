"""Pytest configuration and fixtures."""

import pytest
from dotenv import load_dotenv

# Load .env file before tests are collected
# This ensures OPENROUTER_API_KEY is available for skipif checks
load_dotenv()


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests that require real API calls with the 'integration' marker.

    Detection: any test with a skipif marker whose reason mentions 'API_KEY',
    or marked with vllm_offline, or in a class with 'Integration' in its name.

    This allows skipping all real API tests with: uv run pytest -m 'not integration'
    """
    integration_marker = pytest.mark.integration

    for item in items:
        # Already marked — skip
        if item.get_closest_marker("integration"):
            continue

        # Class name contains "Integration"
        if item.cls and "Integration" in item.cls.__name__:
            item.add_marker(integration_marker)
            continue

        # vLLM offline tests are integration tests
        if item.get_closest_marker("vllm_offline"):
            item.add_marker(integration_marker)
            continue

        # Any skipif marker whose reason mentions an API key
        for marker in item.iter_markers("skipif"):
            reason = marker.kwargs.get("reason", "")
            if "API_KEY" in reason:
                item.add_marker(integration_marker)
                break

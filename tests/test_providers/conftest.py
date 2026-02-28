"""Provider test fixtures and markers."""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()

VLLM_TEST_MODEL = os.getenv("VLLM_TEST_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:7775/v1")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "vllm_offline: requires vLLM (server or offline with GPU)")
    config.addinivalue_line("markers", "openai_api: requires OPENAI_API_KEY")


@pytest.fixture(scope="session")
def vllm_offline_client():
    """Session-scoped vLLM client. Uses a running server if available, else offline.

    Set VLLM_BASE_URL to point at a running vLLM server (default: http://localhost:7775/v1).
    Start one with: bash scripts/vllm_server.sh
    """
    # Try server mode first (fast — model already loaded)
    try:
        import httpx
        r = httpx.get(f"{VLLM_BASE_URL}/models", timeout=2)
        if r.is_success:
            from autochecklist.providers.http_client import LLMHTTPClient
            models = r.json().get("data", [])
            model_id = models[0]["id"] if models else VLLM_TEST_MODEL
            print(f"\n  vLLM: using server at {VLLM_BASE_URL} (model={model_id})")
            client = LLMHTTPClient(provider="vllm", base_url=VLLM_BASE_URL)
            client._test_model = model_id
            yield client
            client.close()
            return
    except Exception:
        pass

    # Fallback: offline mode (slow — loads model into GPU)
    pytest.importorskip("vllm", reason="vLLM not installed and no vLLM server running")
    from autochecklist.providers.vllm_offline import VLLMOfflineClient
    print(f"\n  vLLM: no server found, loading {VLLM_TEST_MODEL} offline...")
    client = VLLMOfflineClient(model=VLLM_TEST_MODEL)
    client._test_model = VLLM_TEST_MODEL
    yield client
    client.close()

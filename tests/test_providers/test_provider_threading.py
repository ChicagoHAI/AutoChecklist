"""Tests for provider parameter threading through generators, scorers, refiners, and pipeline."""


class TestGeneratorProviderThreading:
    """Test provider params in ChecklistGenerator subclasses."""

    def test_generator_stores_provider(self):
        """Generator should store provider param for client creation."""
        from autochecklist.generators.instance_level.direct import DirectGenerator
        tick = DirectGenerator(method_name="tick", provider="vllm", base_url="http://localhost:8000/v1")
        assert tick._provider == "vllm"
        assert tick._base_url == "http://localhost:8000/v1"

    def test_generator_stores_injected_client(self):
        """Generator should store injected client and prefer it."""
        from autochecklist.generators.instance_level.direct import DirectGenerator

        class FakeClient:
            pass

        tick = DirectGenerator(method_name="tick", client=FakeClient())
        assert tick._client is not None
        assert isinstance(tick._client, FakeClient)

    def test_generator_defaults_to_openrouter(self):
        """Without explicit provider, should default to openrouter."""
        from autochecklist.generators.instance_level.direct import DirectGenerator
        tick = DirectGenerator(method_name="tick",)
        assert tick._provider == "openrouter"

    def test_generator_get_client_returns_http_for_vllm_server(self):
        """_get_or_create_client() with vllm+base_url should return HTTP client."""
        from autochecklist.generators.instance_level.direct import DirectGenerator
        from autochecklist.providers.http_client import LLMHTTPClient
        tick = DirectGenerator(method_name="tick", provider="vllm", base_url="http://localhost:8000/v1")
        client = tick._get_or_create_client()
        assert isinstance(client, LLMHTTPClient)
        client.close()

    def test_injected_client_takes_precedence(self):
        """Injected client should be returned by _get_or_create_client()."""
        from autochecklist.generators.instance_level.direct import DirectGenerator

        class FakeClient:
            pass

        fake = FakeClient()
        tick = DirectGenerator(method_name="tick", client=fake, provider="openai")
        assert tick._get_or_create_client() is fake


class TestScorerProviderThreading:
    """Test provider params in ChecklistScorer subclasses."""

    def test_scorer_stores_provider(self):
        """Scorer should store provider param."""
        from autochecklist.scorers import BatchScorer
        scorer = BatchScorer(provider="openai", api_key="test-key")
        assert scorer._provider == "openai"

    def test_scorer_stores_api_format(self):
        """Scorer should store api_format for Responses API."""
        from autochecklist.scorers import BatchScorer
        scorer = BatchScorer(api_format="responses")
        assert scorer._api_format == "responses"

    def test_scorer_defaults_to_openrouter(self):
        """Without explicit provider, should default to openrouter."""
        from autochecklist.scorers import BatchScorer
        scorer = BatchScorer()
        assert scorer._provider == "openrouter"


class TestRefinerProviderThreading:
    """Test provider params in ChecklistRefiner subclasses."""

    def test_refiner_stores_provider(self):
        """Refiner should store provider param."""
        from autochecklist.refiners import Deduplicator
        refiner = Deduplicator(provider="vllm", base_url="http://x:8000/v1")
        assert refiner._provider == "vllm"
        assert refiner._base_url == "http://x:8000/v1"


class TestPipelineProviderThreading:
    """Test provider params in pipeline factory and ChecklistPipeline."""

    def test_pipeline_factory_accepts_provider(self):
        """pipeline() should accept provider param."""
        from autochecklist.pipeline import pipeline
        pipe = pipeline("tick", provider="openai", api_key="test-key")
        assert pipe.generator._provider == "openai"

    def test_pipeline_propagates_provider_to_scorer(self):
        """Provider should propagate from pipeline to scorer."""
        from autochecklist.pipeline import pipeline
        pipe = pipeline("tick", provider="openai", api_key="test-key")
        assert pipe.scorer._provider == "openai"

    def test_pipeline_propagates_client_to_all_components(self):
        """Injected client should be shared across generator and scorer."""
        from autochecklist.pipeline import pipeline

        class FakeClient:
            pass

        fake = FakeClient()
        pipe = pipeline("tick", client=fake)
        assert pipe.generator._client is fake
        assert pipe.scorer._client is fake

    def test_pipeline_propagates_base_url(self):
        """base_url should propagate through pipeline."""
        from autochecklist.pipeline import pipeline
        pipe = pipeline("tick", provider="vllm", base_url="http://gpu:8000/v1")
        assert pipe.generator._base_url == "http://gpu:8000/v1"
        assert pipe.scorer._base_url == "http://gpu:8000/v1"

    def test_pipeline_propagates_api_format(self):
        """api_format should propagate through pipeline."""
        from autochecklist.pipeline import pipeline
        pipe = pipeline("tick", provider="openai", api_key="test", api_format="responses")
        assert pipe.generator._api_format == "responses"
        assert pipe.scorer._api_format == "responses"

"""Tests for ChecklistPipeline.

Tests are organized into:
- Unit tests (no API calls): Test pipeline construction and configuration
- Integration tests (with API calls): Test full pipeline execution
"""

import os
import pytest
from unittest.mock import Mock

from autochecklist import (
    pipeline,
    ChecklistPipeline,
    PipelineResult,
    BatchResult,
    DirectGenerator,
    BatchScorer,
    Checklist,
    ChecklistItem,
    Score,
    ItemScore,
    ChecklistItemAnswer,
)


# ============================================================================
# Unit Tests (no API calls)
# ============================================================================


class TestChecklistPipelineInit:
    """Tests for ChecklistPipeline initialization."""

    # -- String-based generator resolution --

    def test_generator_string_resolves(self):
        """ChecklistPipeline(generator='tick') resolves the generator."""
        pipe = ChecklistPipeline(generator="tick")

        assert pipe.is_instance_level
        assert pipe._generator_name == "tick"
        assert pipe.generator.method_name == "tick"
        # No auto scorer — that's pipeline()'s job
        assert pipe.scorer is None

    def test_generator_string_with_scorer(self):
        """Generator string + scorer string both resolve."""
        pipe = ChecklistPipeline(generator="tick", scorer="batch")

        assert pipe.generator.method_name == "tick"
        assert pipe.scorer is not None
        assert pipe.scorer.scoring_method == "batch"

    def test_generator_string_with_generator_model(self):
        """generator_model applied to string-resolved generator."""
        pipe = ChecklistPipeline(generator="tick", generator_model="openai/gpt-4o")

        assert pipe.generator.model == "openai/gpt-4o"

    def test_generator_string_with_both_models(self):
        """Independent generator and scorer models."""
        pipe = ChecklistPipeline(
            generator="tick",
            scorer="batch",
            generator_model="openai/gpt-4o",
            scorer_model="openai/gpt-4o-mini",
        )

        assert pipe.generator.model == "openai/gpt-4o"
        assert pipe.scorer.model == "openai/gpt-4o-mini"

    def test_from_preset_resolves_generator_and_scorer(self):
        """from_preset= resolves both generator and default scorer."""
        pipe = ChecklistPipeline(from_preset="tick")

        assert pipe._generator_name == "tick"
        assert pipe.generator.method_name == "tick"
        assert pipe.scorer is not None
        assert pipe.scorer.scoring_method == "batch"

    def test_from_preset_scorer_override(self):
        """from_preset= default scorer can be overridden."""
        pipe = ChecklistPipeline(from_preset="tick", scorer="weighted")

        assert pipe.scorer.scoring_method == "weighted"

    def test_generator_instance_without_scorer(self):
        """Generator instance without scorer — scorer is None."""
        gen = DirectGenerator(method_name="tick", model="openai/gpt-4o-mini")
        pipe = ChecklistPipeline(generator=gen)

        assert pipe.generator is gen
        assert pipe.scorer is None

    def test_generator_instance_score_raises_without_scorer(self):
        """pipe.score() raises RuntimeError when no scorer configured."""
        gen = DirectGenerator(method_name="tick", model="openai/gpt-4o-mini")
        pipe = ChecklistPipeline(generator=gen)
        checklist = Checklist(
            items=[ChecklistItem(id="1", question="Test?")],
            source_method="tick",
            generation_level="instance",
        )

        with pytest.raises(RuntimeError, match="No scorer configured"):
            pipe.score(checklist, target="test")

    def test_from_preset_and_generator_raises(self):
        """Providing both from_preset and generator raises ValueError."""
        gen = DirectGenerator(method_name="tick", model="openai/gpt-4o-mini")
        with pytest.raises(ValueError, match="Cannot specify both"):
            ChecklistPipeline(from_preset="tick", generator=gen)

    def test_no_generator_raises(self):
        """Must provide generator."""
        with pytest.raises(ValueError, match="Must provide"):
            ChecklistPipeline()

    def test_pipeline_factory_creates_with_scorer(self):
        """pipeline('tick') auto-attaches default scorer."""
        pipe = pipeline("tick")
        assert isinstance(pipe, ChecklistPipeline)
        assert pipe._generator_name == "tick"
        assert pipe.scorer is not None

    # -- Preserved tests using new API --

    def test_init_with_generator_instance(self):
        """Pipeline accepts generator instance."""
        gen = DirectGenerator(method_name="tick", model="openai/gpt-4o-mini")
        pipe = ChecklistPipeline(generator=gen)

        assert pipe.generator is gen
        assert pipe.generator.method_name == "tick"


    def test_init_with_refiners_list(self):
        """Pipeline accepts list of refiner names."""
        pipe = ChecklistPipeline(
            generator="tick",
            refiners=["deduplicator", "tagger"],
        )

        assert len(pipe.refiners) == 2
        assert pipe.refiners[0].refiner_name == "deduplicator"
        assert pipe.refiners[1].refiner_name == "tagger"

    def test_init_custom_scorer(self):
        """Pipeline accepts custom scorer name."""
        pipe = ChecklistPipeline(generator="tick", scorer="weighted")

        assert pipe.scorer.scoring_method == "weighted"

    def test_init_with_generator_model(self):
        """Pipeline passes generator_model to generator."""
        pipe = ChecklistPipeline(
            generator="tick",
            generator_model="openai/gpt-4o-mini",
        )

        assert pipe.generator.model == "openai/gpt-4o-mini"

    def test_unknown_generator_raises(self):
        """Unknown generator name raises KeyError."""
        with pytest.raises(KeyError):
            ChecklistPipeline(generator="unknown")

    def test_is_instance_level_property(self):
        """is_instance_level returns True for instance-level generators."""
        pipe = ChecklistPipeline(generator="tick")
        assert pipe.is_instance_level
        assert not pipe.is_corpus_level

    def test_is_corpus_level_property(self):
        """is_corpus_level returns True for corpus-level generators."""
        pipe = ChecklistPipeline(generator="feedback")
        assert pipe.is_corpus_level
        assert not pipe.is_instance_level

    def test_generator_instance_with_scorer_string(self):
        """Generator instance + scorer name string resolves scorer."""
        gen = DirectGenerator(method_name="tick", model="openai/gpt-4o-mini")
        pipe = ChecklistPipeline(generator=gen, scorer="batch")

        assert pipe.generator is gen
        assert pipe.scorer is not None
        assert pipe.scorer.scoring_method == "batch"

    def test_generator_instance_with_scorer_instance(self):
        """Generator instance + scorer instance both used directly."""
        gen = DirectGenerator(method_name="tick", model="openai/gpt-4o-mini")
        scorer = BatchScorer(model="openai/gpt-4o-mini")
        pipe = ChecklistPipeline(generator=gen, scorer=scorer)

        assert pipe.generator is gen
        assert pipe.scorer is scorer

    def test_generator_kwargs_passthrough(self):
        """generator_kwargs are passed to generator constructor."""
        pipe = ChecklistPipeline(
            generator="tick",
            generator_kwargs={"model": "openai/gpt-4o"},
        )
        assert pipe.generator.model == "openai/gpt-4o"

    def test_scorer_kwargs_passthrough(self):
        """scorer_kwargs are passed to scorer constructor."""
        pipe = ChecklistPipeline(
            generator="tick",
            scorer="batch",
            scorer_kwargs={"model": "openai/gpt-4o-mini"},
        )
        assert pipe.scorer.model == "openai/gpt-4o-mini"


class TestPipelineFactory:
    """Tests for pipeline() factory function."""

    def test_pipeline_creates_instance(self):
        """pipeline() factory creates ChecklistPipeline."""
        pipe = pipeline("tick")

        assert isinstance(pipe, ChecklistPipeline)
        assert pipe.generator.method_name == "tick"

    def test_pipeline_passes_generator_model(self):
        """pipeline('tick', generator_model=...) threads correctly."""
        pipe = pipeline("tick", generator_model="openai/gpt-4o")

        assert pipe.generator.model == "openai/gpt-4o"

    def test_pipeline_passes_scorer_model(self):
        """pipeline('tick', scorer_model=...) threads correctly."""
        pipe = pipeline("tick", scorer_model="openai/gpt-4o-mini")

        assert pipe.scorer.model == "openai/gpt-4o-mini"

    def test_pipeline_with_refiners(self):
        """pipeline() accepts refiners argument."""
        pipe = pipeline("tick", refiners=["deduplicator"])

        assert len(pipe.refiners) == 1



class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    @pytest.fixture
    def sample_checklist(self):
        """Create a sample checklist."""
        return Checklist(
            items=[
                ChecklistItem(id="1", question="Is it good?"),
                ChecklistItem(id="2", question="Is it complete?"),
            ],
            source_method="tick",
            generation_level="instance",
        )

    @pytest.fixture
    def sample_score(self, sample_checklist):
        """Create a sample score."""
        return Score(
            checklist_id=sample_checklist.id,
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="2", answer=ChecklistItemAnswer.NO),
            ],
            total_score=0.5,
            weighted_score=0.6,
            normalized_score=0.55,
            judge_model="test",
            scoring_method="batch",
        )

    def test_result_with_score(self, sample_checklist, sample_score):
        """PipelineResult with score has convenience properties."""
        result = PipelineResult(checklist=sample_checklist, score=sample_score)

        assert result.pass_rate == sample_score.pass_rate
        assert result.weighted_score == 0.6
        assert result.normalized_score == 0.55

    def test_result_without_score(self, sample_checklist):
        """PipelineResult without score returns None for score properties."""
        result = PipelineResult(checklist=sample_checklist, score=None)

        assert result.pass_rate is None
        assert result.weighted_score is None
        assert result.normalized_score is None


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample scores with different pass rates."""
        def make_score(pass_rate):
            # Create items that give desired pass rate
            yes_count = int(pass_rate * 10)
            no_count = 10 - yes_count
            items = [
                ItemScore(item_id=str(i), answer=ChecklistItemAnswer.YES)
                for i in range(yes_count)
            ] + [
                ItemScore(item_id=str(i + yes_count), answer=ChecklistItemAnswer.NO)
                for i in range(no_count)
            ]
            return Score(
                checklist_id="test",
                item_scores=items,
                total_score=pass_rate,
                judge_model="test",
                scoring_method="batch",
            )

        return [make_score(0.8), make_score(0.6), make_score(0.7)]

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        return [
            {"input": "Write a", "target": "A"},
            {"input": "Write b", "target": "B"},
            {"input": "Write c", "target": "C"},
        ]

    def test_macro_pass_rate(self, sample_scores, sample_data):
        """macro_pass_rate calculates correct average."""
        result = BatchResult(scores=sample_scores, data=sample_data)

        # (0.8 + 0.6 + 0.7) / 3 = 0.7
        assert abs(result.macro_pass_rate - 0.7) < 0.01

    def test_micro_pass_rate(self):
        """micro_pass_rate pools all items across examples."""
        # Example A: 2 yes out of 4 items (pass_rate=0.5)
        score_a = Score(
            checklist_id="a",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="2", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="3", answer=ChecklistItemAnswer.NO),
                ItemScore(item_id="4", answer=ChecklistItemAnswer.NO),
            ],
            total_score=0.5,
            judge_model="test",
            scoring_method="batch",
        )
        # Example B: 3 yes out of 3 items (pass_rate=1.0)
        score_b = Score(
            checklist_id="b",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="2", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="3", answer=ChecklistItemAnswer.YES),
            ],
            total_score=1.0,
            judge_model="test",
            scoring_method="batch",
        )
        data = [{"input": "a", "target": "a"}, {"input": "b", "target": "b"}]
        result = BatchResult(scores=[score_a, score_b], data=data)

        # macro = (0.5 + 1.0) / 2 = 0.75
        assert abs(result.macro_pass_rate - 0.75) < 0.001
        # micro = (2 + 3) / (4 + 3) = 5/7 ≈ 0.714
        assert abs(result.micro_pass_rate - 5 / 7) < 0.001

    def test_empty_scores(self):
        """Empty scores returns 0 for metrics."""
        result = BatchResult(scores=[], data=[])

        assert result.macro_pass_rate == 0.0
        assert result.micro_pass_rate == 0.0

    def test_single_score(self, sample_data):
        """Single score returns correct metrics."""
        score = Score(
            checklist_id="test",
            item_scores=[ItemScore(item_id="1", answer=ChecklistItemAnswer.YES)],
            total_score=1.0,
            judge_model="test",
            scoring_method="batch",
        )
        result = BatchResult(scores=[score], data=[sample_data[0]])

        assert result.macro_pass_rate == 1.0
        assert result.micro_pass_rate == 1.0

    def test_mean_score_default_equals_macro_pass_rate(self, sample_scores, sample_data):
        """mean_score equals macro_pass_rate when primary_metric is 'pass' (default)."""
        result = BatchResult(scores=sample_scores, data=sample_data)
        assert result.mean_score == pytest.approx(result.macro_pass_rate)

    def test_mean_score_weighted(self):
        """mean_score averages weighted_score when primary_metric is 'weighted'."""
        score_a = Score(
            checklist_id="a",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="2", answer=ChecklistItemAnswer.NO),
            ],
            total_score=0.5,
            weighted_score=0.8,  # Different from pass_rate
            judge_model="test",
            scoring_method="weighted",
            primary_metric="weighted",
        )
        score_b = Score(
            checklist_id="b",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
            ],
            total_score=1.0,
            weighted_score=0.6,
            judge_model="test",
            scoring_method="weighted",
            primary_metric="weighted",
        )
        result = BatchResult(
            scores=[score_a, score_b],
            data=[{"input": "a", "target": "a"}, {"input": "b", "target": "b"}],
        )
        # mean_score should average weighted_scores: (0.8 + 0.6) / 2 = 0.7
        assert result.mean_score == pytest.approx(0.7)
        # macro_pass_rate still uses pass_rate: (0.5 + 1.0) / 2 = 0.75
        assert result.macro_pass_rate == pytest.approx(0.75)
        # They should differ
        assert result.mean_score != pytest.approx(result.macro_pass_rate)

    def test_mean_score_normalized(self):
        """mean_score averages normalized_score when primary_metric is 'normalized'."""
        score_a = Score(
            checklist_id="a",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="2", answer=ChecklistItemAnswer.NO),
            ],
            total_score=0.5,
            normalized_score=0.65,
            judge_model="test",
            scoring_method="normalized",
            primary_metric="normalized",
        )
        score_b = Score(
            checklist_id="b",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
            ],
            total_score=1.0,
            normalized_score=0.95,
            judge_model="test",
            scoring_method="normalized",
            primary_metric="normalized",
        )
        result = BatchResult(
            scores=[score_a, score_b],
            data=[{"input": "a", "target": "a"}, {"input": "b", "target": "b"}],
        )
        # mean_score should average normalized_scores: (0.65 + 0.95) / 2 = 0.8
        assert result.mean_score == pytest.approx(0.8)

    def test_mean_score_empty(self):
        """mean_score returns 0.0 for empty scores."""
        result = BatchResult(scores=[], data=[])
        assert result.mean_score == 0.0

    def test_to_dataframe(self, sample_scores, sample_data):
        """to_dataframe() returns pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        result = BatchResult(scores=sample_scores, data=sample_data)
        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "input" in df.columns
        assert "target" in df.columns
        assert "pass_rate" in df.columns

    def test_to_jsonl(self, sample_scores, sample_data, tmp_path):
        """to_jsonl() writes JSONL file."""
        import json

        result = BatchResult(scores=sample_scores, data=sample_data)
        output_path = tmp_path / "results.jsonl"

        result.to_jsonl(str(output_path))

        # Verify file contents
        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert "input" in record
            assert "target" in record
            assert "pass_rate" in record
            assert "item_scores" in record


class TestDefaultScorers:
    """Tests for default scorer selection."""

    def test_pipeline_tick_uses_batch(self):
        """pipeline('tick') auto-attaches BatchScorer."""
        pipe = pipeline("tick", api_key="test")
        assert pipe.scorer.scoring_method == "batch"

    def test_pipeline_rlcf_uses_weighted(self):
        """pipeline('rlcf_direct') auto-attaches WeightedScorer."""
        pipe = pipeline("rlcf_direct", api_key="test")
        assert pipe.scorer.scoring_method == "weighted"

    def test_pipeline_rocketeval_uses_normalized(self):
        """pipeline('rocketeval') auto-attaches NormalizedScorer."""
        pipe = pipeline("rocketeval", api_key="test")
        assert pipe.scorer.scoring_method == "normalized"

    def test_pipeline_corpus_level_uses_batch(self):
        """Corpus-level pipeline() auto-attaches BatchScorer."""
        for gen_name in ["feedback", "checkeval", "interacteval"]:
            pipe = pipeline(gen_name, api_key="test")
            assert pipe.scorer.scoring_method == "batch", f"{gen_name} should use batch"

    def test_from_preset_also_auto_attaches_scorer(self):
        """from_preset= also auto-attaches default scorer."""
        pipe = ChecklistPipeline(from_preset="rlcf_direct", api_key="test")
        assert pipe.scorer.scoring_method == "weighted"

    def test_generator_string_no_auto_scorer(self):
        """ChecklistPipeline(generator='tick') does NOT auto-attach scorer."""
        pipe = ChecklistPipeline(generator="tick")
        assert pipe.scorer is None


class TestPipelineRefine:
    """Tests for refine() method."""

    def test_refine_no_refiners(self):
        """refine() returns checklist unchanged when no refiners."""
        pipe = ChecklistPipeline(generator="tick", refiners=[])
        checklist = Checklist(
            items=[ChecklistItem(id="1", question="Test?")],
            source_method="tick",
            generation_level="instance",
        )

        refined = pipe.refine(checklist)

        assert refined is checklist  # Same object

    def test_refine_chains_refiners(self):
        """refine() applies refiners in sequence."""
        # Create mock refiners that track calls
        mock_refiner1 = Mock()
        mock_refiner1.refine.return_value = Mock(spec=Checklist)

        mock_refiner2 = Mock()
        mock_refiner2.refine.return_value = Mock(spec=Checklist)

        pipe = ChecklistPipeline(generator="tick", refiners=[])
        pipe.refiners = [mock_refiner1, mock_refiner2]

        original = Mock(spec=Checklist)
        pipe.refine(original)

        # First refiner gets original
        mock_refiner1.refine.assert_called_once_with(original)
        # Second refiner gets output of first
        mock_refiner2.refine.assert_called_once_with(mock_refiner1.refine.return_value)


class TestPipelineValidation:
    """Tests for explicit validation branches in batch and generation helpers."""

    def test_generate_requires_instruction_for_instance_generators(self):
        pipe = ChecklistPipeline(generator="tick")
        with pytest.raises(ValueError, match="input is required"):
            pipe.generate()

    def test_run_batch_requires_data_or_lists(self):
        pipe = pipeline("tick")
        with pytest.raises(ValueError, match="Provide either 'data'"):
            pipe.run_batch(data=None, inputs=None, targets=None)

    def test_run_batch_rejects_mismatched_instruction_response_lengths(self):
        pipe = pipeline("tick")
        with pytest.raises(ValueError, match="same length"):
            pipe.run_batch(data=None, inputs=["a", "b"], targets=["x"])


# ============================================================================
# Integration Tests (with API calls)
# ============================================================================


# Skip integration tests if no API key
pytestmark_integration = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)


@pytestmark_integration
class TestPipelineIntegration:
    """Integration tests for full pipeline execution."""

    @pytest.fixture
    def tick_pipeline(self):
        """Create TICK pipeline with cheap model."""
        return pipeline("tick", generator_model="openai/gpt-4o-mini", scorer_model="openai/gpt-4o-mini")

    def test_instance_level_full_pipeline(self, tick_pipeline):
        """Full pipeline: generate → score."""
        result = tick_pipeline(
            input="Write a haiku about autumn",
            target="Leaves fall gently down\nCrisp air whispers through the trees\nNature's final bow"
        )

        assert isinstance(result, PipelineResult)
        assert result.checklist is not None
        assert len(result.checklist.items) > 0
        assert result.score is not None
        assert 0 <= result.pass_rate <= 1

    def test_generate_only_no_response(self, tick_pipeline):
        """Generate without scoring."""
        result = tick_pipeline(input="Write a poem")

        assert result.checklist is not None
        assert len(result.checklist.items) > 0
        assert result.score is None
        assert result.pass_rate is None

    def test_generate_method(self, tick_pipeline):
        """generate() returns checklist only."""
        checklist = tick_pipeline.generate(input="Write a limerick")

        assert isinstance(checklist, Checklist)
        assert checklist.source_method == "tick"
        assert len(checklist.items) > 0

    def test_score_method(self, tick_pipeline):
        """score() scores against existing checklist."""
        checklist = tick_pipeline.generate(input="Write a haiku")
        score = tick_pipeline.score(
            checklist=checklist,
            target="Short poem here\nThree lines of beauty\nEnds with peace",
            input="Write a haiku",
        )

        assert isinstance(score, Score)
        assert 0 <= score.pass_rate <= 1

    def test_score_batch(self, tick_pipeline):
        """score_batch() scores multiple responses."""
        checklist = tick_pipeline.generate(input="Write a short sentence")
        responses = [
            "The cat sat on the mat.",
            "Hello world!",
        ]

        scores = tick_pipeline.score_batch(checklist, responses)

        assert len(scores) == 2
        assert all(isinstance(s, Score) for s in scores)

    def test_run_batch_instance_level(self, tick_pipeline):
        """run_batch() generates checklist per instruction."""
        data = [
            {"input": "Say hello", "target": "Hello there!"},
            {"input": "Say goodbye", "target": "Goodbye!"},
        ]

        result = tick_pipeline.run_batch(data)

        assert isinstance(result, BatchResult)
        assert len(result.scores) == 2
        assert len(result.checklists) == 2
        assert result.macro_pass_rate >= 0

    def test_run_batch_with_shared_checklist(self, tick_pipeline):
        """run_batch() uses shared checklist when provided."""
        checklist = tick_pipeline.generate(input="Write a greeting")
        data = [
            {"target": "Hello!"},
            {"target": "Hi there!"},
        ]

        result = tick_pipeline.run_batch(data, checklist=checklist)

        assert result.checklist is checklist
        assert len(result.scores) == 2
        # All scores should reference the shared checklist
        assert all(s.checklist_id == checklist.id for s in result.scores)

    def test_run_batch_separate_lists(self, tick_pipeline):
        """run_batch() accepts separate instruction/response lists."""
        inputs = ["Say hello", "Say goodbye"]
        targets = ["Hello!", "Bye!"]

        result = tick_pipeline.run_batch(
            inputs=inputs,
            targets=targets,
        )

        assert len(result.scores) == 2
        assert result.data[0]["input"] == "Say hello"


@pytestmark_integration
class TestCorpusLevelPipeline:
    """Integration tests for corpus-level pipelines."""

    def test_feedback_pipeline(self):
        """InductiveGenerator pipeline."""
        pipe = pipeline("feedback", generator_model="openai/gpt-4o-mini", scorer_model="openai/gpt-4o-mini")

        # InductiveGenerator expects list of observations
        observations = [
            "Response was too long",
            "Missing concrete examples",
        ]

        result = pipe(observations=observations, target="This is a test response.")

        assert result.checklist is not None
        assert result.checklist.generation_level == "corpus"

"""Tests for register_custom_pipeline, save/load pipeline config, and CLI --config."""

import json
import warnings

import pytest

from autochecklist import (
    DirectGenerator,
    ContrastiveGenerator,
    BatchScorer,
    WeightedScorer,
    ChecklistPipeline,
    list_generators,
    get_generator,
    register_custom_pipeline,
    save_pipeline_config,
    load_pipeline_config,
)
from autochecklist.registry import (
    _generators,
    _scorers,
    _builtin_generators,
)
from autochecklist.pipeline import DEFAULT_SCORERS


# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_PROMPT = (
    "Generate yes/no evaluation questions for the following input:\n\n{input}"
)

SAMPLE_SCORER_PROMPT = (
    "Evaluate whether the target satisfies the question.\n\n"
    "Input: {input}\nTarget: {target}\nQuestion: {question}"
)


@pytest.fixture(autouse=True)
def _cleanup_custom_registrations():
    """Remove any custom registrations after each test."""
    yield
    for name in list(_generators.keys()):
        if name not in _builtin_generators:
            del _generators[name]
    for name in list(_scorers.keys()):
        if name.endswith("_scorer") and name.replace("_scorer", "") not in _builtin_generators:
            del _scorers[name]
    for name in list(DEFAULT_SCORERS.keys()):
        if name not in _builtin_generators:
            del DEFAULT_SCORERS[name]


# ── prompt_text property ──────────────────────────────────────────────────


class TestPromptText:
    def test_direct_generator_prompt_text(self):
        gen = DirectGenerator(custom_prompt=SAMPLE_PROMPT, method_name="test")
        assert gen.prompt_text == SAMPLE_PROMPT


# ── register_custom_pipeline from config ──────────────────────────────────


class TestRegisterFromConfig:
    def test_register_with_string_prompt(self):
        register_custom_pipeline(
            "test_string",
            generator_prompt=SAMPLE_PROMPT,
        )
        assert "test_string" in list_generators()
        cls = get_generator("test_string")
        assert issubclass(cls, DirectGenerator)

    def test_register_with_path(self, tmp_path):
        prompt_file = tmp_path / "my_prompt.md"
        prompt_file.write_text(SAMPLE_PROMPT)

        register_custom_pipeline(
            "test_path",
            generator_prompt=prompt_file,
        )
        assert "test_path" in list_generators()

    def test_register_with_scorer_mode(self):
        register_custom_pipeline(
            "test_with_scorer",
            generator_prompt=SAMPLE_PROMPT,
            scorer_mode="item",
            primary_metric="weighted",
        )
        assert "test_with_scorer" in list_generators()
        assert DEFAULT_SCORERS["test_with_scorer"] == {
            "mode": "item",
            "primary_metric": "weighted",
        }

    def test_register_with_scorer_legacy(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            register_custom_pipeline(
                "test_legacy_scorer",
                generator_prompt=SAMPLE_PROMPT,
                scorer="weighted",
            )
            assert any("deprecated" in str(x.message).lower() for x in w)
        assert DEFAULT_SCORERS["test_legacy_scorer"] == {
            "mode": "item",
            "primary_metric": "weighted",
        }

    def test_register_with_scorer_prompt_inline(self):
        register_custom_pipeline(
            "test_scorer_prompt",
            generator_prompt=SAMPLE_PROMPT,
            scorer_mode="batch",
            scorer_prompt=SAMPLE_SCORER_PROMPT,
        )
        assert "test_scorer_prompt" in list_generators()
        assert DEFAULT_SCORERS["test_scorer_prompt"]["scorer_prompt"] == SAMPLE_SCORER_PROMPT

    def test_register_with_scorer_prompt_builtin(self):
        register_custom_pipeline(
            "test_scorer_builtin",
            generator_prompt=SAMPLE_PROMPT,
            scorer_mode="item",
            scorer_prompt="rlcf",
        )
        assert "test_scorer_builtin" in list_generators()
        assert DEFAULT_SCORERS["test_scorer_builtin"]["scorer_prompt"] == "rlcf"

    def test_scorer_and_scorer_mode_conflict(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            register_custom_pipeline(
                "test_conflict",
                generator_prompt=SAMPLE_PROMPT,
                scorer="batch",
                scorer_mode="item",
            )

    def test_register_without_scorer(self):
        register_custom_pipeline(
            "test_no_scorer",
            generator_prompt=SAMPLE_PROMPT,
        )
        assert "test_no_scorer" not in DEFAULT_SCORERS

    def test_register_contrastive(self):
        prompt = "Compare candidates for:\n\n{input}\n\n{candidates}"
        register_custom_pipeline(
            "test_contrastive",
            generator_prompt=prompt,
            generator_class="contrastive",
        )
        cls = get_generator("test_contrastive")
        assert issubclass(cls, ContrastiveGenerator)


# ── register_custom_pipeline from instance ────────────────────────────────


class TestRegisterFromInstance:
    def test_register_from_pipeline(self):
        pipe = ChecklistPipeline(
            generator=DirectGenerator(
                custom_prompt=SAMPLE_PROMPT, method_name="orig"
            ),
            scorer=WeightedScorer(),
        )
        register_custom_pipeline("test_from_pipe", pipe)

        assert "test_from_pipe" in list_generators()
        scorer_config = DEFAULT_SCORERS["test_from_pipe"]
        assert isinstance(scorer_config, dict)
        assert scorer_config["mode"] == "item"
        assert scorer_config["primary_metric"] == "weighted"

    def test_register_from_pipeline_no_scorer(self):
        pipe = ChecklistPipeline(
            generator=DirectGenerator(
                custom_prompt=SAMPLE_PROMPT, method_name="orig"
            ),
        )
        register_custom_pipeline("test_pipe_no_scorer", pipe)

        assert "test_pipe_no_scorer" in list_generators()

    def test_extracted_prompt_matches(self):
        pipe = ChecklistPipeline(
            generator=DirectGenerator(
                custom_prompt=SAMPLE_PROMPT, method_name="orig"
            ),
        )
        register_custom_pipeline("test_prompt_match", pipe)

        # Instantiate the registered generator and check prompt
        cls = get_generator("test_prompt_match")
        instance = cls()
        assert instance.prompt_text == SAMPLE_PROMPT


# ── Override protection ───────────────────────────────────────────────────


class TestOverrideProtection:
    def test_cannot_override_builtin(self):
        with pytest.raises(ValueError, match="Cannot override built-in"):
            register_custom_pipeline("tick", generator_prompt=SAMPLE_PROMPT)

    def test_force_override_builtin_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            register_custom_pipeline(
                "tick", generator_prompt=SAMPLE_PROMPT, force=True
            )
            assert len(w) == 1
            assert "Overriding built-in" in str(w[0].message)

    def test_can_override_custom_silently(self):
        register_custom_pipeline("test_overridable", generator_prompt=SAMPLE_PROMPT)
        # Second registration should succeed without error
        register_custom_pipeline(
            "test_overridable",
            generator_prompt=SAMPLE_PROMPT + "\nv2",
        )

    def test_error_without_pipeline_or_prompt(self):
        with pytest.raises(ValueError, match="Must provide"):
            register_custom_pipeline("test_empty")

    def test_error_with_both_pipeline_and_prompt(self):
        pipe = ChecklistPipeline(
            generator=DirectGenerator(
                custom_prompt=SAMPLE_PROMPT, method_name="orig"
            ),
        )
        with pytest.raises(ValueError, match="not both"):
            register_custom_pipeline(
                "test_both", pipeline=pipe, generator_prompt=SAMPLE_PROMPT
            )


# ── save/load pipeline config ────────────────────────────────────────────


class TestSaveLoadConfig:
    def test_round_trip(self, tmp_path):
        register_custom_pipeline(
            "test_roundtrip",
            generator_prompt=SAMPLE_PROMPT,
            scorer_mode="batch",
            primary_metric="pass",
        )

        config_path = tmp_path / "config.json"
        save_pipeline_config("test_roundtrip", config_path)

        # Verify JSON structure
        config = json.loads(config_path.read_text())
        assert config["name"] == "test_roundtrip"
        assert config["generator_class"] == "direct"
        assert config["generator_prompt"] == SAMPLE_PROMPT
        assert config["scorer_mode"] == "batch"
        assert config["primary_metric"] == "pass"

        # Clean up the registration, then reload
        del _generators["test_roundtrip"]
        del DEFAULT_SCORERS["test_roundtrip"]

        loaded_name = load_pipeline_config(config_path)
        assert loaded_name == "test_roundtrip"
        assert "test_roundtrip" in list_generators()
        # Verify scorer config was preserved through round-trip
        assert DEFAULT_SCORERS["test_roundtrip"]["mode"] == "batch"

    def test_round_trip_weighted(self, tmp_path):
        """Test that scorer config with primary_metric survives round-trip."""
        register_custom_pipeline(
            "test_rt_weighted",
            generator_prompt=SAMPLE_PROMPT,
            scorer_mode="item",
            primary_metric="weighted",
        )

        config_path = tmp_path / "config.json"
        save_pipeline_config("test_rt_weighted", config_path)

        config = json.loads(config_path.read_text())
        assert config["scorer_mode"] == "item"
        assert config["primary_metric"] == "weighted"

        # Reload
        del _generators["test_rt_weighted"]
        del DEFAULT_SCORERS["test_rt_weighted"]

        load_pipeline_config(config_path)
        assert DEFAULT_SCORERS["test_rt_weighted"]["mode"] == "item"
        assert DEFAULT_SCORERS["test_rt_weighted"]["primary_metric"] == "weighted"

    def test_round_trip_with_scorer_prompt(self, tmp_path):
        register_custom_pipeline(
            "test_rt_scorer",
            generator_prompt=SAMPLE_PROMPT,
            scorer_mode="batch",
            scorer_prompt=SAMPLE_SCORER_PROMPT,
        )

        config_path = tmp_path / "config.json"
        save_pipeline_config("test_rt_scorer", config_path)

        config = json.loads(config_path.read_text())
        assert config["scorer_prompt"] == SAMPLE_SCORER_PROMPT

    def test_round_trip_with_logprobs(self, tmp_path):
        """Test that use_logprobs is auto-derived from primary_metric='normalized'."""
        register_custom_pipeline(
            "test_rt_logprobs",
            generator_prompt=SAMPLE_PROMPT,
            scorer_mode="item",
            primary_metric="normalized",
        )

        # use_logprobs should be auto-set in the internal config
        assert DEFAULT_SCORERS["test_rt_logprobs"]["use_logprobs"] is True

        config_path = tmp_path / "config.json"
        save_pipeline_config("test_rt_logprobs", config_path)

        config = json.loads(config_path.read_text())
        # use_logprobs is NOT saved to JSON — it's derived from primary_metric
        assert "use_logprobs" not in config
        assert config["primary_metric"] == "normalized"

        del _generators["test_rt_logprobs"]
        del DEFAULT_SCORERS["test_rt_logprobs"]

        load_pipeline_config(config_path)
        # After round-trip, use_logprobs should be auto-derived again
        assert DEFAULT_SCORERS["test_rt_logprobs"]["use_logprobs"] is True

    def test_old_json_with_use_logprobs_ignored(self, tmp_path):
        """Backward compat: old JSON files with use_logprobs are silently ignored."""
        config = {
            "name": "old_logprobs_test",
            "generator_class": "direct",
            "generator_prompt": SAMPLE_PROMPT,
            "scorer_mode": "item",
            "primary_metric": "normalized",
            "use_logprobs": True,  # old format included this
            "capture_reasoning": False,
        }
        config_path = tmp_path / "old_logprobs.json"
        config_path.write_text(json.dumps(config))

        load_pipeline_config(config_path)
        # use_logprobs auto-derived from primary_metric="normalized"
        assert DEFAULT_SCORERS["old_logprobs_test"]["use_logprobs"] is True

    def test_load_old_format(self, tmp_path):
        """Backward compat: old JSON format with 'scorer' key still works."""
        config = {
            "name": "old_format_test",
            "generator_class": "direct",
            "generator_prompt": SAMPLE_PROMPT,
            "scorer": "weighted",
            "scorer_prompt": None,
        }
        config_path = tmp_path / "old_config.json"
        config_path.write_text(json.dumps(config))

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            loaded_name = load_pipeline_config(config_path)

        assert loaded_name == "old_format_test"
        assert "old_format_test" in list_generators()
        # Legacy scorer name gets converted to config dict
        scorer_config = DEFAULT_SCORERS["old_format_test"]
        assert isinstance(scorer_config, dict)
        assert scorer_config["primary_metric"] == "weighted"

    def test_save_unknown_pipeline_raises(self, tmp_path):
        with pytest.raises(KeyError):
            save_pipeline_config("nonexistent_pipeline", tmp_path / "x.json")


# ── CLI --config flag ────────────────────────────────────────────────────


class TestCLIConfig:
    def test_config_flag_exists_on_run(self):
        from autochecklist.cli import main

        # Verify --config is recognized by the parser (will fail on unknown args)
        # We test by checking the parser doesn't crash on --config
        with pytest.raises(SystemExit):
            # Missing --data, but --config should be recognized
            main(["run", "--config", "some.json"])

    def test_config_flag_exists_on_generate(self):
        from autochecklist.cli import main

        with pytest.raises(SystemExit):
            # Missing --data, but --config should be recognized
            main(["generate", "--config", "some.json"])

    def test_config_loads_pipeline(self, tmp_path):
        # Write a config file
        config = {
            "name": "cli_test_pipeline",
            "generator_class": "direct",
            "generator_prompt": SAMPLE_PROMPT,
            "scorer": "batch",
            "scorer_prompt": None,
        }
        config_path = tmp_path / "pipeline.json"
        config_path.write_text(json.dumps(config))

        # Create a minimal data file
        data_path = tmp_path / "data.jsonl"
        data_path.write_text('{"input": "test", "target": "test"}\n')

        # Test that _maybe_load_config works correctly
        from autochecklist.cli import _maybe_load_config
        import argparse

        args = argparse.Namespace(
            config=str(config_path), pipeline=None
        )
        _maybe_load_config(args)

        assert args.pipeline == "cli_test_pipeline"
        assert "cli_test_pipeline" in list_generators()

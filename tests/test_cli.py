"""Tests for autochecklist CLI."""

import os

import pytest

from autochecklist.cli import _load_jsonl, main


# ---------------------------------------------------------------------------
# _load_jsonl tests
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    def test_basic_load(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            '{"input": "Write a haiku", "target": "Leaves fall"}\n'
            '{"input": "Write a poem", "target": "Roses are red"}\n'
        )
        data = _load_jsonl(str(f))
        assert len(data) == 2
        assert data[0]["input"] == "Write a haiku"
        assert data[0]["target"] == "Leaves fall"
        assert data[1]["input"] == "Write a poem"

    def test_key_remapping(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"instruction": "Hello", "response": "Hi there"}\n')
        data = _load_jsonl(str(f), input_key="instruction", target_key="response")
        assert data[0]["input"] == "Hello"
        assert data[0]["target"] == "Hi there"

    def test_preserves_extra_fields(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"input": "Q", "target": "A", "reference": "ref", "id": 42}\n')
        data = _load_jsonl(str(f))
        assert data[0]["reference"] == "ref"
        assert data[0]["id"] == 42

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"input": "Q", "target": "A"}\n\n\n{"input": "Q2", "target": "A2"}\n')
        data = _load_jsonl(str(f))
        assert len(data) == 2


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------

class TestArgParsing:
    def test_run_required_flags(self):
        """--data is required for run."""
        with pytest.raises(SystemExit):
            main(["run"])

    def test_run_parses_all_flags(self, capsys):
        """Verify run subcommand parses all flags without error."""
        # We can't actually run the pipeline without API keys, so we test
        # that parsing succeeds by catching the error from missing data file
        with pytest.raises((SystemExit, FileNotFoundError)):
            main([
                "run",
                "--pipeline", "tick",
                "--data", "/nonexistent.jsonl",
                "--generator-model", "openai/gpt-4o-mini",
                "--scorer-model", "openai/gpt-4o-mini",
                "--scorer", "batch",
                "--provider", "openrouter",
                "--api-format", "chat",
                "--input-key", "instruction",
                "--target-key", "response",
                "--generator-prompt", "my_prompt.md",
                "--scorer-prompt", "my_scorer.md",
                "-o", "out.jsonl",
                "--overwrite",
            ])

    def test_generate_parses_flags(self):
        with pytest.raises((SystemExit, FileNotFoundError)):
            main([
                "generate",
                "--pipeline", "tick",
                "--data", "/nonexistent.jsonl",
                "--generator-model", "openai/gpt-4o-mini",
                "--generator-prompt", "my_prompt.md",
            ])

    def test_score_requires_checklist(self):
        """--checklist is required for score."""
        with pytest.raises(SystemExit):
            main(["score", "--data", "data.jsonl"])

    def test_score_parses_flags(self):
        with pytest.raises((SystemExit, FileNotFoundError)):
            main([
                "score",
                "--data", "/nonexistent.jsonl",
                "--checklist", "checklist.json",
                "--scorer", "item",
                "--scorer-model", "openai/gpt-4o-mini",
                "--scorer-prompt", "my_scorer.md",
            ])

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "autochecklist" in captured.out


# ---------------------------------------------------------------------------
# cmd_list tests
# ---------------------------------------------------------------------------

class TestCmdList:
    def test_list_generators(self, capsys):
        main(["list"])
        captured = capsys.readouterr()
        assert "tick" in captured.out
        assert "Name" in captured.out

    def test_list_generators_explicit(self, capsys):
        main(["list", "--component", "generators"])
        captured = capsys.readouterr()
        assert "tick" in captured.out

    def test_list_scorers(self, capsys):
        main(["list", "--component", "scorers"])
        captured = capsys.readouterr()
        assert "batch" in captured.out

    def test_list_refiners(self, capsys):
        main(["list", "--component", "refiners"])
        captured = capsys.readouterr()
        # Just verify it runs without error and produces output
        assert "Name" in captured.out


# ---------------------------------------------------------------------------
# Integration tests (require API key)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
class TestCLIIntegration:
    @pytest.fixture
    def sample_data(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            '{"input": "Write a haiku about nature", "target": "Leaves fall gently down\\nWhispering winds carry them far\\nNature rests in peace"}\n'
            '{"input": "Write a short greeting", "target": "Hello! How are you doing today?"}\n'
        )
        return str(f)

    def test_run_end_to_end(self, sample_data, tmp_path, capsys):
        output = str(tmp_path / "results.jsonl")
        main([
            "run",
            "--pipeline", "tick",
            "--data", sample_data,
            "--generator-model", "openai/gpt-4o-mini",
            "--scorer-model", "openai/gpt-4o-mini",
            "-o", output,
        ])
        captured = capsys.readouterr()
        assert "Macro pass rate" in captured.err
        assert os.path.exists(output)

    def test_generate_end_to_end(self, sample_data, tmp_path, capsys):
        output = str(tmp_path / "checklists.jsonl")
        main([
            "generate",
            "--pipeline", "tick",
            "--data", sample_data,
            "--generator-model", "openai/gpt-4o-mini",
            "-o", output,
        ])
        captured = capsys.readouterr()
        assert "Generated 2 checklists" in captured.err
        assert os.path.exists(output)

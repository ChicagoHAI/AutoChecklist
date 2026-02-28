"""Tests for prompt loading and format files."""

import pytest

from autochecklist.prompts import load_format


class TestFormatFiles:
    """Verify format prompt files load correctly."""

    @pytest.mark.parametrize("name", [
        "checklist",
        "weighted_checklist",
        "batch_scoring",
        "batch_scoring_reasoned",
        "item_scoring",
        "item_scoring_reasoned",
    ])
    def test_format_files_load(self, name):
        text = load_format(name)
        assert len(text) > 10
        assert "JSON" in text or "json" in text

    def test_load_format_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_format("nonexistent_format")
"""Tests for Checklist.by_category(), save(), and load()."""

import json
import os
import tempfile

import pytest

from autochecklist.models import Checklist, ChecklistItem


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_items():
    """Create a mixed set of categorized and uncategorized items."""
    return [
        ChecklistItem(id="1", question="Is the text coherent?", category="coherence"),
        ChecklistItem(id="2", question="Are transitions smooth?", category="coherence"),
        ChecklistItem(id="3", question="Is the text clear?", category="clarity"),
        ChecklistItem(id="4", question="Is the text fluent?", category="fluency"),
        ChecklistItem(id="5", question="No typos?"),  # category=None
    ]


_SENTINEL = object()

def _make_checklist(items=_SENTINEL):
    return Checklist(
        id="test-checklist",
        items=_make_items() if items is _SENTINEL else items,
        source_method="checkeval",
        generation_level="corpus",
        corpus_description="Test corpus",
        metadata={"task": "summarization"},
    )


# ── by_category() ────────────────────────────────────────────────────────


class TestByCategory:

    def test_groups_items_by_category(self):
        cl = _make_checklist()
        groups = cl.by_category()
        assert set(groups.keys()) == {"coherence", "clarity", "fluency", "ungrouped"}

    def test_coherence_has_two_items(self):
        cl = _make_checklist()
        groups = cl.by_category()
        assert len(groups["coherence"]) == 2

    def test_ungrouped_contains_none_category(self):
        cl = _make_checklist()
        groups = cl.by_category()
        assert len(groups["ungrouped"]) == 1
        assert groups["ungrouped"].items[0].question == "No typos?"

    def test_sub_checklist_inherits_metadata(self):
        cl = _make_checklist()
        groups = cl.by_category()
        sub = groups["coherence"]
        assert sub.source_method == "checkeval"
        assert sub.generation_level == "corpus"
        assert sub.corpus_description == "Test corpus"
        assert sub.metadata["parent_id"] == "test-checklist"
        assert sub.metadata["category"] == "coherence"
        # Original metadata key preserved
        assert sub.metadata["task"] == "summarization"

    def test_sub_checklist_id_format(self):
        cl = _make_checklist()
        groups = cl.by_category()
        assert groups["clarity"].id == "test-checklist_clarity"

    def test_preserves_item_order(self):
        cl = _make_checklist()
        groups = cl.by_category()
        questions = [item.question for item in groups["coherence"].items]
        assert questions == ["Is the text coherent?", "Are transitions smooth?"]

    def test_category_order_matches_first_occurrence(self):
        cl = _make_checklist()
        groups = cl.by_category()
        keys = list(groups.keys())
        assert keys == ["coherence", "clarity", "fluency", "ungrouped"]

    def test_all_same_category(self):
        items = [
            ChecklistItem(id="1", question="Q1", category="quality"),
            ChecklistItem(id="2", question="Q2", category="quality"),
        ]
        cl = _make_checklist(items)
        groups = cl.by_category()
        assert list(groups.keys()) == ["quality"]
        assert len(groups["quality"]) == 2

    def test_all_ungrouped(self):
        items = [
            ChecklistItem(id="1", question="Q1"),
            ChecklistItem(id="2", question="Q2"),
        ]
        cl = _make_checklist(items)
        groups = cl.by_category()
        assert list(groups.keys()) == ["ungrouped"]
        assert len(groups["ungrouped"]) == 2

    def test_empty_checklist(self):
        cl = _make_checklist(items=[])
        groups = cl.by_category()
        assert groups == {}

    def test_sub_checklist_input_preserved(self):
        cl = Checklist(
            id="inst",
            items=[ChecklistItem(id="1", question="Q?", category="c")],
            source_method="tick",
            generation_level="instance",
            input="Write a haiku",
        )
        groups = cl.by_category()
        assert groups["c"].input == "Write a haiku"


# ── save() / load() ──────────────────────────────────────────────────────


class TestSaveLoad:

    def test_round_trip(self):
        cl = _make_checklist()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cl.save(path)
            loaded = Checklist.load(path)
            assert loaded.id == cl.id
            assert len(loaded) == len(cl)
            assert loaded.source_method == cl.source_method
            assert loaded.generation_level == cl.generation_level
            assert loaded.corpus_description == cl.corpus_description
            assert loaded.metadata == cl.metadata
        finally:
            os.remove(path)

    def test_items_preserved(self):
        cl = _make_checklist()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cl.save(path)
            loaded = Checklist.load(path)
            for orig, loaded_item in zip(cl.items, loaded.items):
                assert loaded_item.id == orig.id
                assert loaded_item.question == orig.question
                assert loaded_item.category == orig.category
                assert loaded_item.weight == orig.weight
        finally:
            os.remove(path)

    def test_save_creates_valid_json(self):
        cl = _make_checklist()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cl.save(path)
            with open(path) as f:
                data = json.load(f)
            assert "items" in data
            assert "source_method" in data
            assert isinstance(data["items"], list)
        finally:
            os.remove(path)

    def test_by_category_after_round_trip(self):
        """Verify by_category works on loaded checklists."""
        cl = _make_checklist()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cl.save(path)
            loaded = Checklist.load(path)
            groups = loaded.by_category()
            assert set(groups.keys()) == {"coherence", "clarity", "fluency", "ungrouped"}
            assert len(groups["coherence"]) == 2
        finally:
            os.remove(path)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            Checklist.load("/tmp/nonexistent_checklist_abc123.json")

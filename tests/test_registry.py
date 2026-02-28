"""Tests for component registry."""

import pytest

from autochecklist import (
    BatchScorer,
    DeductiveGenerator,
    Deduplicator,
    InductiveGenerator,
    InteractiveGenerator,
    ItemScorer,
    NormalizedScorer,
    DirectGenerator,
    ContrastiveGenerator,
    Selector,
    Tagger,
    UnitTester,
    WeightedScorer,
    get_generator,
    get_refiner,
    get_scorer,
    list_generators,
    list_refiners,
    list_scorers,
    register_generator,
    register_refiner,
    register_scorer,
)
from autochecklist.registry import (
    _generators,
    _refiners,
    _scorers,
    get_generator_info,
    get_refiner_info,
    get_scorer_info,
    list_generators_with_info,
    list_refiners_with_info,
    list_scorers_with_info,
)


def test_list_functions_include_expected_components():
    assert set(list_generators()) >= {
        "tick",
        "rlcf_direct",
        "rocketeval",
        "feedback",
        "checkeval",
        "interacteval",
    }
    assert set(list_scorers()) >= {"batch", "item", "weighted", "normalized"}
    assert set(list_refiners()) >= {"deduplicator", "tagger", "unit_tester", "selector"}


@pytest.mark.parametrize(
    "getter,name,expected",
    [
        (get_generator, "feedback", InductiveGenerator),
        (get_generator, "checkeval", DeductiveGenerator),
        (get_generator, "interacteval", InteractiveGenerator),
        (get_scorer, "batch", BatchScorer),
        (get_scorer, "item", ItemScorer),
        (get_scorer, "weighted", WeightedScorer),
        (get_scorer, "normalized", NormalizedScorer),
        (get_refiner, "deduplicator", Deduplicator),
        (get_refiner, "tagger", Tagger),
        (get_refiner, "unit_tester", UnitTester),
        (get_refiner, "selector", Selector),
    ],
)
def test_get_functions_return_expected_class(getter, name, expected):
    cls = getter(name)
    assert cls is expected


@pytest.mark.parametrize(
    "name,base_class",
    [
        ("tick", DirectGenerator),
        ("rlcf_direct", DirectGenerator),
        ("rocketeval", DirectGenerator),
        ("rlcf_candidate", ContrastiveGenerator),
        ("rlcf_candidates_only", ContrastiveGenerator),
    ],
)
def test_instance_generator_factories_are_subclasses(name, base_class):
    cls = get_generator(name)
    assert issubclass(cls, base_class)


@pytest.mark.parametrize(
    "getter,bad_name,expected_text",
    [
        (get_generator, "nope", "Unknown generator"),
        (get_scorer, "nope", "Unknown scorer"),
        (get_refiner, "nope", "Unknown refiner"),
    ],
)
def test_get_unknown_component_raises_helpful_error(getter, bad_name, expected_text):
    with pytest.raises(KeyError, match=expected_text):
        getter(bad_name)


@pytest.mark.parametrize(
    "register_fn,list_fn,get_fn,store,name",
    [
        (register_generator, list_generators, get_generator, _generators, "test_custom_gen"),
        (register_scorer, list_scorers, get_scorer, _scorers, "test_custom_scorer"),
        (register_refiner, list_refiners, get_refiner, _refiners, "test_custom_refiner"),
    ],
)
def test_register_decorators_register_and_return_class(register_fn, list_fn, get_fn, store, name):
    @register_fn(name)
    class CustomClass:
        pass

    try:
        assert name in list_fn()
        assert get_fn(name) is CustomClass
        assert CustomClass.__name__ == "CustomClass"
    finally:
        del store[name]


def test_generator_info_contains_required_fields():
    info = get_generator_info("tick")
    assert info["name"] == "tick"
    assert info["level"] == "instance"
    assert "description" in info


def test_scorer_info_contains_required_fields():
    info = get_scorer_info("batch")
    assert info["name"] == "batch"
    assert info["method"] == "batch"
    assert "description" in info


def test_refiner_info_contains_required_fields():
    info = get_refiner_info("deduplicator")
    assert info["name"] == "deduplicator"
    assert "description" in info


def test_list_with_info_shapes():
    generators = list_generators_with_info()
    scorers = list_scorers_with_info()
    refiners = list_refiners_with_info()

    assert all({"name", "level", "description"}.issubset(g) for g in generators)
    assert all({"name", "method", "description"}.issubset(s) for s in scorers)
    assert all({"name", "description"}.issubset(r) for r in refiners)

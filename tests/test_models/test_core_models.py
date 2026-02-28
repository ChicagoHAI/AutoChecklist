"""Tests for core data models."""

import pytest
from pydantic import ValidationError

from autochecklist.models import (
    Score,
    ItemScore,
    ChecklistItemAnswer,
    GeneratedQuestion,
    GeneratedWeightedQuestion,
    ChecklistResponse,
    WeightedChecklistResponse,
    BatchAnswerItem,
    BatchScoringResponse,
    BatchAnswerItemReasoned,
    BatchScoringResponseReasoned,
    ItemScoringResponse,
    ItemScoringResponseReasoned,
    WeightedScoringResponse,
    to_response_format,
)


class TestScoreScaling:
    """Tests for Score.scaled_score_1_5 property (InteractEval-style)."""

    def test_scaled_score_all_yes(self):
        """Pass rate of 1.0 should give scaled score of 5.0."""
        score = Score(
            checklist_id="test",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="2", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="3", answer=ChecklistItemAnswer.YES),
            ],
            total_score=1.0,
            judge_model="test",
            scoring_method="binary",
        )
        assert score.pass_rate == 1.0
        assert score.scaled_score_1_5 == 5.0

    def test_scaled_score_all_no(self):
        """Pass rate of 0.0 should give scaled score of 1.0."""
        score = Score(
            checklist_id="test",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.NO),
                ItemScore(item_id="2", answer=ChecklistItemAnswer.NO),
            ],
            total_score=0.0,
            judge_model="test",
            scoring_method="binary",
        )
        assert score.pass_rate == 0.0
        assert score.scaled_score_1_5 == 1.0

    def test_scaled_score_half(self):
        """Pass rate of 0.5 should give scaled score of 3.0."""
        score = Score(
            checklist_id="test",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="2", answer=ChecklistItemAnswer.NO),
            ],
            total_score=0.5,
            judge_model="test",
            scoring_method="binary",
        )
        assert score.pass_rate == 0.5
        assert score.scaled_score_1_5 == 3.0

    def test_scaled_score_formula(self):
        """Verify the formula: scaled = pass_rate * 4 + 1."""
        # 3 yes, 1 no = 75% pass rate
        score = Score(
            checklist_id="test",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="2", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="3", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="4", answer=ChecklistItemAnswer.NO),
            ],
            total_score=0.75,
            judge_model="test",
            scoring_method="binary",
        )
        assert score.pass_rate == 0.75
        assert score.scaled_score_1_5 == 0.75 * 4 + 1  # = 4.0

    def test_scaled_score_bounds(self):
        """Scaled score should always be between 1 and 5."""
        # All yes
        all_yes = Score(
            checklist_id="test",
            item_scores=[ItemScore(item_id="1", answer=ChecklistItemAnswer.YES)],
            total_score=1.0,
            judge_model="test",
            scoring_method="binary",
        )
        assert 1.0 <= all_yes.scaled_score_1_5 <= 5.0

        # All no
        all_no = Score(
            checklist_id="test",
            item_scores=[ItemScore(item_id="1", answer=ChecklistItemAnswer.NO)],
            total_score=0.0,
            judge_model="test",
            scoring_method="binary",
        )
        assert 1.0 <= all_no.scaled_score_1_5 <= 5.0


class TestResponseSchemas:
    """Tests for LLM response schemas (structured output contracts)."""

    def test_checklist_response_valid(self):
        resp = ChecklistResponse(questions=[
            GeneratedQuestion(question="Is the response accurate?"),
            GeneratedQuestion(question="Does it address the topic?"),
        ])
        assert len(resp.questions) == 2
        assert resp.questions[0].question == "Is the response accurate?"

    def test_checklist_response_empty(self):
        resp = ChecklistResponse(questions=[])
        assert len(resp.questions) == 0

    def test_weighted_checklist_response_valid(self):
        resp = WeightedChecklistResponse(questions=[
            GeneratedWeightedQuestion(question="Is it accurate?", weight=100),
            GeneratedWeightedQuestion(question="Is the tone right?", weight=50),
        ])
        assert resp.questions[0].weight == 100
        assert resp.questions[1].weight == 50

    def test_weighted_question_weight_bounds(self):
        # Valid bounds
        GeneratedWeightedQuestion(question="test?", weight=0)
        GeneratedWeightedQuestion(question="test?", weight=100)
        # Out of bounds
        with pytest.raises(ValidationError):
            GeneratedWeightedQuestion(question="test?", weight=-1)
        with pytest.raises(ValidationError):
            GeneratedWeightedQuestion(question="test?", weight=101)

    def test_batch_scoring_response_valid(self):
        resp = BatchScoringResponse(answers=[
            BatchAnswerItem(question_index=1, answer="YES"),
            BatchAnswerItem(question_index=2, answer="NO"),
        ])
        assert len(resp.answers) == 2
        assert resp.answers[0].answer == "YES"

    def test_batch_answer_item_rejects_invalid(self):
        with pytest.raises(ValidationError):
            BatchAnswerItem(question_index=1, answer="MAYBE")

    # -- ItemScoringResponse (answer only, renamed from WeightedScoringResponse) --

    def test_item_scoring_response_valid(self):
        resp = ItemScoringResponse(answer="NO")
        assert resp.answer == "NO"

    def test_item_scoring_response_rejects_invalid(self):
        with pytest.raises(ValidationError):
            ItemScoringResponse(answer="yes")  # Must be uppercase

    # -- ItemScoringResponseReasoned (answer + reasoning, renamed from old ItemScoringResponse) --

    def test_item_scoring_response_reasoned_valid(self):
        resp = ItemScoringResponseReasoned(
            answer="YES",
            reasoning="The text correctly addresses the requirement.",
        )
        assert resp.answer == "YES"
        assert "correctly" in resp.reasoning

    def test_item_scoring_response_reasoned_rejects_invalid(self):
        with pytest.raises(ValidationError):
            ItemScoringResponseReasoned(answer="NA", reasoning="test")

    # -- BatchScoringResponseReasoned (batch answers with reasoning) --

    def test_batch_scoring_response_reasoned_valid(self):
        resp = BatchScoringResponseReasoned(answers=[
            BatchAnswerItemReasoned(question_index=1, answer="YES", reasoning="Good coverage."),
            BatchAnswerItemReasoned(question_index=2, answer="NO", reasoning="Missing detail."),
        ])
        assert len(resp.answers) == 2
        assert resp.answers[0].reasoning == "Good coverage."
        assert resp.answers[1].answer == "NO"

    def test_batch_answer_item_reasoned_rejects_invalid(self):
        with pytest.raises(ValidationError):
            BatchAnswerItemReasoned(question_index=1, answer="MAYBE", reasoning="test")

    def test_batch_scoring_response_reasoned_round_trip(self):
        """Verify round-trip: dict → model → dict."""
        data = {"answers": [
            {"question_index": 1, "answer": "YES", "reasoning": "Good."},
            {"question_index": 2, "answer": "NO", "reasoning": "Bad."},
        ]}
        resp = BatchScoringResponseReasoned.model_validate(data)
        assert resp.answers[0].reasoning == "Good."
        dumped = resp.model_dump()
        assert dumped == data

    # -- Deprecated aliases --

    def test_weighted_scoring_response_deprecated_alias(self):
        """WeightedScoringResponse is a deprecated alias for ItemScoringResponse."""
        resp = WeightedScoringResponse(answer="NO")
        assert resp.answer == "NO"
        assert WeightedScoringResponse is ItemScoringResponse

    def test_to_response_format_shape(self):
        result = to_response_format(ChecklistResponse, "checklist_response")
        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "checklist_response"
        assert result["json_schema"]["strict"] is True
        assert "schema" in result["json_schema"]

    def test_to_response_format_schema_content(self):
        result = to_response_format(ChecklistResponse, "test")
        schema = result["json_schema"]["schema"]
        assert "properties" in schema
        assert "questions" in schema["properties"]

    def test_to_response_format_weighted(self):
        result = to_response_format(WeightedChecklistResponse, "weighted")
        schema = result["json_schema"]["schema"]
        # Pydantic uses $defs with $ref for nested models
        assert "$defs" in schema
        weighted_q_schema = schema["$defs"]["GeneratedWeightedQuestion"]
        assert "weight" in weighted_q_schema["properties"]

    def test_batch_scoring_from_json(self):
        """Verify round-trip: dict → model → dict."""
        data = {"answers": [
            {"question_index": 1, "answer": "YES"},
            {"question_index": 2, "answer": "NO"},
        ]}
        resp = BatchScoringResponse.model_validate(data)
        assert resp.answers[0].question_index == 1
        dumped = resp.model_dump()
        assert dumped == data

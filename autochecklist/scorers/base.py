"""Unified checklist scorer — single configurable class for all scoring modes."""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config import get_config
from ..models import (
    BatchScoringResponse,
    BatchScoringResponseReasoned,
    Checklist,
    ChecklistItemAnswer,
    ConfidenceLevel,
    ItemScore,
    ItemScoringResponse,
    ItemScoringResponseReasoned,
    Score,
    to_response_format,
)
from ..prompts import PromptTemplate, load_format, load_template
from ..utils.parsing import extract_json

logger = logging.getLogger(__name__)

_VALID_MODES = ("batch", "item")
_VALID_PRIMARY_METRICS = ("pass", "weighted", "normalized")


class ChecklistScorer:
    """Configurable checklist scorer that supports batch and per-item modes.

    Consolidates the former BatchScorer, ItemScorer, WeightedScorer, and
    NormalizedScorer into a single class.  All three aggregate metrics
    (pass_rate, weighted_score, normalized_score) are always computed.

    Args:
        mode: ``"batch"`` (one LLM call) or ``"item"`` (one call per item).
        capture_reasoning: Item mode only — include per-item reasoning.
        use_logprobs: Item mode only — use logprobs for confidence scoring.
        primary_metric: Which metric ``Score.primary_score`` aliases.
            One of ``"pass"`` (pass_rate), ``"weighted"`` (weighted_score),
            ``"normalized"`` (normalized_score).
        custom_prompt: Override the default prompt template (str text or Path).
        model: LLM model identifier.
        temperature: Sampling temperature.
        api_key: Provider API key.
        provider: LLM provider name.
        base_url: Override base URL.
        client: Pre-configured LLM client.
        api_format: API format (``"chat"`` or ``"responses"``).
        max_tokens: Maximum response tokens.
        reasoning_effort: Reasoning effort hint for supported models.

    Example:
        >>> scorer = ChecklistScorer(mode="batch")
        >>> score = scorer.score(checklist, target="The response text...")
        >>> print(score.primary_score)  # uses primary_metric
    """

    def __init__(
        self,
        mode: str = "batch",
        capture_reasoning: bool = False,
        use_logprobs: bool = False,
        primary_metric: str = "pass",
        custom_prompt: Optional[Union[str, Path]] = None,
        # LLM config
        model: Optional[str] = None,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
        api_format: Optional[str] = None,
        max_tokens: int = 2048,
        reasoning_effort: Optional[str] = None,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES!r}, got {mode!r}"
            )
        if primary_metric not in _VALID_PRIMARY_METRICS:
            raise ValueError(
                f"primary_metric must be one of {_VALID_PRIMARY_METRICS!r}, "
                f"got {primary_metric!r}"
            )

        self.mode = mode
        self.capture_reasoning = capture_reasoning
        self.use_logprobs = use_logprobs
        self.primary_metric = primary_metric

        # LLM settings
        config = get_config()
        self.model = model or config.scorer_model.model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self._client = client
        self._provider = provider or config.scorer_model.provider or "openrouter"
        self._base_url = base_url
        self._api_format = api_format or "chat"
        self.reasoning_effort = reasoning_effort

        # Load prompt template and format
        self._template, self._format_text = self._load_prompt_and_format(
            custom_prompt
        )

        # Logprobs availability check (deferred to avoid API call unless needed)
        self._logprobs_available = False
        if self.use_logprobs:
            client_instance = self._get_or_create_client()
            if client_instance.supports_logprobs(self.model):
                self._logprobs_available = True
            else:
                warnings.warn(
                    f"Model '{self.model}' does not support logprobs. "
                    "Scorer will use text-based scoring with confidence=None. "
                    "normalized_score will equal pass_rate (unweighted).",
                    UserWarning,
                    stacklevel=2,
                )

    # ── Public properties ──────────────────────────────────────────────────

    @property
    def scoring_method(self) -> str:
        """Backward-compat scoring method string for Score metadata."""
        if self.mode == "batch":
            return "batch"
        if self.use_logprobs:
            return "normalized"
        if self.primary_metric == "weighted":
            return "weighted"
        if self.capture_reasoning:
            return "item"
        return "item"

    @property
    def prompt_text(self) -> str:
        """The raw prompt template text."""
        return self._template.template

    # ── Core scoring API ───────────────────────────────────────────────────

    def score(
        self,
        checklist: Checklist,
        target: str,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> Score:
        """Score a target response against a checklist.

        Args:
            checklist: The checklist to evaluate against.
            target: The target text to score.
            input: Optional input/context (falls back to checklist.input).
            **kwargs: Additional arguments (ignored).

        Returns:
            Score object with item-level and all aggregate scores.
        """
        if self.mode == "batch":
            return self._score_batch(checklist, target, input)
        else:
            return self._score_items_dispatch(checklist, target, input)

    def score_batch(
        self,
        checklist: Checklist,
        targets: List[str],
        inputs: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Score]:
        """Score multiple targets sequentially."""
        results = []
        for i, target in enumerate(targets):
            input_text = inputs[i] if inputs else None
            results.append(self.score(checklist, target, input_text, **kwargs))
        return results

    # ── Batch scoring ──────────────────────────────────────────────────────

    def _score_batch(
        self,
        checklist: Checklist,
        target: str,
        input: Optional[str],
    ) -> Score:
        """Evaluate ALL checklist items in a single LLM call."""
        inst = input or checklist.input or ""

        checklist_text = "\n".join(
            f"Q{i}: {item.question}"
            for i, item in enumerate(checklist.items, 1)
        )

        prompt = self._template.format(
            input=inst, target=target, checklist=checklist_text,
        )
        full_prompt = prompt + "\n\n" + self._format_text

        if self.capture_reasoning:
            rf = to_response_format(
                BatchScoringResponseReasoned, "batch_scoring_reasoned"
            )
            raw = self._call_model(full_prompt, response_format=rf)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = extract_json(raw)
            validated = BatchScoringResponseReasoned.model_validate(data)

            answer_map = {a.question_index: a.answer for a in validated.answers}
            reasoning_map = {
                a.question_index: a.reasoning for a in validated.answers
            }
        else:
            rf = to_response_format(BatchScoringResponse, "batch_scoring")
            raw = self._call_model(full_prompt, response_format=rf)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = extract_json(raw)
            validated = BatchScoringResponse.model_validate(data)

            answer_map = {a.question_index: a.answer for a in validated.answers}
            reasoning_map = {}

        item_scores = []
        for i, item in enumerate(checklist.items):
            answer_str = answer_map.get(i + 1, "NO")
            answer = (
                ChecklistItemAnswer.YES
                if answer_str == "YES"
                else ChecklistItemAnswer.NO
            )
            reasoning = reasoning_map.get(i + 1)
            item_scores.append(
                ItemScore(item_id=item.id, answer=answer, reasoning=reasoning)
            )

        return self._build_score(checklist, item_scores, raw_response=raw)

    # ── Item scoring ───────────────────────────────────────────────────────

    def _score_items_dispatch(
        self,
        checklist: Checklist,
        target: str,
        input: Optional[str],
    ) -> Score:
        """Evaluate each checklist item individually."""
        inst = input or checklist.input or ""
        item_scores = []

        client = self._get_or_create_client()

        for item in checklist.items:
            prompt = self._template.format(
                input=inst, target=target, question=item.question,
                # normalized template may use {history}
                history="",
            )

            if self.use_logprobs and self._logprobs_available:
                # Logprobs path — confidence scoring
                messages = [{"role": "user", "content": prompt}]
                probs = client.get_logprobs(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                confidence, answer, level = self._interpret_probs(probs)
                item_scores.append(
                    ItemScore(
                        item_id=item.id,
                        answer=answer,
                        confidence=confidence,
                        confidence_level=level,
                    )
                )
            else:
                # Structured output path
                full_prompt = prompt + "\n\n" + self._format_text

                if self.capture_reasoning:
                    rf = to_response_format(
                        ItemScoringResponseReasoned, "item_scoring_reasoned"
                    )
                    raw = self._call_model(full_prompt, response_format=rf)
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = extract_json(raw)
                    validated = ItemScoringResponseReasoned.model_validate(data)
                    answer = (
                        ChecklistItemAnswer.YES
                        if validated.answer == "YES"
                        else ChecklistItemAnswer.NO
                    )
                    item_scores.append(
                        ItemScore(
                            item_id=item.id,
                            answer=answer,
                            reasoning=validated.reasoning,
                        )
                    )
                else:
                    rf = to_response_format(
                        ItemScoringResponse, "item_scoring"
                    )
                    raw = self._call_model(full_prompt, response_format=rf)
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = extract_json(raw)
                    validated = ItemScoringResponse.model_validate(data)
                    answer = (
                        ChecklistItemAnswer.YES
                        if validated.answer == "YES"
                        else ChecklistItemAnswer.NO
                    )
                    item_scores.append(
                        ItemScore(item_id=item.id, answer=answer)
                    )

        return self._build_score(
            checklist,
            item_scores,
            num_calls=len(checklist.items),
        )

    # ── Score assembly ─────────────────────────────────────────────────────

    def _build_score(
        self,
        checklist: Checklist,
        item_scores: List[ItemScore],
        raw_response: Optional[str] = None,
        num_calls: Optional[int] = None,
    ) -> Score:
        """Build a Score with all three aggregate metrics computed."""
        yes_count = sum(
            1 for s in item_scores if s.answer == ChecklistItemAnswer.YES
        )
        total = len(item_scores)
        total_score = yes_count / total if total > 0 else 0.0

        # Weighted score
        weighted_score = self._calculate_weighted_score(item_scores, checklist)

        # Normalized score
        confidences = [
            s.confidence for s in item_scores if s.confidence is not None
        ]
        normalized_score = (
            sum(confidences) / len(confidences) if confidences else total_score
        )

        metadata: Dict[str, Any] = {}
        if raw_response is not None:
            metadata["raw_response"] = raw_response
        if num_calls is not None:
            metadata["num_calls"] = num_calls

        return Score(
            checklist_id=checklist.id,
            item_scores=item_scores,
            total_score=total_score,
            weighted_score=weighted_score,
            normalized_score=normalized_score,
            primary_metric=self.primary_metric,
            judge_model=self.model,
            scoring_method=self.scoring_method,
            metadata=metadata,
        )

    # ── Weighted score calculation ─────────────────────────────────────────

    def _calculate_weighted_score(
        self,
        item_scores: List[ItemScore],
        checklist: Checklist,
    ) -> float:
        """Calculate weighted score: sum(weight_i * score_i) / sum(weight_i)."""
        item_weights = {item.id: item.weight for item in checklist.items}

        weighted_sum = 0.0
        total_weight = 0.0

        for s in item_scores:
            weight = item_weights.get(s.item_id, 1.0)
            total_weight += weight
            if s.answer == ChecklistItemAnswer.YES:
                weighted_sum += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    # ── Logprobs helpers ───────────────────────────────────────────────────

    def _interpret_probs(
        self,
        probs: Dict[str, float],
    ) -> tuple:
        """Interpret Yes/No probabilities into confidence and answer.

        Confidence = P(Yes) / (P(Yes) + P(No))
        """
        yes_prob = probs.get("yes", 0.0)
        no_prob = probs.get("no", 0.0)

        if yes_prob + no_prob < 1e-10:
            return 0.5, ChecklistItemAnswer.NO, ConfidenceLevel.UNSURE

        confidence = yes_prob / (yes_prob + no_prob)
        answer, level = self._confidence_to_level(confidence)
        return confidence, answer, level

    def _confidence_to_level(
        self,
        confidence: float,
    ) -> tuple:
        """Map confidence value to answer and confidence level."""
        if confidence < 0.2:
            return ChecklistItemAnswer.NO, ConfidenceLevel.NO_10
        elif confidence < 0.4:
            return ChecklistItemAnswer.NO, ConfidenceLevel.NO_30
        elif confidence < 0.6:
            return ChecklistItemAnswer.NO, ConfidenceLevel.UNSURE
        elif confidence < 0.8:
            return ChecklistItemAnswer.YES, ConfidenceLevel.YES_70
        else:
            return ChecklistItemAnswer.YES, ConfidenceLevel.YES_90

    # ── Prompt loading ─────────────────────────────────────────────────────

    def _load_prompt_and_format(
        self,
        custom_prompt: Optional[Union[str, Path]],
    ) -> tuple:
        """Load prompt template and format instruction based on mode."""
        # Determine template text
        if custom_prompt is not None:
            if isinstance(custom_prompt, Path):
                template_text = custom_prompt.read_text(encoding="utf-8")
            else:
                template_text = custom_prompt
        else:
            template_text = self._default_template_text()

        template = PromptTemplate(template_text)

        # Determine format text
        format_text = self._default_format_text()

        return template, format_text

    def _default_template_text(self) -> str:
        """Load the default prompt template for the current mode config.

        Simplified: just batch vs item. Pipeline presets override via
        ``scorer_prompt`` key for paper-specific prompts (rlcf, rocketeval).
        """
        if self.mode == "batch":
            return load_template("scoring", "batch")
        return load_template("scoring", "item")

    def _default_format_text(self) -> str:
        """Load the default format instruction for the current mode config."""
        if self.mode == "batch":
            if self.capture_reasoning:
                return load_format("batch_scoring_reasoned")
            return load_format("batch_scoring")
        if self.capture_reasoning:
            return load_format("item_scoring_reasoned")
        return load_format("item_scoring")

    # ── LLM client management ─────────────────────────────────────────────

    def _get_or_create_client(self) -> Any:
        """Get injected client or create one from provider settings."""
        if self._client is not None:
            return self._client
        from ..providers.factory import get_client
        return get_client(
            provider=self._provider,
            api_key=self.api_key,
            base_url=self._base_url,
            model=self.model,
            api_format=self._api_format,
        )

    def _call_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Call the LLM and return the response text.

        Handles fallback from structured output to schema-in-prompt on
        400 errors.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        client = self._get_or_create_client()

        try:
            response = client.chat_completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
        except (ValueError, KeyError, TypeError) as e:
            if response_format is not None:
                logger.warning(
                    "Structured output failed (%s), retrying without "
                    "response_format (fallback to schema-in-prompt).",
                    e,
                )
                fallback_kwargs = {
                    k: v for k, v in kwargs.items() if k != "response_format"
                }
                response = client.chat_completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **fallback_kwargs,
                )
            else:
                raise
        except Exception as e:
            import httpx
            if (
                response_format is not None
                and isinstance(e, httpx.HTTPStatusError)
                and e.response.status_code == 400
            ):
                logger.warning(
                    "Structured output failed (%s), retrying without "
                    "response_format (fallback to schema-in-prompt).",
                    e,
                )
                fallback_kwargs = {
                    k: v for k, v in kwargs.items() if k != "response_format"
                }
                response = client.chat_completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **fallback_kwargs,
                )
            else:
                raise

        return response["choices"][0]["message"]["content"]

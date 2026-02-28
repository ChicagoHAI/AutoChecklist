"""Response parsing utilities."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..models import ChecklistItem


def extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from text that may contain surrounding prose.

    Tries in order:
    1. Direct ``json.loads`` (text is pure JSON)
    2. Fenced code block (```json ... ```)
    3. First ``{ ... }`` substring

    Raises ``ValueError`` if no valid JSON found.
    """
    # 1. Direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. First { ... } substring (greedy from first { to last })
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from text: {text[:200]}")


def parse_checklist_items(
    response: str,
    max_items: int = None,
) -> List[ChecklistItem]:
    """Parse checklist items from LLM response.

    Supports multiple formats:
    - [[item]] bracketed format (TICK, checklist-effectiveness-study)
    - Numbered list: "1. Question?"
    - Bulleted list: "- Question?"
    - RLCF weighted format: "- Question? (weight)"

    Args:
        response: Raw LLM response text
        max_items: Maximum items to return

    Returns:
        List of ChecklistItem objects
    """
    items: List[ChecklistItem] = []

    # Method 1: [[item]] bracketed format
    bracketed = re.findall(r"\[\[(.*?)\]\]", response, re.DOTALL)
    if bracketed:
        for text in bracketed:
            question = text.strip()
            if question and _looks_like_question(question):
                items.append(ChecklistItem(question=_ensure_question_mark(question)))

    # Method 2: RLCF weighted format "- Question? (weight)"
    if not items:
        weighted_pattern = r"[-•]\s*(.+?\?)\s*\((\d+)\)"
        for match in re.finditer(weighted_pattern, response):
            question = match.group(1).strip()
            weight = int(match.group(2))
            items.append(ChecklistItem(
                question=question,
                weight=float(weight)
            ))

    # Method 3: Numbered list "1. Question?"
    if not items:
        numbered = re.findall(r"^\s*\d+\.\s*(.+)$", response, re.MULTILINE)
        for text in numbered:
            question = text.strip()
            if _looks_like_question(question):
                items.append(ChecklistItem(question=_ensure_question_mark(question)))

    # Method 4: Bulleted list "- Question?"
    if not items:
        bulleted = re.findall(r"^\s*[-•]\s*(.+)$", response, re.MULTILINE)
        for text in bulleted:
            question = text.strip()
            if _looks_like_question(question):
                items.append(ChecklistItem(question=_ensure_question_mark(question)))

    # Limit to max_items
    if max_items:
        return items[:max_items]
    return items


def parse_yes_no_answers(
    response: str,
    num_questions: int,
) -> List[Tuple[int, str, Optional[str]]]:
    """Parse yes/no answers from LLM response.

    Supports formats:
    - "Q1: Yes" or "Q1: No"
    - "1. Yes" or "1. No"
    - JSON with "answer" field

    Args:
        response: Raw LLM response text
        num_questions: Expected number of questions

    Returns:
        List of (question_index, answer, reasoning) tuples
    """
    results: List[Tuple[int, str, Optional[str]]] = []

    # Pattern 1: "Q1: Yes/No" or "Q1: Yes/No - reasoning"
    pattern1 = r"Q(\d+):\s*(Yes|No)(?:\s*[-–]\s*(.+))?$"
    for match in re.finditer(pattern1, response, re.MULTILINE | re.IGNORECASE):
        idx = int(match.group(1)) - 1  # Convert to 0-indexed
        answer = match.group(2).lower()
        reasoning = match.group(3).strip() if match.group(3) else None
        results.append((idx, answer, reasoning))

    # Pattern 2: "1. Yes/No"
    if not results:
        pattern2 = r"^\s*(\d+)\.\s*(Yes|No)(?:\s*[-–]\s*(.+))?$"
        for match in re.finditer(pattern2, response, re.MULTILINE | re.IGNORECASE):
            idx = int(match.group(1)) - 1
            answer = match.group(2).lower()
            reasoning = match.group(3).strip() if match.group(3) else None
            results.append((idx, answer, reasoning))

    return results


def _looks_like_question(text: str) -> bool:
    """Check if text looks like a question."""
    text = text.strip()
    if not text:
        return False
    # Must end with ? or contain question words
    if text.endswith("?"):
        return True
    question_starters = ["does", "is", "are", "was", "were", "has", "have", "did", "do", "can", "could", "would", "should", "will"]
    first_word = text.split()[0].lower() if text.split() else ""
    return first_word in question_starters


def _ensure_question_mark(text: str) -> str:
    """Ensure text ends with a question mark."""
    text = text.strip()
    if not text.endswith("?"):
        text += "?"
    return text

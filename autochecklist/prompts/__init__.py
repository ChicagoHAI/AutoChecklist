"""Prompt template loading utilities."""

from pathlib import Path
from typing import Optional
import re

PROMPTS_DIR = Path(__file__).parent


def load_template(
    method: str,
    template_name: str,
    custom_path: Optional[str] = None
) -> str:
    """Load a prompt template.

    Args:
        method: Method name (e.g., "tick", "rlcf")
        template_name: Template file name without extension
        custom_path: Optional path to override default template

    Returns:
        Template string with {placeholders}
    """
    if custom_path:
        path = Path(custom_path)
    else:
        # Try multiple extensions
        for ext in [".txt", ".md", ".yaml"]:
            path = PROMPTS_DIR / method / f"{template_name}{ext}"
            if path.exists():
                break
        else:
            path = PROMPTS_DIR / method / f"{template_name}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")

    return path.read_text(encoding="utf-8")


class PromptTemplate:
    """Reusable prompt template with validation."""

    def __init__(self, template: str):
        self.template = template
        self._placeholders = self._extract_placeholders()

    def _extract_placeholders(self) -> set:
        """Extract placeholder names, ignoring escaped braces {{...}}."""
        # Use negative lookbehind/lookahead to skip double-brace patterns
        # e.g., {var} matches, but {{var}} does not
        return set(re.findall(r"(?<!\{)\{(\w+)\}(?!\})", self.template))

    def format(self, **kwargs) -> str:
        missing = self._placeholders - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.template.format(**kwargs)


def load_format(format_name: str) -> str:
    """Load a format prompt file from the formats/ directory.

    Args:
        format_name: Format file name without extension (e.g., "checklist")

    Returns:
        Format instruction text to append to prompts
    """
    return load_template("formats", format_name)


__all__ = [
    "load_template",
    "load_format",
    "PromptTemplate",
    "PROMPTS_DIR",
]

"""I/O utilities for JSON and JSONL files."""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Save data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file as a list of dictionaries."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Iterate over a JSONL file, yielding one item at a time."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(path: Union[str, Path], data: List[Dict[str, Any]]) -> None:
    """Save a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: Union[str, Path], item: Dict[str, Any]) -> None:
    """Append a single item to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

"""Simple filesystem storage utilities for the UI backend."""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional


def slugify(name: str) -> str:
    """Convert a name to a URL/filesystem-safe slug.

    E.g. "InteractEval: Consistency (SummEval)" -> "interacteval-consistency-summeval"
    """
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "untitled"

def _resolve_data_dir() -> Path:
    """Resolve data directory: UI_DATA_DIR env var > .env file > default (ui-dev/data/)."""
    env_key = "UI_DATA_DIR"
    if env_key in os.environ:
        return Path(os.environ[env_key])
    env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{env_key}="):
                    value = line.partition("=")[2].strip().strip("'\"")
                    if value:
                        return Path(value)
    return Path(__file__).parent.parent.parent.parent / "data"


DATA_DIR = _resolve_data_dir()


def ensure_dirs():
    """Create data directories if they don't exist."""
    for subdir in ["evaluations", "checklists", "batches", "uploads", "prompt_templates", "pipelines"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Optional[dict]:
    """Read a JSON file, return None if not found."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (write to .tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)


def delete_data_dir(subdir: str, item_id: str) -> bool:
    """Delete a data directory or file by ID. Returns True if found and deleted."""
    import shutil

    item_dir = DATA_DIR / subdir / item_id
    if item_dir.is_dir():
        shutil.rmtree(item_dir)
        return True

    # Also handle single JSON files (e.g., old-style evaluations)
    item_file = DATA_DIR / subdir / f"{item_id}.json"
    if item_file.is_file():
        item_file.unlink()
        return True

    return False


def list_dir(path: Path, suffix: str = ".json") -> list[Path]:
    """List files in directory with given suffix, sorted by mtime desc."""
    if not path.exists():
        return []
    files = [f for f in path.iterdir() if f.suffix == suffix]
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

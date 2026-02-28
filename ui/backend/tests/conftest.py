"""Shared fixtures for backend tests."""

import sys
from pathlib import Path

import pytest

# Ensure app is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Create a temporary data directory and patch storage.DATA_DIR."""
    import app.services.storage as storage_mod

    monkeypatch.setattr(storage_mod, "DATA_DIR", tmp_path)
    return tmp_path

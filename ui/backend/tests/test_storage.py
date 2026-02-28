"""Tests for storage utilities."""

import json


from app.services.storage import ensure_dirs, read_json, write_json, list_dir


class TestEnsureDirs:
    def test_creates_all_subdirs(self, tmp_data_dir):
        """ensure_dirs creates the expected subdirectories."""
        ensure_dirs()
        for subdir in ["evaluations", "checklists", "batches", "uploads"]:
            assert (tmp_data_dir / subdir).is_dir()

    def test_idempotent(self, tmp_data_dir):
        """Calling ensure_dirs twice does not raise."""
        ensure_dirs()
        ensure_dirs()
        assert (tmp_data_dir / "evaluations").is_dir()


class TestReadJson:
    def test_reads_valid_json(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text('{"key": "value"}')
        result = read_json(path)
        assert result == {"key": "value"}

    def test_returns_none_for_missing_file(self, tmp_path):
        result = read_json(tmp_path / "nonexistent.json")
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all {{{")
        result = read_json(path)
        assert result is None


class TestWriteJson:
    def test_writes_json(self, tmp_path):
        path = tmp_path / "out.json"
        write_json(path, {"hello": "world"})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == {"hello": "world"}

    def test_atomic_write_no_tmp_left(self, tmp_path):
        """After write_json, no .tmp file should remain."""
        path = tmp_path / "out.json"
        write_json(path, {"a": 1})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "out.json"
        write_json(path, {"v": 1})
        write_json(path, {"v": 2})
        data = json.loads(path.read_text())
        assert data["v"] == 2

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "file.json"
        write_json(path, {"nested": True})
        assert path.exists()
        assert json.loads(path.read_text()) == {"nested": True}

    def test_serializes_non_json_types(self, tmp_path):
        """default=str should handle non-serializable types."""
        from datetime import datetime

        path = tmp_path / "dates.json"
        write_json(path, {"ts": datetime(2024, 1, 1)})
        data = json.loads(path.read_text())
        assert "2024" in data["ts"]


class TestListDir:
    def test_empty_dir(self, tmp_path):
        result = list_dir(tmp_path)
        assert result == []

    def test_nonexistent_dir(self, tmp_path):
        result = list_dir(tmp_path / "nope")
        assert result == []

    def test_filters_by_suffix(self, tmp_path):
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.txt").write_text("hello")
        (tmp_path / "c.json").write_text("{}")
        result = list_dir(tmp_path, suffix=".json")
        names = [f.name for f in result]
        assert "a.json" in names
        assert "c.json" in names
        assert "b.txt" not in names

    def test_sorted_by_mtime_desc(self, tmp_path):
        import time

        (tmp_path / "old.json").write_text("{}")
        time.sleep(0.05)
        (tmp_path / "new.json").write_text("{}")
        result = list_dir(tmp_path)
        assert result[0].name == "new.json"
        assert result[1].name == "old.json"

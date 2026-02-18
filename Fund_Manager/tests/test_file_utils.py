"""
Tests for the file_utils module.
"""
import pytest
import os
import json
import time
import threading
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_utils import (
    FileLock, atomic_write, safe_json_load, safe_json_save,
    safe_append_json, ensure_file_exists
)


class TestFileLock:
    """Tests for the FileLock class."""

    def test_acquire_and_release(self, temp_dir):
        """Test basic lock acquire and release."""
        filepath = os.path.join(temp_dir, "test.json")
        lock = FileLock(filepath, timeout=1.0)

        assert lock.acquire() is True
        assert os.path.exists(lock.lock_path)

        lock.release()
        assert not os.path.exists(lock.lock_path)

    def test_context_manager(self, temp_dir):
        """Test using FileLock as context manager."""
        filepath = os.path.join(temp_dir, "test.json")

        with FileLock(filepath) as lock:
            assert os.path.exists(lock.lock_path)

        assert not os.path.exists(lock.lock_path)

    def test_timeout_when_locked(self, temp_dir):
        """Test that lock times out when already held."""
        filepath = os.path.join(temp_dir, "test.json")
        lock1 = FileLock(filepath, timeout=0.2)
        lock2 = FileLock(filepath, timeout=0.2)

        assert lock1.acquire() is True
        assert lock2.acquire() is False

        lock1.release()

    def test_stale_lock_removal(self, temp_dir):
        """Test that stale locks are removed."""
        filepath = os.path.join(temp_dir, "test.json")

        # Create a stale lock file manually
        lock_path = filepath + ".lock"
        with open(lock_path, 'w') as f:
            f.write(f"{os.getpid()}:{time.time() - 120}\n")  # 2 minutes old

        # Modify the file's timestamp to make it old
        old_time = time.time() - 120
        os.utime(lock_path, (old_time, old_time))

        lock = FileLock(filepath, timeout=0.5)
        assert lock.acquire() is True
        lock.release()


class TestAtomicWrite:
    """Tests for the atomic_write function."""

    def test_successful_write(self, temp_dir):
        """Test successful atomic write."""
        filepath = os.path.join(temp_dir, "test.txt")

        with atomic_write(filepath) as f:
            f.write("test content")

        with open(filepath, 'r') as f:
            assert f.read() == "test content"

    def test_cleanup_on_error(self, temp_dir):
        """Test that temp file is cleaned up on error."""
        filepath = os.path.join(temp_dir, "test.txt")

        try:
            with atomic_write(filepath) as f:
                f.write("partial content")
                raise ValueError("Intentional error")
        except ValueError:
            pass

        # Original file should not exist
        assert not os.path.exists(filepath)

        # Temp file should be cleaned up
        temp_files = [f for f in os.listdir(temp_dir) if f.endswith('.tmp')]
        assert len(temp_files) == 0


class TestSafeJsonLoad:
    """Tests for the safe_json_load function."""

    def test_load_valid_json(self, temp_dir):
        """Test loading valid JSON file."""
        filepath = os.path.join(temp_dir, "test.json")
        data = {"key": "value", "number": 42}

        with open(filepath, 'w') as f:
            json.dump(data, f)

        loaded = safe_json_load(filepath)
        assert loaded == data

    def test_returns_default_for_missing_file(self, temp_dir):
        """Test that default is returned for missing file."""
        filepath = os.path.join(temp_dir, "nonexistent.json")

        result = safe_json_load(filepath, default={"default": True})
        assert result == {"default": True}

    def test_returns_default_for_corrupted_json(self, temp_dir):
        """Test that default is returned for corrupted JSON."""
        filepath = os.path.join(temp_dir, "corrupted.json")

        with open(filepath, 'w') as f:
            f.write("not valid json {{{")

        result = safe_json_load(filepath, default={"fallback": True})
        assert result == {"fallback": True}

    def test_empty_default(self, temp_dir):
        """Test that empty dict is default when no default specified."""
        filepath = os.path.join(temp_dir, "nonexistent.json")

        result = safe_json_load(filepath)
        assert result == {}


class TestSafeJsonSave:
    """Tests for the safe_json_save function."""

    def test_save_json(self, temp_dir):
        """Test saving JSON file."""
        filepath = os.path.join(temp_dir, "test.json")
        data = {"key": "value", "list": [1, 2, 3]}

        result = safe_json_save(filepath, data)
        assert result is True

        with open(filepath, 'r') as f:
            loaded = json.load(f)

        assert loaded == data

    def test_creates_backup(self, temp_dir):
        """Test that backup is created when saving over existing file."""
        filepath = os.path.join(temp_dir, "test.json")

        # Create initial file
        with open(filepath, 'w') as f:
            json.dump({"original": True}, f)

        # Save new content
        safe_json_save(filepath, {"new": True}, create_backup=True)

        # Check backup exists
        backup_path = filepath + ".bak"
        assert os.path.exists(backup_path)

        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
        assert backup_data == {"original": True}

    def test_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created."""
        filepath = os.path.join(temp_dir, "subdir", "nested", "test.json")

        result = safe_json_save(filepath, {"data": 123}, ensure_dir=True)
        assert result is True
        assert os.path.exists(filepath)


class TestSafeAppendJson:
    """Tests for the safe_append_json function."""

    def test_append_to_new_file(self, temp_dir):
        """Test appending to a new file."""
        filepath = os.path.join(temp_dir, "list.json")

        safe_append_json(filepath, {"entry": 1})
        safe_append_json(filepath, {"entry": 2})

        with open(filepath, 'r') as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["entry"] == 1
        assert data[1]["entry"] == 2

    def test_max_entries_limit(self, temp_dir):
        """Test that max_entries limit is enforced."""
        filepath = os.path.join(temp_dir, "list.json")

        for i in range(10):
            safe_append_json(filepath, {"entry": i}, max_entries=5)

        with open(filepath, 'r') as f:
            data = json.load(f)

        assert len(data) == 5
        # Should have entries 5-9 (latest 5)
        assert data[0]["entry"] == 5
        assert data[4]["entry"] == 9


class TestEnsureFileExists:
    """Tests for the ensure_file_exists function."""

    def test_creates_file_with_default(self, temp_dir):
        """Test that file is created with default content."""
        filepath = os.path.join(temp_dir, "new.json")

        result = ensure_file_exists(filepath, default_content={"initialized": True})
        assert result is True
        assert os.path.exists(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)
        assert data == {"initialized": True}

    def test_does_not_overwrite_existing(self, temp_dir):
        """Test that existing file is not overwritten."""
        filepath = os.path.join(temp_dir, "existing.json")

        with open(filepath, 'w') as f:
            json.dump({"existing": True}, f)

        result = ensure_file_exists(filepath, default_content={"new": True})
        assert result is True

        with open(filepath, 'r') as f:
            data = json.load(f)
        assert data == {"existing": True}

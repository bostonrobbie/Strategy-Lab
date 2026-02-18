"""
File utilities with locking and atomic operations for the NHHF system.

Provides thread-safe and process-safe file operations to prevent
data corruption from concurrent access.

Usage:
    from file_utils import safe_json_load, safe_json_save, FileLock

    # Simple usage
    data = safe_json_load("portfolio.json", default={})
    safe_json_save("portfolio.json", data)

    # With explicit locking
    with FileLock("myfile.json"):
        data = load_data()
        process(data)
        save_data(data)
"""
import json
import os
import time
import platform
from contextlib import contextmanager
from typing import Any, Optional
from pathlib import Path
import logging
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


class FileLock:
    """
    Cross-platform file locking using lock files.

    Uses a simple lock file approach that works on both Windows and Unix.
    Not as robust as fcntl/msvcrt locks but more portable.

    Usage:
        with FileLock("data.json", timeout=10):
            # File is locked
            data = load_data()
            save_data(data)
        # Lock is released

    Args:
        path: Path to the file to lock
        timeout: Maximum seconds to wait for lock (default 10)
        poll_interval: Seconds between lock attempts (default 0.1)
    """

    def __init__(self, path: str, timeout: float = 10.0, poll_interval: float = 0.1):
        self.path = path
        self.lock_path = path + ".lock"
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._acquired = False

    def acquire(self) -> bool:
        """
        Acquire the lock.

        Returns:
            True if lock was acquired, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            try:
                # Try to create lock file exclusively
                # On both Windows and Unix, 'x' mode fails if file exists
                fd = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644
                )
                # Write PID and timestamp for debugging
                os.write(fd, f"{os.getpid()}:{time.time()}\n".encode())
                os.close(fd)
                self._acquired = True
                return True

            except FileExistsError:
                # Lock file exists - check if it's stale
                if self._is_stale_lock():
                    self._remove_stale_lock()
                    continue

                time.sleep(self.poll_interval)

            except OSError as e:
                logger.warning(f"[FileLock] Error acquiring lock for {self.path}: {e}")
                time.sleep(self.poll_interval)

        return False

    def _is_stale_lock(self, max_age: float = 60.0) -> bool:
        """Check if lock file is stale (older than max_age seconds)."""
        try:
            stat = os.stat(self.lock_path)
            age = time.time() - stat.st_mtime
            return age > max_age
        except OSError:
            return False

    def _remove_stale_lock(self):
        """Remove a stale lock file."""
        try:
            os.remove(self.lock_path)
            logger.info(f"[FileLock] Removed stale lock: {self.lock_path}")
        except OSError:
            pass

    def release(self):
        """Release the lock."""
        if self._acquired:
            try:
                os.remove(self.lock_path)
            except OSError:
                pass
            finally:
                self._acquired = False

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock for {self.path} within {self.timeout}s")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


@contextmanager
def atomic_write(path: str, mode: str = 'w'):
    """
    Context manager for atomic file writes.

    Writes to a temporary file then renames to target path.
    This prevents partial writes if the process crashes.

    Usage:
        with atomic_write("data.json") as f:
            json.dump(data, f)

    Args:
        path: Target file path
        mode: File mode ('w' for text, 'wb' for binary)

    Yields:
        File handle for writing
    """
    temp_path = path + f".tmp.{os.getpid()}"
    success = False

    try:
        with open(temp_path, mode) as f:
            yield f
        success = True
    finally:
        if success:
            try:
                # Atomic rename
                if platform.system() == 'Windows':
                    # Windows can't rename over existing file
                    if os.path.exists(path):
                        os.remove(path)
                os.rename(temp_path, path)
            except OSError as e:
                logger.error(f"[atomic_write] Failed to rename {temp_path} to {path}: {e}")
                # Try to clean up temp file
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                raise
        else:
            # Write failed - clean up temp file
            try:
                os.remove(temp_path)
            except OSError:
                pass


def safe_json_load(path: str, default: Any = None, create_backup: bool = False) -> Any:
    """
    Load JSON file with locking and error recovery.

    If the file is corrupted, attempts to restore from backup.

    Args:
        path: Path to JSON file
        default: Default value if file doesn't exist or is corrupted
        create_backup: Whether to create a backup before reading

    Returns:
        Parsed JSON data or default value
    """
    if not os.path.exists(path):
        return default if default is not None else {}

    try:
        with FileLock(path, timeout=5.0):
            # Optionally create backup
            if create_backup:
                backup_path = path + ".bak"
                try:
                    shutil.copy2(path, backup_path)
                except OSError:
                    pass

            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

    except json.JSONDecodeError as e:
        logger.warning(f"[safe_json_load] Corrupted JSON in {path}: {e}")

        # Try to restore from backup
        backup_path = path + ".bak"
        if os.path.exists(backup_path):
            try:
                logger.info(f"[safe_json_load] Attempting to restore from backup")
                with open(backup_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

        # Create corrupted backup for debugging
        try:
            corrupted_backup = path + f".corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(path, corrupted_backup)
            logger.info(f"[safe_json_load] Saved corrupted file to {corrupted_backup}")
        except OSError:
            pass

        return default if default is not None else {}

    except TimeoutError:
        logger.warning(f"[safe_json_load] Timeout acquiring lock for {path}")
        # Try without lock as fallback
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default if default is not None else {}

    except Exception as e:
        logger.error(f"[safe_json_load] Unexpected error loading {path}: {e}")
        return default if default is not None else {}


def safe_json_save(
    path: str,
    data: Any,
    indent: int = 2,
    create_backup: bool = True,
    ensure_dir: bool = True
) -> bool:
    """
    Save JSON file with locking and atomic write.

    Creates a backup of the existing file before overwriting.

    Args:
        path: Path to JSON file
        data: Data to serialize
        indent: JSON indent level
        create_backup: Whether to backup existing file
        ensure_dir: Create parent directories if needed

    Returns:
        True if save was successful, False otherwise
    """
    try:
        # Ensure parent directory exists
        if ensure_dir:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        with FileLock(path, timeout=10.0):
            # Create backup of existing file
            if create_backup and os.path.exists(path):
                backup_path = path + ".bak"
                try:
                    shutil.copy2(path, backup_path)
                except OSError as e:
                    logger.warning(f"[safe_json_save] Could not create backup: {e}")

            # Atomic write
            with atomic_write(path) as f:
                json.dump(data, f, indent=indent, default=str)

            return True

    except TimeoutError:
        logger.error(f"[safe_json_save] Timeout acquiring lock for {path}")
        return False

    except Exception as e:
        logger.error(f"[safe_json_save] Error saving {path}: {e}")
        return False


def safe_append_json(path: str, entry: Any, max_entries: int = 100) -> bool:
    """
    Safely append an entry to a JSON array file.

    Maintains a maximum number of entries (circular buffer behavior).

    Args:
        path: Path to JSON file (should contain an array)
        entry: Entry to append
        max_entries: Maximum entries to keep

    Returns:
        True if successful
    """
    try:
        with FileLock(path, timeout=10.0):
            # Load existing data
            data = []
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    data = []

            # Ensure it's a list
            if not isinstance(data, list):
                data = []

            # Append and trim
            data.append(entry)
            data = data[-max_entries:]

            # Save with atomic write
            with atomic_write(path) as f:
                json.dump(data, f, indent=2, default=str)

            return True

    except Exception as e:
        logger.error(f"[safe_append_json] Error appending to {path}: {e}")
        return False


def ensure_file_exists(path: str, default_content: Any = None) -> bool:
    """
    Ensure a JSON file exists, creating it with default content if needed.

    Args:
        path: Path to JSON file
        default_content: Default content if file doesn't exist

    Returns:
        True if file exists (or was created)
    """
    if os.path.exists(path):
        return True

    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(default_content if default_content is not None else {}, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"[ensure_file_exists] Error creating {path}: {e}")
        return False

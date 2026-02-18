"""
Enhanced logging system for the NHHF Fund Manager.

Features:
- Severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Full stack traces for errors
- Operation context tracking (step A, B, C, etc.)
- Error rate tracking for alerting
- Thread-safe operations
- Automatic log rotation
"""

import json
import os
import traceback
import threading
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import deque

from file_utils import safe_json_load, safe_json_save


# Log file paths
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
ENHANCED_ERROR_LOG = os.path.join(LOG_DIR, "enhanced_errors.json")
ACTIVITY_LOG = os.path.join(LOG_DIR, "activity.json")
ERROR_RATES_LOG = os.path.join(LOG_DIR, "error_rates.json")


class Severity(Enum):
    """Log severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        """Numeric level for comparison."""
        levels = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50
        }
        return levels.get(self.value, 0)


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    severity: str
    agent: str
    operation: str
    step: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    error_type: Optional[str] = None
    resolved: bool = False
    resolution_notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ErrorRateTracker:
    """
    Tracks error rates for alerting purposes.

    Maintains sliding windows of error counts for different time periods.
    """

    def __init__(self, window_minutes: int = 60):
        """
        Initialize the error rate tracker.

        Args:
            window_minutes: Time window for tracking (default 60 min)
        """
        self.window_minutes = window_minutes
        self._errors: deque = deque()
        self._lock = threading.Lock()

    def record_error(self, agent: str, error_type: str) -> None:
        """Record an error occurrence."""
        with self._lock:
            self._errors.append({
                "timestamp": datetime.now(),
                "agent": agent,
                "error_type": error_type
            })
            self._cleanup_old_errors()

    def _cleanup_old_errors(self) -> None:
        """Remove errors outside the time window."""
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
        while self._errors and self._errors[0]["timestamp"] < cutoff:
            self._errors.popleft()

    def get_error_rate(self, agent: Optional[str] = None) -> Dict[str, Any]:
        """
        Get error rate statistics.

        Args:
            agent: Optional filter by agent name

        Returns:
            Dictionary with error rate statistics
        """
        with self._lock:
            self._cleanup_old_errors()

            errors = list(self._errors)
            if agent:
                errors = [e for e in errors if e["agent"] == agent]

            # Count by type
            type_counts: Dict[str, int] = {}
            agent_counts: Dict[str, int] = {}

            for error in errors:
                error_type = error.get("error_type", "unknown")
                agent_name = error.get("agent", "unknown")

                type_counts[error_type] = type_counts.get(error_type, 0) + 1
                agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1

            return {
                "window_minutes": self.window_minutes,
                "total_errors": len(errors),
                "errors_per_minute": len(errors) / max(self.window_minutes, 1),
                "by_type": type_counts,
                "by_agent": agent_counts
            }

    def is_alert_threshold_exceeded(
        self,
        threshold: int = 10,
        agent: Optional[str] = None
    ) -> bool:
        """
        Check if error rate exceeds alert threshold.

        Args:
            threshold: Maximum errors in window before alerting
            agent: Optional filter by agent

        Returns:
            True if threshold exceeded
        """
        stats = self.get_error_rate(agent)
        return stats["total_errors"] >= threshold


class EnhancedLogger:
    """
    Enhanced logging system with severity levels, stack traces, and context.

    Thread-safe singleton implementation.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._min_severity = Severity.DEBUG
        self._current_operation: Optional[str] = None
        self._current_step: Optional[str] = None
        self._error_tracker = ErrorRateTracker()
        self._log_lock = threading.Lock()

        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)

    def set_min_severity(self, severity: Severity) -> None:
        """Set minimum severity level to log."""
        self._min_severity = severity

    def set_operation_context(self, operation: str, step: str = "start") -> None:
        """
        Set the current operation context.

        Args:
            operation: Name of the current operation (e.g., "evolve_strategy")
            step: Current step (e.g., "A", "B", "generate_code")
        """
        self._current_operation = operation
        self._current_step = step

    def clear_operation_context(self) -> None:
        """Clear the current operation context."""
        self._current_operation = None
        self._current_step = None

    def _should_log(self, severity: Severity) -> bool:
        """Check if this severity level should be logged."""
        return severity.level >= self._min_severity.level

    def _create_entry(
        self,
        severity: Severity,
        agent: str,
        message: str,
        context: Optional[Dict] = None,
        exception: Optional[Exception] = None
    ) -> LogEntry:
        """Create a log entry."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            severity=severity.value,
            agent=agent,
            operation=self._current_operation or "unknown",
            step=self._current_step or "unknown",
            message=message,
            context=context or {}
        )

        if exception:
            entry.error_type = type(exception).__name__
            entry.stack_trace = traceback.format_exc()

        return entry

    def _save_entry(self, entry: LogEntry, log_file: str, max_entries: int = 500) -> None:
        """Save a log entry to file."""
        with self._log_lock:
            entries = safe_json_load(log_file, default=[])
            if not isinstance(entries, list):
                entries = []

            entries.append(entry.to_dict())

            # Rotate logs - keep only recent entries
            entries = entries[-max_entries:]

            safe_json_save(log_file, entries)

    def debug(self, agent: str, message: str, context: Optional[Dict] = None) -> None:
        """Log a debug message."""
        if not self._should_log(Severity.DEBUG):
            return

        entry = self._create_entry(Severity.DEBUG, agent, message, context)
        self._save_entry(entry, ACTIVITY_LOG, max_entries=200)

    def info(self, agent: str, message: str, context: Optional[Dict] = None) -> None:
        """Log an info message."""
        if not self._should_log(Severity.INFO):
            return

        entry = self._create_entry(Severity.INFO, agent, message, context)
        self._save_entry(entry, ACTIVITY_LOG, max_entries=200)

    def warning(self, agent: str, message: str, context: Optional[Dict] = None) -> None:
        """Log a warning message."""
        if not self._should_log(Severity.WARNING):
            return

        entry = self._create_entry(Severity.WARNING, agent, message, context)
        self._save_entry(entry, ACTIVITY_LOG, max_entries=200)

    def error(
        self,
        agent: str,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict] = None
    ) -> None:
        """
        Log an error with full stack trace.

        Args:
            agent: Agent name
            message: Error message
            exception: Optional exception object for stack trace
            context: Optional additional context
        """
        if not self._should_log(Severity.ERROR):
            return

        entry = self._create_entry(Severity.ERROR, agent, message, context, exception)
        self._save_entry(entry, ENHANCED_ERROR_LOG)

        # Track error rate
        error_type = entry.error_type or "unknown"
        self._error_tracker.record_error(agent, error_type)

    def critical(
        self,
        agent: str,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict] = None
    ) -> None:
        """
        Log a critical error.

        Args:
            agent: Agent name
            message: Error message
            exception: Optional exception object
            context: Optional additional context
        """
        if not self._should_log(Severity.CRITICAL):
            return

        entry = self._create_entry(Severity.CRITICAL, agent, message, context, exception)
        self._save_entry(entry, ENHANCED_ERROR_LOG)

        # Track error rate
        error_type = entry.error_type or "critical_unknown"
        self._error_tracker.record_error(agent, error_type)

    def get_recent_errors(
        self,
        limit: int = 20,
        severity_filter: Optional[Severity] = None,
        agent_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recent error log entries.

        Args:
            limit: Maximum number of entries to return
            severity_filter: Optional filter by severity
            agent_filter: Optional filter by agent name

        Returns:
            List of error log entries
        """
        entries = safe_json_load(ENHANCED_ERROR_LOG, default=[])
        if not isinstance(entries, list):
            return []

        # Apply filters
        if severity_filter:
            entries = [e for e in entries if e.get("severity") == severity_filter.value]

        if agent_filter:
            entries = [e for e in entries if e.get("agent") == agent_filter]

        return entries[-limit:]

    def get_activity_log(self, limit: int = 50) -> List[Dict]:
        """Get recent activity log entries."""
        entries = safe_json_load(ACTIVITY_LOG, default=[])
        if not isinstance(entries, list):
            return []
        return entries[-limit:]

    def get_error_rate_stats(self, agent: Optional[str] = None) -> Dict[str, Any]:
        """
        Get error rate statistics.

        Args:
            agent: Optional filter by agent

        Returns:
            Error rate statistics dictionary
        """
        return self._error_tracker.get_error_rate(agent)

    def is_error_rate_high(self, threshold: int = 10, agent: Optional[str] = None) -> bool:
        """
        Check if error rate is high (for alerting).

        Args:
            threshold: Error count threshold
            agent: Optional agent filter

        Returns:
            True if error rate exceeds threshold
        """
        return self._error_tracker.is_alert_threshold_exceeded(threshold, agent)

    def mark_error_resolved(self, timestamp: str, notes: str = "") -> bool:
        """
        Mark an error as resolved.

        Args:
            timestamp: Timestamp of the error to mark resolved
            notes: Optional resolution notes

        Returns:
            True if error was found and marked resolved
        """
        with self._log_lock:
            entries = safe_json_load(ENHANCED_ERROR_LOG, default=[])
            if not isinstance(entries, list):
                return False

            for entry in entries:
                if entry.get("timestamp") == timestamp:
                    entry["resolved"] = True
                    entry["resolution_notes"] = notes
                    safe_json_save(ENHANCED_ERROR_LOG, entries)
                    return True

            return False

    def get_unresolved_errors(self, limit: int = 20) -> List[Dict]:
        """Get unresolved errors."""
        entries = safe_json_load(ENHANCED_ERROR_LOG, default=[])
        if not isinstance(entries, list):
            return []

        unresolved = [e for e in entries if not e.get("resolved", False)]
        return unresolved[-limit:]

    def save_error_rates_snapshot(self) -> None:
        """Save current error rates to file for dashboard consumption."""
        stats = self._error_tracker.get_error_rate()
        stats["snapshot_time"] = datetime.now().isoformat()
        safe_json_save(ERROR_RATES_LOG, stats)


# Singleton accessor
def get_logger() -> EnhancedLogger:
    """Get the singleton enhanced logger instance."""
    return EnhancedLogger()


# Convenience functions for common logging patterns
def log_agent_error(
    agent: str,
    message: str,
    exception: Optional[Exception] = None,
    context: Optional[Dict] = None
) -> None:
    """Log an agent error with full context."""
    get_logger().error(agent, message, exception, context)


def log_agent_activity(
    agent: str,
    message: str,
    context: Optional[Dict] = None
) -> None:
    """Log agent activity."""
    get_logger().info(agent, message, context)


def set_operation_step(operation: str, step: str) -> None:
    """Set the current operation and step context."""
    get_logger().set_operation_context(operation, step)


def clear_operation_step() -> None:
    """Clear operation context."""
    get_logger().clear_operation_context()


class OperationContext:
    """
    Context manager for tracking operation steps.

    Usage:
        with OperationContext("evolve_strategy", "A_generate"):
            # ... code ...
    """

    def __init__(self, operation: str, step: str):
        self.operation = operation
        self.step = step
        self._logger = get_logger()

    def __enter__(self):
        self._logger.set_operation_context(self.operation, self.step)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Log the exception with full context
            self._logger.error(
                agent="system",
                message=f"Exception in {self.operation}/{self.step}: {exc_val}",
                exception=exc_val,
                context={"operation": self.operation, "step": self.step}
            )
        self._logger.clear_operation_context()
        return False  # Don't suppress exceptions

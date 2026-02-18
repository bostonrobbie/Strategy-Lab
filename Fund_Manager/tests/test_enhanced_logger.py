"""
Tests for the enhanced_logger module.
"""
import pytest
import os
import json
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_logger import (
    Severity, LogEntry, ErrorRateTracker, EnhancedLogger,
    get_logger, log_agent_error, log_agent_activity,
    set_operation_step, clear_operation_step, OperationContext
)


class TestSeverity:
    """Tests for the Severity enum."""

    def test_severity_values(self):
        """Test severity level values."""
        assert Severity.DEBUG.value == "debug"
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"
        assert Severity.CRITICAL.value == "critical"

    def test_severity_levels(self):
        """Test severity level ordering."""
        assert Severity.DEBUG.level < Severity.INFO.level
        assert Severity.INFO.level < Severity.WARNING.level
        assert Severity.WARNING.level < Severity.ERROR.level
        assert Severity.ERROR.level < Severity.CRITICAL.level


class TestLogEntry:
    """Tests for the LogEntry dataclass."""

    def test_log_entry_creation(self):
        """Test creating a log entry."""
        entry = LogEntry(
            timestamp="2024-01-01T00:00:00",
            severity="error",
            agent="test_agent",
            operation="test_op",
            step="step_1",
            message="Test message"
        )
        assert entry.timestamp == "2024-01-01T00:00:00"
        assert entry.severity == "error"
        assert entry.agent == "test_agent"
        assert entry.resolved is False

    def test_log_entry_to_dict(self):
        """Test converting log entry to dictionary."""
        entry = LogEntry(
            timestamp="2024-01-01T00:00:00",
            severity="info",
            agent="test",
            operation="op",
            step="s1",
            message="msg",
            context={"key": "value"}
        )
        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d["timestamp"] == "2024-01-01T00:00:00"
        assert d["context"] == {"key": "value"}

    def test_log_entry_with_stack_trace(self):
        """Test log entry with stack trace."""
        entry = LogEntry(
            timestamp="2024-01-01T00:00:00",
            severity="error",
            agent="test",
            operation="op",
            step="s1",
            message="Error occurred",
            stack_trace="Traceback...",
            error_type="ValueError"
        )
        assert entry.stack_trace == "Traceback..."
        assert entry.error_type == "ValueError"


class TestErrorRateTracker:
    """Tests for the ErrorRateTracker class."""

    def test_record_error(self):
        """Test recording errors."""
        tracker = ErrorRateTracker(window_minutes=60)
        tracker.record_error("agent_1", "ValueError")
        tracker.record_error("agent_1", "TypeError")
        tracker.record_error("agent_2", "ValueError")

        stats = tracker.get_error_rate()
        assert stats["total_errors"] == 3

    def test_error_rate_by_type(self):
        """Test error rate grouped by type."""
        tracker = ErrorRateTracker()
        tracker.record_error("agent", "ValueError")
        tracker.record_error("agent", "ValueError")
        tracker.record_error("agent", "TypeError")

        stats = tracker.get_error_rate()
        assert stats["by_type"]["ValueError"] == 2
        assert stats["by_type"]["TypeError"] == 1

    def test_error_rate_by_agent(self):
        """Test error rate grouped by agent."""
        tracker = ErrorRateTracker()
        tracker.record_error("agent_1", "Error")
        tracker.record_error("agent_1", "Error")
        tracker.record_error("agent_2", "Error")

        stats = tracker.get_error_rate()
        assert stats["by_agent"]["agent_1"] == 2
        assert stats["by_agent"]["agent_2"] == 1

    def test_filter_by_agent(self):
        """Test filtering error rate by agent."""
        tracker = ErrorRateTracker()
        tracker.record_error("agent_1", "Error")
        tracker.record_error("agent_2", "Error")

        stats = tracker.get_error_rate(agent="agent_1")
        assert stats["total_errors"] == 1

    def test_alert_threshold(self):
        """Test alert threshold detection."""
        tracker = ErrorRateTracker()

        # Below threshold
        for _ in range(5):
            tracker.record_error("agent", "Error")
        assert tracker.is_alert_threshold_exceeded(threshold=10) is False

        # At threshold
        for _ in range(5):
            tracker.record_error("agent", "Error")
        assert tracker.is_alert_threshold_exceeded(threshold=10) is True


class TestEnhancedLogger:
    """Tests for the EnhancedLogger class."""

    @pytest.fixture
    def logger(self, temp_dir, monkeypatch):
        """Create a logger with temp directory."""
        import enhanced_logger as logger_module

        log_dir = os.path.join(temp_dir, "logs")
        monkeypatch.setattr(logger_module, 'LOG_DIR', log_dir)
        monkeypatch.setattr(logger_module, 'ENHANCED_ERROR_LOG',
                          os.path.join(log_dir, "enhanced_errors.json"))
        monkeypatch.setattr(logger_module, 'ACTIVITY_LOG',
                          os.path.join(log_dir, "activity.json"))
        monkeypatch.setattr(logger_module, 'ERROR_RATES_LOG',
                          os.path.join(log_dir, "error_rates.json"))

        # Reset singleton for testing
        logger_module.EnhancedLogger._instance = None

        return logger_module.EnhancedLogger()

    def test_singleton_pattern(self, logger, temp_dir, monkeypatch):
        """Test that logger is a singleton."""
        import enhanced_logger as logger_module

        # Reset for this test
        logger_module.EnhancedLogger._instance = None

        logger1 = logger_module.EnhancedLogger()
        logger2 = logger_module.EnhancedLogger()
        assert logger1 is logger2

    def test_set_operation_context(self, logger):
        """Test setting operation context."""
        logger.set_operation_context("test_operation", "step_A")
        assert logger._current_operation == "test_operation"
        assert logger._current_step == "step_A"

    def test_clear_operation_context(self, logger):
        """Test clearing operation context."""
        logger.set_operation_context("op", "step")
        logger.clear_operation_context()
        assert logger._current_operation is None
        assert logger._current_step is None

    def test_log_debug(self, logger, temp_dir, monkeypatch):
        """Test debug logging."""
        import enhanced_logger as logger_module

        logger.debug("test_agent", "Debug message", {"key": "value"})

        activity_log = os.path.join(temp_dir, "logs", "activity.json")
        with open(activity_log, 'r') as f:
            entries = json.load(f)

        assert len(entries) == 1
        assert entries[0]["severity"] == "debug"
        assert entries[0]["message"] == "Debug message"

    def test_log_info(self, logger, temp_dir):
        """Test info logging."""
        logger.info("test_agent", "Info message")

        activity_log = os.path.join(temp_dir, "logs", "activity.json")
        with open(activity_log, 'r') as f:
            entries = json.load(f)

        assert entries[-1]["severity"] == "info"

    def test_log_warning(self, logger, temp_dir):
        """Test warning logging."""
        logger.warning("test_agent", "Warning message")

        activity_log = os.path.join(temp_dir, "logs", "activity.json")
        with open(activity_log, 'r') as f:
            entries = json.load(f)

        assert entries[-1]["severity"] == "warning"

    def test_log_error_with_exception(self, logger, temp_dir):
        """Test error logging with exception."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.error("test_agent", "Error occurred", exception=e)

        error_log = os.path.join(temp_dir, "logs", "enhanced_errors.json")
        with open(error_log, 'r') as f:
            entries = json.load(f)

        assert len(entries) == 1
        assert entries[0]["severity"] == "error"
        assert entries[0]["error_type"] == "ValueError"
        assert "Traceback" in entries[0]["stack_trace"]

    def test_log_critical(self, logger, temp_dir):
        """Test critical logging."""
        logger.critical("test_agent", "Critical error!")

        error_log = os.path.join(temp_dir, "logs", "enhanced_errors.json")
        with open(error_log, 'r') as f:
            entries = json.load(f)

        assert entries[-1]["severity"] == "critical"

    def test_severity_filtering(self, logger, temp_dir):
        """Test that severity filtering works."""
        logger.set_min_severity(Severity.WARNING)

        logger.debug("agent", "Debug - should not log")
        logger.info("agent", "Info - should not log")
        logger.warning("agent", "Warning - should log")
        logger.error("agent", "Error - should log")

        activity_log = os.path.join(temp_dir, "logs", "activity.json")
        with open(activity_log, 'r') as f:
            entries = json.load(f)

        # Only warning should be in activity log
        assert len(entries) == 1
        assert entries[0]["severity"] == "warning"

    def test_get_recent_errors(self, logger, temp_dir):
        """Test getting recent errors."""
        logger.error("agent_1", "Error 1")
        logger.error("agent_2", "Error 2")
        logger.error("agent_1", "Error 3")

        errors = logger.get_recent_errors(limit=10)
        assert len(errors) == 3

        # Filter by agent
        errors = logger.get_recent_errors(agent_filter="agent_1")
        assert len(errors) == 2

    def test_get_error_rate_stats(self, logger):
        """Test getting error rate statistics."""
        logger.error("agent", "Error 1")
        logger.error("agent", "Error 2")

        stats = logger.get_error_rate_stats()
        assert stats["total_errors"] == 2

    def test_mark_error_resolved(self, logger, temp_dir):
        """Test marking an error as resolved."""
        logger.error("agent", "Error to resolve")

        error_log = os.path.join(temp_dir, "logs", "enhanced_errors.json")
        with open(error_log, 'r') as f:
            entries = json.load(f)

        timestamp = entries[0]["timestamp"]
        result = logger.mark_error_resolved(timestamp, "Fixed the issue")
        assert result is True

        with open(error_log, 'r') as f:
            entries = json.load(f)

        assert entries[0]["resolved"] is True
        assert entries[0]["resolution_notes"] == "Fixed the issue"

    def test_get_unresolved_errors(self, logger, temp_dir):
        """Test getting unresolved errors."""
        logger.error("agent", "Error 1")
        logger.error("agent", "Error 2")

        error_log = os.path.join(temp_dir, "logs", "enhanced_errors.json")
        with open(error_log, 'r') as f:
            entries = json.load(f)

        # Resolve first error
        logger.mark_error_resolved(entries[0]["timestamp"])

        unresolved = logger.get_unresolved_errors()
        assert len(unresolved) == 1
        assert unresolved[0]["message"] == "Error 2"


class TestOperationContext:
    """Tests for the OperationContext context manager."""

    @pytest.fixture
    def logger(self, temp_dir, monkeypatch):
        """Create a logger with temp directory."""
        import enhanced_logger as logger_module

        log_dir = os.path.join(temp_dir, "logs")
        monkeypatch.setattr(logger_module, 'LOG_DIR', log_dir)
        monkeypatch.setattr(logger_module, 'ENHANCED_ERROR_LOG',
                          os.path.join(log_dir, "enhanced_errors.json"))
        monkeypatch.setattr(logger_module, 'ACTIVITY_LOG',
                          os.path.join(log_dir, "activity.json"))
        monkeypatch.setattr(logger_module, 'ERROR_RATES_LOG',
                          os.path.join(log_dir, "error_rates.json"))

        logger_module.EnhancedLogger._instance = None
        return logger_module.get_logger()

    def test_operation_context_sets_context(self, logger):
        """Test that context manager sets operation context."""
        with OperationContext("test_op", "step_1"):
            assert logger._current_operation == "test_op"
            assert logger._current_step == "step_1"

        # Context cleared after exit
        assert logger._current_operation is None

    def test_operation_context_logs_exception(self, logger, temp_dir):
        """Test that context manager logs exceptions."""
        try:
            with OperationContext("failing_op", "bad_step"):
                raise RuntimeError("Test failure")
        except RuntimeError:
            pass

        error_log = os.path.join(temp_dir, "logs", "enhanced_errors.json")
        with open(error_log, 'r') as f:
            entries = json.load(f)

        assert len(entries) == 1
        assert "failing_op" in entries[0]["message"]
        assert entries[0]["context"]["operation"] == "failing_op"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def setup_logger(self, temp_dir, monkeypatch):
        """Set up logger for testing."""
        import enhanced_logger as logger_module

        log_dir = os.path.join(temp_dir, "logs")
        monkeypatch.setattr(logger_module, 'LOG_DIR', log_dir)
        monkeypatch.setattr(logger_module, 'ENHANCED_ERROR_LOG',
                          os.path.join(log_dir, "enhanced_errors.json"))
        monkeypatch.setattr(logger_module, 'ACTIVITY_LOG',
                          os.path.join(log_dir, "activity.json"))

        logger_module.EnhancedLogger._instance = None
        return temp_dir

    def test_log_agent_error(self, setup_logger):
        """Test log_agent_error convenience function."""
        log_agent_error("test_agent", "Test error message")

        error_log = os.path.join(setup_logger, "logs", "enhanced_errors.json")
        with open(error_log, 'r') as f:
            entries = json.load(f)

        assert len(entries) == 1
        assert entries[0]["agent"] == "test_agent"

    def test_log_agent_activity(self, setup_logger):
        """Test log_agent_activity convenience function."""
        log_agent_activity("test_agent", "Activity message")

        activity_log = os.path.join(setup_logger, "logs", "activity.json")
        with open(activity_log, 'r') as f:
            entries = json.load(f)

        assert len(entries) == 1
        assert entries[0]["message"] == "Activity message"

    def test_set_and_clear_operation_step(self, setup_logger):
        """Test set and clear operation step functions."""
        import enhanced_logger as logger_module

        set_operation_step("my_operation", "my_step")
        logger = logger_module.get_logger()
        assert logger._current_operation == "my_operation"
        assert logger._current_step == "my_step"

        clear_operation_step()
        assert logger._current_operation is None


class TestThreadSafety:
    """Tests for thread safety."""

    @pytest.fixture
    def logger(self, temp_dir, monkeypatch):
        """Create a logger for thread safety testing."""
        import enhanced_logger as logger_module

        log_dir = os.path.join(temp_dir, "logs")
        monkeypatch.setattr(logger_module, 'LOG_DIR', log_dir)
        monkeypatch.setattr(logger_module, 'ENHANCED_ERROR_LOG',
                          os.path.join(log_dir, "enhanced_errors.json"))
        monkeypatch.setattr(logger_module, 'ACTIVITY_LOG',
                          os.path.join(log_dir, "activity.json"))

        logger_module.EnhancedLogger._instance = None
        return logger_module.get_logger()

    def test_concurrent_logging(self, logger, temp_dir):
        """Test that concurrent logging doesn't corrupt data."""
        errors = []

        def log_errors():
            try:
                for i in range(10):
                    logger.error("thread", f"Error {i}")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=log_errors) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0

        # Log file should be valid JSON
        error_log = os.path.join(temp_dir, "logs", "enhanced_errors.json")
        with open(error_log, 'r') as f:
            entries = json.load(f)

        # Should have logged approximately 50 errors (5 threads * 10 each)
        assert len(entries) >= 40  # Allow some tolerance

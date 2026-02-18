"""
Tests for the circuit breaker module.
"""
import pytest
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitConfig, CircuitOpenError,
    RateLimiter, RateLimitError
)


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Test that circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert not cb.is_open

    def test_allows_requests_when_closed(self):
        """Test that requests are allowed when circuit is closed."""
        cb = CircuitBreaker("test")
        assert cb.allow_request() is True

    def test_opens_after_threshold_failures(self):
        """Test that circuit opens after reaching failure threshold."""
        config = CircuitConfig(failure_threshold=3, recovery_timeout=1.0)
        cb = CircuitBreaker("test", config)

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.is_open

    def test_rejects_requests_when_open(self):
        """Test that requests are rejected when circuit is open."""
        config = CircuitConfig(failure_threshold=2, recovery_timeout=60.0)
        cb = CircuitBreaker("test", config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        assert cb.allow_request() is False

    def test_transitions_to_half_open_after_timeout(self):
        """Test that circuit transitions to HALF_OPEN after recovery timeout."""
        config = CircuitConfig(failure_threshold=2, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Request should transition to half-open
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_successful_recovery(self):
        """Test that circuit closes after successful calls in half-open state."""
        config = CircuitConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=2
        )
        cb = CircuitBreaker("test", config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for recovery timeout
        time.sleep(0.15)

        # Make successful calls in half-open state
        cb.allow_request()
        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_during_half_open(self):
        """Test that circuit reopens if failure occurs during half-open."""
        config = CircuitConfig(failure_threshold=2, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for recovery timeout
        time.sleep(0.15)

        # Make a call in half-open state
        cb.allow_request()
        cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_reset_closes_circuit(self):
        """Test that manual reset closes the circuit."""
        config = CircuitConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        cb = CircuitBreaker("test", CircuitConfig(failure_threshold=3))

        cb.allow_request()
        cb.record_success()
        cb.allow_request()
        cb.record_failure()

        stats = cb.stats
        assert stats["total_calls"] == 2
        assert stats["total_failures"] == 1

    def test_decorator_usage(self):
        """Test using circuit breaker as decorator."""
        cb = CircuitBreaker("test", CircuitConfig(failure_threshold=2))

        @cb
        def may_fail(should_fail=False):
            if should_fail:
                raise ValueError("Intentional failure")
            return "success"

        # Successful call
        assert may_fail() == "success"

        # Failing calls
        with pytest.raises(ValueError):
            may_fail(should_fail=True)
        with pytest.raises(ValueError):
            may_fail(should_fail=True)

        # Circuit should be open now
        with pytest.raises(CircuitOpenError):
            may_fail()


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_allows_requests_under_limit(self):
        """Test that requests are allowed when under the limit."""
        limiter = RateLimiter(max_calls=5, period_seconds=1.0)

        for _ in range(5):
            wait = limiter.wait_if_needed()
            assert wait == 0.0

    def test_raises_error_when_not_blocking(self):
        """Test that RateLimitError is raised when blocking is disabled."""
        limiter = RateLimiter(max_calls=2, period_seconds=60.0, block_when_limited=False)

        limiter.wait_if_needed()
        limiter.wait_if_needed()

        with pytest.raises(RateLimitError):
            limiter.wait_if_needed()

    def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        limiter = RateLimiter(max_calls=10, period_seconds=60.0)

        for _ in range(5):
            limiter.wait_if_needed()

        stats = limiter.stats
        assert stats["total_calls"] == 5
        assert stats["current_calls_in_window"] == 5

    def test_decorator_usage(self):
        """Test using rate limiter as decorator."""
        limiter = RateLimiter(max_calls=3, period_seconds=1.0, block_when_limited=False)

        @limiter
        def limited_function():
            return "success"

        # Should work for first 3 calls
        assert limited_function() == "success"
        assert limited_function() == "success"
        assert limited_function() == "success"

        # Fourth call should raise
        with pytest.raises(RateLimitError):
            limited_function()

    def test_sliding_window(self):
        """Test that sliding window correctly expires old calls."""
        limiter = RateLimiter(max_calls=2, period_seconds=0.1, block_when_limited=False)

        limiter.wait_if_needed()
        limiter.wait_if_needed()

        # Wait for the window to expire
        time.sleep(0.15)

        # Should be able to make calls again
        wait = limiter.wait_if_needed()
        assert wait == 0.0

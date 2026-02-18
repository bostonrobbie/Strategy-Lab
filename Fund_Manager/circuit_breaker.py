"""
Circuit Breaker and Rate Limiter patterns for the NHHF system.

These patterns help prevent cascading failures and API overload:
- CircuitBreaker: Stops calls to failing services, allows recovery
- RateLimiter: Throttles API calls to prevent overwhelming services

Usage:
    from circuit_breaker import ollama_breaker, rate_limiter

    @ollama_breaker
    @rate_limiter
    def call_ollama(prompt):
        return ollama.chat(...)
"""
import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass, field
from functools import wraps
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation - calls pass through
    OPEN = "OPEN"          # Failing - calls are rejected
    HALF_OPEN = "HALF_OPEN"  # Testing recovery - limited calls allowed


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker behavior."""
    # Number of failures before opening circuit
    failure_threshold: int = 5

    # Seconds to wait before trying to recover
    recovery_timeout: float = 120.0

    # Number of test calls allowed in half-open state
    half_open_max_calls: int = 3

    # Whether to log state transitions
    log_transitions: bool = True


class CircuitOpenError(Exception):
    """Raised when circuit is open and rejecting calls."""

    def __init__(self, circuit_name: str, time_until_retry: float = 0):
        self.circuit_name = circuit_name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit '{circuit_name}' is OPEN. "
            f"Retry in {time_until_retry:.1f}s"
        )


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Service is failing, calls are rejected immediately
    - HALF_OPEN: Testing if service has recovered

    Usage:
        breaker = CircuitBreaker("ollama_api")

        @breaker
        def call_api(prompt):
            return api.call(prompt)

        # Or manual control:
        if breaker.allow_request():
            try:
                result = api.call(prompt)
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """

    def __init__(self, name: str, config: CircuitConfig = None):
        self.name = name
        self.config = config or CircuitConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejections = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self._state == CircuitState.OPEN

    @property
    def stats(self) -> dict:
        """Get circuit statistics."""
        return {
            'name': self.name,
            'state': self._state.value,
            'failure_count': self._failure_count,
            'total_calls': self._total_calls,
            'total_failures': self._total_failures,
            'total_rejections': self._total_rejections,
            'time_since_last_failure': time.time() - self._last_failure_time if self._last_failure_time else None
        }

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state with optional logging."""
        old_state = self._state
        self._state = new_state

        if self.config.log_transitions and old_state != new_state:
            logger.info(f"[CircuitBreaker:{self.name}] {old_state.value} -> {new_state.value}")

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed.

        Returns:
            True if request is allowed, False if circuit is open
        """
        with self._lock:
            self._total_calls += 1

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                time_since_failure = time.time() - self._last_failure_time

                if time_since_failure >= self.config.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_calls = 0
                    return True

                self._total_rejections += 1
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    return True
                return False

            return False

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

                if self._half_open_calls >= self.config.half_open_max_calls:
                    # Recovered - close the circuit
                    self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0

            elif self._state == CircuitState.CLOSED:
                # Reduce failure count on success (gradual recovery)
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Still failing - reopen the circuit
                self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def reset(self):
        """Manually reset the circuit to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._half_open_calls = 0

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator for protected functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.allow_request():
                time_until_retry = max(
                    0,
                    self.config.recovery_timeout - (time.time() - self._last_failure_time)
                )
                raise CircuitOpenError(self.name, time_until_retry)

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        return wrapper


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    # Maximum calls allowed in the time window
    max_calls: int = 20

    # Time window in seconds
    period_seconds: float = 60.0

    # Whether to block or raise error when limit reached
    block_when_limited: bool = True


class RateLimitError(Exception):
    """Raised when rate limit is exceeded and blocking is disabled."""

    def __init__(self, wait_time: float):
        self.wait_time = wait_time
        super().__init__(f"Rate limit exceeded. Wait {wait_time:.1f}s")


class RateLimiter:
    """
    Sliding window rate limiter for API calls.

    Ensures that no more than max_calls are made within period_seconds.
    Can either block until allowed, or raise RateLimitError.

    Usage:
        limiter = RateLimiter(max_calls=20, period_seconds=60)

        @limiter
        def call_api():
            ...

        # Or manual:
        limiter.wait_if_needed()
        call_api()
    """

    def __init__(
        self,
        max_calls: int = 20,
        period_seconds: float = 60.0,
        block_when_limited: bool = True
    ):
        self.max_calls = max_calls
        self.period = period_seconds
        self.block_when_limited = block_when_limited
        self._calls = deque()
        self._lock = threading.Lock()

        # Statistics
        self._total_calls = 0
        self._total_waits = 0
        self._total_wait_time = 0.0

    @property
    def stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            'max_calls': self.max_calls,
            'period_seconds': self.period,
            'current_calls_in_window': len(self._calls),
            'total_calls': self._total_calls,
            'total_waits': self._total_waits,
            'total_wait_time': self._total_wait_time
        }

    def _cleanup_old_calls(self, now: float):
        """Remove calls outside the time window."""
        cutoff = now - self.period
        while self._calls and self._calls[0] < cutoff:
            self._calls.popleft()

    def wait_if_needed(self) -> float:
        """
        Wait if rate limit would be exceeded.

        Returns:
            Time spent waiting (0 if no wait needed)

        Raises:
            RateLimitError: If blocking is disabled and limit is exceeded
        """
        with self._lock:
            now = time.time()
            self._cleanup_old_calls(now)

            wait_time = 0.0

            if len(self._calls) >= self.max_calls:
                # Need to wait until oldest call expires
                wait_time = self._calls[0] + self.period - now

                if wait_time > 0:
                    if not self.block_when_limited:
                        raise RateLimitError(wait_time)

                    self._total_waits += 1
                    self._total_wait_time += wait_time

                    # Release lock while sleeping
                    self._lock.release()
                    try:
                        time.sleep(wait_time)
                    finally:
                        self._lock.acquire()

                    # Re-check after sleep
                    now = time.time()
                    self._cleanup_old_calls(now)

            self._calls.append(now)
            self._total_calls += 1

            return wait_time

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator for rate-limited functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)

        return wrapper


# ============================================
# Pre-configured instances for common uses
# ============================================

# Circuit breaker for Ollama API
ollama_breaker = CircuitBreaker(
    "ollama_api",
    CircuitConfig(
        failure_threshold=5,
        recovery_timeout=120.0,
        half_open_max_calls=3
    )
)

# Circuit breaker for backtest execution
backtest_breaker = CircuitBreaker(
    "backtest",
    CircuitConfig(
        failure_threshold=3,
        recovery_timeout=300.0,
        half_open_max_calls=2
    )
)

# Rate limiter for Ollama API (20 calls per minute)
ollama_rate_limiter = RateLimiter(
    max_calls=20,
    period_seconds=60.0,
    block_when_limited=True
)


def get_all_breaker_stats() -> dict:
    """Get statistics for all circuit breakers."""
    return {
        'ollama_api': ollama_breaker.stats,
        'backtest': backtest_breaker.stats,
        'rate_limiter': ollama_rate_limiter.stats
    }

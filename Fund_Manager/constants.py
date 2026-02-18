"""
Centralized configuration constants for the NHHF Fund Manager system.

This module provides frozen dataclasses for configuration values that were
previously hardcoded throughout the codebase. Using centralized constants:
- Makes thresholds easy to adjust
- Provides documentation for magic numbers
- Enables configuration validation
- Supports future externalization to config files
"""
from dataclasses import dataclass
from typing import Dict, Any
import os


@dataclass(frozen=True)
class RiskThresholds:
    """
    Risk management thresholds used by the Risk Manager agent.

    These are the criteria for VETO/APPROVE decisions on strategies.
    A strategy must pass ALL criteria to be approved.
    """
    # Maximum allowable drawdown (as decimal, 0.25 = 25%)
    MAX_DRAWDOWN: float = 0.25

    # Minimum required Sharpe ratio
    MIN_SHARPE: float = 1.0

    # Minimum number of trades for statistical significance
    MIN_TRADE_COUNT: int = 30

    # Minimum profit factor (gross profit / gross loss)
    MIN_PROFIT_FACTOR: float = 1.2

    # Maximum allocation to any single strategy (as decimal)
    MAX_SINGLE_STRATEGY_WEIGHT: float = 0.30

    # Minimum win rate (as decimal, 0.4 = 40%)
    MIN_WIN_RATE: float = 0.40

    # Maximum consecutive losses before strategy review
    MAX_CONSECUTIVE_LOSSES: int = 10


@dataclass(frozen=True)
class PortfolioLimits:
    """
    Portfolio construction constraints used by Portfolio Architect.
    """
    # Target portfolio-level Sharpe ratio
    TARGET_SHARPE: float = 1.5

    # Maximum portfolio-level drawdown
    MAX_PORTFOLIO_DD: float = 0.20

    # Minimum number of strategies for diversification
    MIN_STRATEGIES: int = 3

    # Maximum number of strategies (complexity limit)
    MAX_STRATEGIES: int = 7

    # Default starting capital
    DEFAULT_CAPITAL: float = 100000.0

    # Minimum weight for any strategy to be meaningful
    MIN_STRATEGY_WEIGHT: float = 0.05

    # Maximum correlation between strategies (ideal)
    MAX_CORRELATION: float = 0.5


@dataclass(frozen=True)
class SystemLimits:
    """
    System operation limits and timeouts.
    """
    # Timeout for running research scripts (seconds)
    SCRIPT_TIMEOUT_SECONDS: int = 300

    # Maximum consecutive errors before pause
    MAX_ERROR_COUNT: int = 5

    # Pause duration after hitting error threshold (seconds)
    ERROR_PAUSE_SECONDS: int = 300

    # Interval between continuous cycle iterations (seconds)
    CYCLE_INTERVAL_SECONDS: int = 300

    # Maximum entries in memory bank
    MAX_MEMORY_ENTRIES: int = 100

    # Maximum improvement requests to keep
    MAX_REQUEST_ENTRIES: int = 50

    # Maximum errors to keep in log
    MAX_ERROR_ENTRIES: int = 100

    # Top strategies to track on leaderboard
    MAX_LEADERBOARD_ENTRIES: int = 5

    # Maximum portfolio history entries
    MAX_HISTORY_ENTRIES: int = 500

    # Memory query similarity threshold
    MEMORY_SIMILARITY_THRESHOLD: float = 0.8

    # Number of evolution iterations per cycle
    EVOLUTION_ITERATIONS: int = 1


@dataclass(frozen=True)
class FileConfig:
    """
    File operation limits and paths.
    """
    # Maximum code snippet length for logging
    MAX_CODE_SNIPPET_LENGTH: int = 500

    # Maximum prompt length to prevent context overflow
    MAX_PROMPT_LENGTH: int = 4000

    # Maximum response length to display
    MAX_RESPONSE_DISPLAY: int = 2000

    # Maximum dialog entries to keep
    MAX_DIALOG_ENTRIES: int = 100

    # Days to keep sandbox candidates before archiving
    SANDBOX_RETENTION_DAYS: int = 7

    # Maximum sandbox files before cleanup
    MAX_SANDBOX_FILES: int = 50


@dataclass(frozen=True)
class APIConfig:
    """
    API and LLM configuration.
    """
    # Default Ollama model
    DEFAULT_MODEL: str = "llama3"

    # Rate limiting: max calls per period
    MAX_API_CALLS_PER_MINUTE: int = 20

    # Circuit breaker failure threshold
    CIRCUIT_BREAKER_THRESHOLD: int = 5

    # Circuit breaker recovery timeout (seconds)
    CIRCUIT_BREAKER_TIMEOUT: float = 120.0

    # Request timeout (seconds)
    REQUEST_TIMEOUT: int = 60


@dataclass(frozen=True)
class LoggingConfig:
    """
    Logging and reporting configuration.
    """
    # Board log categories
    CATEGORY_IMPROVEMENT: str = "[IMPROVEMENT]"
    CATEGORY_FRICTION: str = "[FRICTION]"
    CATEGORY_REQUEST: str = "[REQUEST]"
    CATEGORY_PORTFOLIO: str = "[PORTFOLIO]"
    CATEGORY_TEAM: str = "[TEAM]"
    CATEGORY_WARNING: str = "[WARNING]"
    CATEGORY_CLEANUP: str = "[CLEANUP]"

    # Agent response markers
    MARKER_APPROVED: str = "APPROVED"
    MARKER_VETO: str = "VETO"
    MARKER_COMPLIANT: str = "[COMPLIANT]"
    MARKER_CONCERN: str = "[CONCERN]"
    MARKER_VIOLATION: str = "[VIOLATION]"
    MARKER_BUG: str = "[BUG]"
    MARKER_WARNING: str = "[WARNING]"
    MARKER_PAUSE: str = "[PAUSE]"
    MARKER_RESET: str = "[RESET]"


# ============================================
# Singleton instances for easy import
# ============================================

RISK = RiskThresholds()
PORTFOLIO = PortfolioLimits()
SYSTEM = SystemLimits()
FILES = FileConfig()
API = APIConfig()
LOGGING = LoggingConfig()


# ============================================
# Helper functions
# ============================================

def get_all_config() -> Dict[str, Any]:
    """
    Get all configuration values as a dictionary.

    Useful for logging current configuration or saving to file.

    Returns:
        Dictionary with all configuration values
    """
    return {
        'risk': {
            'max_drawdown': RISK.MAX_DRAWDOWN,
            'min_sharpe': RISK.MIN_SHARPE,
            'min_trade_count': RISK.MIN_TRADE_COUNT,
            'min_profit_factor': RISK.MIN_PROFIT_FACTOR,
            'max_single_strategy_weight': RISK.MAX_SINGLE_STRATEGY_WEIGHT,
            'min_win_rate': RISK.MIN_WIN_RATE,
            'max_consecutive_losses': RISK.MAX_CONSECUTIVE_LOSSES,
        },
        'portfolio': {
            'target_sharpe': PORTFOLIO.TARGET_SHARPE,
            'max_portfolio_dd': PORTFOLIO.MAX_PORTFOLIO_DD,
            'min_strategies': PORTFOLIO.MIN_STRATEGIES,
            'max_strategies': PORTFOLIO.MAX_STRATEGIES,
            'default_capital': PORTFOLIO.DEFAULT_CAPITAL,
            'min_strategy_weight': PORTFOLIO.MIN_STRATEGY_WEIGHT,
            'max_correlation': PORTFOLIO.MAX_CORRELATION,
        },
        'system': {
            'script_timeout': SYSTEM.SCRIPT_TIMEOUT_SECONDS,
            'max_errors': SYSTEM.MAX_ERROR_COUNT,
            'error_pause': SYSTEM.ERROR_PAUSE_SECONDS,
            'cycle_interval': SYSTEM.CYCLE_INTERVAL_SECONDS,
            'max_memory_entries': SYSTEM.MAX_MEMORY_ENTRIES,
            'evolution_iterations': SYSTEM.EVOLUTION_ITERATIONS,
        },
        'api': {
            'default_model': API.DEFAULT_MODEL,
            'max_calls_per_minute': API.MAX_API_CALLS_PER_MINUTE,
            'circuit_breaker_threshold': API.CIRCUIT_BREAKER_THRESHOLD,
            'circuit_breaker_timeout': API.CIRCUIT_BREAKER_TIMEOUT,
        }
    }


def validate_strategy_metrics(metrics: Dict[str, float]) -> tuple:
    """
    Validate strategy metrics against risk thresholds.

    Args:
        metrics: Dictionary with strategy metrics

    Returns:
        Tuple of (is_valid, list_of_failures)
    """
    failures = []

    sharpe = metrics.get('sharpe', 0)
    if sharpe < RISK.MIN_SHARPE:
        failures.append(f"Sharpe {sharpe:.2f} < {RISK.MIN_SHARPE}")

    max_dd = abs(metrics.get('max_dd', 0))
    if max_dd > RISK.MAX_DRAWDOWN * 100:  # Convert to percentage
        failures.append(f"Max DD {max_dd:.1f}% > {RISK.MAX_DRAWDOWN * 100}%")

    trade_count = metrics.get('trade_count', 0)
    if trade_count < RISK.MIN_TRADE_COUNT:
        failures.append(f"Trades {trade_count} < {RISK.MIN_TRADE_COUNT}")

    profit_factor = metrics.get('profit_factor', 0)
    if profit_factor < RISK.MIN_PROFIT_FACTOR:
        failures.append(f"PF {profit_factor:.2f} < {RISK.MIN_PROFIT_FACTOR}")

    return len(failures) == 0, failures


# Directory paths (computed at import time)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(_BASE_DIR, "reports")
SANDBOX_DIR = os.path.join(_BASE_DIR, "Strategy_Sandbox")

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SANDBOX_DIR, exist_ok=True)

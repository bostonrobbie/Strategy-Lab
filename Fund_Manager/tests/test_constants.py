"""
Tests for the constants module.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import (
    RISK, PORTFOLIO, SYSTEM, FILES, API, LOGGING,
    RiskThresholds, PortfolioLimits, SystemLimits, FileConfig, APIConfig, LoggingConfig,
    get_all_config, validate_strategy_metrics, SANDBOX_DIR, REPORTS_DIR
)


class TestRiskThresholds:
    """Tests for risk threshold constants."""

    def test_defaults_exist(self):
        """Test that default risk thresholds are defined."""
        assert RISK.MAX_DRAWDOWN == 0.25
        assert RISK.MIN_SHARPE == 1.0
        assert RISK.MIN_TRADE_COUNT == 30
        assert RISK.MIN_PROFIT_FACTOR == 1.2
        assert RISK.MAX_SINGLE_STRATEGY_WEIGHT == 0.30
        assert RISK.MIN_WIN_RATE == 0.40
        assert RISK.MAX_CONSECUTIVE_LOSSES == 10

    def test_max_drawdown_reasonable(self):
        """Test that max drawdown is a reasonable value."""
        assert 0 < RISK.MAX_DRAWDOWN <= 0.5

    def test_min_sharpe_positive(self):
        """Test that minimum Sharpe ratio is positive."""
        assert RISK.MIN_SHARPE > 0

    def test_min_trade_count_reasonable(self):
        """Test that minimum trade count is reasonable."""
        assert RISK.MIN_TRADE_COUNT >= 10

    def test_profit_factor_above_one(self):
        """Test that minimum profit factor is above 1."""
        assert RISK.MIN_PROFIT_FACTOR >= 1.0

    def test_single_strategy_weight_limit(self):
        """Test that single strategy weight is limited."""
        assert 0 < RISK.MAX_SINGLE_STRATEGY_WEIGHT < 1.0

    def test_frozen_dataclass(self):
        """Test that RiskThresholds is frozen (immutable)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            RISK.MAX_DRAWDOWN = 0.5


class TestPortfolioLimits:
    """Tests for portfolio limit constants."""

    def test_defaults_exist(self):
        """Test that default portfolio limits are defined."""
        assert PORTFOLIO.TARGET_SHARPE == 1.5
        assert PORTFOLIO.MAX_PORTFOLIO_DD == 0.20
        assert PORTFOLIO.MIN_STRATEGIES == 3
        assert PORTFOLIO.MAX_STRATEGIES == 7
        assert PORTFOLIO.DEFAULT_CAPITAL == 100000.0
        assert PORTFOLIO.MIN_STRATEGY_WEIGHT == 0.05
        assert PORTFOLIO.MAX_CORRELATION == 0.5

    def test_max_strategies_reasonable(self):
        """Test that max strategies is reasonable."""
        assert PORTFOLIO.MAX_STRATEGIES >= 3

    def test_default_capital_positive(self):
        """Test that default capital is positive."""
        assert PORTFOLIO.DEFAULT_CAPITAL > 0

    def test_frozen_dataclass(self):
        """Test that PortfolioLimits is frozen (immutable)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            PORTFOLIO.MAX_STRATEGIES = 10


class TestSystemLimits:
    """Tests for system limit constants."""

    def test_defaults_exist(self):
        """Test that default system limits are defined."""
        assert SYSTEM.SCRIPT_TIMEOUT_SECONDS == 300
        assert SYSTEM.MAX_ERROR_COUNT == 5
        assert SYSTEM.ERROR_PAUSE_SECONDS == 300
        assert SYSTEM.CYCLE_INTERVAL_SECONDS == 300
        assert SYSTEM.MAX_MEMORY_ENTRIES == 100
        assert SYSTEM.MAX_REQUEST_ENTRIES == 50
        assert SYSTEM.MAX_ERROR_ENTRIES == 100
        assert SYSTEM.MAX_LEADERBOARD_ENTRIES == 5
        assert SYSTEM.MAX_HISTORY_ENTRIES == 500
        assert SYSTEM.MEMORY_SIMILARITY_THRESHOLD == 0.8
        assert SYSTEM.EVOLUTION_ITERATIONS == 1

    def test_script_timeout_reasonable(self):
        """Test that script timeout is reasonable."""
        assert 60 <= SYSTEM.SCRIPT_TIMEOUT_SECONDS <= 600

    def test_max_error_count_positive(self):
        """Test that max error count is positive."""
        assert SYSTEM.MAX_ERROR_COUNT > 0

    def test_cycle_interval_reasonable(self):
        """Test that cycle interval is reasonable."""
        assert 60 <= SYSTEM.CYCLE_INTERVAL_SECONDS <= 3600


class TestFileConfig:
    """Tests for file configuration constants."""

    def test_defaults_exist(self):
        """Test that default file config values are defined."""
        assert FILES.MAX_CODE_SNIPPET_LENGTH == 500
        assert FILES.MAX_PROMPT_LENGTH == 4000
        assert FILES.MAX_RESPONSE_DISPLAY == 2000
        assert FILES.MAX_DIALOG_ENTRIES == 100
        assert FILES.SANDBOX_RETENTION_DAYS == 7
        assert FILES.MAX_SANDBOX_FILES == 50


class TestAPIConfig:
    """Tests for API configuration constants."""

    def test_defaults_exist(self):
        """Test that default API config values are defined."""
        assert API.DEFAULT_MODEL == "llama3"
        assert API.MAX_API_CALLS_PER_MINUTE == 20
        assert API.CIRCUIT_BREAKER_THRESHOLD == 5
        assert API.CIRCUIT_BREAKER_TIMEOUT == 120.0
        assert API.REQUEST_TIMEOUT == 60


class TestLoggingConfig:
    """Tests for logging configuration constants."""

    def test_category_constants(self):
        """Test that logging category constants are defined."""
        assert LOGGING.CATEGORY_IMPROVEMENT == "[IMPROVEMENT]"
        assert LOGGING.CATEGORY_FRICTION == "[FRICTION]"
        assert LOGGING.CATEGORY_REQUEST == "[REQUEST]"
        assert LOGGING.CATEGORY_PORTFOLIO == "[PORTFOLIO]"
        assert LOGGING.CATEGORY_TEAM == "[TEAM]"
        assert LOGGING.CATEGORY_WARNING == "[WARNING]"
        assert LOGGING.CATEGORY_CLEANUP == "[CLEANUP]"

    def test_marker_constants(self):
        """Test that marker constants are defined."""
        assert LOGGING.MARKER_APPROVED == "APPROVED"
        assert LOGGING.MARKER_VETO == "VETO"
        assert LOGGING.MARKER_COMPLIANT == "[COMPLIANT]"
        assert LOGGING.MARKER_CONCERN == "[CONCERN]"
        assert LOGGING.MARKER_VIOLATION == "[VIOLATION]"
        assert LOGGING.MARKER_BUG == "[BUG]"
        assert LOGGING.MARKER_WARNING == "[WARNING]"
        assert LOGGING.MARKER_PAUSE == "[PAUSE]"
        assert LOGGING.MARKER_RESET == "[RESET]"


class TestGetAllConfig:
    """Tests for the get_all_config function."""

    def test_returns_dict(self):
        """Test that get_all_config returns a dictionary."""
        config = get_all_config()
        assert isinstance(config, dict)

    def test_contains_all_sections(self):
        """Test that all config sections are present."""
        config = get_all_config()
        assert "risk" in config
        assert "portfolio" in config
        assert "system" in config
        assert "api" in config

    def test_risk_section_values(self):
        """Test that risk section has correct values."""
        config = get_all_config()
        assert config["risk"]["max_drawdown"] == 0.25
        assert config["risk"]["min_sharpe"] == 1.0

    def test_system_section_values(self):
        """Test that system section has correct values."""
        config = get_all_config()
        assert config["system"]["script_timeout"] == 300


class TestValidateStrategyMetrics:
    """Tests for the validate_strategy_metrics function."""

    def test_valid_strategy(self):
        """Test validation of a valid strategy."""
        metrics = {
            "sharpe": 1.5,
            "max_dd": -15.0,
            "trade_count": 50,
            "profit_factor": 1.5
        }
        is_valid, failures = validate_strategy_metrics(metrics)
        assert is_valid is True
        assert len(failures) == 0

    def test_low_sharpe_fails(self):
        """Test that low Sharpe ratio fails validation."""
        metrics = {
            "sharpe": 0.5,  # Below MIN_SHARPE of 1.0
            "max_dd": -15.0,
            "trade_count": 50,
            "profit_factor": 1.5
        }
        is_valid, failures = validate_strategy_metrics(metrics)
        assert is_valid is False
        assert any("Sharpe" in f for f in failures)

    def test_high_drawdown_fails(self):
        """Test that high drawdown fails validation."""
        metrics = {
            "sharpe": 1.5,
            "max_dd": -30.0,  # Above MAX_DRAWDOWN of 25%
            "trade_count": 50,
            "profit_factor": 1.5
        }
        is_valid, failures = validate_strategy_metrics(metrics)
        assert is_valid is False
        assert any("DD" in f for f in failures)

    def test_low_trade_count_fails(self):
        """Test that low trade count fails validation."""
        metrics = {
            "sharpe": 1.5,
            "max_dd": -15.0,
            "trade_count": 10,  # Below MIN_TRADE_COUNT of 30
            "profit_factor": 1.5
        }
        is_valid, failures = validate_strategy_metrics(metrics)
        assert is_valid is False
        assert any("Trades" in f for f in failures)

    def test_low_profit_factor_fails(self):
        """Test that low profit factor fails validation."""
        metrics = {
            "sharpe": 1.5,
            "max_dd": -15.0,
            "trade_count": 50,
            "profit_factor": 0.9  # Below MIN_PROFIT_FACTOR of 1.2
        }
        is_valid, failures = validate_strategy_metrics(metrics)
        assert is_valid is False
        assert any("PF" in f for f in failures)

    def test_multiple_failures(self):
        """Test that multiple failures are all reported."""
        metrics = {
            "sharpe": 0.5,
            "max_dd": -30.0,
            "trade_count": 10,
            "profit_factor": 0.9
        }
        is_valid, failures = validate_strategy_metrics(metrics)
        assert is_valid is False
        assert len(failures) == 4

    def test_empty_metrics(self):
        """Test validation with empty metrics."""
        metrics = {}
        is_valid, failures = validate_strategy_metrics(metrics)
        assert is_valid is False
        # Should fail on all checks with default 0 values


class TestDirectoryPaths:
    """Tests for directory path constants."""

    def test_sandbox_dir_non_empty(self):
        """Test that sandbox directory path is non-empty."""
        assert len(SANDBOX_DIR) > 0

    def test_reports_dir_non_empty(self):
        """Test that reports directory path is non-empty."""
        assert len(REPORTS_DIR) > 0

    def test_sandbox_dir_is_absolute(self):
        """Test that sandbox directory is an absolute path."""
        assert os.path.isabs(SANDBOX_DIR)

    def test_reports_dir_is_absolute(self):
        """Test that reports directory is an absolute path."""
        assert os.path.isabs(REPORTS_DIR)

"""
Tests for the utils module.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import safe_get, extract_nested_metrics, truncate_string, MetricsExtractor, parse_float_safe, parse_int_safe


class TestSafeGet:
    """Tests for the safe_get function."""

    def test_basic_access(self):
        """Test basic dictionary access."""
        data = {"a": {"b": {"c": 123}}}
        assert safe_get(data, "a", "b", "c") == 123

    def test_missing_key_returns_default(self):
        """Test that missing keys return the default value."""
        data = {"a": {"b": 1}}
        assert safe_get(data, "a", "x", default=None) is None
        assert safe_get(data, "a", "x", default=42) == 42

    def test_none_value_in_path(self):
        """Test handling of None values in the path."""
        data = {"a": None}
        assert safe_get(data, "a", "b", default="default") == "default"

    def test_empty_dict(self):
        """Test access on empty dictionary."""
        data = {}
        assert safe_get(data, "a", "b", default="empty") == "empty"

    def test_non_dict_returns_default(self):
        """Test that non-dict values return default."""
        data = {"a": "string_value"}
        assert safe_get(data, "a", "b", default="fail") == "fail"

    def test_single_key(self):
        """Test single key access."""
        data = {"key": "value"}
        assert safe_get(data, "key") == "value"


class TestExtractNestedMetrics:
    """Tests for the extract_nested_metrics function."""

    def test_with_full_data(self, sample_portfolio_data):
        """Test extraction with complete portfolio data."""
        metrics = extract_nested_metrics(sample_portfolio_data)
        assert metrics["sharpe"] == 1.8
        assert metrics["total_return"] == 27.5
        assert metrics["max_drawdown"] == -0.12
        assert metrics["num_strategies"] == 2

    def test_with_empty_dict(self):
        """Test extraction with empty dictionary."""
        metrics = extract_nested_metrics({})
        assert metrics["sharpe"] == 0.0
        assert metrics["total_return"] == 0.0
        assert metrics["num_strategies"] == 0

    def test_with_none(self):
        """Test extraction with None input."""
        metrics = extract_nested_metrics(None)
        assert metrics["sharpe"] == 0.0

    def test_with_missing_portfolio_metrics(self):
        """Test extraction when portfolio_metrics key is missing."""
        data = {"strategies": [], "total_capital": 100000}
        metrics = extract_nested_metrics(data)
        assert metrics["sharpe"] == 0.0
        assert metrics["num_strategies"] == 0


class TestTruncateString:
    """Tests for the truncate_string function."""

    def test_short_string_unchanged(self):
        """Test that short strings are not truncated."""
        assert truncate_string("hello", 10) == "hello"

    def test_long_string_truncated(self):
        """Test that long strings are truncated with ellipsis."""
        result = truncate_string("hello world", 8)
        assert len(result) == 8
        assert result.endswith("...")

    def test_exact_length(self):
        """Test string exactly at max length."""
        assert truncate_string("hello", 5) == "hello"

    def test_empty_string(self):
        """Test empty string input."""
        assert truncate_string("", 10) == ""

    def test_none_input(self):
        """Test None input."""
        assert truncate_string(None, 10) == ""


class TestParseFloatSafe:
    """Tests for the parse_float_safe function."""

    def test_valid_float_string(self):
        """Test parsing valid float string."""
        assert parse_float_safe("3.14") == 3.14

    def test_valid_int_string(self):
        """Test parsing int string to float."""
        assert parse_float_safe("42") == 42.0

    def test_none_returns_default(self):
        """Test that None returns default."""
        assert parse_float_safe(None) == 0.0
        assert parse_float_safe(None, default=1.5) == 1.5

    def test_invalid_string_returns_default(self):
        """Test invalid string returns default."""
        assert parse_float_safe("not a number") == 0.0

    def test_already_float(self):
        """Test passing a float value."""
        assert parse_float_safe(3.14) == 3.14


class TestParseIntSafe:
    """Tests for the parse_int_safe function."""

    def test_valid_int_string(self):
        """Test parsing valid int string."""
        assert parse_int_safe("42") == 42

    def test_float_string_truncated(self):
        """Test parsing float string to int."""
        assert parse_int_safe("3.9") == 3

    def test_none_returns_default(self):
        """Test that None returns default."""
        assert parse_int_safe(None) == 0
        assert parse_int_safe(None, default=10) == 10

    def test_invalid_string_returns_default(self):
        """Test invalid string returns default."""
        assert parse_int_safe("not a number") == 0


class TestMetricsExtractor:
    """Tests for the MetricsExtractor class."""

    def test_from_dict_with_standard_keys(self):
        """Test extraction with standard key names."""
        data = {
            "sharpe": 1.5,
            "return_pct": 25.0,
            "max_dd": -15.0,
            "trade_count": 100
        }
        metrics = MetricsExtractor.from_dict(data)
        assert metrics["sharpe"] == 1.5
        assert metrics["return_pct"] == 25.0
        assert metrics["max_dd"] == -15.0
        assert metrics["trade_count"] == 100

    def test_from_dict_with_alternate_keys(self):
        """Test extraction with alternate key names."""
        data = {
            "Sharpe Ratio": 2.0,
            "Total Return": 30.0,
            "Max Drawdown": -10.0,
            "Trade Count": 50
        }
        metrics = MetricsExtractor.from_dict(data)
        assert metrics["sharpe"] == 2.0
        assert metrics["return_pct"] == 30.0
        assert metrics["max_dd"] == -10.0
        assert metrics["trade_count"] == 50

    def test_from_dict_with_empty_dict(self):
        """Test extraction with empty dictionary."""
        metrics = MetricsExtractor.from_dict({})
        assert metrics == {}

    def test_from_dict_with_none(self):
        """Test extraction with None input."""
        metrics = MetricsExtractor.from_dict(None)
        assert metrics == {}

    def test_from_dict_partial_data(self):
        """Test extraction with partial data."""
        data = {"sharpe": 1.5}
        metrics = MetricsExtractor.from_dict(data)
        assert metrics["sharpe"] == 1.5
        assert "return_pct" not in metrics

"""
Integration tests for the NHHF Fund Manager system.
Tests the interaction between multiple components.
"""
import pytest
import os
import json
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPortfolioIntegration:
    """Integration tests for portfolio management."""

    @pytest.fixture
    def setup_integration(self, temp_dir, monkeypatch):
        """Set up the integration test environment."""
        import portfolio as portfolio_module
        import file_utils as file_utils_module

        # Set up temp paths
        portfolio_path = os.path.join(temp_dir, 'portfolio.json')
        history_path = os.path.join(temp_dir, 'portfolio_history.json')

        monkeypatch.setattr(portfolio_module, 'PORTFOLIO_FILE', portfolio_path)
        monkeypatch.setattr(portfolio_module, 'PORTFOLIO_HISTORY_FILE', history_path)
        monkeypatch.setattr(portfolio_module, '_portfolio_instance', None)

        return {
            'temp_dir': temp_dir,
            'portfolio_path': portfolio_path,
            'history_path': history_path
        }

    def test_portfolio_persistence(self, setup_integration):
        """Test that portfolio changes are persisted to disk."""
        from portfolio import Portfolio

        # Create and modify portfolio
        p1 = Portfolio()
        p1.add_strategy("persistent_strategy", {"sharpe": 1.5, "return_pct": 25.0})

        # Reload from disk
        p2 = Portfolio()

        # Verify data was persisted
        strategy = p2.get_strategy("persistent_strategy")
        assert strategy is not None
        assert strategy["metrics"]["sharpe"] == 1.5

    def test_portfolio_history_logging(self, setup_integration):
        """Test that portfolio changes are logged to history."""
        from portfolio import Portfolio

        p = Portfolio()
        p.add_strategy("logged_strategy", {"sharpe": 1.5})
        p.remove_strategy("logged_strategy")

        # Check history file
        with open(setup_integration['history_path'], 'r') as f:
            history = json.load(f)

        assert len(history) >= 2  # add and remove
        actions = [h["action"] for h in history]
        assert "add" in actions
        assert "remove" in actions


class TestFileOperationsIntegration:
    """Integration tests for file operations."""

    def test_safe_json_round_trip(self, temp_dir):
        """Test saving and loading JSON with file locking."""
        from file_utils import safe_json_save, safe_json_load

        filepath = os.path.join(temp_dir, 'data.json')
        original_data = {
            "key": "value",
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}]
        }

        # Save
        result = safe_json_save(filepath, original_data)
        assert result is True

        # Load
        loaded_data = safe_json_load(filepath)
        assert loaded_data == original_data

    def test_concurrent_access_safety(self, temp_dir):
        """Test that concurrent access doesn't corrupt data."""
        import threading
        from file_utils import safe_json_save, safe_json_load

        filepath = os.path.join(temp_dir, 'concurrent.json')
        safe_json_save(filepath, {"counter": 0})

        errors = []

        def increment():
            try:
                for _ in range(5):
                    data = safe_json_load(filepath, default={"counter": 0})
                    data["counter"] += 1
                    safe_json_save(filepath, data)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=increment) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not have errors (locks should prevent corruption)
        assert len(errors) == 0

        # Data should still be valid JSON
        final_data = safe_json_load(filepath)
        assert "counter" in final_data


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker pattern."""

    def test_circuit_breaker_with_real_function(self):
        """Test circuit breaker integration with actual function calls."""
        from circuit_breaker import CircuitBreaker, CircuitConfig, CircuitOpenError

        call_count = 0
        config = CircuitConfig(failure_threshold=3, recovery_timeout=0.1)
        cb = CircuitBreaker("test_integration", config)

        @cb
        def flaky_function(should_fail=False):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Intentional failure")
            return "success"

        # Successful calls
        assert flaky_function() == "success"
        assert flaky_function() == "success"

        # Failing calls to trip the circuit
        for _ in range(3):
            try:
                flaky_function(should_fail=True)
            except ValueError:
                pass

        # Circuit should be open now
        with pytest.raises(CircuitOpenError):
            flaky_function()

        # Wait for recovery
        time.sleep(0.15)

        # Should be able to try again (half-open state)
        assert flaky_function() == "success"


class TestCodeValidationIntegration:
    """Integration tests for code validation."""

    def test_validate_and_execute(self, temp_dir):
        """Test validating code and then executing it."""
        from code_validator import validate_and_fix_code

        code = '''
import warnings
warnings.filterwarnings('ignore')

def calculate(x, y):
    return x + y

result = calculate(5, 3)
print(f"Result: {result}")
'''
        is_valid, fixed_code, issues = validate_and_fix_code(code)
        assert is_valid is True

        # If valid, we should be able to write and execute
        filepath = os.path.join(temp_dir, 'validated_script.py')
        with open(filepath, 'w') as f:
            f.write(code)

        # Execute and capture output
        import subprocess
        result = subprocess.run(
            ['python', filepath],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0
        assert "Result: 8" in result.stdout


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_extract_metrics_from_real_portfolio_data(self, sample_portfolio_data):
        """Test metric extraction with realistic data."""
        from utils import extract_nested_metrics

        metrics = extract_nested_metrics(sample_portfolio_data)

        assert metrics["sharpe"] == 1.8
        assert metrics["total_return"] == 27.5
        assert metrics["max_drawdown"] == -0.12
        assert metrics["num_strategies"] == 2

    def test_safe_get_with_complex_structure(self):
        """Test safe_get with complex nested structures."""
        from utils import safe_get

        complex_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": 42
                    },
                    "list": [1, 2, 3]
                }
            }
        }

        # Deep access
        assert safe_get(complex_data, "level1", "level2", "level3", "value") == 42

        # Access with missing intermediate key
        assert safe_get(complex_data, "level1", "missing", "value", default="not found") == "not found"

        # Access stopping at non-dict
        assert safe_get(complex_data, "level1", "level2", "list", "item", default="nope") == "nope"


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.fixture
    def workflow_setup(self, temp_dir, monkeypatch):
        """Set up for end-to-end tests."""
        import portfolio as portfolio_module

        portfolio_path = os.path.join(temp_dir, 'portfolio.json')
        history_path = os.path.join(temp_dir, 'portfolio_history.json')

        monkeypatch.setattr(portfolio_module, 'PORTFOLIO_FILE', portfolio_path)
        monkeypatch.setattr(portfolio_module, 'PORTFOLIO_HISTORY_FILE', history_path)
        monkeypatch.setattr(portfolio_module, '_portfolio_instance', None)

        return temp_dir

    def test_strategy_lifecycle(self, workflow_setup):
        """Test complete strategy lifecycle: add, update, remove."""
        from portfolio import Portfolio

        portfolio = Portfolio()

        # 1. Add strategy
        strategy = portfolio.add_strategy(
            "lifecycle_test",
            {
                "sharpe": 1.5,
                "return_pct": 20.0,
                "max_dd": -0.15,
                "trade_count": 50
            },
            logic_summary="Test RSI strategy for lifecycle testing"
        )

        assert strategy["status"] == "active"
        initial_weight = strategy["weight"]

        # 2. Update with new metrics
        portfolio.add_strategy(
            "lifecycle_test",
            {
                "sharpe": 2.0,  # Improved
                "return_pct": 30.0,
                "max_dd": -0.10,
                "trade_count": 75
            }
        )

        updated = portfolio.get_strategy("lifecycle_test")
        assert updated["metrics"]["sharpe"] == 2.0

        # 3. Pause strategy
        portfolio.update_strategy_status("lifecycle_test", "paused")
        paused = portfolio.get_strategy("lifecycle_test")
        assert paused["status"] == "paused"

        # 4. Verify it's not in active strategies
        active = portfolio.get_active_strategies()
        assert not any(s["name"] == "lifecycle_test" for s in active)

        # 5. Reactivate
        portfolio.update_strategy_status("lifecycle_test", "active")
        reactivated = portfolio.get_strategy("lifecycle_test")
        assert reactivated["status"] == "active"

        # 6. Remove
        removed = portfolio.remove_strategy("lifecycle_test")
        assert removed is True

        # 7. Verify removal
        gone = portfolio.get_strategy("lifecycle_test")
        assert gone is None

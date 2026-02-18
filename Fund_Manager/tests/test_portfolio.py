"""
Tests for the portfolio module.
"""
import pytest
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio import Portfolio


class TestPortfolio:
    """Tests for the Portfolio class."""

    @pytest.fixture
    def portfolio(self, temp_dir, monkeypatch):
        """Create a fresh portfolio instance with temp directory."""
        # Monkeypatch the file paths
        import portfolio as portfolio_module
        monkeypatch.setattr(portfolio_module, 'PORTFOLIO_FILE',
                          os.path.join(temp_dir, 'portfolio.json'))
        monkeypatch.setattr(portfolio_module, 'PORTFOLIO_HISTORY_FILE',
                          os.path.join(temp_dir, 'portfolio_history.json'))

        # Reset the singleton
        monkeypatch.setattr(portfolio_module, '_portfolio_instance', None)

        return Portfolio()

    def test_initial_state(self, portfolio):
        """Test portfolio initializes with correct defaults."""
        assert portfolio.data["strategies"] == []
        assert portfolio.data["total_capital"] == 100000.0
        assert portfolio.data["portfolio_metrics"]["num_strategies"] == 0

    def test_add_strategy(self, portfolio):
        """Test adding a strategy to the portfolio."""
        metrics = {
            "sharpe": 1.5,
            "return_pct": 25.0,
            "max_dd": -0.15,
            "profit_factor": 1.8,
            "win_rate": 0.55,
            "trade_count": 100
        }

        strategy = portfolio.add_strategy("test_strategy", metrics)

        assert strategy["name"] == "test_strategy"
        assert strategy["metrics"]["sharpe"] == 1.5
        assert strategy["status"] == "active"

    def test_add_multiple_strategies(self, portfolio):
        """Test adding multiple strategies with rebalancing."""
        portfolio.add_strategy("strategy_1", {"sharpe": 1.5, "return_pct": 20.0})
        portfolio.add_strategy("strategy_2", {"sharpe": 2.0, "return_pct": 30.0})

        strategies = portfolio.get_active_strategies()
        assert len(strategies) == 2

        # Force rebalance to ensure weights sum to 1.0 (lazy rebalancing defers this)
        portfolio.force_rebalance()

        # Weights should sum to 1.0 after rebalancing
        total_weight = sum(s["weight"] for s in strategies)
        assert abs(total_weight - 1.0) < 0.01

    def test_lazy_rebalancing_threshold(self, portfolio):
        """Test that lazy rebalancing triggers after threshold changes."""
        # Add strategies below threshold (3) - no automatic rebalance
        portfolio.add_strategy("strat_1", {"sharpe": 1.0})
        portfolio.add_strategy("strat_2", {"sharpe": 1.5})

        # Should need rebalance but not triggered yet
        assert portfolio.needs_rebalance() is True
        assert portfolio._pending_changes == 2

        # Add third strategy - should trigger automatic rebalance
        portfolio.add_strategy("strat_3", {"sharpe": 2.0})

        # After threshold reached, rebalance should have happened
        assert portfolio._pending_changes == 0
        assert portfolio.needs_rebalance() is False

        # Weights should now sum to 1.0
        strategies = portfolio.get_active_strategies()
        total_weight = sum(s["weight"] for s in strategies)
        assert abs(total_weight - 1.0) < 0.01

    def test_force_rebalance(self, portfolio):
        """Test that force_rebalance works before threshold."""
        portfolio.add_strategy("strat_1", {"sharpe": 1.0})

        # Manually force rebalance
        portfolio.force_rebalance()

        # Should reset tracking
        assert portfolio._pending_changes == 0
        assert portfolio.needs_rebalance() is False

    def test_remove_strategy(self, portfolio):
        """Test removing a strategy from the portfolio."""
        portfolio.add_strategy("to_remove", {"sharpe": 1.0})
        portfolio.add_strategy("to_keep", {"sharpe": 1.5})

        result = portfolio.remove_strategy("to_remove")
        assert result is True

        strategies = portfolio.get_all_strategies()
        assert len(strategies) == 1
        assert strategies[0]["name"] == "to_keep"

    def test_remove_nonexistent_strategy(self, portfolio):
        """Test removing a strategy that doesn't exist."""
        result = portfolio.remove_strategy("nonexistent")
        assert result is False

    def test_update_strategy_status(self, portfolio):
        """Test updating strategy status."""
        portfolio.add_strategy("test", {"sharpe": 1.5})

        result = portfolio.update_strategy_status("test", "paused")
        assert result is True

        strategy = portfolio.get_strategy("test")
        assert strategy["status"] == "paused"

    def test_get_strategy(self, portfolio):
        """Test getting a specific strategy."""
        portfolio.add_strategy("find_me", {"sharpe": 1.5})

        strategy = portfolio.get_strategy("find_me")
        assert strategy is not None
        assert strategy["name"] == "find_me"

        not_found = portfolio.get_strategy("nonexistent")
        assert not_found is None

    def test_get_active_strategies(self, portfolio):
        """Test getting only active strategies."""
        portfolio.add_strategy("active_1", {"sharpe": 1.5})
        portfolio.add_strategy("active_2", {"sharpe": 2.0})
        portfolio.add_strategy("paused", {"sharpe": 1.0})
        portfolio.update_strategy_status("paused", "paused")

        active = portfolio.get_active_strategies()
        assert len(active) == 2
        assert all(s["status"] == "active" for s in active)

    def test_portfolio_metrics_calculation(self, portfolio):
        """Test that portfolio metrics are calculated correctly."""
        portfolio.add_strategy("high_sharpe", {
            "sharpe": 2.0,
            "return_pct": 30.0,
            "max_dd": -0.10
        })
        portfolio.add_strategy("low_sharpe", {
            "sharpe": 1.0,
            "return_pct": 15.0,
            "max_dd": -0.20
        })

        summary = portfolio.get_portfolio_summary()
        metrics = summary["portfolio_metrics"]

        # Weighted average should be between the individual values
        assert 1.0 <= metrics["sharpe"] <= 2.0
        assert metrics["num_strategies"] == 2

    def test_set_total_capital(self, portfolio):
        """Test setting total capital."""
        portfolio.set_total_capital(250000.0)
        assert portfolio.data["total_capital"] == 250000.0

    def test_get_allocation_breakdown(self, portfolio):
        """Test getting capital allocation breakdown."""
        portfolio.set_total_capital(100000.0)
        portfolio.add_strategy("strat_a", {"sharpe": 1.5}, weight=0.6)
        portfolio.add_strategy("strat_b", {"sharpe": 1.0}, weight=0.4)

        allocations = portfolio.get_allocation_breakdown()
        assert len(allocations) == 2

        total_allocated = sum(a["allocated_capital"] for a in allocations)
        assert abs(total_allocated - 100000.0) < 0.01

    def test_weight_calculation_by_sharpe(self, portfolio):
        """Test that strategies with higher Sharpe get higher allocation."""
        # Add all strategies together so weights are calculated relative to each other
        portfolio.add_strategy("high", {"sharpe": 2.5})
        portfolio.add_strategy("medium", {"sharpe": 1.5})
        portfolio.add_strategy("low", {"sharpe": 0.5})

        # Get current weights
        s1 = portfolio.get_strategy("high")
        s2 = portfolio.get_strategy("medium")
        s3 = portfolio.get_strategy("low")

        # Higher Sharpe should get higher or equal weight
        assert s1["weight"] >= s2["weight"]
        assert s2["weight"] >= s3["weight"]

        # All weights should be positive and sum to 1
        total_weight = s1["weight"] + s2["weight"] + s3["weight"]
        assert abs(total_weight - 1.0) < 0.01

    def test_update_strategy_equity(self, portfolio):
        """Test updating strategy equity curve."""
        portfolio.add_strategy("test", {"sharpe": 1.5})

        equity_point = {
            "timestamp": "2024-01-01T00:00:00",
            "equity": 10500.0,
            "pnl": 500.0
        }

        result = portfolio.update_strategy_equity("test", equity_point)
        assert result is True

        curve = portfolio.get_strategy_equity_curve("test")
        assert len(curve) == 1
        assert curve[0]["equity"] == 10500.0

    def test_equity_curve_limit(self, portfolio):
        """Test that equity curve is limited to 1000 points."""
        portfolio.add_strategy("test", {"sharpe": 1.5})

        for i in range(1100):
            portfolio.update_strategy_equity("test", {
                "timestamp": f"2024-01-01T{i:05d}",
                "equity": 10000 + i
            })

        curve = portfolio.get_strategy_equity_curve("test")
        assert len(curve) == 1000

    def test_portfolio_summary(self, portfolio):
        """Test portfolio summary generation."""
        portfolio.add_strategy("active", {"sharpe": 1.5})
        portfolio.add_strategy("paused", {"sharpe": 1.0})
        portfolio.update_strategy_status("paused", "paused")

        summary = portfolio.get_portfolio_summary()

        assert summary["total_strategies"] == 2
        assert summary["active_count"] == 1
        assert summary["paused_count"] == 1
        assert "strategies" in summary
        assert len(summary["strategies"]) == 2

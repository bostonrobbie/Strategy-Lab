"""
Portfolio Management System
Tracks approved strategies, their allocations, and overall portfolio performance.
"""

import json
import os
from datetime import datetime

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")
PORTFOLIO_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "portfolio_history.json")


class Portfolio:
    """
    Manages the collection of approved trading strategies.

    Features lazy rebalancing to avoid unnecessary computation.
    Rebalancing is deferred until:
    1. A threshold number of changes accumulate (default: 3)
    2. force_rebalance() is explicitly called
    3. A portfolio summary is requested
    """

    # Number of changes before automatic rebalancing
    REBALANCE_THRESHOLD = 3

    def __init__(self):
        self.data = self._load()
        self._pending_changes = 0  # Count of changes since last rebalance
        self._needs_rebalance = False  # Flag for deferred rebalancing

    def _load(self):
        if os.path.exists(PORTFOLIO_FILE):
            try:
                with open(PORTFOLIO_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"[Portfolio] Warning: Corrupted portfolio file, creating backup and starting fresh: {e}")
                # Create backup of corrupted file
                import shutil
                backup_path = PORTFOLIO_FILE + f".corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    shutil.copy2(PORTFOLIO_FILE, backup_path)
                except Exception:
                    pass
            except IOError as e:
                print(f"[Portfolio] Warning: Could not read portfolio file: {e}")
            except Exception as e:
                print(f"[Portfolio] Unexpected error loading portfolio: {e}")
        return {
            "strategies": [],
            "total_capital": 100000.0,
            "last_updated": None,
            "portfolio_metrics": {
                "sharpe": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "num_strategies": 0
            }
        }

    def _save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(self.data, f, indent=2)

    def add_strategy(self, name, metrics, weight=None, code_path=None, logic_summary=""):
        """
        Add an approved strategy to the portfolio.

        Args:
            name: Strategy name/identifier
            metrics: Dict with sharpe, return_pct, max_dd, etc.
            weight: Allocation weight (0-1). If None, auto-calculated.
            code_path: Path to the strategy code file
            logic_summary: Brief description of the strategy logic
        """
        # Check if strategy already exists
        for s in self.data["strategies"]:
            if s["name"] == name:
                # Update existing
                s["metrics"] = metrics
                s["weight"] = weight or s.get("weight", 0.1)
                s["updated_at"] = datetime.now().isoformat()
                if code_path:
                    s["code_path"] = code_path
                if logic_summary:
                    s["logic_summary"] = logic_summary
                self._rebalance()
                self._save()
                return s

        # Add new strategy
        strategy = {
            "name": name,
            "metrics": {
                "sharpe": metrics.get("sharpe", metrics.get("Sharpe Ratio", 0)),
                "return_pct": metrics.get("return_pct", metrics.get("Total Return", 0)),
                "max_dd": metrics.get("max_dd", metrics.get("Max Drawdown", 0)),
                "profit_factor": metrics.get("profit_factor", metrics.get("Profit Factor", 1.0)),
                "win_rate": metrics.get("win_rate", metrics.get("Win Rate", 0.5)),
                "trade_count": metrics.get("trade_count", metrics.get("Trade Count", 0))
            },
            "weight": weight or self._calculate_weight(metrics),
            "status": "active",
            "added_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "code_path": code_path or "",
            "logic_summary": logic_summary or "",
            "equity_curve": [],  # Will store equity data points
            "performance_history": []  # Will store periodic performance snapshots
        }

        self.data["strategies"].append(strategy)
        self._mark_needs_rebalance()
        self._update_portfolio_metrics()
        self._save()
        self._log_history("add", name, strategy)
        return strategy

    def remove_strategy(self, name):
        """Remove a strategy from the portfolio."""
        for i, s in enumerate(self.data["strategies"]):
            if s["name"] == name:
                removed = self.data["strategies"].pop(i)
                self._mark_needs_rebalance()
                self._update_portfolio_metrics()
                self._save()
                self._log_history("remove", name, removed)
                return True
        return False

    def update_strategy_status(self, name, status):
        """Update strategy status (active, paused, deprecated)."""
        for s in self.data["strategies"]:
            if s["name"] == name:
                s["status"] = status
                s["updated_at"] = datetime.now().isoformat()
                self._save()
                return True
        return False

    def update_strategy_equity(self, name, equity_point):
        """
        Add an equity data point for a strategy.
        equity_point: {"timestamp": iso_string, "equity": float, "pnl": float}
        """
        for s in self.data["strategies"]:
            if s["name"] == name:
                if "equity_curve" not in s:
                    s["equity_curve"] = []
                s["equity_curve"].append(equity_point)
                # Keep last 1000 data points
                s["equity_curve"] = s["equity_curve"][-1000:]
                self._save()
                return True
        return False

    def _calculate_weight(self, metrics):
        """Calculate initial weight based on strategy quality."""
        sharpe = float(metrics.get("sharpe", metrics.get("Sharpe Ratio", 0)))

        # Simple weight based on Sharpe ratio
        if sharpe >= 2.0:
            return 0.25
        elif sharpe >= 1.5:
            return 0.20
        elif sharpe >= 1.0:
            return 0.15
        else:
            return 0.10

    def _mark_needs_rebalance(self):
        """Mark that rebalancing is needed and check threshold."""
        self._pending_changes += 1
        self._needs_rebalance = True

        # Auto-rebalance if threshold reached
        if self._pending_changes >= self.REBALANCE_THRESHOLD:
            self._rebalance()

    def _rebalance(self):
        """
        Ensure weights sum to 1.0 for active strategies.

        Called automatically when threshold reached or explicitly via force_rebalance().
        """
        active = [s for s in self.data["strategies"] if s.get("status") == "active"]
        if not active:
            self._pending_changes = 0
            self._needs_rebalance = False
            return

        total_weight = sum(s.get("weight", 0) for s in active)
        if total_weight > 0:
            for s in active:
                s["weight"] = s.get("weight", 0) / total_weight

        # Reset tracking
        self._pending_changes = 0
        self._needs_rebalance = False

    def force_rebalance(self):
        """
        Force immediate rebalancing regardless of threshold.

        Call this when you need weights to be accurate immediately.
        """
        self._rebalance()
        self._update_portfolio_metrics()
        self._save()

    def needs_rebalance(self) -> bool:
        """Check if rebalancing is pending."""
        return self._needs_rebalance

    def _update_portfolio_metrics(self):
        """Calculate aggregate portfolio metrics."""
        active = [s for s in self.data["strategies"] if s.get("status") == "active"]

        if not active:
            self.data["portfolio_metrics"] = {
                "sharpe": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "num_strategies": 0
            }
            return

        # Weighted average of metrics
        total_weight = sum(s.get("weight", 0) for s in active)

        weighted_sharpe = sum(
            s["metrics"].get("sharpe", 0) * s.get("weight", 0)
            for s in active
        ) / max(total_weight, 0.01)

        weighted_return = sum(
            s["metrics"].get("return_pct", 0) * s.get("weight", 0)
            for s in active
        ) / max(total_weight, 0.01)

        # For max DD, we'd need correlation data for accurate calc
        # Using simple weighted average as approximation
        weighted_dd = sum(
            abs(s["metrics"].get("max_dd", 0)) * s.get("weight", 0)
            for s in active
        ) / max(total_weight, 0.01)

        self.data["portfolio_metrics"] = {
            "sharpe": round(weighted_sharpe, 2),
            "total_return": round(weighted_return, 2),
            "max_drawdown": round(-weighted_dd, 2),
            "num_strategies": len(active)
        }

    def _log_history(self, action, name, data):
        """Log portfolio changes for audit trail."""
        history = []
        if os.path.exists(PORTFOLIO_HISTORY_FILE):
            try:
                with open(PORTFOLIO_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                # History file corrupted or unreadable - start fresh
                print(f"[Portfolio] Warning: Could not load history file: {e}")
                history = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "strategy_name": name,
            "data_snapshot": {
                "metrics": data.get("metrics", {}),
                "weight": data.get("weight", 0)
            }
        }
        history.append(entry)

        # Keep last 500 entries
        history = history[-500:]

        with open(PORTFOLIO_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

    def get_all_strategies(self):
        """Get all strategies in the portfolio."""
        return self.data["strategies"]

    def get_active_strategies(self):
        """Get only active strategies."""
        return [s for s in self.data["strategies"] if s.get("status") == "active"]

    def get_strategy(self, name):
        """Get a specific strategy by name."""
        for s in self.data["strategies"]:
            if s["name"] == name:
                return s
        return None

    def get_portfolio_summary(self):
        """Get a summary of the portfolio.

        Note: Forces rebalancing if needed to ensure accurate weights.
        """
        # Ensure weights are accurate before returning summary
        if self._needs_rebalance:
            self._rebalance()

        active = self.get_active_strategies()
        paused = [s for s in self.data["strategies"] if s.get("status") == "paused"]
        deprecated = [s for s in self.data["strategies"] if s.get("status") == "deprecated"]

        return {
            "total_strategies": len(self.data["strategies"]),
            "active_count": len(active),
            "paused_count": len(paused),
            "deprecated_count": len(deprecated),
            "total_capital": self.data.get("total_capital", 100000),
            "portfolio_metrics": self.data.get("portfolio_metrics", {}),
            "last_updated": self.data.get("last_updated"),
            "strategies": [
                {
                    "name": s["name"],
                    "weight": round(s.get("weight", 0) * 100, 1),
                    "sharpe": s["metrics"].get("sharpe", 0),
                    "return_pct": s["metrics"].get("return_pct", 0),
                    "status": s.get("status", "active")
                }
                for s in self.data["strategies"]
            ]
        }

    def get_strategy_equity_curve(self, name):
        """Get equity curve data for a strategy."""
        for s in self.data["strategies"]:
            if s["name"] == name:
                return s.get("equity_curve", [])
        return []

    def set_total_capital(self, amount):
        """Set the total portfolio capital."""
        self.data["total_capital"] = amount
        self._save()

    def get_allocation_breakdown(self):
        """Get capital allocation for each strategy.

        Note: Forces rebalancing if needed to ensure accurate weights.
        """
        # Ensure weights are accurate before returning allocation
        if self._needs_rebalance:
            self._rebalance()

        total = self.data.get("total_capital", 100000)
        active = self.get_active_strategies()

        return [
            {
                "name": s["name"],
                "weight_pct": round(s.get("weight", 0) * 100, 1),
                "allocated_capital": round(total * s.get("weight", 0), 2),
                "sharpe": s["metrics"].get("sharpe", 0),
                "status": s.get("status", "active")
            }
            for s in active
        ]


# Singleton instance for easy access
_portfolio_instance = None

def get_portfolio():
    global _portfolio_instance
    if _portfolio_instance is None:
        _portfolio_instance = Portfolio()
    return _portfolio_instance

"""
Pytest fixtures for NHHF Fund Manager tests.
"""
import pytest
import os
import sys
import json
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing."""
    return {
        "strategies": [
            {
                "name": "test_strategy_1",
                "metrics": {
                    "sharpe": 1.5,
                    "return_pct": 25.0,
                    "max_dd": -0.15,
                    "profit_factor": 1.8,
                    "win_rate": 0.55,
                    "trade_count": 100
                },
                "weight": 0.4,
                "status": "active",
                "added_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "code_path": "/path/to/strategy.py",
                "logic_summary": "Test RSI strategy",
                "equity_curve": [],
                "performance_history": []
            },
            {
                "name": "test_strategy_2",
                "metrics": {
                    "sharpe": 2.0,
                    "return_pct": 30.0,
                    "max_dd": -0.10,
                    "profit_factor": 2.0,
                    "win_rate": 0.60,
                    "trade_count": 80
                },
                "weight": 0.6,
                "status": "active",
                "added_at": "2024-01-02T00:00:00",
                "updated_at": "2024-01-02T00:00:00",
                "code_path": "/path/to/strategy2.py",
                "logic_summary": "Test momentum strategy",
                "equity_curve": [],
                "performance_history": []
            }
        ],
        "total_capital": 100000.0,
        "last_updated": "2024-01-02T00:00:00",
        "portfolio_metrics": {
            "sharpe": 1.8,
            "total_return": 27.5,
            "max_drawdown": -0.12,
            "num_strategies": 2
        }
    }


@pytest.fixture
def sample_leaderboard_data():
    """Sample leaderboard data for testing."""
    return {
        "strategies": [
            {"name": "strategy_1", "sharpe": 2.5, "return_pct": 35.0, "max_dd": -0.08},
            {"name": "strategy_2", "sharpe": 2.0, "return_pct": 28.0, "max_dd": -0.12},
            {"name": "strategy_3", "sharpe": 1.5, "return_pct": 20.0, "max_dd": -0.15}
        ],
        "last_updated": "2024-01-01T00:00:00"
    }


@pytest.fixture
def sample_memory_data():
    """Sample memory bank data for testing."""
    return [
        {
            "timestamp": "2024-01-01T00:00:00",
            "summary": "Tested RSI period 14 on NQ",
            "outcome": "SUCCESS",
            "metrics": {"sharpe": 1.5, "return_pct": 25.0},
            "embedding": [0.1] * 1024
        },
        {
            "timestamp": "2024-01-02T00:00:00",
            "summary": "Tested momentum strategy on ES",
            "outcome": "FAILURE",
            "metrics": {"sharpe": 0.5, "return_pct": -5.0},
            "embedding": [0.2] * 1024
        }
    ]


@pytest.fixture
def sample_backtest_metrics():
    """Sample backtest metrics output for testing."""
    return {
        "Sharpe Ratio": 1.75,
        "Total Return": 28.5,
        "Max Drawdown": -0.12,
        "Trade Count": 150,
        "Win Rate": 0.58,
        "Profit Factor": 1.65
    }


@pytest.fixture
def sample_strategy_code():
    """Sample strategy code for testing."""
    return '''
import warnings
import os
import sys
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

def run_strategy(df):
    """Simple moving average crossover strategy."""
    df['sma_fast'] = df['close'].rolling(10).mean()
    df['sma_slow'] = df['close'].rolling(50).mean()
    df['signal'] = np.where(df['sma_fast'] > df['sma_slow'], 1, -1)
    return df

if __name__ == "__main__":
    print("Running strategy...")
'''


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response for testing."""
    return {
        "message": {
            "content": "This is a test response from the LLM."
        }
    }


@pytest.fixture
def sample_error_log():
    """Sample error log entries for testing."""
    return [
        {
            "timestamp": "2024-01-01T10:00:00",
            "agent": "System",
            "error_type": "cycle_error",
            "message": "KeyError: 'num_strategies'",
            "context": {},
            "resolved": False
        },
        {
            "timestamp": "2024-01-01T11:00:00",
            "agent": "Code Reviewer",
            "error_type": "code_bug",
            "message": "Missing import detected",
            "context": {"candidate": "test_strategy.py"},
            "resolved": False
        }
    ]

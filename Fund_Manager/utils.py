"""
Utility functions for the Fund Manager system.
"""
from typing import Any, Dict, List, Optional, TypeVar
import logging

T = TypeVar('T')

logger = logging.getLogger(__name__)


def safe_get(data: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely traverse nested dictionaries without raising KeyError.

    Args:
        data: The dictionary or nested structure to traverse
        *keys: Variable number of keys to traverse
        default: Value to return if any key is not found

    Returns:
        The value at the nested key path, or default if not found

    Examples:
        >>> data = {'a': {'b': {'c': 1}}}
        >>> safe_get(data, 'a', 'b', 'c')
        1
        >>> safe_get(data, 'a', 'x', 'y', default=0)
        0
        >>> safe_get({}, 'portfolio_metrics', 'sharpe', default=0.0)
        0.0
    """
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return default
        if data is None:
            return default
    return data if data is not None else default


def safe_dict_get(d: Optional[Dict], key: str, default: T = None) -> T:
    """
    Safely get a value from a dictionary that might be None.

    Args:
        d: Dictionary (can be None)
        key: Key to retrieve
        default: Default value if dict is None or key not found

    Returns:
        The value or default
    """
    if d is None:
        return default
    return d.get(key, default)


def extract_nested_metrics(portfolio_summary: Dict) -> Dict[str, Any]:
    """
    Extract portfolio metrics safely from a portfolio summary.

    This function handles cases where:
    - portfolio_summary is None
    - portfolio_metrics key doesn't exist
    - Individual metric keys don't exist

    Args:
        portfolio_summary: The portfolio summary dictionary

    Returns:
        Dictionary with all metrics, using defaults for missing values
    """
    if not portfolio_summary:
        return {
            'sharpe': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'num_strategies': 0,
            'active_count': 0
        }

    pm = portfolio_summary.get('portfolio_metrics', {}) or {}

    return {
        'sharpe': pm.get('sharpe', 0.0),
        'total_return': pm.get('total_return', 0.0),
        'max_drawdown': pm.get('max_drawdown', 0.0),
        'num_strategies': pm.get('num_strategies', 0),
        'active_count': portfolio_summary.get('active_count', 0)
    }


def truncate_string(s: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length, adding suffix if truncated.

    Args:
        s: String to truncate
        max_length: Maximum length (including suffix)
        suffix: String to append if truncated

    Returns:
        Truncated string
    """
    if not s:
        return ""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def parse_float_safe(value: Any, default: float = 0.0) -> float:
    """
    Safely parse a value to float.

    Args:
        value: Value to parse (can be str, int, float, or None)
        default: Default value if parsing fails

    Returns:
        Parsed float or default
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_int_safe(value: Any, default: int = 0) -> int:
    """
    Safely parse a value to int.

    Args:
        value: Value to parse (can be str, int, float, or None)
        default: Default value if parsing fails

    Returns:
        Parsed int or default
    """
    if value is None:
        return default
    try:
        return int(float(value))  # Handle "1.0" -> 1
    except (ValueError, TypeError):
        return default


class MetricsExtractor:
    """
    Helper class for extracting metrics from various output formats.
    """

    @staticmethod
    def from_dict(d: Optional[Dict], key_mappings: Optional[Dict[str, List[str]]] = None) -> Dict[str, float]:
        """
        Extract metrics from a dictionary, handling multiple possible key names.

        Args:
            d: Source dictionary
            key_mappings: Dict mapping canonical names to list of possible key names

        Returns:
            Dictionary with canonical metric names
        """
        if not d:
            return {}

        default_mappings = {
            'sharpe': ['sharpe', 'Sharpe Ratio', 'sharpe_ratio', 'SR'],
            'return_pct': ['return_pct', 'Total Return', 'total_return', 'return', 'CAGR'],
            'max_dd': ['max_dd', 'Max Drawdown', 'max_drawdown', 'MDD'],
            'profit_factor': ['profit_factor', 'Profit Factor'],
            'win_rate': ['win_rate', 'Win Rate', 'winrate'],
            'trade_count': ['trade_count', 'Trade Count', 'trades', 'num_trades']
        }

        mappings = key_mappings or default_mappings
        result = {}

        for canonical_name, possible_keys in mappings.items():
            for key in possible_keys:
                if key in d and d[key] is not None:
                    result[canonical_name] = parse_float_safe(d[key])
                    break

        return result

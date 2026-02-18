import pandas as pd
import numpy as np
from datetime import datetime, time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

def evaluate_sharpe_ratio(m):
    return m['Sharpe Ratio'] > 1.0

def evaluate_max_drawdown(m):
    return m['Max Drawdown'] > -0.25

def evaluate_profit_factor(m):
    return m['Profit Factor'] > 1.2

def evaluate_trade_count(m):
    return m['Trade Count'] > 30

def filter_results(results):
    approved = [name for name, m in results.items() if 
                evaluate_sharpe_ratio(m) and 
                evaluate_max_drawdown(m) and 
                evaluate_profit_factor(m) and 
                evaluate_trade_count(m)]
    
    if approved:
        return f"DEPLOY: {', '.join(approved)} meet all Risk Manager criteria"
    elif best_strat[1]['Sharpe Ratio'] > 0.5:
        return f"REFINE: {best_strat[0]} shows promise but needs optimization\n  Suggestions:\n  - Adjust entry/exit criteria\n  - Optimize ATR multipliers for stops/targets\n  - Add additional filters (volume, time-of-day)"
    else:
        return "RESEARCH: All strategies need fundamental redesign\n  Suggestions:\n  - Review market regime detection\n  - Consider different timeframes\n  - Explore alternative signal generation methods"

def run_multi_backtest(start_date, end_date):
    # Your code here
    results = {}
    for name in strategy_names:
        m = backtest(name)
        results[name] = m
    
    best_strat = max(results.values(), key=lambda x: x['Sharpe Ratio'])
    
    return filter_results(results), best_strat

if __name__ == "__main__":
    run_multi_backtest("2022-01-01", "2024-12-31")
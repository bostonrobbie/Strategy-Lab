import pandas as pd
import numpy as np
from datetime import datetime, time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

def evaluate_strategy(name, m):
    if m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] > -0.25 and m['Profit Factor'] > 1.2 and m['Trade Count'] > 30:
        return "APPROVED"
    else:
        return "VETO"

def run_multi_backtest(start, end):
    results = {}
    best_strat = None
    best_sharpe = -1

    for name, m in results.items():
        passes = []
        fails = []

        if m['Sharpe Ratio'] > 0.5:
            passes.append(f"Sharpe Ratio={m['Sharpe Ratio']}")
        else:
            fails.append("Sharpe Ratio=<0.5")

        if m['Max Drawdown'] > -0.25:
            passes.append(f"Max Drawdown={m['Max Drawdown']}")
        else:
            fails.append(f"Max Drawdown=<{-0.25}")

        if m['Profit Factor'] > 1.2:
            passes.append(f"Profit Factor={m['Profit Factor']}")
        else:
            fails.append(f"Profit Factor=<1.2")

        if m['Trade Count'] > 30:
            passes.append(f"Trades={m['Trade Count']} (>30)")
        else:
            fails.append(f"Trades={m['Trade Count']} (<30)")

        status = evaluate_strategy(name, m)
        print(f"{name}: {status}")
        if passes:
            print(f"  [+] Passes: {', '.join(passes)}")
        if fails:
            print(f"  [-] Fails: {', '.join(fails)}")
        print("")

    # Recommendation
    print(f"{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")

    approved = [name for name, m in results.items() if evaluate_strategy(name, m) == "APPROVED"]

    if approved:
        print(f"DEPLOY: {', '.join(approved)} meet all Risk Manager criteria")
    elif best_strat[1]['Sharpe Ratio'] > 0.5:
        print(f"REFINE: {best_strat[0]} shows promise but needs optimization")
        print("  Suggestions:")
        print("  - Adjust entry/exit criteria")
        print("  - Optimize ATR multipliers for stops/targets")
        print("  - Add additional filters (volume, time-of-day)")
    else:
        print("RESEARCH: All strategies need fundamental redesign")
        print("  Suggestions:")
        print("  - Review market regime detection")
        print("  - Consider different timeframes")
        print("  - Explore alternative signal generation methods")

    return results


if __name__ == "__main__":
    run_multi_backtest("2022-01-01", "2024-12-31")
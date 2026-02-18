import warnings

import pandas as pd
import numpy as np
from datetime import time
warnings.filterwarnings('ignore')

...

def run_multi_backtest(start_date='2022-01-01', end_date='2024-12-31'):
    df = load_data('NQ', '5m', start_date, end_date)
    results = {}

    ...

    for name, m in results.items():
        passes = []
        fails = []

        if m['Sharpe Ratio'] > 1.0:
            passes.append(f"Sharpe={m['Sharpe Ratio']:.2f}")
        else:
            fails.append(f"Sharpe={m['Sharpe Ratio']:.2f} (<1.0)")

        if m['Max Drawdown'] < -0.25:
            passes.append(f"MaxDD={m['Max Drawdown']:.2%}")
        else:
            fails.append(f"MaxDD={m['Max Drawdown']:.2%} (>25%)")

        if m['Profit Factor'] > 1.2:
            passes.append(f"PF={m['Profit Factor']:.2f}")
        else:
            fails.append(f"PF={m['Profit Factor']:.2f} (<1.2)")

        if m['Trade Count'] > 30:
            passes.append(f"Trades={m['Trade Count']}")
        else:
            fails.append(f"Trades={m['Trade Count']} (<30)")

        status = "APPROVED" if len(fails) == 0 else "VETO"
        print(f"{name}: {status}")
        if passes:
            print(f"  [+] Passes: {', '.join(passes)}")
        if fails:
            print(f"  [-] Fails: {', '.join(fails)}")
        print("")

    approved = [name for name, m in results.items()
                if m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] < -0.25
                and m['Profit Factor'] > 1.2 and m['Trade Count'] > 30]

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
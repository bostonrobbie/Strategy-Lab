import pandas as pd
import numpy as np
from datetime import datetime, time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

def run_multi_backtest(start_date, end_date):
    results = {}
    
    for name in ["Strategy 1", "Strategy 2"]:
        m = {"Trade Count": 0}
        
        if m["Trade Count"] < 30:
            fails.append(f"Trades={m['Trade Count']} (<30)")
        else:
            passes.append(name)
        
        status = "APPROVED" if len(fails) == 0 else "VETO"
        print(f"{name}: {status}")
        if passes:
            print(f"  [+] Passes: {' '.join(passes)}")
        if fails:
            print(f"  [-] Fails: {' '.join(fails)}")
        print("")
        
    approved = [name for name, m in results.items()
                if m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] > -0.25
                and m['Profit Factor'] > 1.2 and m['Trade Count'] > 30]

    if approved:
        print(f"DEPLOY: {' '.join(approved)} meet all Risk Manager criteria")
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
    # Use 3-year period for robust testing
    run_multi_backtest(datetime(2022,1,1), datetime(2024,12,31))
import pandas as pd
import numpy as np
from datetime import datetime, time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

def run_multi_backtest(start_date, end_date):
    results = {}
    
    for name in ['Strategy A', 'Strategy B', 'Strategy C']:
        df = pd.read_csv(f'{name}_data.csv', index_col='Date', parse_dates=['Date'])
        df['Date'] = df['Date'].apply(lambda x: datetime.combine(x, time(9, 30)))
        
        passes = []
        fails = []
        for m in df.groupby('Method'):
            if len(m) > 10:
                pass_count = len([1 for i in range(len(m))])
                if pass_count > 0.8 * len(m):
                    passes.append(f"Trades={pass_count} (>30)")
                else:
                    fails.append(f"Trades={pass_count} (<30)")
        
        status = "APPROVED" if len(fails) == 0 else "VETO"
        print(f"{name}: {status}")
        if passes:
            print(f"  [+] Passes: {', '.join(passes)}")
        if fails:
            print(f"  [-] Fails: {', '.join(fails)}")
        print("")

    approved = [name for name, m in results.items()
                if m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] > -0.25
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

if __name__ == "__main__":
    run_multi_backtest("2022-01-01", "2024-12-31")
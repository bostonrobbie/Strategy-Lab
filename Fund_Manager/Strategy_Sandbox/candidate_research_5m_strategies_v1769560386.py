import pandas as pd
import numpy as np
from datetime import datetime, time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

def run_multi_backtest(start_date, end_date):
    results = {}
    
    for name in ['Strategy A', 'Strategy B', 'Strategy C']:
        m = {}
        
        try:
            data = pd.read_csv(f'NQ_{name}_2020.csv', index_col='Date', parse_dates=['Date'])
            data['Close'].replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(inplace=True)

            if len(data) < 250:
                print(f"Skipped {name}: Insufficient data (only {len(data)} points)")
                continue

            m['Trade Count'] = len(data) - 1
            m['Sharpe Ratio'] = np.nanmean(data.pct_change().dropna().iloc[1:].apply(lambda x: (x - x.mean()) / x.std()))
            m['Max Drawdown'] = np.max(np.maximum.accumulate(data['Close'].cummax() - data['Close']) / data['Close'])
            m['Profit Factor'] = np.nanmean((data['Close'].pct_change().dropna().iloc[1:].apply(lambda x: x if x > 0 else 0)) / (data['Close'].pct_change().dropna().iloc[1:].apply(lambda x: abs(x) if x < 0 else 0)))

            results[name] = m

        except FileNotFoundError:
            print(f"Skipped {name}: File not found")
        
    for name, m in results.items():
        passes = []
        fails = []

        if m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] > -0.25 and m['Profit Factor'] > 1.2 and m['Trade Count'] > 30:
            passes.append(f"Trades={m['Trade Count']} (>30)")
        else:
            fails.append(f"Trades={m['Trade Count']} (<30)")

        status = "APPROVED" if len(fails) == 0 else "VETO"
        print(f"{name}: {status}")
        if passes:
            print(f"  [+] Passes: {', '.join(passes)}")
        if fails:
            print(f"  [-] Fails: {', '.join(fails)}")
        print("")

    for name in ['Strategy A', 'Strategy B', 'Strategy C']:
        pass

    print(f"{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")

    approved = [name for name, m in results.items() if m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] > -0.25 and m['Profit Factor'] > 1.2 and m['Trade Count'] > 30]

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
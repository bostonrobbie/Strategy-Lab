import pandas as pd
import numpy as np
from datetime import datetime, time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

def run_multi_backtest(start_date, end_date):
    # Load data
    df = pd.read_csv('data.csv')

    # Check for missing values and gaps
    if df.isna().any():
        print("Missing values detected. Handling...")
        # Implement strategy for handling missing values here
        
    if df.index.gap().any():
        print("Gaps detected. Handling...")
        # Implement strategy for handling gaps here

    # Evaluate performance metrics
    results = {}
    passes = []
    fails = []
    best_strat = ('', {'Sharpe Ratio': 0, 'Max Drawdown': 0, 'Profit Factor': 0, 'Trade Count': 0})

    for name in df['Name'].unique():
        m = df[df['Name'] == name]
        if (m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] > -0.25
            and m['Profit Factor'] > 1.2 and m['Trade Count'] > 30):
            passes.append(name)
            if (m['Sharpe Ratio'] > best_strat[1]['Sharpe Ratio'] or
                (m['Sharpe Ratio'] == best_strat[1]['Sharpe Ratio']
                 and m['Max Drawdown'] > best_strat[1]['Max Drawdown'])):
                best_strat = (name, m.iloc[0].to_dict())
        else:
            fails.append(f"Trades={m['Trade Count']} (<30)")

    status = "APPROVED" if len(fails) == 0 else "VETO"
    print(f"{best_strat[0]}: {status}")
    if passes:
        print(f"  [+] Passes: {', '.join(passes)}")
    if fails:
        print(f"  [-] Fails: {', '.join(fails)}")
    print("")

    # Recommendation
    print(f"{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")

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
import pandas as pd
import numpy as np
from datetime import time, timedelta
from collections import defaultdict

# Load data once
df = load_data('NQ', '5m', '2022-01-01', '2024-12-31')

if df is None or df.empty:
    print("ERROR: Could not load data")
    return

results = {}

# --- Strategy 1: Opening Range Breakout ---
print("\n>>> Testing Strategy 1: Opening Range Breakout (ORB)...")
signals1 = orb_signals(df)
equity1, returns1 = run_vectorized_backtest(df, signals1)
metrics1 = calculate_metrics(equity1, returns1, signals1)
results['ORB'] = metrics1

print(f"   Total Return: {metrics1['Total Return']:.2%}")
print(f"   CAGR: {metrics1['CAGR']:.2%}")
print(f"   Sharpe Ratio: {metrics1['Sharpe Ratio']:.2f}")
print(f"   Max Drawdown: {metrics1['Max Drawdown']:.2%}")
print(f"   Win Rate: {metrics1['Win Rate']:.1%}")
print(f"   Profit Factor: {metrics1['Profit Factor']:.2f}")
print(f"   Trade Count: {metrics1['Trade Count']}")

# --- Strategy 2: Bollinger Mean Reversion ---
print("\n>>> Testing Strategy 2: Bollinger Mean Reversion...")
signals2 = bollinger_mean_reversion_signals(df)
equity2, returns2 = run_vectorized_backtest(df, signals2)
metrics2 = calculate_metrics(equity2, returns2, signals2)
results['Bollinger MR'] = metrics2

print(f"   Total Return: {metrics2['Total Return']:.2%}")
print(f"   CAGR: {metrics2['CAGR']:.2%}")
print(f"   Sharpe Ratio: {metrics2['Sharpe Ratio']:.2f}")
print(f"   Max Drawdown: {metrics2['Max Drawdown']:.2%}")
print(f"   Win Rate: {metrics2['Win Rate']:.1%}")
print(f"   Profit Factor: {metrics2['Profit Factor']:.2f}")
print(f"   Trade Count: {metrics2['Trade Count']}")

# --- Strategy 3: ADX Trend Following ---
print("\n>>> Testing Strategy 3: ADX Trend Following...")
signals3 = adx_trend_following_signals(df)
equity3, returns3 = run_vectorized_backtest(df, signals3)
metrics3 = calculate_metrics(equity3, returns3, signals3)
results['ADX Trend'] = metrics3

print(f"   Total Return: {metrics3['Total Return']:.2%}")
print(f"   CAGR: {metrics3['CAGR']:.2%}")
print(f"   Sharpe Ratio: {metrics3['Sharpe Ratio']:.2f}")
print(f"   Max Drawdown: {metrics3['Max Drawdown']:.2%}")
print(f"   Win Rate: {metrics3['Win Rate']:.1%}")
print(f"   Profit Factor: {metrics3['Profit Factor']:.2f}")
print(f"   Trade Count: {metrics3['Trade Count']}")

# --- Summary ---
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

# Find best by Sharpe
best_strat = max(results.items(), key=lambda x: x[1]['Sharpe Ratio'])
print(f"\nBest by Sharpe: {best_strat[0]} (Sharpe: {best_strat[1]['Sharpe Ratio']:.2f})")

# Find best by Return
best_return = max(results.items(), key=lambda x: x[1]['Total Return'])
print(f"Best by Return: {best_return[0]} (Return: {best_return[1]['Total Return']:.2%})")

# Risk-adjusted ranking
print("\n--- Risk-Adjusted Rankings ---")
ranked = sorted(results.items(), key=lambda x: x[1]['Sharpe Ratio'], reverse=True)
for i, (name, m) in enumerate(ranked, 1):
    status = "PASS" if m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] > -0.25 else "FAIL"
    print(f"  {i}. [{status}] {name}: Sharpe={m['Sharpe Ratio']:.2f}, Return={m['Total Return']:.2%}, MaxDD={m['Max Drawdown']:.2%}, PF={m['Profit Factor']:.2f}")

# Risk Manager Criteria Check
print(f"\n{'='*60}")
print("RISK MANAGER CRITERIA")
print(f"{'='*60}")
print("Required: Sharpe > 1.0, MaxDD < 25%, ProfitFactor > 1.2, Trades > 30")
print("")

for name, m in results.items():
    passes = []
    fails = []

    if m['Sharpe Ratio'] > 1.0:
        passes.append(f"Sharpe={m['Sharpe Ratio']:.2f}")
    else:
        fails.append(f"Sharpe={m['Sharpe Ratio']:.2f} (<1.0)")

    if m['Max Drawdown'] > -0.25:
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

# Recommendation
print(f"{'='*60}")
print("RECOMMENDATION")
print(f"{'='*60}")

approved = [name for name, m in results.items()
            if m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] > -0.25
            and m['Profit Factor'] > 1.2 and m['Trade Count'] > 30]

if approved:
    print(f"DEPLOY: {', '.join(approved)} meet all Risk Manager criteria")
else:
    if best_strat[1]['Sharpe Ratio'] > 0.5:
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
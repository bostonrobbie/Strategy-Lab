import pandas as pd
import numpy as np
from datetime import time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

def load_data(symbol='NQ', interval='5m', start_date='2020-01-01', end_date='2024-12-31'):
    """Load data efficiently with date filtering"""
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')

    # Map interval to filename
    int_map = {'1m': 'm1', '5m': 'm5', '15m': 'm15'}
    suffix = int_map.get(interval, 'm5')

    filepath = os.path.join(csv_dir, f"A2API-{symbol.upper()}-{suffix}.csv")

    if not os.path.exists(filepath):
        print(f"Error: Data file not found: {filepath}")
        return None

    print(f"Loading data from {filepath}...")

    # Read only needed columns for speed
    df = pd.read_csv(filepath, usecols=['time', 'open', 'high', 'low', 'close', 'volume'])

    # Parse dates
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert(None)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    # Standardize column names
    df.columns = [c.capitalize() for c in df.columns]

    # Filter date range
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    print(f"Loaded {len(df):,} bars from {df.index.min()} to {df.index.max()}")

    return df

def run_vectorized_backtest(df, signals, initial_capital=100000, commission=2.05, slippage=1.0):
    """
    Fast vectorized backtest with position-aware returns
    """
    # Returns
    returns = df['Close'].pct_change().fillna(0)

    # Position held for NEXT bar after signal
    position = signals.shift(1).fillna(0)

    # Turnover for costs
    turnover = position.diff().abs().fillna(0)

    # Transaction costs as % of price
    cost_pct = (commission + slippage) / df['Close']
    transaction_costs = turnover * cost_pct

    # Strategy returns
    strategy_returns = position * returns - transaction_costs

    # Equity curve
    equity = initial_capital * (1 + strategy_returns).cumprod()

    return equity, strategy_returns

def run_multi_backtest(start_date='2022-01-01', end_date='2024-12-31'):
    """
    Run comparative backtest on multiple strategies
    Using 3-year period for robust yet fast testing
    """
    print(f"\n{'='*60}")
    print(f"Running Optimized 5m Strategy Research")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")

    # Load data once
    df = load_data('NQ', '5m', start_date, end_date)
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
    run_multi_backtest("2022-01-01", "2024-12-31")
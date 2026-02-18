Here is the improved Python script with added input validation and error handling:


import sys
import os
from datetime import datetime, time
import queue
import pandas as pd
import numpy as np
from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet

# Add Src to Path
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline'))

class TrendPullback5m(Strategy):
    # ...

class VolBreakout5m(Strategy):
    # ...

def run_multi_backtest(start_date, end_date):
    if start_date > end_date:
        print("Error: Start date must be earlier than or equal to the end date.")
        return

    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    if start_date < datetime(1970, 1, 1) or end_date > datetime.today():
        print("Error: Start date must be later than January 1, 1970 and end date must not be in the future.")
        return

    print(f"Running Comparative Research: {start_date} to {end_date}")
    
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    # If not found, try others
    search_dirs = [
        csv_dir,
        os.path.join(os.getcwd(), 'examples'),
        os.path.join(os.getcwd())
    ]
    
    symbol_list = ['NQ']
    # 5m Data
    data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    try:
        if not data.has_bars():
            print("No bars found for the given date range.")
            return
    except Exception as e:
        print(f"Data Error: {e}")
        return

    # --- Strategy 1: Trend Pullback ---
    print("\n>>> Testing Trend Pullback (5m)...")
    events_1 = queue.Queue()
    data1 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port1 = Portfolio(data1, events_1, initial_capital=100000.0)
    strat1 = TrendPullback5m(data1, events_1, verbose=False)
    exec1 = SimulatedExecutionHandler(events_1, data1, commission_model=FixedCommission(2.05))
    engine1 = BacktestEngine(data1, strat1, port1, exec1)
    engine1.run()
    
    ts1 = TearSheet(port1)
    stats1 = ts1.analyze()
    print(f"Pullback Return: {stats1.get('Total Return', 0):.2%}")
    print(f"Pullback MaxDD:  {stats1.get('Max Drawdown', 0):.2%}")
    
    # --- Strategy 2: Volatility Breakout ---
    print("\n>>> Testing Volatility Breakout (5m)...")
    events_2 = queue.Queue()
    data2 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port2 = Portfolio(data2, events_2, initial_capital=100000.0)
    strat2 = VolBreakout5m(data2, events_2, verbose=False)
    exec2 = SimulatedExecutionHandler(events_2, data2, commission_model=FixedCommission(2.05))
    engine2 = BacktestEngine(data2, strat2, port2, exec2)
    engine2.run()
    
    ts2 = TearSheet(port2)
    stats2 = ts2.analyze()
    print(f"VolBreakout Return: {stats2.get('Total Return', 0):.2%}")
    print(f"VolBreakout MaxDD:  {stats2.get('Max Drawdown', 0):.2%}")
    
    # Winner?
    ret1 = stats1.get('Total Return', 0)
    ret2 = stats2.get('Total Return', 0)
    
    if ret1 > ret2:
        print("\nWinner: Trend Pullback")
    else:
        print("\nWinner: Volatility Breakout")

if __name__ == "__main__":
    # 2015 to 2024
    run_multi_backtest("2015-01-01", "2024-12-31")
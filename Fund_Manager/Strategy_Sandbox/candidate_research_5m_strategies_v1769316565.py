Here is the modified code with the suggested change:


import sys
import os
import pandas as pd
import numpy as np
from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet
import queue

# Add Src to Path
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline'))

class MeanReversion(Strategy):
    def __init__(self, bars, events, lookback=50, threshold=2.0, verbose=False):
        super().__init__(bars, events)
        self.lookback = lookback
        self.threshold = threshold
        self.verbose = verbose
        
        self.entry_price = 0.0
        self.current_pos = 0 # 0, 1, -1

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        
        # Session Filter (9:30 - 15:45)
        if not (time(9, 30) <= current_time < time(15, 45)):
            if self.current_pos != 0 and current_time >= time(15, 50):
                self.exit(symbol)
                self.current_pos = 0
            return

        # Data
        bars = self.bars.get_latest_bars(symbol, N=self.lookback)
        if len(bars) < self.lookback: 
            return
        
        closes = np.array([b.close for b in bars])
        
        # Indicators
        s = pd.Series(closes)
        ema = s.ewm(span=20, adjust=False).mean().iloc[-1]
        std_dev = s.std()
        
        current_price = event.close
        
        if self.current_pos == 0:
            # Mean Reversion Strategy: Long when Price < EMA - Threshold*std_dev
            if current_price < ema - (self.threshold * std_dev):
                if self.verbose: print(f"[{ts}] Mean Reversion Buy Signal: {current_price} < {ema:.2f} - {self.threshold * std_dev:.2f}")
                self.buy(symbol, 1)
                self.entry_price = current_price
                self.current_pos = 1
                
            # Mean Reversion Strategy: Short when Price > EMA + Threshold*std_dev
            elif current_price > ema + (self.threshold * std_dev):
               if self.verbose: print(f"[{ts}] Mean Reversion Sell Signal: {current_price} > {ema:.2f} + {self.threshold * std_dev:.2f}")
               self.sell(symbol, 1)
               self.entry_price = current_price
               self.current_pos = -1
               
        else:
             # Basic Risk Management
             if self.current_pos > 0:
                 sl = self.entry_price - (self.threshold * std_dev)
                 tp = self.entry_price + (self.threshold * std_dev)
                 if event.low < sl:
                     self.exit(symbol)
                     self.current_pos = 0
                 elif event.high > tp:
                     self.exit(symbol)
                     self.current_pos = 0
             elif self.current_pos < 0:
                 sl = self.entry_price + (self.threshold * std_dev)
                 tp = self.entry_price - (self.threshold * std_dev)
                 if event.high > sl:
                     self.exit(symbol)
                     self.current_pos = 0
                 elif event.low < tp:
                     self.exit(symbol)
                     self.current_pos = 0


# ==========================================
# Runner
# ==========================================

from datetime import time

def run_multi_backtest(start_date, end_date):
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
    try:
        data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    except Exception as e:
        print(f"Data Error: {e}")
        return

    # --- Strategy: Mean Reversion ---
    print("\n>>> Testing Mean Reversion (5m)...")
    events = queue.Queue()
    data1 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port = Portfolio(data1, events, initial_capital=100000.0)
    strat = MeanReversion(data1, events, verbose=False)
    exec = SimulatedExecutionHandler(events, data1, commission_model=FixedCommission(2.05))
    engine = BacktestEngine(data1, strat, port, exec)
    engine.run()
    
    ts = TearSheet(port)
    stats = ts.analyze()
    print(f"Mean Reversion Return: {stats.get('Total Return', 0):.2%}")
    print(f"Mean Reversion MaxDD:  {stats.get('Max Drawdown', 0):.2%}")

if __name__ == "__main__":
    # 2015 to 2024
    run_multi_backtest("2015-01-01", "2024-12-31")
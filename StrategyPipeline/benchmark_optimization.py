
import time
import os
import sys

# Setup Path
sys.path.insert(0, os.path.abspath('src'))

from backtesting.optimizer import GridSearch
from backtesting.data import SmartDataHandler
from backtesting.strategy import Strategy
from backtesting.schema import OrderEvent, OrderType, OrderSide

# Define simple mock strategy inline
class MockStrategy(Strategy):
    def __init__(self, bars, events, short_window=10, long_window=50):
        self.bars = bars
        self.events = events
        self.short_window = short_window
        self.long_window = long_window

    def calculate_signals(self, event):
        # Minimal logic to consume CPU cycles but keep it fast
        pass

def benchmark():
    print("Running Optimization Benchmark...")
    
    # Define a small but meaningful grid
    # Reduced grid for quick verification
    param_grid = {
        'short_window': range(10, 60, 5),  # 10 items
        'long_window': range(100, 200, 10) # 10 items
    }
    
    start_time = time.time()
    
    # Use examples/SPY.csv usually available
    
    optimizer = GridSearch(
        data_handler_cls=SmartDataHandler,
        data_handler_args=(['benchmark_data'], ['.'], '2023-01-01', '2023-01-10', '1d'),
        strategy_cls=MockStrategy,
        param_grid=param_grid,
        n_jobs=-1 # Use all cores
    )
    
    # Run
    df = optimizer.run(parallel=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n[BENCHMARK] Completed {len(df)} iterations in {duration:.4f} seconds.")
    if not df.empty:
        print(f"Top Result: {df.iloc[0]['Total Return']:.2%} with params {df.iloc[0].to_dict()}")
    else:
        print("No results returned.")

if __name__ == "__main__":
    benchmark()

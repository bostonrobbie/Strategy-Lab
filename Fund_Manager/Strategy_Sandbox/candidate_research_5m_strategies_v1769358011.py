Here's the modified code with the requested changes incorporated:

Python
import sys
import os
import queue
from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet
import pandas as pd
import numpy as np

# Add Src to Path
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline', 'src'))
sys.path.append(os.path.join(os.getcwd(), 'StrategyPipeline'))

class TrendPullback5m(Strategy):
    def __init__(self, bars, events, ema_trend=50, ema_pullback=20, sl_atr=2.0, tp_atr=4.0, verbose=False):
        super().__init__(bars, events)
        self.ema_trend_len = ema_trend
        self.ema_pb_len = ema_pullback
        self.sl_atr_mult = sl_atr
        self.tp_atr_mult = tp_atr
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
        lookback = self.ema_trend_len + 5
        bars = self.bars.get_latest_bars(symbol, N=lookback)
        if len(bars) < lookback: 
            return
        
        closes = np.array([b.close for b in bars])
        
        # Indicators
        s = pd.Series(closes)
        ema_trend = s.ewm(span=self.ema_trend_len, adjust=False).mean().iloc[-1]
        ema_pb = s.ewm(span=self.ema_pb_len, adjust=False).mean().iloc[-1]
        
        # ATR (Simple TR approx for speed)
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))[1:]
        atr = tr[-14:].mean() if len(tr) > 14 else tr.mean()

        current_price = event.close
        
        if self.current_pos == 0:
            # Long Setup: Uptrend (Price > EMA50) + Pullback (Price < EMA20)
            if current_price > ema_trend and current_price < ema_pb:
                if self.verbose: print(f"[{ts}] Pullback Buy Signal: {current_price} < {ema_pb:.2f} (Trend {ema_trend:.2f})")
                self.buy(symbol, 1)
                self.entry_price = current_price
                self.current_pos = 1
                
            # Short Setup: Downtrend (Price < EMA50) + Pullback (Price > EMA20)
            elif current_price < ema_trend and current_price > ema_pb:
               if self.verbose: print(f"[{ts}] Pullback Sell Signal: {current_price} > {ema_pb:.2f} (Trend {ema_trend:.2f})")
               self.sell(symbol, 1)
               self.entry_price = current_price
               self.current_pos = -1
               
        else:
             # Basic Risk Management
             if self.current_pos > 0:
                 sl = self.entry_price - (atr * self.sl_atr_mult)
                 tp = self.entry_price + (atr * self.tp_atr_mult)
                 if event.low < sl:
                     self.exit(symbol)
                     self.current_pos = 0
                 elif event.high > tp:
                     self.exit(symbol)
                     self.current_pos = 0
             elif self.current_pos < 0:
                 sl = self.entry_price + (atr * self.sl_atr_mult)
                 tp = self.entry_price - (atr * self.tp_atr_mult)
                 if event.high > sl:
                     self.exit(symbol)
                     self.current_pos = 0
                 elif event.low < tp:
                     self.exit(symbol)
                     self.current_pos = 0


class VolBreakout5m(Strategy):
    def __init__(self, bars, events, sl_atr=1.5, tp_atr=3.0, verbose=False):
        super().__init__(bars, events)
        self.sl_atr_mult = sl_atr
        self.tp_atr_mult = tp_atr
        self.verbose = verbose
        
        self.entry_price = 0.0
        self.current_pos = 0 # 0, 1, -1
        
        # State necessary for Inside Bar
        self.prev_high = None
        self.prev_low = None

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
            
        # Get ATR
        bars = self.bars.get_latest_bars(symbol, N=20)
        if len(bars) < 20: 
            self.prev_high = event.high
            self.prev_low = event.low
            return
            
        closes = np.array([b.close for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))[1:]
        atr = tr[-14:].mean()
        
        current_high = event.high
        current_low = event.low
        
        # Manage Existing Trade
        if self.current_pos != 0:
             if self.current_pos > 0:
                 sl = self.entry_price - (atr * self.sl_atr_mult)
                 tp = self.entry_price + (atr * self.tp_atr_mult)
                 if event.low < sl:
                     self.exit(symbol)
                     self.current_pos = 0
                 elif event.high > tp:
                     self.exit(symbol)
                     self.current_pos = 0
             elif self.current_pos < 0:
                 sl = self.entry_price + (atr * self.sl_atr_mult)
                 tp = self.entry_price - (atr * self.tp_atr_mult)
                 if event.high > sl:
                     self.exit(symbol)
                     self.current_pos = 0
                 elif event.low < tp:
                     self.exit(symbol)
                     self.current_pos = 0
                     
             # Update state even during trade logic? 
             # Inside bar logic usually applies to NEW entries.
             self.prev_high = current_high
             self.prev_low = current_low
             return

        # Entry Logic: Inside Bar
        # Current bar is "Inside" if High < PrevHigh and Low > PrevLow
        # Wait, if we are processing bar `event`, this is the JUST COMPLETED bar.
        # So we check if THIS bar was an inside bar. If so, we place stops for NEXT bar.
        # But our simplistic engine executes MARKET.
        # So we will check if the PREVIOUS bar was an inside bar, and if CURRENT bar broke it.
        
        # Let's look at bars[-2] (Previous) and bars[-1] (Current/Event)
        # bars[-1] is the event bar.
        if self.prev_high is None:
            self.prev_high = current_high
            self.prev_low = current_low
            return

        # Was previous bar an inside bar relative to the one before it?
        # That requires 3 bars of history.
        # Let's simplify: 
        # Strategy: Trading the breakout of the PREVIOUS bar if it was an Inside Bar?
        # No, "Inside Bar" pattern: Bar 1 (Mother), Bar 2 (Inside).
        # Entry: Break of Bar 2 High/Low.
        # So we need to detect if Bar[-2] was inside Bar[-3].
        # If so, check if Bar[-1] broke Bar[-2].
        
        if len(bars) < 3: return
        
        bar_mother = bars[-3]
        bar_inside = bars[-2]
        bar_current = bars[-1] # The event bar
        
        is_inside = (bar_inside.high < bar_mother.high) and (bar_inside.low > bar_mother.low)
        
        # Indicators
        s = pd.Series(closes)
        ema_trend = s.ewm(span=200, adjust=False).mean().iloc[-1]
        
        if is_inside:
            # Check Breakout in Current Bar
            # Breakout Long (Only if Trend is UP)
            if bar_current.close > bar_inside.high and bar_current.close > ema_trend:
                 if self.verbose: print(f"[{ts}] IB Breakout Long (Trend OK)")
                 self.buy(symbol, 1)
                 self.entry_price = bar_current.close
                 self.current_pos = 1
                 
            # Breakout Short (Only if Trend is DOWN)
            elif bar_current.close < bar_inside.low and bar_current.close < ema_trend:
                 if self.verbose: print(f"[{ts}] IB Breakout Short (Trend OK)")
                 self.sell(symbol, 1)
                 self.entry_price = bar_current.close
                 self.current_pos = -1

# ==========================================
# Runner
# ==========================================

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

    # --- Strategy 1: Trend Pullback ---
    print("\n>>> Testing Trend Pullback (5m)...")
    events_1 = queue.Queue()
    # Re-init data? SmartDataHandler is stateful (generators). 
    # We need fresh data handlers or reset them.
    # Easiest is to create new handler instance.
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
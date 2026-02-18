Here is the modified code:


import sys
import os
from datetime import time
import pandas as pd
import numpy as np
from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet

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
        
        if not (time(9, 30) <= current_time < time(15, 45)):
            if self.current_pos != 0 and current_time >= time(15, 50):
                self.exit(symbol)
                self.current_pos = 0
            return

        lookback = self.ema_trend_len + 5
        bars = self.bars.get_latest_bars(symbol, N=lookback)
        if len(bars) < lookback: return
        
        closes = np.array([b.close for b in bars])
        
        s = pd.Series(closes)
        ema_trend = s.ewm(span=self.ema_trend_len, adjust=False).mean().iloc[-1]
        ema_pb = s.ewm(span=self.ema_pb_len, adjust=False).mean().iloc[-1]
        
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))[1:]
        atr = tr.mean()
        
        current_price = event.close
        
        if self.current_pos == 0:
            if current_price > ema_trend and current_price < ema_pb:
                if self.verbose: print(f"[{ts}] Pullback Buy Signal: {current_price} < {ema_pb:.2f} (Trend {ema_trend:.2f})")
                self.buy(symbol, 1)
                self.entry_price = current_price
                self.current_pos = 1
                
            elif current_price < ema_trend and current_price > ema_pb:
                if self.verbose: print(f"[{ts}] Pullback Sell Signal: {current_price} > {ema_pb:.2f} (Trend {ema_trend:.2f})")
                self.sell(symbol, 1)
                self.entry_price = current_price
                self.current_pos = -1
        else:
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
        
        self.prev_high = None
        self.prev_low = None

    def calculate_signals(self, event):
        symbol = event.symbol
        ts = event.timestamp
        current_time = ts.time()
        
        if not (time(9, 35) <= current_time < time(15, 30)): # Start slightly later
            if self.current_pos != 0 and current_time >= time(15, 45):
                self.exit(symbol)
                self.current_pos = 0
            return
            
        bars = self.bars.get_latest_bars(symbol, N=20)
        if len(bars) < 20: 
            self.prev_high = event.high
            self.prev_low = event.low
            return
            
        closes = np.array([b.close for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))[1:]
        atr = tr.mean()
        
        current_high = event.high
        current_low = event.low
        
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
                    
            self.prev_high = current_high
            self.prev_low = current_low
            return

        # Entry Logic: Inside Bar
        if self.prev_high is None:
            self.prev_high = current_high
            self.prev_low = current_low
            return

        if len(bars) < 3: return
        
        bar_mother = bars[-3]
        bar_inside = bars[-2]
        bar_current = bars[-1] # The event bar
        
        is_inside = (bar_inside.high < bar_mother.high) and (bar_inside.low > bar_mother.low)
        
        if is_inside:
            if bar_current.close > bar_inside.high and bar_current.close > ema_trend:
                 if self.verbose: print(f"[{ts}] IB Breakout Long (Trend OK)")
                 self.buy(symbol, 1)
                 self.entry_price = bar_current.close
                 self.current_pos = 1
                 
            elif bar_current.close < bar_inside.low and bar_current.close < ema_trend:
                 if self.verbose: print(f"[{ts}] IB Breakout Short (Trend OK)")
                 self.sell(symbol, 1)
                 self.entry_price = bar_current.close
                 self.current_pos = -1


def run_multi_backtest(start_date, end_date):
    print(f"Running Comparative Research: {start_date} to {end_date}")
    
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    search_dirs = [
        csv_dir,
        os.path.join(os.getcwd(), 'examples'),
        os.path.join(os.getcwd())
    ]
    
    symbol_list = ['NQ']
    data = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    events_1 = queue.Queue()
    port1 = Portfolio(data, events_1, initial_capital=100000.0)
    strat1 = TrendPullback5m(data, events_1, verbose=False)
    exec1 = SimulatedExecutionHandler(events_1, data, commission_model=FixedCommission(2.05))
    engine1 = BacktestEngine(data, strat1, port1, exec1)
    engine1.run()
    
    ts1 = TearSheet(port1)
    stats1 = ts1.analyze()
    print(f"Pullback Return: {stats1.get('Total Return', 0):.2%}")
    print(f"Pullback MaxDD:  {stats1.get('Max Drawdown', 0):.2%}")
    
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
    
    ret1 = stats1.get('Total Return', 0)
    ret2 = stats2.get('Total Return', 0)
    
    if ret1 > ret2:
        print("\nWinner: Trend Pullback")
    else:
        print("\nWinner: Volatility Breakout")

if __name__ == "__main__":
    run_multi_backtest("2015-01-01", "2024-12-31")
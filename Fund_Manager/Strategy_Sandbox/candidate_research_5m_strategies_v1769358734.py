import pandas as pd
import numpy as np
from backtesting.strategy import Strategy
from backtesting.data import SmartDataHandler
from backtesting.portfolio import Portfolio
from backtesting.execution import SimulatedExecutionHandler, FixedCommission
from backtesting.engine import BacktestEngine
from backtesting.performance import TearSheet
import queue

class RefactoredStrategy(Strategy):
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
        if len(bars) < lookback: return
        
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

    # ... rest of the code ...
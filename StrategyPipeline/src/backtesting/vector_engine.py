import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import time
from .accelerate import get_dataframe_library, get_array_library
from .monitor import PipelineMonitor

monitor = PipelineMonitor()

try:
    from numba import jit
    monitor.log_gpu_status(True, "Numba JIT available. GPU/Fast CPU acceleration enabled.")
except ImportError:
    monitor.log_gpu_status(False, "Numba not found. Vector Engine will run in strict CPU mode (Slow).")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

class VectorStrategy(ABC):
    """
    Abstract Base Class for Vectorized Strategies.
    Generates signals for an entire DataFrame at once.
    """
    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def generate_signals(self, df):
        """
        Takes a DataFrame (cudf or pandas) and returns a Series of signals.
        1 = LONG, -1 = SHORT, 0 = FLAT
        """
        raise NotImplementedError

class VectorEngine:
    """
    High-Performance Backtest Engine using Vectorized Operations.
    Ideal for GPU acceleration.
    Includes cost modeling (Slippage + Commissions).
    """
    def __init__(self, strategy: VectorStrategy, initial_capital=100000.0, commission=1.0, slippage=1.0, volatility_factor=0.01):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_per_unit = commission
        self.slippage_per_unit = slippage
        self.volatility_factor = volatility_factor
        self.pd = get_dataframe_library()
        self.np = get_array_library()

    def run(self, df):
        """
        Runs the vectorized backtest on the provided DataFrame.
        """
        # 1. Generate Signals
        signals = self.strategy.generate_signals(df)
        
        # 2. Calculate Returns
        # We assume execution on NEXT OPEN (shift signals by 1)
        if 'Close' not in df.columns and 'close' in df.columns:
            df.rename(columns={'close': 'Close'}, inplace=True)
            
        # Returns from Open to Close or Close to Close? 
        # Standard: Close to Close returns applied to position held at shift(1)
        returns = df['Close'].pct_change().fillna(0)
        
        # Position is held for the bar AFTER the signal
        pos = signals.shift(1).fillna(0)
        
        # --- COST MODELING (Dynamic Volatility Slippage) ---
        prices = df['Close']
        highs = df['High']
        lows = df['Low']
        
        # Calculate Volatility (Range)
        # Slippage is likely to be higher when the bar range is large
        volatility = (highs - lows).abs()
        
        # Dynamic Slippage
        # If slippage_factor is provided (e.g. 0.05 of range), use it.
        # Otherwise treat self.slippage_per_unit as a fixed dollar amount per trade (fallback)
        # We will assume self.slippage_per_unit is a 'factor' if < 1.0, else fixed ticks
        # Actually, let's just make it robust:
        # Cost = Commission + (Volatility * 0.1) + Fixed_Slippage
        # Let's assume standard 'slippage_per_unit' is the fixed component (e.g. 1 tick)
        # And we add a volatility component: 5% of the bar range.
        
        # Cost = Commission + (Volatility * Factor) + Fixed_Slippage
        # Adjusted to 1% (0.01) based on user feedback for "balanced" realism.
        
        vol_slippage = volatility * self.volatility_factor
        
        # Total Cost per Unit = Fixed Comm + Fixed Slip + Dynamic Vol Slip
        total_cost_per_unit = self.commission_per_unit + self.slippage_per_unit + vol_slippage
        
        # Cost as % of Price
        cost_pct = total_cost_per_unit / prices
        
        # Transaction Costs
        transaction_costs = turnover * cost_pct
        
        # Strategy Returns (Gross)
        strat_returns = pos * returns
        
        # Net Returns
        net_returns = strat_returns - transaction_costs
        
        # cumulative equity
        equity_curve = self.initial_capital * (1 + net_returns).cumprod()
        
        return {
            'equity_curve': equity_curve,
            'signals': signals,
            'returns': net_returns,
            'turnover': turnover
        }

class VectorizedMA(VectorStrategy):
    """
    Vectorized implementation of Moving Average Crossover.
    """
    def __init__(self, short_window=50, long_window=200, **kwargs):
        super().__init__(short_window=short_window, long_window=long_window, **kwargs)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df):
        pd_lib = get_dataframe_library()
        short_ma = df['Close'].rolling(window=self.short_window).mean()
        long_ma = df['Close'].rolling(window=self.long_window).mean()
        
        signals = pd_lib.Series(0, index=df.index)
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = 0 
        return signals

class VectorizedNQORB(VectorStrategy):
    """
    Vectorized implementation of 15m Opening Range Breakout (ORB).
    Optimized for GPU (cuDF/Pandas).
    """
    def __init__(self, orb_start="09:30", orb_end="09:45", ema_filter=50, atr_filter=14, sl_atr_mult=2.0, tp_atr_mult=4.0, atr_max_mult=2.5,
                 use_htf=False, htf_ma=200, 
                 use_rvol=False, rvol_thresh=1.5,
                 use_hurst=False, hurst_thresh=0.5,
                 use_adx=False, adx_thresh=20,
                 use_trailing_stop=False, ts_atr_mult=3.0,
                 **kwargs):
        super().__init__(orb_start=orb_start, orb_end=orb_end, ema_filter=ema_filter, atr_filter=atr_filter, sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, atr_max_mult=atr_max_mult, **kwargs)
        self.orb_start = orb_start
        self.orb_end = orb_end
        self.ema_filter = int(ema_filter)
        self.atr_filter = int(atr_filter)
        self.sl_atr_mult = float(sl_atr_mult)
        self.tp_atr_mult = float(tp_atr_mult)
        self.atr_max_mult = float(atr_max_mult)
        
        # Phase 2 Features
        self.use_htf = use_htf
        self.htf_ma = int(htf_ma)
        self.use_rvol = use_rvol
        self.rvol_thresh = float(rvol_thresh)
        self.use_hurst = use_hurst
        self.hurst_thresh = float(hurst_thresh)
        self.use_adx = use_adx
        self.adx_thresh = float(adx_thresh)
        self.use_trailing_stop = use_trailing_stop
        self.ts_atr_mult = float(ts_atr_mult)

    def generate_signals(self, df):
        from src.backtesting import ta

        pd_lib = get_dataframe_library()
        
        # Prepare Data
        # We need numpy arrays for Numba
        # Ensure we have all necessary columns
        if 'close' not in df.columns: df.rename(columns={'Close': 'close'}, inplace=True)
        if 'high' not in df.columns: df.rename(columns={'High': 'high'}, inplace=True)
        if 'low' not in df.columns: df.rename(columns={'Low': 'low'}, inplace=True)
        
        # Calculate Indicators (Pandas/TA-Lib is fast enough for vector calc)
        if 'volume' not in df.columns and 'Volume' in df.columns: df.rename(columns={'Volume': 'volume'}, inplace=True)
        if 'volume' not in df.columns: df['volume'] = 1.0 # fallback

        # Calculate Base Indicators
        # Using custom ta which returns Series
        ema = ta.ema(df['close'], length=self.ema_filter).fillna(0).values
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.atr_filter).fillna(0).values

        # --- Phase 2: Advanced Indicators ---
        # 1. HTF Trend (Daily MA)
        # Resample to Daily -> Calc MA -> Reindex to Intraday
        if self.use_htf:
            daily_close = df['close'].resample('D').last().dropna()
            daily_ma = ta.sma(daily_close, length=self.htf_ma)
            
            if daily_ma is None or daily_ma.empty:
                # Not enough data for MA
                daily_ma = pd.Series(0, index=daily_close.index)
            
            # Reindex and ffill (careful: avoid lookahead bias, shift 1 day)
            # Daily MA for today should be based on YESTERDAY's close
            daily_ma = daily_ma.shift(1).reindex(df.index).ffill().fillna(0).values
        else:
            daily_ma = np.zeros(len(df))

        # 2. RVOL (Relative Volume)
        if self.use_rvol:
            avg_vol = ta.sma(df['volume'], length=20).fillna(1.0) # Avoid div/0
            rvol = df['volume'] / avg_vol
            rvol = rvol.fillna(0).values
        else:
            rvol = np.zeros(len(df))

        # 3. Hurst Exponent (requires specialized calc, ta libs usually lack rolling hurst)
        # We will use a simplified Efficiency Ratio (ER) as a proxy if simple Hurst isn't avail.
        # ER = Change / Sum of ranges. High ER => Trend.
        # To strictly do Hurst: log(R/S) / log(T)
        # For speed in Numba/Vector, implementation is tricky. 
        # Let's placeholder with 0.5 (neutral) if disabled, or use ER.
        if self.use_hurst:
            # Using KAMA Efficiency Ratio as fast Trend Persistence proxy
            # ER = abs(Change) / Volatility-Sum
            change = df['close'].diff(10).abs()
            volatility = df['close'].diff().abs().rolling(10).sum()
            er = change / volatility
            hurst_proxy = er.fillna(0.5).values 
            # Note: ER ranges 0-1. ER > 0.5 implies trendiness similar to Hurst > 0.5
        else:
            hurst_proxy = np.zeros(len(df))

        # 4. ADX
        if self.use_adx:
            # Custom ta.adx returns a Series of ADX values
            adx_series = ta.adx(df['high'], df['low'], df['close'], length=14)
            adx_val = adx_series.fillna(0).values
        else:
            adx_val = np.zeros(len(df))

        volume = df['volume'].values
        
        # Convert Time logic to integers for faster comparison
        # We'll use minute of day: 9:30 = 9*60 + 30 = 570
        times = df.index.hour * 60 + df.index.minute
        dates = df.index.date
        
        # Create a unique integer ID for each day to detect day changes fast
        # (Assuming dates are sorted)
        # We can map unique dates to ints or just use change detection
        # Actually, simpler: pass day_changed boolean array
        # day_changed[i] = dates[i] != dates[i-1]
        
        # For Numba, we need comparable arrays. 
        # Converting dates to int64 timestamps (ns)
        timestamps = df.index.values.astype(np.int64)
        
        # Parse Strategy Times
        t_start = pd.to_datetime(self.orb_start).time()
        start_min = t_start.hour * 60 + t_start.minute
        
        t_end = pd.to_datetime(self.orb_end).time()
        end_min = t_end.hour * 60 + t_end.minute
        
        # Exit time typically 15:45
        exit_min = 15 * 60 + 45
        
        # extract numpy arrays
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # Run Numba Core
        signals = _numba_orb_logic(
            timestamps, times.values, closes, highs, lows, ema, atr, 
            start_min, end_min, exit_min, 
            self.atr_max_mult, self.sl_atr_mult, self.tp_atr_mult,
            self.use_htf, daily_ma,
            self.use_rvol, rvol, self.rvol_thresh,
            self.use_hurst, hurst_proxy, self.hurst_thresh,
            self.use_adx, adx_val, self.adx_thresh,
            self.use_trailing_stop, self.ts_atr_mult
        )
        
        return pd.Series(signals, index=df.index)

@jit(nopython=True)
def _numba_orb_logic(timestamps, times, closes, highs, lows, ema, atr, 
                     start_min, end_min, exit_min, 
                     atr_max_mult, sl_mult, tp_mult,
                     use_htf, daily_ma,
                     use_rvol, rvol, rvol_thresh,
                     use_hurst, hurst, hurst_thresh,
                     use_adx, adx, adx_thresh,
                     use_ts, ts_mult):
    n = len(closes)
    signals = np.zeros(n, dtype=np.int32)
    
    # State Variables
    orb_high = -1.0
    orb_low = 1e9
    
    # Session State
    traded_today = False
    in_pos = 0 # 1 long, -1 short
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    
    # We iterate chronologically
    for i in range(1, n):
        # 1. Detect New Day
        # Timestamp change big enough or just check if time went backward (new day)
        # Simple check: times[i] < times[i-1] usually means new day, 
        # BUT if data has gaps it might not.
        # Check timestamps roughly > 1 day gap or use day change logic?
        # Actually, let's track previous "day" via timestamp / 86400e9 (ns per day)
        # or simplified: reset if time < start_min
        
        t = times[i]
        
        # Reset Day State
        # If time is exactly start_min, we treat it as start of ORB? 
        # NQ opens 9:30. ORB is 9:30-9:45. 
        # We need to capture H/L during [start_min, end_min).
        
        # Reset Logic: If we see a time < previous time, it's a new day
        if times[i] < times[i-1]:
            orb_high = -1.0
            orb_low = 1e9
            traded_today = False
            in_pos = 0
        
        # ORB Calculation Window
        if t >= start_min and t < end_min:
            if orb_high == -1.0: # first bar of ORB
                orb_high = highs[i]
                orb_low = lows[i]
            else:
                if highs[i] > orb_high: orb_high = highs[i]
                if lows[i] < orb_low: orb_low = lows[i]
                
        # Trading Window
        elif t >= end_min and t < exit_min:
            # Check Exits First
            if in_pos != 0:
                if in_pos == 1:
                    # Long Exit
                    
                    # Trailing Stop Update
                    if use_ts:
                        # For long, SL moves UP only. 
                        # TS Level = High - (ATR * mult)
                        # We used entry-based SL initially.
                        # Dynamic SL based on Current High - TS
                        # Check if we assume 'high' of this bar contributes to TS move?
                        # Usually TS is calculated on close or high of PREV bar to avoid peeking?
                        # Or intra-bar high? 
                        # If intra-bar, we can survive provided Low doesn't hit SL first.
                        # Simplistic Conservative: Update SL for NEXT bar based on THIS bar's High.
                        new_sl = highs[i] - (atr[i] * ts_mult)
                        if new_sl > sl_price:
                            sl_price = new_sl
                            
                    if lows[i] <= sl_price:
                        # Stop Hit
                        signals[i] = 0 
                        in_pos = 0
                    elif not use_ts and highs[i] >= tp_price:
                        # TP Hit (Only if Fixed Target is used)
                        in_pos = 0
                        
                elif in_pos == -1:
                    # Short Exit // Trailing Stop
                    if use_ts:
                        new_sl = lows[i] + (atr[i] * ts_mult)
                        if new_sl < sl_price:
                            sl_price = new_sl
                            
                    if highs[i] >= sl_price:
                        in_pos = 0
                    elif not use_ts and lows[i] <= tp_price:
                        in_pos = 0
            
            # Check Entries
            if in_pos == 0 and not traded_today and orb_high != -1.0:
                range_size = orb_high - orb_low
                cur_atr = atr[i]
                
                if range_size > 0 and cur_atr > 0 and range_size <= (cur_atr * atr_max_mult):
                    # Long Entry
                    if closes[i] > orb_high and closes[i] > ema[i]:
                        # Check Filters
                        valid = True
                        if use_htf and closes[i] <= daily_ma[i]: valid = False
                        if use_rvol and rvol[i] <= rvol_thresh: valid = False
                        if use_hurst and hurst[i] <= hurst_thresh: valid = False
                        if use_adx and adx[i] <= adx_thresh: valid = False
                        
                        if valid:
                            in_pos = 1
                            entry_price = closes[i]
                            sl_price = entry_price - (cur_atr * sl_mult)
                            tp_price = entry_price + (cur_atr * tp_mult)
                            traded_today = True
                            
                    # Short Entry
                    elif closes[i] < orb_low and closes[i] < ema[i]:
                        # Check Filters
                        valid = True
                        if use_htf and closes[i] >= daily_ma[i]: valid = False
                        if use_rvol and rvol[i] <= rvol_thresh: valid = False
                        if use_hurst and hurst[i] <= hurst_thresh: valid = False
                        if use_adx and adx[i] <= adx_thresh: valid = False

                        if valid:
                            in_pos = -1
                            entry_price = closes[i]
                            sl_price = entry_price + (cur_atr * sl_mult)
                            tp_price = entry_price - (cur_atr * tp_mult)
                            traded_today = True
                        
        # Session End
        elif t >= exit_min:
            in_pos = 0
            
        # Record Signal
        # VectorEngine logic: 'pos' is held for NEXT bar. 
        # So signals[i] determines position at i+1.
        signals[i] = in_pos
        
    return signals

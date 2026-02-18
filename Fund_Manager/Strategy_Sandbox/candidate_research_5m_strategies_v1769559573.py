import warnings

import pandas as pd
import numpy as np
from datetime import time
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), 'StrategyPipeline', 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'StrategyPipeline'))

def calculate_sharpe(returns, periods_per_year=252*78):
    if returns.std() == 0:
        return 0
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()

def calculate_max_drawdown(equity_curve):
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def calculate_metrics(equity_curve, returns, signals):
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    sharpe = calculate_sharpe(returns)
    max_dd = calculate_max_drawdown(equity_curve)

    # Trade analysis (count position changes)
    position_changes = signals.diff().abs()
    trade_count = (position_changes > 0).sum() // 2  # Entry + exit = 1 round trip

    # Win rate from returns where we had a position
    active_returns = returns[signals.shift(1).fillna(0) != 0]
    if len(active_returns) > 0:
        winning_bars = (active_returns > 0).sum()
        total_bars = len(active_returns)
        win_rate = winning_bars / max(total_bars, 1)
    else:
        win_rate = 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / max(gross_loss, 0.0001)

    # CAGR
    years = len(equity_curve) / (252 * 78)  # Approximate trading bars per year
    if years > 0 and equity_curve.iloc[0] > 0:
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1
    else:
        cagr = 0

    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Trade Count': trade_count
    }

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr

def orb_signals(df, orb_minutes=15, ema_filter=50, atr_period=14, atr_filter_mult=2.5):
    close = df['Close']
    high = df['High']
    low = df['Low']

    # EMA for trend filter
    ema = close.ewm(span=ema_filter, adjust=False).mean()

    # ATR for volatility filter
    atr = calculate_atr(df, atr_period)

    # Time components
    times = df.index.time
    dates = df.index.date

    # Initialize
    signals = pd.Series(0.0, index=df.index)
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    # Group by date to calculate ORB
    for date in pd.unique(dates):
        day_mask = [d == date for d in dates]
        day_data = df[day_mask]

        if len(day_data) < 20:  # Need enough bars
            continue

        # Get ORB window (9:30 - 9:45)
        orb_mask = [(time(9, 30) <= t < time(9, 45)) for t in day_data.index.time]
        orb_data = day_data[orb_mask]

        if len(orb_data) == 0:
            continue

        orb_high = orb_data['High'].max()
        orb_low = orb_data['Low'].min()
        orb_range = orb_high - orb_low

        # Get trading window (9:45 - 15:30)
        trade_mask = [(time(9, 45) <= t < time(15, 30)) for t in day_data.index.time]
        trade_indices = day_data.index[trade_mask]

        for idx in trade_indices:
            current_close = close[idx]
            current_high = high[idx]
            current_low = low[idx]
            current_ema = ema[idx]
            current_atr = atr[idx]

            # ATR filter - skip very volatile days
            if pd.isna(current_atr) or orb_range > current_atr * atr_filter_mult:
                continue

            if position == 0:
                # Long entry: breakout above ORB high + trend filter
                if current_close > orb_high and current_close > current_ema:
                    position = 1
                    entry_price = current_close
                    stop_loss = entry_price - (current_atr * 2)
                    take_profit = entry_price + (current_atr * 3)

                # Short entry: breakdown below ORB low + trend filter
                elif current_close < orb_low and current_close < current_ema:
                    position = -1
                    entry_price = current_close
                    stop_loss = entry_price + (current_atr * 2)
                    take_profit = entry_price - (current_atr * 3)

            elif position == 1:
                # Long exit: stop or target
                if current_low <= stop_loss or current_high >= take_profit:
                    position = 0

            elif position == -1:
                # Short exit: stop or target
                if current_high >= stop_loss or current_low <= take_profit:
                    position = 0

            signals[idx] = position

        # End of day: flatten
        eod_mask = [t >= time(15, 30) for t in day_data.index.time]
        eod_indices = day_data.index[eod_mask]
        signals[eod_indices] = 0
        position = 0

    return signals

def bollinger_mean_reversion_signals(df, bb_period=20, bb_std=2.0, rsi_period=14, oversold=30, overbought=70, atr_period=14):
    close = df['Close']
    high = df['High']
    low = df['Low']

    # Bollinger Bands
    sma = close.rolling(window=bb_period).mean()
    std = close.rolling(window=bb_period).std()
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)

    # ATR for stops
    atr = calculate_atr(df, atr_period)

    # Time filter (avoid overnight)
    times = df.index.time
    in_session = pd.Series([(time(9, 30) <= t < time(15, 45)) for t in times], index=df.index)

    # Initialize
    signals = pd.Series(0.0, index=df.index)
    position = 0
    entry_price = 0
    stop_loss = 0

    for i in range(bb_period + atr_period, len(df)):
        idx = df.index[i]

        if not in_session[idx]:
            if position != 0:
                position = 0
            signals[idx] = 0
            continue

        current_close = close.iloc[i]
        current_low = low.iloc[i]
        current_high = high.iloc[i]
        current_lower = lower_band.iloc[i]
        current_upper = upper_band.iloc[i]
        current_rsi = rsi.iloc[i]

        if pd.isna(current_atr):
            signals[idx] = position
            continue

        if position == 0:
            # Long: RSI oversold + Bollinger breakout
            if current_close <= current_lower and current_rsi < oversold:
                position = 1
                entry_price = current_close
                stop_loss = entry_price - (current_atr * 2)

            # Short: RSI overbought + Bollinger breakdown
            elif current_close >= current_upper and current_rsi > overbought:
                position = -1
                entry_price = current_close
                stop_loss = entry_price + (current_atr * 2)

        elif position == 1:
            # Long exit: stop or target
            if current_low <= stop_loss or current_high >= current_upper:
                position = 0

        elif position == -1:
            # Short exit: stop or target
            if current_high >= stop_loss or current_low <= current_lower:
                position = 0

        signals[idx] = position

    return signals

def adx_trend_following_signals(df, atr_period=14):
    close = df['Close']
    high = df['High']
    low = df['Low']

    # ATR for trend filter
    atr = calculate_atr(df, atr_period)

    # Time components
    times = df.index.time
    dates = df.index.date

    # Initialize
    signals = pd.Series(0.0, index=df.index)
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    for date in pd.unique(dates):
        day_mask = [d == date for d in dates]
        day_data = df[day_mask]

        if len(day_data) < 20:  # Need enough bars
            continue

        # Get trading window (9:30 - 15:30)
        trade_mask = [(time(9, 30) <= t < time(15, 30)) for t in day_data.index.time]
        trade_indices = day_data.index[trade_mask]

        for idx in trade_indices:
            current_close = close[idx]
            current_high = high[idx]
            current_low = low[idx]
            current_atr = atr[idx]

            # ATR filter - skip very volatile days
            if pd.isna(current_atr):
                continue

            if position == 0:
                # Long entry: ADX above 25 + trend filter
                if close.iloc[idx] > (high.iloc[idx] + low.iloc[idx]) / 2 and current_atr < 1.5 * close.iloc[idx]:
                    position = 1
                    entry_price = current_close
                    stop_loss = entry_price - (current_atr * 2)
                    take_profit = entry_price + (current_atr * 3)

                # Short entry: ADX below 25 + trend filter
                elif close.iloc[idx] < (high.iloc[idx] + low.iloc[idx]) / 2 and current_atr < 1.5 * close.iloc[idx]:
                    position = -1
                    entry_price = current_close
                    stop_loss = entry_price + (current_atr * 2)
                    take_profit = entry_price - (current_atr * 3)

            elif position == 1:
                # Long exit: stop or target
                if close.iloc[idx] <= stop_loss or close.iloc[idx] >= take_profit:
                    position = 0

            elif position == -1:
                # Short exit: stop or target
                if close.iloc[idx] >= stop_loss or close.iloc[idx] <= take_profit:
                    position = 0

            signals[idx] = position

        # End of day: flatten
        eod_mask = [t >= time(15, 30) for t in day_data.index.time]
        eod_indices = day_data.index[eod_mask]
        signals[eod_indices] = 0
        position = 0

    return signals

def run_vectorized_backtest(df, signals):
    equity = pd.Series(index=df.index)
    returns = pd.Series(index=df.index)

    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    for i in range(len(df)):
        idx = df.index[i]

        if not signals[idx]:
            continue

        if position == 0:
            if signals[idx] > 0:
                position = 1
                entry_price = close.iloc[idx]
            elif signals[idx] < 0:
                position = -1
                entry_price = close.iloc[idx]

        elif position == 1:
            # Long exit: stop or target
            if (signals[idx] < 0 and close.iloc[idx] <= entry_price - stop_loss) or \
               (signals[idx] > 0 and close.iloc[idx] >= entry_price + take_profit):
                position = 0

        elif position == -1:
            # Short exit: stop or target
            if (signals[idx] > 0 and close.iloc[idx] >= entry_price - stop_loss) or \
               (signals[idx] < 0 and close.iloc[idx] <= entry_price + take_profit):
                position = 0

        returns[idx] = close.iloc[idx] - entry_price if position != 0 else 0
        equity[idx] = (1 + returns[idx]).cumprod()

    return equity, returns

def calculate_metrics(equity_curve, returns, signals):
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    sharpe = calculate_sharpe(returns)
    max_dd = calculate_max_drawdown(equity_curve)

    # Trade analysis (count position changes)
    position_changes = signals.diff().abs()
    trade_count = (position_changes > 0).sum() // 2  # Entry + exit = 1 round trip

    # Win rate from returns where we had a position
    active_returns = returns[signals.shift(1).fillna(0) != 0]
    if len(active_returns) > 0:
        winning_bars = (active_returns > 0).sum()
        total_bars = len(active_returns)
        win_rate = winning_bars / max(total_bars, 1)
    else:
        win_rate = 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / max(gross_loss, 0.0001)

    # CAGR
    years = len(equity_curve) / (252 * 78)  # Approximate trading bars per year
    if years > 0 and equity_curve.iloc[0] > 0:
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1
    else:
        cagr = 0

    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Trade Count': trade_count
    }

def main():
    run_multi_backtest("2022-01-01", "2024-12-31")

if __name__ == "__main__":
    main()
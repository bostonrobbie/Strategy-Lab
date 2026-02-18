import pandas as pd
import numpy as np

def market_regime_detection(df):
    # Use a simple moving average to detect regime changes
    ma_50 = df['Close'].rolling(window=50).mean()
    ma_200 = df['Close'].rolling(window=200).mean()

    regime_changes = []
    prev_regime = None

    for i in range(1, len(df)):
        if (ma_50.iloc[i] > ma_200.iloc[i] and
            prev_regime != 'bullish'):
            regime_changes.append((df.index[i], 'bullish'))
            prev_regime = 'bullish'
        elif (ma_50.iloc[i] < ma_200.iloc[i] and
              prev_regime != 'bearish'):
            regime_changes.append((df.index[i], 'bearish'))
            prev_regime = 'bearish'

    return pd.DataFrame(regime_changes, columns=['Date', 'Regime'])

def strategy1_orb_signals(df):
    orb_minutes = 15
    ema_filter = 50
    atr_period = 14
    atr_filter_mult = 2.5

    # Calculate opening range and breakout signals
    close = df['Close']
    high = df['High']
    low = df['Low']

    orb_data = df[:orb_minutes]
    orb_high = orb_data['High'].max()
    orb_low = orb_data['Low'].min()

    signals = pd.Series(0.0, index=df.index)

    for i in range(len(df)):
        if (close.iloc[i] > orb_high and
            ema_filter > close.iloc[i-1]):
            signals.iloc[i] = 1
        elif (close.iloc[i] < orb_low and
              ema_filter < close.iloc[i-1]):
            signals.iloc[i] = -1

    return signals

def strategy2_bollinger_mean_reversion_signals(df):
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    oversold = 30
    overbought = 70
    atr_period = 14

    # Calculate Bollinger Bands and RSI signals
    close = df['Close']
    high = df['High']
    low = df['Low']

    sma = close.rolling(window=bb_period).mean()
    std = close.rolling(window=bb_period).std()
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    signals = pd.Series(0.0, index=df.index)

    for i in range(len(df)):
        if close.iloc[i] < lower_band.iloc[i] and rsi.iloc[i] < oversold:
            signals.iloc[i] = 1
        elif close.iloc[i] > upper_band.iloc[i] and rsi.iloc[i] > overbought:
            signals.iloc[i] = -1

    return signals

def strategy3_adx_trend_following_signals(df):
    ema_fast = 20
    ema_slow = 50
    adx_period = 14
    adx_threshold = 25
    atr_period = 14

    # Calculate ADX trend following signals
    close = df['Close']
    high = df['High']
    low = df['Low']

    ema_fast_line = close.ewm(span=ema_fast, adjust=False).mean()
    ema_slow_line = close.ewm(span=ema_slow, adjust=False).mean()

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_di = 100 * (plus_dm.ewm(span=adx_period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=adx_period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=adx_period).mean()

    signals = pd.Series(0.0, index=df.index)

    for i in range(len(df)):
        if ema_fast_line.iloc[i] > ema_slow_line.iloc[i] and adx.iloc[i] > adx_threshold:
            signals.iloc[i] = 1
        elif ema_fast_line.iloc[i] < ema_slow_line.iloc[i] and adx.iloc[i] > adx_threshold:
            signals.iloc[i] = -1

    return signals
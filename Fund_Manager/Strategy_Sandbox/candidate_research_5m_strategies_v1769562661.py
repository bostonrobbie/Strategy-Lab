import pandas as pd

from datetime import time
def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr

# Time components
times = df.index.time
dates = df.index.date

for date in pd.unique(dates):
    day_mask = [(t >= time(9, 30) for t in times if d == date)]
    day_data = df[day_mask]

    # Get ORB window (9:30 - 9:45)
    orb_mask = [(time(9, 30) <= t < time(9, 45)) for t in day_data.index.time]
    orb_data = day_data[orb_mask]
import pandas as pd
from datetime import datetime, time

def load_data(symbol='NQ', interval='5m', start_date='2020-01-01', end_date='2024-12-31'):
    """Load data efficiently with date filtering"""
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')

    # Map interval to filename
    int_map = {'1m': 'm1', '5m': 'm5', '15m': 'm15'}
    suffix = int_map.get(interval, 'm5')

    filepath = os.path.join(csv_dir, f"A2API-{symbol.upper()}-{suffix}.csv")

    if not os.path.exists(filepath):
        print(f"Error: Data file not found: {filepath}")
        return None

    print(f"Loading data from {filepath}...")

    # Read only needed columns for speed
    df = pd.read_csv(filepath, usecols=['time', 'open', 'high', 'low', 'close', 'volume'])

    # Check if all timestamps are in the same timezone
    timezones = set([t.tzinfo for t in [datetime.strptime(s, '%Y-%m-%d %H:%M') for s in df['time'].tolist()]])
    if len(timezones) > 1:
        print(f"Warning: Multiple timezones detected. This may affect backtesting results.")
        return None

    # Parse dates
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert(None)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    # Standardize column names
    df.columns = [c.capitalize() for c in df.columns]

    # Filter date range
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    print(f"Loaded {len(df):,} bars from {df.index.min()} to {df.index.max()}")

    return df


# ... rest of the code remains the same ...
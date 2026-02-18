import pandas as pd

def load_data(symbol='NQ', interval='5m', start_date='2020-01-01', end_date='2024-12-31'):
    """Load data efficiently with date filtering"""
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')

    # Map interval to filename
    int_map = {'1m': 'm1', '5m': 'm5', '15m': 'm15'}
    suffix = int_map.get(interval, 'm5')

    filepath = os.path.join(csv_dir, f"A2API-{symbol.upper()}-{suffix}.csv")

    # Validate date format
    if not (start_date == end_date and len(start_date) == 10):
        raise ValueError(f"Invalid start/end dates. Should be in YYYY-MM-DD format.")

    if not os.path.exists(filepath):
        print(f"Error: Data file not found: {filepath}")
        return None

    # Read only needed columns for speed
    df = pd.read_csv(filepath, usecols=['time', 'open', 'high', 'low', 'close', 'volume'])

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
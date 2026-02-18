import pandas as pd

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

    # Check timestamp consistency
    try:
        df = pd.read_csv(filepath, usecols=['time', 'open', 'high', 'low', 'close', 'volume'])
    except ValueError as e:
        print(f"Invalid data: {e}")
        return None

    # Parse dates and check for invalid timestamps
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert(None)
    df.set_index('time', inplace=True)

    if not (df.index.min() <= start_date <= df.index.max() and df.index.min() <= end_date <= df.index.max()):
        print(f"Invalid date range: {start_date} to {end_date}")
        return None

    # Standardize column names
    df.columns = [c.capitalize() for c in df.columns]

    return df
import pandas as pd
from datetime import datetime, time

def validate_data(df):
    if df.index.min() < datetime(2022, 1, 1) or df.index.max() > datetime(2024, 12, 31):
        return False
    if not all([isinstance(x, pd.Timedelta) for x in df['time']]):
        return False
    if not all(df.columns == ['open', 'high', 'low', 'close']):
        return False
    if len(df) < 10000:
        return False
    return True


def load_data(symbol='NQ', interval='5m'):
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
    df = pd.read_csv(filepath, usecols=['time', 'open', 'high', 'low', 'close'])

    # Parse dates
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert(None)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    if not validate_data(df):
        print("Error: Data is invalid")
        return None

    return df


# Rest of the code remains the same
import pandas as pd

def load_data(symbol='NQ', interval='5m', start_date='2020-01-01', end_date='2024-12-31'):
    """
    Load data efficiently with date filtering
    """
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')

    # Map interval to filename
    int_map = {'1m': 'm1', '5m': 'm5', '15m': 'm15'}
    suffix = int_map.get(interval, 'm5')

    filepath = os.path.join(csv_dir, f"A2API-{symbol.upper()}-{suffix}.csv")

    # Validate data file
    if not os.path.exists(filepath):
        print(f"Error: Data file not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error: Unable to load data from {filepath} ({str(e)})")
        return None

    # Validate timestamps
    if df.index.min() < pd.to_datetime(start_date) or df.index.max() > pd.to_datetime(end_date):
        print("Error: Data timestamps do not match the requested period.")
        return None

    return df
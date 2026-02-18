import pandas as pd
import numpy as np
from datetime import time
import warnings
warnings.filterwarnings('ignore')

def load_data(symbol='NQ', interval='5m', start_date='2020-01-01', end_date='2024-12-31'):
    csv_dir = os.path.join(os.getcwd(), 'data', 'Intra OHLC')
    int_map = {'1m': 'm1', '5m': 'm5', '15m': 'm15'}
    suffix = int_map.get(interval, 'm5')
    filepath = os.path.join(csv_dir, f"A2API-{symbol.upper()}-{suffix}.csv")
    if not os.path.exists(filepath):
        print(f"Error: Data file not found: {filepath}")
        return None
    df = pd.read_csv(filepath, usecols=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert(None)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    df.columns = [c.capitalize() for c in df.columns]
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df

def data_quality_check(df):
    if not df.index.is_unique:
        print("Error: Duplicate timestamps found. Please clean the dataset.")
        return None
    return df

def run_multi_backtest(start_date='2022-01-01', end_date='2024-12-31'):
    df = load_data()
    df = data_quality_check(df)
    if df is None:
        print("Error: Data quality check failed. Please re-run with cleaned dataset.")
        return
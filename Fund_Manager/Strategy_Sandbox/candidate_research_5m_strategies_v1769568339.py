import pandas as pd
import numpy as np
from datetime import time
import warnings
warnings.filterwarnings('ignore')

def validate_data(df):
    """
    Validate NQ 5-minute data for missing or inconsistent timestamps
    """
    if df.isna().any().any():
        print("Error: Missing values detected")
        return False

    # Check for duplicate timestamps
    if len(df) != len(df.index.unique()):
        print("Error: Duplicate timestamps found")
        return False

    return True


def run_multi_backtest(start_date='2022-01-01', end_date='2024-12-31'):
    ...
    df = load_data('NQ', '5m', start_date, end_date)
    if df is None or df.empty:
        print("Error: Could not load data")
        return

    # Validate data
    if not validate_data(df):
        print("Data validation failed. Aborting.")
        return

    ...
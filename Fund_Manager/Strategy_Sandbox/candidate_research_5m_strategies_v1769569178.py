import pandas as pd
import numpy as np
from datetime import datetime, time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

def run_multi_backtest(start_date_str, end_date_str):
    results = {}

    # ... (rest of the code remains unchanged)

    return results

if __name__ == "__main__":
    start_time = datetime.strptime(f"{start_date_str} 09:30", "%Y-%m-%d %H:%M").time()
    end_time = datetime.strptime(f"{end_date_str} 16:00", "%Y-%m-%d %H:%M").time()
    run_multi_backtest(start_time, end_time)
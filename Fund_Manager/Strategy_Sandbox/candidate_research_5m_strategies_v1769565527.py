import pandas as pd
import numpy as np
from datetime import datetime, time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

def data_validation(df):
    if df.empty:
        return False

    for col in df.columns:
        if not isinstance(df[col].iloc[0], (int, float)):
            return False

        if pd.isnull(df[col]).any():
            return False

    return True


# rest of the code remains the same
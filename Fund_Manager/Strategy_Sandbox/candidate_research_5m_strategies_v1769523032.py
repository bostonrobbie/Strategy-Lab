import sys
import os
from datetime import time
import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.getcwd(), 'StrategyPipeline', 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'StrategyPipeline'))

import pandas.tseries.offsets as toffset

# ... (rest of the code remains the same)
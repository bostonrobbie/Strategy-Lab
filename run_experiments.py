
import sys
import os
import pandas as pd
import numpy as np
from backtesting.data import SmartDataHandler
from backtesting.optimizer import VectorizedGridSearch
from examples.nqorb_15m import NqOrb15m
from backtesting.vector_engine import VectorizedNQORB

def run_experiment(name, param_grid, baseline_sharpe=None):
    print(f"\n[EXPERIMENT] {name}...")
    
    # Setup Data
    symbol_list = ['NQ']
    # Use 1 year of data for fast iteration or full history? 
    # User said "systematically test... make sure legit". 
    # Let's use a decent chunk, maybe 2020-2023 (post-covid volatility included)
    start_date = "2020-01-01" 
    end_date = "2023-12-31"
    interval = "15m"
    
    csv_dir = os.path.join(os.getcwd(), 'examples')
    # Assuming standard data location logic in SmartDataHandler
    
    # Initialize Optimizer
    # We pass the class, but we need to ensure the grid is correct
    optimizer = VectorizedGridSearch(
        data_handler_cls=SmartDataHandler,
        data_handler_args=(symbol_list, [csv_dir], start_date, end_date, interval),
        strategy_cls=NqOrb15m, # This is just for metadata, the engine uses mappings
        param_grid=param_grid,
        initial_capital=100000.0,
        n_jobs=-1
    )
    
    results = optimizer.run()
    
    if results.empty:
        print(f"  No results for {name}")
        return None
        
    best_res = results.iloc[0]
    print(f"  Best Params: {best_res.to_dict()}")
    print(f"  Total Return: {best_res['Total Return']:.2%}")
    print(f"  Final Equity: ${best_res['Final Equity']:,.2f}")
    
    return best_res

if __name__ == "__main__":
    print("starting Systematic Feature Testing (2020-2023)...")
    
    # 0. Baseline
    # Fixed params from previous optimization
    baseline_grid = {
        'orb_start': ["09:30"],
        'orb_end': ["09:45"],
        'sl_atr_mult': [2.0],
        'tp_atr_mult': [4.0],
        'ema_filter': [50],
        'atr_max_mult': [2.5]
    }
    
    print("\n--- BASELINE ---")
    base_res = run_experiment("Baseline", baseline_grid)
    base_ret = base_res['Total Return']
    
    experiments = [
        {
            "name": "Exp 1: HTF Trend (Daily MA)",
            "grid": {
                **baseline_grid,
                "use_htf": [True],
                "htf_ma": [50, 100, 200]
            }
        },
        {
            "name": "Exp 2: RVOL Filter",
            "grid": {
                **baseline_grid,
                "use_rvol": [True],
                "rvol_thresh": [1.0, 1.5, 2.0]
            }
        },
        {
            "name": "Exp 3: Hurst Exponent (Trend Persistence)",
            "grid": {
                **baseline_grid,
                "use_hurst": [True],
                "hurst_thresh": [0.45, 0.5, 0.55]
            }
        },
        {
            "name": "Exp 4: ADX Filter (Volatility)",
            "grid": {
                **baseline_grid,
                "use_adx": [True],
                "adx_thresh": [15, 20, 25, 30]
            }
        },
        {
            "name": "Exp 5: ATR Trailing Stop",
            "grid": {
                **baseline_grid,
                "use_trailing_stop": [True],
                "ts_atr_mult": [2.0, 3.0, 4.0, 5.0],
            }
        },
        {
            "name": "Exp 6: SYNTHESIS (HTF + RVOL + TS)",
            "grid": {
                **baseline_grid,
                "use_htf": [True], "htf_ma": [100],
                "use_rvol": [True], "rvol_thresh": [1.5],
                "use_trailing_stop": [True], "ts_atr_mult": [2.0, 2.5, 3.0]
            }
        }
    ]
    
    summary = []
    summary.append(base_res)
    
    for exp in experiments:
        res = run_experiment(exp['name'], exp['grid'])
        if res is not None:
            # Calculate Delta
            delta = res['Total Return'] - base_ret
            res['Delta Return'] = delta
            res['Experiment'] = exp['name']
            summary.append(res)
            
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    df_summary = pd.DataFrame(summary)
    cols = ['Experiment', 'Total Return', 'Delta Return', 'Final Equity'] 
    # Add param cols dynamically
    for c in ['htf_ma', 'rvol_thresh', 'hurst_thresh', 'adx_thresh', 'ts_atr_mult']:
        if c in df_summary.columns: cols.append(c)
        
    print(df_summary[cols].to_string())

import pandas as pd
import numpy as np
from datetime import datetime, time
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

def run_multi_backtest(start_date, end_date):
    results = {}
    
    for symbol in ["NQ"]:
        name = f"{symbol}_5min"
        m = pd.read_csv(f"input_data/{name}.csv", index_col='Date', parse_dates=['Date'])
        
        if not isinstance(m.index[0], datetime):
            m.index = pd.to_datetime(m.index)
        
        m.sort_index(inplace=True)
        
        opens = m['Open']
        highs = m['High']
        lows = m['Low']
        closes = m['Close']

        for i in range(1, len(closes)):
            if closes[i] < 0.9*highs[i-1]:
                breaks.append((i, 'sell'))
            elif closes[i] > 1.1*lows[i-1]:
                breaks.append((i, 'buy'))

    m['Break'] = np.where(m.index.isin([x[0] for x in breaks]), x[1], None)

    if not all(m['Break'].isna()):
        m.dropna(subset=['Close'], inplace=True)
        
    else:
        m.interpolate(method='linear', inplace=True, limit_direction='both')
        
    backtested = m.copy()
    
    # Perform backtesting
    for i in range(30):
        if backtested['Break'][i] == 'buy':
            buy_price = backtested['Close'][i]
            sell_price = 0.0
            
            for j in range(i+1, len(backtested)):
                if backtested['Break'][j] == 'sell' and backtested['Close'][j] < 0.9*backtested['High'][j]:
                    sell_price = backtested['Close'][j]
                    break
                    
            if sell_price > 0.0:
                profit = (sell_price - buy_price) / buy_price
                backtested.loc[i, 'Profit'] = profit
                backtested.loc[i, 'Sharpe Ratio'] = np.sqrt(252)*profit/np.std(backtested['Close'].pct_change())
                
    status = "APPROVED" if len(fails) == 0 else "VETO"
    print(f"{name}: {status}")
    if passes:
        print(f"  [+] Passes: {', '.join(passes)}")
    if fails:
        print(f"  [-] Fails: {', '.join(fails)}")
    print("")

    approved = [name for name, m in results.items() if m['Sharpe Ratio'] > 1.0 and m['Max Drawdown'] > -0.25 and m['Profit Factor'] > 1.2 and m['Trade Count'] > 30]

    if approved:
        print(f"DEPLOY: {', '.join(approved)} meet all Risk Manager criteria")
    elif best_strat[1]['Sharpe Ratio'] > 0.5:
        print(f"REFINE: {best_strat[0]} shows promise but needs optimization")
        print("  Suggestions:")
        print("  - Adjust entry/exit criteria")
        print("  - Optimize ATR multipliers for stops/targets")
        print("  - Add additional filters (volume, time-of-day)")
    else:
        print("RESEARCH: All strategies need fundamental redesign")
        print("  Suggestions:")
        print("  - Review market regime detection")
        print("  - Consider different timeframes")
        print("  - Explore alternative signal generation methods")

    return results


if __name__ == "__main__":
    run_multi_backtest("2022-01-01", "2024-12-31")
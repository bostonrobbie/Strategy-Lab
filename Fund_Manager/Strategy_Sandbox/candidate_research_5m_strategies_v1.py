
from queue import Queue

def run_multi_backtest(start_date, end_date):
    ...
    # --- Strategy 2: Volatility Breakout (with RSI filter) ---
    print("\n>>> Testing Volatility Breakout with RSI Filter (5m)...")
    events_3 = Queue()
    data3 = SmartDataHandler(symbol_list, search_dirs=search_dirs, start_date=start_date, end_date=end_date, interval='5m')
    port3 = Portfolio(data3, events_3, initial_capital=100000.0)
    strat3 = VolBreakoutWithRSI5m(data3, events_3, verbose=False, rsi_period=14, rsi_upper=70, rsi_lower=30)
    exec3 = SimulatedExecutionHandler(events_3, data3, commission_model=FixedCommission(2.05))
    engine3 = BacktestEngine(data3, strat3, port3, exec3)
    engine3.run()
    
    ts3 = TearSheet(port3)
    stats3 = ts3.analyze()
    print(f"VolBreakout with RSI Return: {stats3.get('Total Return', 0):.2%}")
    print(f"VolBreakout with RSI MaxDD:  {stats3.get('Max Drawdown', 0):.2%}")
    
    # Winner?
    ret1 = stats1.get('Total Return', 0)
    ret2 = stats2.get('Total Return', 0)
    ret3 = stats3.get('Total Return', 0)
    
    if ret1 > ret2 and ret1 > ret3:
        print("\nWinner: Trend Pullback")
    elif ret2 > ret1 and ret2 > ret3:
        print("\nWinner: Volatility Breakout")
    else:
        print("\nWinner: VolBreakout with RSI Filter")

if __name__ == "__main__":
    run_multi_backtest("2015-01-01", "2024-12-31")
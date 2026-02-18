from datetime import datetime, time

def run_multi_backtest(start_date='2022-01-01', end_date='2024-12-31'):
    if start_date > end_date:
        print("ERROR: Start date cannot be later than end date. Please correct.")
        return None

    if not (datetime.strptime(start_date, '%Y-%m-%d') >= datetime.min and
            datetime.strptime(end_date, '%Y-%m-%d') <= datetime.max):
        print(f"ERROR: Invalid date range. Dates must be between {datetime.min.date()} and {datetime.max.date()}.")
        return None

    # ... (rest of the code remains the same) ...
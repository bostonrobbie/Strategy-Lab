import numpy as np

def early_exit(df, signals, returns, atr_period=14):
    """
    Early Exit mechanism: exit trades if expected return not met within 2-3 days
    """
    mask = (signals.diff().abs() > 0).shift(1).fillna(False)
    entry_returns = returns[mask]
    exit_returns = returns[~mask]

    for i, (entry, exit) in enumerate(zip(entry_returns, exit_returns)):
        if i % 2 == 0:
            # Long or short position
            expected_return = np.mean([returns.iloc[i-1], returns.iloc[i+1]])
            if entry > expected_return or exit < expected_return:
                signals.iloc[i] = 0  # Exit trade

    return signals
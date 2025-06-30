import numpy as np

def estimate_market_volatility(prcSoFar, window=20):
    """
    Estimate market-wide volatility as the average rolling standard deviation of log returns across instruments.

    Parameters:
    prcSoFar (numpy.ndarray): Historical price data for the instruments.
    window (int): Rolling window size for volatility calculation.

    Returns:
    float: Estimated market-wide volatility, or 0.01 if insufficient data.
    """
    log_returns = np.diff(np.log(prcSoFar), axis=1)
    if log_returns.shape[1] < window:
        print(f"Warning: Insufficient returns ({log_returns.shape[1]}) at t={prcSoFar.shape[1]}. Using default volatility.")
        return 0.01
    rolling_std_devs = np.array([np.std(log_returns[i, -window:]) for i in range(log_returns.shape[0])])
    rolling_std_devs = np.nan_to_num(rolling_std_devs, nan=0.01, posinf=0.01, neginf=0.01)
    market_volatility = np.mean(rolling_std_devs)
    market_volatility = float(market_volatility if not np.isnan(market_volatility) else 0.01)
    print(f"Computed volatility {market_volatility:.6f} at t={prcSoFar.shape[1]} with {window} returns.")
    return market_volatility
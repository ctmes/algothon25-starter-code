import numpy as np

def estimate_market_volatility(prcSoFar, window=20):
    """
    Estimate the market-wide volatility as the average rolling standard deviation of log returns across instruments.

    Parameters:
    prcSoFar (numpy.ndarray): Historical price data for the instruments.
    window (int): Rolling window size for volatility calculation.

    Returns:
    float: Estimated market-wide volatility.
    """
    # Compute log returns
    log_returns = np.diff(np.log(prcSoFar), axis=1)
    
    # Compute rolling standard deviation for each instrument
    rolling_std_devs = np.array([np.std(log_returns[i, -window:]) for i in range(log_returns.shape[0])])
    
    # Compute average volatility across instruments
    market_volatility = np.mean(rolling_std_devs)
    
    return market_volatility

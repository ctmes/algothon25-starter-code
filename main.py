import numpy as np

nInst = 50
currentPos = np.zeros(nInst)


def compute_rsi(prices, period=14):
    """
    Compute RSI for each instrument over the last period days.
    Returns: RSI values for the last timestep.
    """
    delta = np.diff(prices, axis=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(prices.shape[0])
    avg_loss = np.zeros(prices.shape[0])
    for i in range(period, delta.shape[1]):
        avg_gain = (avg_gain * (period - 1) + gain[:, i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[:, i]) / period
    rsi = np.where(avg_loss != 0, 100 - (100 * avg_gain / (avg_gain + avg_loss)), 100)
    return rsi


def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    # Validate input shape and transpose if necessary
    if nins != nInst:
        if nt == nInst:
            prcSoFar = prcSoFar.T
            nins, nt = prcSoFar.shape
        else:
            raise ValueError(f"Expected {nInst} instruments, got {nins}")

    if nt < 20:  # Need at least 20 days for volatility and RSI
        return np.zeros(nins)

    # Calculate momentum: 5-day log returns
    lookback = 5
    returns = np.log(prcSoFar[:, -1] / prcSoFar[:, -lookback - 1])

    # Calculate volatility: standard deviation of daily log returns over 20 days
    daily_returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
    volatility = np.std(daily_returns[:, -20:], axis=1)
    volatility = np.where(volatility == 0, 1e-6, volatility)

    # Trend filtering: t-statistic for momentum
    mean_returns = np.mean(daily_returns[:, -lookback:], axis=1)
    t_stat = mean_returns / (volatility / np.sqrt(lookback))
    trend_threshold = 1.0
    momentum_signal = np.where(np.abs(t_stat) > trend_threshold, returns / volatility, 0)

    # Mean-reversion: RSI-based signal for non-trending instruments
    rsi = compute_rsi(prcSoFar) if nt > 14 else np.full(nins, 50.0)
    reversion_signal = np.zeros(nins)
    reversion_mask = (np.abs(t_stat) <= trend_threshold) & ((rsi < 35) | (rsi > 65))
    reversion_signal[reversion_mask] = -np.sign(rsi[reversion_mask] - 50) / volatility[reversion_mask]

    # Combine signals
    signal = momentum_signal
    signal[reversion_mask] = reversion_signal[reversion_mask]

    # Log trading activity (for debugging)
    trending_count = np.sum(np.abs(t_stat) > trend_threshold)
    reversion_count = np.sum(reversion_mask)
    if nt % 100 == 0:  # Log periodically
        print(f"Day {nt}: Trending instruments: {trending_count}/50, Mean-reverting: {reversion_count}/50")

    # Normalize signals
    signal_norm = np.sqrt(np.sum(signal ** 2))
    if signal_norm > 0:
        signal = signal / signal_norm

    # Calculate target positions
    current_prices = prcSoFar[:, -1]
    target_dollar_pos = 7500 * signal  # Increased for moderate volatility
    target_shares = np.array([int(x / p) if p != 0 else 0 for x, p in zip(target_dollar_pos, current_prices)])

    # Enforce position limits
    pos_limits = np.array([int(10000 / p) if p != 0 else 0 for p in current_prices])
    target_shares = np.clip(target_shares, -pos_limits, pos_limits)

    # Commission management
    delta_pos = target_shares - currentPos
    trade_threshold = 0.015 * pos_limits
    delta_pos = np.where(np.abs(delta_pos) > trade_threshold, delta_pos, 0)

    # Update positions
    currentPos = np.array([int(x) for x in currentPos + delta_pos])

    return currentPos
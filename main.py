import numpy as np
from volatility import estimate_market_volatility


def compute_z_scores(prices, lookback=5):
    """Compute z-scores for mean-reversion signals based on short-term returns."""
    returns = np.diff(np.log(prices), axis=1)
    mean_returns = np.mean(returns[:, -lookback:], axis=1)
    std_returns = np.std(returns[:, -lookback:], axis=1)
    std_returns = np.where(std_returns == 0, 1e-6, std_returns)  # Avoid division by zero
    z_scores = mean_returns / std_returns
    return z_scores


def compute_correlations(prices, lookback=20):
    """Compute correlation matrix for recent returns."""
    returns = np.diff(np.log(prices), axis=1)[:, -lookback:]
    corr_matrix = np.corrcoef(returns)
    return corr_matrix


def find_pairs(corr_matrix, threshold=0.75):
    """Identify highly correlated pairs for pairs trading."""
    n = corr_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i, j] > threshold:
                pairs.append((i, j))
    return pairs


def getMyPosition(prices):
    """
    Mean-reversion strategy with pairs trading and dynamic risk scaling.
    Parameters:
        prices: np.array of shape (50, nDays) - historical prices
    Returns:
        positions: np.array of shape (50,) - number of shares to hold
    """
    nInst, nDays = prices.shape
    positions = np.zeros(nInst)

    # Use global variable to track rebalance state
    global last_rebalance
    if 'last_rebalance' not in globals():
        last_rebalance = -30

    # Rebalance every 30 days
    if nDays - last_rebalance < 30 or nDays < 20:
        return positions

    # Compute mean returns for instrument selection
    returns = np.diff(np.log(prices), axis=1)
    mean_returns = np.mean(returns[:, -20:], axis=1)  # Use recent 20 days for stability

    # Select top 10 instruments with least negative returns
    top_indices = np.argsort(mean_returns)[-10:] if nDays >= 20 else np.arange(nInst)

    # Compute z-scores for mean-reversion
    z_scores = compute_z_scores(prices[top_indices], lookback=5)

    # Compute correlations for pairs trading
    corr_matrix = compute_correlations(prices[top_indices], lookback=20)
    pairs = find_pairs(corr_matrix, threshold=0.75)

    # Estimate market volatility and adjust risk
    market_vol = estimate_market_volatility(prices, window=20)
    target_risk = 2500 / (market_vol + 1e-6)  # Conservative for volatility ~0.5428

    # Mean-reversion signals
    for idx, inst in enumerate(top_indices):
        z = z_scores[idx]
        if z > 0.6:  # Overbought: short
            positions[inst] = -target_risk / prices[inst, -1]
        elif z < -0.6:  # Oversold: long
            positions[inst] = target_risk / prices[inst, -1]

    # Pairs trading for correlated instruments
    for i, j in pairs:
        inst_i, inst_j = top_indices[i], top_indices[j]
        z_i, z_j = z_scores[i], z_scores[j]
        spread = z_i - z_j
        if spread > 0.6:  # Spread overbought: short i, long j
            positions[inst_i] -= target_risk / prices[inst_i, -1]
            positions[inst_j] += target_risk / prices[inst_j, -1]
        elif spread < -0.6:  # Spread oversold: long i, short j
            positions[inst_i] += target_risk / prices[inst_i, -1]
            positions[inst_j] -= target_risk / prices[inst_j, -1]

    # Cap position sizes to reduce turnover
    positions = np.clip(positions, -1000, 1000)

    # Update rebalance tracker
    last_rebalance = nDays

    return positions
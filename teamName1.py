import numpy as np
from sklearn.decomposition import PCA
from volatility import estimate_market_volatility


def getMyPosition(prcSoFar, n_pca_components=1, z_score_threshold=0.7, lookback_window_for_stats=50,
                  base_volatility=0.005, momentum_window=10, signal_weights=(0.0, 1.0), vol_window=5,
                  use_ema=True, return_all_z_scores=False):
    """
    Generate positions based on a mean-reversion strategy using PCA residuals.

    Parameters:
        prcSoFar (np.array): Historical price data (shape: nInst, nt).
        n_pca_components (int): Number of PCA components.
        z_score_threshold (float): Threshold for trading based on z-score.
        lookback_window_for_stats (int): Lookback window for volatility and mean calculations.
        base_volatility (float): Base volatility for position sizing.
        momentum_window (int): Lookback window for momentum (unused if momentum_weight=0).
        signal_weights (tuple): (momentum_weight, mean_reversion_weight).
        vol_window (int): Window for volatility estimation.
        use_ema (bool): If True, use EMA for volatility estimation.
        return_all_z_scores (bool): If True, return z-scores for all time steps.

    Returns:
        tuple: (position, prcSoFar, z_scores)
            - position (np.array): Positions for each instrument.
            - prcSoFar (np.array): Input price array (unchanged).
            - z_scores (np.array): Z-scores (1D for last day or 2D if return_all_z_scores=True).
    """
    nInst, nt = prcSoFar.shape
    position = np.zeros(nInst)

    # Check if enough data is available
    min_days = max(n_pca_components, 2, lookback_window_for_stats, momentum_window, vol_window)
    if nt < min_days:
        return position, prcSoFar, np.zeros(nInst) if not return_all_z_scores else np.zeros((nInst, 0))

    # Calculate returns for PCA
    returns = np.diff(prcSoFar, axis=1) / prcSoFar[:, :-1]  # Shape: (nInst, nt-1)

    # Apply PCA to returns
    pca = PCA(n_components=n_pca_components)
    pca.fit(returns.T)
    residuals = returns.T - pca.inverse_transform(pca.transform(returns.T))
    residuals = residuals.T  # Shape: (nInst, nt-1)

    # Get volatility from estimate_market_volatility
    _, instrument_volatilities = estimate_market_volatility(prcSoFar, window=vol_window, use_ema=use_ema)
    safe_vol = np.where(instrument_volatilities == 0, 1e-9, instrument_volatilities)

    # Mean-reversion z-score
    if return_all_z_scores:
        z_scores = np.zeros((nInst, nt - min_days + 1))
        for t in range(min_days - 1, nt):
            z_scores[:, t - (min_days - 1)] = residuals[:, t - (min_days - 1)] / safe_vol
    else:
        z_scores = residuals[:, -1] / safe_vol  # Shape: (nInst,)

    # Generate positions based on z-score threshold (only for last day)
    if not return_all_z_scores:
        cur_prices = prcSoFar[:, -1]
        safe_cur_prices = np.where(cur_prices == 0, 1e-9, cur_prices)
        for i in range(nInst):
            if np.abs(z_scores[i]) > z_score_threshold:
                position[i] = -np.sign(z_scores[i]) * np.floor((base_volatility / safe_vol[i]) * 1000)

        # Apply daily turnover cap (10,000) with turnover penalty
        daily_dvolume = np.sum(np.abs(position * safe_cur_prices))
        if daily_dvolume > 10000:
            scaling_factor = 10000 / (daily_dvolume + 1e-9)
            position = position * scaling_factor
            position = np.array([int(x) for x in position])

    return position, prcSoFar, z_scores
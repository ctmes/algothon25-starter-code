import numpy as np
from sklearn.decomposition import PCA
from volatility import estimate_market_volatility

def getMyPosition(prcSoFar):
    """
    Enhanced version of a residual-based mean-reversion strategy using PCA.
    Dynamically scales notional exposure based on market-wide volatility.
    """

    # Step 1: Compute log returns
    returns = np.diff(np.log(prcSoFar), axis=1)

    # Step 2: Apply PCA to extract 2 market factors
    pca = PCA(n_components=2)
    pca.fit(returns.T)
    reconstructed = pca.inverse_transform(pca.transform(returns.T)).T

    # Step 3: Compute residuals (idiosyncratic returns)
    residuals = returns - reconstructed

    # Step 4: Z-score the most recent residual
    mean = np.mean(residuals, axis=1)
    std = np.std(residuals, axis=1)
    z = (residuals[:, -1] - mean) / std
    z = np.nan_to_num(z)  # handle divide-by-zero

    # Step 5: Apply z-score threshold
    threshold = 1.0
    positions = np.where(np.abs(z) > threshold, -z, 0)

    # Step 6: Estimate market-wide volatility
    market_volatility = estimate_market_volatility(prcSoFar)
    base_volatility = 0.01  # target market volatility
    scaling_factor = base_volatility / market_volatility

    # Step 7: Dynamically scale notional exposure
    notional_per_unit = 5000 * scaling_factor
    positions = positions * notional_per_unit / prcSoFar[:, -1]

    # Step 8: Convert positions to integer for execution
    return positions.astype(int)

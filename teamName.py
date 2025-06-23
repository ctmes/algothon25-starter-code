import numpy as np
from sklearn.decomposition import PCA

def getMyPosition(prcSoFar):
    """
    MVP version of a residual-based mean-reversion strategy using PCA.
    No scaling, no clipping, no risk model — just raw signal logic.
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

    # Step 5: Position = -z-score (mean-reversion)
    return (-z).astype(int)

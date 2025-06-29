import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from teamName1 import getMyPosition  # Note: Assuming teamName1.py is your modified file
from volatility import estimate_market_volatility  # Assuming volatility.py exists for market volatility


def loadPrices(fn="prices.txt"):
    """Loads prices from a text file."""
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    return (df.values).T


def plot_diagnostics(prcAll, num_days_to_plot=200):
    """
    Generates plots for strategy diagnostics:
    - Residuals for selected instruments
    - Z-scores for selected instruments
    - Market Volatility over time
    - Histogram of Z-scores
    """
    nInst, nt = prcAll.shape

    # Store historical residuals, z-scores, and market volatility
    all_residuals = []
    all_z_scores = []
    market_vol_history = []

    # Simulate strategy execution to collect data
    startDay = nt - num_days_to_plot
    for t in range(startDay, nt):
        prcHistSoFar = prcAll[:, :t + 1]

        # Get positions and diagnostics from your strategy
        _, residuals, z = getMyPosition(prcHistSoFar)

        # Store the current state's diagnostics
        all_residuals.append(residuals[:, -1])  # Store the most recent residual for each instrument
        all_z_scores.append(z)  # Store the most recent z-score for each instrument

        # Estimate and store market volatility
        market_vol = estimate_market_volatility(prcHistSoFar)
        market_vol_history.append(market_vol)

    all_residuals = np.array(all_residuals).T  # Shape: (nInst, num_days_to_plot)
    all_z_scores = np.array(all_z_scores).T  # Shape: (nInst, num_days_to_plot)
    market_vol_history = np.array(market_vol_history)

    # --- Plotting ---

    # 1. Residuals for a few instruments
    plt.figure(figsize=(15, 8))
    num_instruments_to_plot = min(nInst, 5)  # Plot up to 5 instruments
    for i in range(num_instruments_to_plot):
        plt.plot(all_residuals[i], label=f'Instrument {i + 1} Residuals')
    plt.title('Residuals Over Time (Last 200 Days)')
    plt.xlabel('Days')
    plt.ylabel('Residual Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Z-scores for a few instruments
    plt.figure(figsize=(15, 8))
    for i in range(num_instruments_to_plot):
        plt.plot(all_z_scores[i], label=f'Instrument {i + 1} Z-Score')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold +1.0')
    plt.axhline(y=-1.0, color='r', linestyle='--', label='Threshold -1.0')
    plt.title('Z-Scores Over Time (Last 200 Days)')
    plt.xlabel('Days')
    plt.ylabel('Z-Score')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Market Volatility
    plt.figure(figsize=(15, 6))
    plt.plot(market_vol_history, label='Estimated Market Volatility', color='purple')
    plt.title('Estimated Market Volatility Over Time')
    plt.xlabel('Days')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 4. Histogram of Z-scores (across all instruments and days)
    plt.figure(figsize=(10, 6))
    plt.hist(all_z_scores.flatten(), bins=50, density=True, alpha=0.7, color='skyblue')
    plt.title('Distribution of Z-Scores')
    plt.xlabel('Z-Score Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    prcAll = loadPrices("prices.txt")  # Make sure prices.txt is in the same directory
    print("Loaded %d instruments for %d days" % (prcAll.shape[0], prcAll.shape[1]))
    plot_diagnostics(prcAll, num_days_to_plot=200)  # Plot for the last 200 days of data
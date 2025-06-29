import numpy as np
import matplotlib.pyplot as plt
import os
import json
from volatility import estimate_market_volatility

# Create plots directory
if not os.path.exists('plots'):
 os.makedirs('plots')

# Load prices
prices = np.loadtxt('prices.txt')  # Shape: (50 instruments, >=200 days)
nInst, nDays = prices.shape

# Compute daily returns
returns = np.diff(np.log(prices), axis=1)
mean_returns = np.mean(returns, axis=1)
std_returns = np.std(returns, axis=1)

# Plot 1: Price series for first 5 instruments
plt.figure()
for i in range(min(5, nInst)):
 plt.plot(prices[i], label=f'Instrument {i}')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.savefig('plots/price_series.png')
plt.close()

# Plot 2: Rolling volatility (20-day window)
window = 20
rolling_vol = np.zeros((nInst, nDays - 1))
for t in range(nDays - 1):
 start = max(0, t - window + 1)
 rolling_vol[:, t] = np.std(returns[:, start:t + 1], axis=1)
plt.figure()
for i in range(min(5, nInst)):
 plt.plot(rolling_vol[i], label=f'Instrument {i}')
plt.xlabel('Day')
plt.ylabel('Volatility')
plt.legend()
plt.savefig('plots/rolling_volatility.png')
plt.close()

# Plot 3: Correlation heatmap of daily returns
corr_matrix = np.corrcoef(returns)
plt.figure()
plt.imshow(corr_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Heatmap')
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# Plot 4: Returns histogram for instrument 0
plt.figure()
plt.hist(returns[0], bins=30)
plt.xlabel('Daily Returns')
plt.ylabel('Frequency')
plt.title('Returns Histogram (Instrument 0)')
plt.savefig('plots/returns_histogram.png')
plt.close()

# Plot 5: Z-score histogram for instrument 0
z_scores = (returns[0] - mean_returns[0]) / std_returns[0]
plt.figure()
plt.hist(z_scores, bins=30)
plt.xlabel('Z-Scores')
plt.ylabel('Frequency')
plt.title('Z-Score Histogram (Instrument 0)')
plt.savefig('plots/zscore_histogram.png')
plt.close()

# Plot 6: Market volatility
market_vol = [estimate_market_volatility(prices[:, :t + 1]) for t in range(1, nDays)]
plt.figure()
plt.plot(market_vol)
plt.xlabel('Day')
plt.ylabel('Market Volatility')
plt.title('Market Volatility Over Time')
plt.savefig('plots/market_volatility.png')
plt.close()

# Save analysis results
results = {
 'n_instruments': nInst,
 'n_days': nDays,
 'statistics': {
     'mean_returns': mean_returns.tolist(),
     'std_returns': std_returns.tolist(),
     'mean_market_vol': float(np.mean(market_vol))
 },
 'plot_paths': [
     'plots/price_series.png',
     'plots/rolling_volatility.png',
     'plots/correlation_heatmap.png',
     'plots/returns_histogram.png',
     'plots/zscore_histogram.png',
     'plots/market_volatility.png'
 ]
}
with open('analysis_results.json', 'w') as f:
 json.dump(results, f, indent=4)
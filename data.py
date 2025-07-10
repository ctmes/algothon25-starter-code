import numpy as np
import pandas as pd
import os


def load_price_data(file_path="prices.txt"):
    """
    Load price data from a file or generate synthetic data if file is unavailable.
    Returns: numpy array of shape [nInst, nDays].
    """
    nInst = 50
    nDays = 1000  # Updated to match provided data

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, delimiter=r'\s+', skipinitialspace=True, header=None, engine='python')
            prices = df.to_numpy()
            print(f"Loaded prices.txt with shape {prices.shape}")
            if prices.shape[0] == nDays and prices.shape[1] == nInst:
                print(f"Transposing from [{prices.shape[0]}, {prices.shape[1]}] to [{nInst}, {nDays}]")
                prices = prices.T
            elif prices.shape[0] != nInst:
                raise ValueError(f"Expected {nInst} instruments, got {prices.shape[0]}")
            if prices.shape[1] != nDays:
                print(f"Warning: Expected {nDays} days, got {prices.shape[1]}")
            if np.any(prices <= 0):
                raise ValueError("Negative or zero prices detected")
            return prices
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Generating synthetic data.")

    np.random.seed(42)
    prices = np.zeros((nInst, nDays))
    for i in range(nInst):
        trend = np.random.choice([0, 0.0005, -0.0005])
        volatility = np.random.uniform(0.005, 0.02)
        initial_price = np.random.uniform(10, 100)
        returns = np.random.normal(trend, volatility, nDays)
        prices[i] = initial_price * np.exp(np.cumsum(returns))
    return prices


def compute_rsi(prices, period=14):
    """
    Compute Relative Strength Index (RSI) for each instrument.
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


def analyze_prices(prices, lookback_momentum=5, lookback_volatility=20, trend_threshold=1.0):
    """
    Analyze price data to compute per-instrument and portfolio metrics.
    Args:
        prices: numpy array [nInst, nDays]
        lookback_momentum: Days for momentum calculation
        lookback_volatility: Days for volatility calculation
        trend_threshold: t-statistic threshold for trend significance
    Returns: Dictionary with analysis results
    """
    nInst, nDays = prices.shape
    results = {
        'instrument_metrics': [],
        'summary': {}
    }

    daily_returns = np.log(prices[:, 1:] / prices[:, :-1])

    for i in range(nInst):
        ret_5d = np.log(prices[i, -1] / prices[i, -lookback_momentum - 1]) if nDays > lookback_momentum else 0
        ret_10d = np.log(prices[i, -1] / prices[i, -10 - 1]) if nDays > 10 else 0
        vol = np.std(daily_returns[i, -lookback_volatility:]) if nDays > lookback_volatility else 0
        vol = max(vol, 1e-6)
        mean_ret = np.mean(daily_returns[i, -lookback_momentum:]) if nDays > lookback_momentum else 0
        t_stat = mean_ret / (vol / np.sqrt(lookback_momentum)) if vol > 0 else 0
        rsi = compute_rsi(prices[i:i + 1, :])[0] if nDays > 14 else 50
        behavior = 'Trending' if abs(t_stat) > trend_threshold else (
            'Mean-Reverting' if rsi < 30 or rsi > 70 else 'Neutral')

        results['instrument_metrics'].append({
            'instrument': i,
            'current_price': prices[i, -1],
            '5d_return': ret_5d,
            '10d_return': ret_10d,
            'volatility': vol,
            't_stat': t_stat,
            'rsi': rsi,
            'behavior': behavior
        })

    vol_avg = np.mean([m['volatility'] for m in results['instrument_metrics']])
    trending_count = sum(1 for m in results['instrument_metrics'] if m['behavior'] == 'Trending')
    corr_matrix = np.corrcoef(daily_returns[:, -60:]) if nDays > 60 else np.zeros((nInst, nInst))
    avg_corr = np.mean(corr_matrix[np.triu_indices(nInst, k=1)]) if nDays > 60 else 0
    max_daily_ret = np.max(np.abs(daily_returns[:, -20:]), axis=1).mean() if nDays > 20 else 0

    results['summary'] = {
        'avg_volatility': vol_avg,
        'trending_instruments': trending_count,
        'avg_correlation': avg_corr,
        'n_days': nDays,
        'max_daily_return': max_daily_ret
    }

    return results


def save_analysis(results, output_file="price_analysis.csv"):
    """
    Save analysis results to a CSV file.
    """
    df_instruments = pd.DataFrame(results['instrument_metrics'])
    df_instruments.to_csv(output_file, index=False)
    with open(output_file, 'a') as f:
        f.write("\nSummary Metrics:\n")
        for key, value in results['summary'].items():
            f.write(f"{key},{value}\n")
    print(f"Analysis saved to {output_file}")


def main():
    prices = load_price_data("prices.txt")
    results = analyze_prices(prices, lookback_momentum=5, lookback_volatility=20, trend_threshold=1.0)
    save_analysis(results, "price_analysis.csv")
    trending_count = results['summary']['trending_instruments']
    avg_vol = results['summary']['avg_volatility']
    max_ret = results['summary']['max_daily_return']
    print("Parameter Suggestions:")
    print(f"- Trending instruments: {trending_count}/50")
    print(f"- Average volatility: {avg_vol:.6f}")
    print(f"- Max daily return: {max_ret:.6f}")
    print("- Suggested lookback_momentum: 5 (stable trends)")
    print(f"- Suggested trend_threshold: 1.0 (relaxed due to low trends)")
    if trending_count < 10:
        print("- Consider hybrid strategy (momentum + mean-reversion) due to low trending instruments.")
    if avg_vol > 0.02:
        print("- High volatility detected; consider tighter position limits (e.g., $6000) and verify price data.")


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from main import getMyPosition
from eval import calcPL


def run_backtest(prices, lookback=200, test_days=50):
    """Simplified non-parallel backtest for reliability."""
    results = []
    total_days = prices.shape[1]

    for start in range(lookback, total_days - test_days, test_days):
        end = start + test_days
        segment = prices[:, :end]

        mean_pl, _, pl_std, sharpe, _ = calcPL(segment, test_days)
        results.append({
            'start': start, 'end': end,
            'sharpe': sharpe,
            'score': mean_pl - 0.1 * pl_std
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Fix the warning by using raw string for regex separator
    prices = pd.read_csv(r'prices.txt', sep=r'\s+', header=None).values.T
    results = run_backtest(prices)
    print(f"Median Sharpe: {results['sharpe'].median():.2f}")
    print(f"Median Score: {results['score'].median():.2f}")
    results.to_csv('backtest_results.csv', index=False)
# walk_forward_orchestrator.py
import numpy as np
import pandas as pd

# Import constants and loadPrices from your existing eval.py
from eval import loadPrices, nInst, nt, commRate, dlrPosLimit

# Import your modified strategy function (ensure teamName1.py is the one I provided previously)
from teamName1 import getMyPosition

# --- Walk-Forward Configuration ---
IN_SAMPLE_WINDOW_DAYS = 500  # Example: Use 500 days for training/optimization
OUT_OF_SAMPLE_WINDOW_DAYS = 250  # CRITICAL: Now matches lookback_window_for_stats in teamName1.py
STEP_DAYS = 50  # How many days to advance the window each time (can be same as OUT_OF_SAMPLE_WINDOW_DAYS for non-overlapping)

# Define parameter grids for optimization
PCA_COMPONENTS_RANGE = range(1, 6)  # Test 1 to 5 PCA components
Z_SCORE_THRESHOLD_RANGE = np.arange(0.25, 2, 0.25)
# CRITICAL ADDITION: Optimize base_volatility
BASE_VOLATILITY_RANGE = np.arange(0.001, 0.051, 0.005) # Example range: 0.001 to 0.05, step 0.005


# --- Helper function to run a backtest for a given period and parameters ---
def run_backtest_segment(prc_data_segment, n_pca, z_threshold, base_volatility_param, initial_cash=100000.0): # Add base_volatility_param
    """
    Runs a backtest on a given price data segment with specified parameters.
    Returns the annualized Net Sharpe Ratio, daily Net Returns,
    annualized Gross Sharpe Ratio, daily Gross Returns, and total dollar volume for the segment.
    """
    n_instruments, n_days_segment = prc_data_segment.shape

    # Initialize variables for NET P&L calculation
    cash_net = initial_cash
    curPos_net = np.zeros(n_instruments)
    daily_values_net = []
    totDVolume_net = 0

    # Initialize variables for GROSS P&L calculation
    cash_gross = initial_cash
    curPos_gross = np.zeros(n_instruments)
    daily_values_gross = []

    # Minimum history required by getMyPosition for meaningful PCA/stats (e.g., 2 for returns, n_pca for PCA, lookback for stats)
    # lookback_window_for_stats is currently hardcoded to 250 in getMyPosition, so we must respect that.
    min_days_for_strategy = max(n_pca, 2, 250)

    for t_relative in range(n_days_segment):
        prcHistSoFar_for_strategy = prc_data_segment[:, :t_relative + 1]
        curPrices = prcHistSoFar_for_strategy[:, -1]

        # Determine new positions for the next day, if applicable
        newPosOrig = np.zeros(n_instruments)  # Default to no new positions
        if t_relative < n_days_segment - 1:  # Don't trade on the very last day of the segment
            # Call getMyPosition only if enough historical data is available
            if prcHistSoFar_for_strategy.shape[1] >= min_days_for_strategy:  # Ensure enough data for full lookback
                # Pass the new base_volatility_param to getMyPosition
                newPosOrig, _, _ = getMyPosition(prcHistSoFar_for_strategy,
                                                 n_pca_components=n_pca,
                                                 z_score_threshold=z_threshold,
                                                 lookback_window_for_stats=250,
                                                 base_volatility=base_volatility_param) # Pass base_volatility

            # Apply dollar position limits (using constants from eval.py)
            safe_cur_prices = np.where(curPrices == 0, 1e-9, curPrices)  # Avoid div by zero
            posLimits = np.array([int(x) for x in dlrPosLimit / safe_cur_prices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
        else:  # On the very last day of the segment, no new trades are initiated
            newPos = np.array(curPos_net)  # Keep existing net positions for final valuation

        # --- NET P&L Calculation ---
        deltaPos_net = newPos - curPos_net
        dvolume_net_day = np.sum(curPrices * np.abs(deltaPos_net))
        totDVolume_net += dvolume_net_day  # Accumulate total dollar volume
        comm_net = dvolume_net_day * commRate

        cash_net -= curPrices.dot(deltaPos_net) + comm_net
        curPos_net = np.array(newPos)  # Update current positions for net calculation

        posValue_net = curPos_net.dot(curPrices)
        daily_portfolio_value_net = cash_net + posValue_net
        daily_values_net.append(daily_portfolio_value_net)

        # --- GROSS P&L Calculation ---
        deltaPos_gross = newPos - curPos_gross

        cash_gross -= curPrices.dot(deltaPos_gross)  # No commission for gross calculation
        curPos_gross = np.array(newPos)  # Update current positions for gross calculation

        posValue_gross = curPos_gross.dot(curPrices)
        daily_portfolio_value_gross = cash_gross + posValue_gross
        daily_values_gross.append(daily_portfolio_value_gross)

    # Calculate daily returns from the daily_values series
    daily_returns_net = pd.Series(daily_values_net).pct_change().dropna().values
    daily_returns_gross = pd.Series(daily_values_gross).pct_change().dropna().values

    # Calculate NET Sharpe Ratio
    annSharpe_net = 0.0
    if len(daily_returns_net) > 0 and np.std(daily_returns_net) > 0:
        annSharpe_net = np.sqrt(249) * np.mean(daily_returns_net) / np.std(daily_returns_net)

    # Calculate GROSS Sharpe Ratio
    annSharpe_gross = 0.0
    if len(daily_returns_gross) > 0 and np.std(daily_returns_gross) > 0:
        annSharpe_gross = np.sqrt(249) * np.mean(daily_returns_gross) / np.std(daily_returns_gross)

    # Always return all expected values, even if some are 0.0 or empty arrays
    return annSharpe_net, daily_returns_net, annSharpe_gross, daily_returns_gross, totDVolume_net


# --- Main Walk-Forward Execution ---
if __name__ == "__main__":
    pricesFile = "prices.txt"
    prcAll = loadPrices(pricesFile)  # Load your entire price data using eval.loadPrices
    nInst, nt = prcAll.shape  # nInst and nt are also available from eval.py

    all_oos_net_returns = []  # To store all out-of-sample daily NET returns
    all_oos_gross_returns = []  # To store all out-of-sample daily GROSS returns

    # Loop through the data for walk-forward periods
    start_of_in_sample = 0
    # Ensure we have at least IN_SAMPLE_WINDOW_DAYS + OUT_OF_SAMPLE_WINDOW_DAYS for a full cycle
    while (start_of_in_sample + IN_SAMPLE_WINDOW_DAYS + OUT_OF_SAMPLE_WINDOW_DAYS) <= nt:

        in_sample_start_idx = start_of_in_sample
        in_sample_end_idx = in_sample_start_idx + IN_SAMPLE_WINDOW_DAYS - 1  # Inclusive

        oos_start_idx = in_sample_end_idx + 1
        oos_end_idx = oos_start_idx + OUT_OF_SAMPLE_WINDOW_DAYS - 1  # Inclusive

        print(
            f"\n--- Walk-Forward Window: In-Sample Days [{in_sample_start_idx}-{in_sample_end_idx}], Out-of-Sample Days [{oos_start_idx}-{oos_end_idx}] ---")

        # --- Optimize Parameters on In-Sample Data ---
        best_sharpe_in_sample = -np.inf  # Track net sharpe for optimization
        best_params = {}
        best_in_sample_totDVolume = 0  # Track totDVolume for best in-sample params

        # The data segment for in-sample optimization
        prc_in_sample_segment = prcAll[:, in_sample_start_idx: in_sample_end_idx + 1]

        # Determine minimum history required for getMyPosition (for PCA and statistics lookback)
        min_history_for_pca = max(PCA_COMPONENTS_RANGE[-1], 2)  # Max components or 2 for returns
        min_history_for_stats_lookback = 250  # Default lookback_window_for_stats from teamName1.py
        min_segment_length_required = max(min_history_for_pca, min_history_for_stats_lookback)

        if prc_in_sample_segment.shape[1] < min_segment_length_required:
            print(
                f"  Not enough in-sample data ({prc_in_sample_segment.shape[1]} days) for minimum history ({min_segment_length_required} days). Skipping.")
            start_of_in_sample += STEP_DAYS
            continue

        for n_pca in PCA_COMPONENTS_RANGE:
            for z_threshold in Z_SCORE_THRESHOLD_RANGE:
                for base_volatility_param in BASE_VOLATILITY_RANGE: # Iterate over base_volatility_param
                    # Run backtest for this parameter combination on in-sample data
                    sharpe_net, _, sharpe_gross, _, totDVolume = run_backtest_segment(
                        prc_in_sample_segment,
                        n_pca=n_pca,
                        z_threshold=z_threshold,
                        base_volatility_param=base_volatility_param # Pass the parameter
                    )

                    print(
                        f"  In-Sample Params: PCA={n_pca}, Z-Thresh={z_threshold:.2f}, BaseVol={base_volatility_param:.4f} -> Net Sharpe={sharpe_net:.4f}, Gross Sharpe={sharpe_gross:.4f}, Total DVolume={totDVolume:.0f}")

                    # Optimize based on Net Sharpe Ratio
                    if sharpe_net > best_sharpe_in_sample:
                        best_sharpe_in_sample = sharpe_net
                        best_params = {'n_pca': n_pca, 'z_threshold': z_threshold, 'base_volatility': base_volatility_param} # Store new param
                        best_in_sample_totDVolume = totDVolume  # Store the totDVolume for the best params

        if not best_params or best_sharpe_in_sample == -np.inf:
            print(
                "  Warning: No valid or profitable parameters found for in-sample optimization. Skipping this window.")
            start_of_in_sample += STEP_DAYS
            continue

        print(
            f"  Best In-Sample Params: {best_params} with Net Sharpe={best_sharpe_in_sample:.4f}, Total DVolume={best_in_sample_totDVolume:.0f}")

        # --- Test Best Parameters on Out-of-Sample Data ---
        prc_oos_segment = prcAll[:, oos_start_idx: oos_end_idx + 1]

        # Ensure enough OOS data to run the strategy
        if prc_oos_segment.shape[1] < min_segment_length_required:
            print(
                f"  Not enough out-of-sample data ({prc_oos_segment.shape[1]} days) for minimum history ({min_segment_length_required} days). Skipping OOS test.")
            start_of_in_sample += STEP_DAYS
            continue

        oos_sharpe_net, oos_daily_returns_net, oos_sharpe_gross, oos_daily_returns_gross, oos_totDVolume = run_backtest_segment(
            prc_oos_segment,
            n_pca=best_params['n_pca'],
            z_threshold=best_params['z_threshold'],
            base_volatility_param=best_params['base_volatility'] # Pass best param
        )
        all_oos_net_returns.extend(oos_daily_returns_net)
        all_oos_gross_returns.extend(oos_daily_returns_gross)

        print(
            f"  Out-of-Sample Net Sharpe for these params: {oos_sharpe_net:.4f}, Gross Sharpe: {oos_sharpe_gross:.4f}, Total DVolume={oos_totDVolume:.0f}")
        # Add diagnostic prints to verify OOS returns are being collected
        print(
            f"  OOS Net Returns collected for this window: {len(oos_daily_returns_net)} days, Total Collected: {len(all_oos_net_returns)} days")
        print(
            f"  OOS Gross Returns collected for this window: {len(all_oos_gross_returns)} days, Total Collected: {len(all_oos_gross_returns)} days")

        # Advance the window for the next iteration
        start_of_in_sample += STEP_DAYS

    # --- Final Aggregate Performance ---
    # Overall Net Sharpe
    final_oos_net_returns = np.array(all_oos_net_returns)
    overall_net_sharpe = 0.0
    if len(final_oos_net_returns) > 0 and np.std(final_oos_net_returns) > 0:
        overall_net_sharpe = np.sqrt(249) * np.mean(final_oos_net_returns) / np.std(final_oos_net_returns)
        print(f"\n--- Overall Walk-Forward Out-of-Sample NET Sharpe Ratio: {overall_net_sharpe:.4f} ---")
    else:
        print(
            "\n--- No valid out-of-sample net returns to calculate overall Sharpe. This usually means not enough data or all segments skipped. ---")

    # Overall Gross Sharpe
    final_oos_gross_returns = np.array(all_oos_gross_returns)
    overall_gross_sharpe = 0.0
    if len(final_oos_gross_returns) > 0 and np.std(final_oos_gross_returns) > 0:
        overall_gross_sharpe = np.sqrt(249) * np.mean(final_oos_gross_returns) / np.std(final_oos_gross_returns)
        print(f"--- Overall Walk-Forward Out-of-Sample GROSS Sharpe Ratio: {overall_gross_sharpe:.4f} ---")
    else:
        print("--- No valid out-of-sample gross returns to calculate overall Sharpe. ---")

    # You can also add plotting here using your plotting.py if desired
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(np.cumsum(final_oos_net_returns), label='Net Cumulative Returns')
    # plt.plot(np.cumsum(final_oos_gross_returns), label='Gross Cumulative Returns', linestyle='--')
    # plt.title('Cumulative P&L (Walk-Forward Out-of-Sample)')
    # plt.xlabel('Days')
    # plt.ylabel('Cumulative Return')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

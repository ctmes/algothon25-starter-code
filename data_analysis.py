import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def load_prices(file_path="prices.txt"):
    """
    Loads price data from a specified file.
    Assumes the file is space-separated with no header and returns a transposed numpy array.
    """
    try:
        df = pd.read_csv(file_path, sep='\s+', header=None, index_col=None)
        # Fill forward NaNs and replace 0s with NaN, then drop columns that are all NaN
        df = df.ffill().replace(0, np.nan).dropna(axis=1, how='all')
        if np.all(df.values == df.values[0, 0]):
            print("Warning: Price data is flat. Check input file.", file=sys.stderr)
        print(f"Loaded {df.shape[1]} instruments for {df.shape[0]} days.", file=sys.stderr)
        return df.values.T  # Transpose to nInst x nt
    except FileNotFoundError:
        print(f"Error: prices.txt not found at {file_path}. Please ensure the file exists.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred while loading prices: {e}", file=sys.stderr)
        return None


def analyze_data_characteristics(prc_all, output_file="data_analysis_output.txt"):
    """
    Performs a deep dive into the statistical characteristics of the price data
    and writes the output to a specified file.

    Args:
        prc_all (np.array): Price data (nInst x nt).
        output_file (str): Path to the file where the analysis output will be saved.
    """
    if prc_all is None:
        return

    n_inst, n_t = prc_all.shape

    with open(output_file, 'w') as f:
        def print_to_file(*args, **kwargs):
            print(*args, file=f, **kwargs)

        print_to_file(f"\n--- Data Characteristics Analysis ({n_inst} instruments, {n_t} days) ---")

        # --- 1. Calculate Returns (Log Returns for better properties) ---
        # Handle potential zeros in prices by replacing with a small number before log
        safe_prc_all = np.where(prc_all == 0, 1e-9, prc_all)
        log_returns = np.diff(np.log(safe_prc_all), axis=1)  # nInst x (nt-1)

        # Calculate weekly returns (e.g., every 5 trading days)
        if n_t >= 6:
            weekly_returns = log_returns[:, 4::5]  # Take every 5th daily return as a proxy for weekly
        else:
            weekly_returns = np.array([])
            print_to_file("Not enough data for weekly return analysis.")

        print_to_file("\n--- 2. Distributions of Returns ---")
        print_to_file("Daily Log Returns Descriptive Statistics:")
        for i in range(n_inst):  # Iterate through all instruments
            print_to_file(
                f"  Instrument {i + 1}: Mean={np.mean(log_returns[i]):.6f}, Std={np.std(log_returns[i]):.6f}, "
                f"Skew={pd.Series(log_returns[i]).skew():.4f}, Kurt={pd.Series(log_returns[i]).kurt():.4f}")
        print_to_file(f"Overall Daily Log Returns Mean={np.mean(log_returns):.6f}, Std={np.std(log_returns):.6f}")

        if weekly_returns.size > 0:
            print_to_file("\nWeekly Log Returns Descriptive Statistics (proxy):")
            for i in range(n_inst):  # Iterate through all instruments
                if weekly_returns[i].size > 0:
                    print_to_file(
                        f"  Instrument {i + 1}: Mean={np.mean(weekly_returns[i]):.6f}, Std={np.std(weekly_returns[i]):.6f}")

        # Optional: Plot histograms for a few instruments
        # plt.figure(figsize=(12, 6))
        # for i in range(min(n_inst, 3)): # Still limit plotting for practical reasons
        #     sns.histplot(log_returns[i], bins=50, kde=True, stat="density", label=f'Inst {i+1}')
        # plt.title('Distribution of Daily Log Returns')
        # plt.legend()
        # plt.show()

        # --- 3. Correlation Matrix Analysis ---
        print_to_file("\n--- 3. Correlation Matrix Analysis ---")
        if log_returns.shape[1] > 1:
            overall_corr_matrix = np.corrcoef(log_returns)
            print_to_file("Overall Correlation Matrix (showing full matrix if small, else top 5x5 sub-matrix):")
            if n_inst <= 10:  # Print full matrix if n_inst is small
                print_to_file(overall_corr_matrix)
            else:
                print_to_file(overall_corr_matrix[:5, :5])  # Otherwise, just a sub-matrix for brevity

            # Rolling correlation (example for first two instruments)
            window_size_corr = 60  # ~3 months
            if n_t - 1 >= window_size_corr:
                if n_inst >= 2:
                    print_to_file(
                        f"\nRolling Correlation (Window={window_size_corr} days, e.g., between Inst 1 and Inst 2):")
                    rolling_corr = pd.Series(log_returns[0, :]).rolling(window=window_size_corr).corr(
                        pd.Series(log_returns[1, :]))
                    print_to_file(rolling_corr.dropna().tail())  # Show last few rolling correlations
                    # Optional: Plot rolling correlations
                    # plt.figure(figsize=(10, 5))
                    # rolling_corr.plot(title=f'Rolling Correlation (Inst 1 vs Inst 2, Window={window_size_corr})')
                    # plt.show()
                else:
                    print_to_file("Need at least 2 instruments for rolling correlation example.")
            else:
                print_to_file(
                    f"Not enough data for rolling correlation (needs at least {window_size_corr} days of returns).")
        else:
            print_to_file("Not enough data for correlation analysis.")

        # --- 4. Volatility Clustering (Autocorrelation of squared returns) ---
        print_to_file("\n--- 4. Volatility Clustering (Autocorrelation of Squared Log Returns) ---")
        if log_returns.shape[1] > 1:
            squared_returns = log_returns ** 2
            for i in range(n_inst):  # Iterate through all instruments
                # Using pandas autocorrelation function
                autocorr_val = pd.Series(squared_returns[i]).autocorr(lag=1)
                print_to_file(f"  Instrument {i + 1} Autocorrelation of Squared Returns (Lag 1): {autocorr_val:.4f}")
                if autocorr_val > 0.1:  # Heuristic for potential clustering
                    print_to_file(f"    (Suggests some volatility persistence for Instrument {i + 1})")
        else:
            print_to_file("Not enough data for volatility clustering analysis.")

        # --- 5. Stationarity Tests (Augmented Dickey-Fuller Test) ---
        print_to_file("\n--- 5. Stationarity Tests (Augmented Dickey-Fuller - ADF Test) ---")
        print_to_file("Testing Daily Log Returns for Stationarity:")
        for i in range(n_inst):  # Iterate through all instruments
            if len(log_returns[i]) > 10:  # ADF test needs sufficient observations
                adf_result = adfuller(log_returns[i])
                p_value = adf_result[1]
                print_to_file(f"  Instrument {i + 1} ADF P-value: {p_value:.4f} "
                              f"({'Stationary' if p_value < 0.05 else 'Non-stationary'})")
            else:
                print_to_file(f"  Instrument {i + 1}: Not enough data for ADF test.")

        # Test stationarity of PCA residuals (if PCA is feasible)
        if n_t - 1 > n_inst and n_inst > 0:  # Need more observations than features for PCA and at least one instrument
            try:
                # Fit PCA on the transposed log returns (columns are features/instruments)
                pca = PCA(n_components=min(1, n_inst))  # Using 1 component or min(1, n_inst) if n_inst is 0
                if n_inst > 0:  # Only fit if there are instruments
                    pca.fit(log_returns.T)
                    # Reconstruct and get residuals
                    reconstructed_returns = pca.inverse_transform(pca.transform(log_returns.T)).T
                    pca_residuals = log_returns - reconstructed_returns

                    print_to_file("\nTesting PCA Residuals for Stationarity (ADF Test for all instruments):")
                    for i in range(n_inst):  # Iterate through all instruments
                        if len(pca_residuals[i]) > 10:
                            adf_result_res = adfuller(pca_residuals[i])
                            p_value_res = adf_result_res[1]
                            print_to_file(f"  Residual {i + 1} ADF P-value: {p_value_res:.4f} "
                                          f"({'Stationary' if p_value_res < 0.05 else 'Non-stationary'})")
                        else:
                            print_to_file(f"  Residual {i + 1}: Not enough data for ADF test.")
                else:
                    print_to_file("No instruments to perform PCA residual stationarity test.")

            except Exception as e:
                print_to_file(f"Could not perform PCA residual stationarity test: {e}")
        else:
            print_to_file(
                "Not enough data points relative to instruments for meaningful PCA and residual stationarity test (or no instruments).")

        # --- 6. Outlier Detection and Handling ---
        print_to_file("\n--- 6. Outlier Detection (Z-score based) ---")
        z_score_threshold_outlier = 3.0  # Common threshold for outliers (e.g., 3 standard deviations)
        outliers_found = False
        for i in range(n_inst):  # Iterate through all instruments
            mean_ret = np.mean(log_returns[i])
            std_ret = np.std(log_returns[i])
            if std_ret > 1e-9:  # Avoid division by zero
                z_scores = np.abs((log_returns[i] - mean_ret) / std_ret)
                outlier_indices = np.where(z_scores > z_score_threshold_outlier)[0]
                if len(outlier_indices) > 0:
                    print_to_file(
                        f"  Instrument {i + 1}: Found {len(outlier_indices)} outliers (Z-score > {z_score_threshold_outlier}) at indices: {outlier_indices[:5]}...")  # Show first 5
                    outliers_found = True
        if not outliers_found:
            print_to_file(
                f"  No significant outliers found (Z-score > {z_score_threshold_outlier}) in daily log returns.")

        print_to_file("\n--- Outlier Handling Notes ---")
        print_to_file("  - The current code includes `np.where(prcSoFar == 0, 1e-9, prcSoFar)` for robustness.")
        print_to_file("  - For more explicit outlier handling, consider:")
        print_to_file(
            "    - **Winsorization:** Capping extreme values to a certain percentile (e.g., 1st and 99th percentile).")
        print_to_file("    - **Trimming:** Removing extreme observations.")
        print_to_file(
            "    - **Robust Statistics:** Using metrics like median and MAD (Median Absolute Deviation) instead of mean and standard deviation, as they are less sensitive to outliers.")
        print_to_file(
            "    - **Imputation:** Replacing outliers with more reasonable values (e.g., interpolated values).")
        print_to_file("  - The choice of method depends on the nature of outliers and downstream model sensitivity.")

    print(f"\nAnalysis complete. Output saved to '{output_file}'.", file=sys.stderr)
    print("Remember to replace 'prices.txt' with your actual data and uncomment plotting lines if desired.",
          file=sys.stderr)


if __name__ == "__main__":
    output_file_name = "data_analysis_output.txt"
    # Example usage:
    # Ensure you have a 'prices.txt' file in the same directory or provide the correct path.
    # For demonstration, creating a dummy prices.txt if not found.
    try:
        with open("prices.txt", "x") as f:
            f.write("100 101 102 103 104 105 106 107 108 109 110 111\n")
            f.write("50 51 52 51 50 49 50 51 52 53 54 55\n")
            f.write("200 202 204 200 198 196 194 192 190 188 186 184\n")
            f.write("70 71 72 73 74 75 76 77 78 79 80 81\n")
            f.write("150 149 148 147 146 145 144 143 142 141 140 139\n")
            # Add more instruments to demonstrate "all instruments"
            f.write("10 11 12 13 14 15 16 17 18 19 20 21\n")
            f.write("30 31 32 33 34 35 36 37 38 39 40 41\n")
            f.write("250 255 260 250 240 230 220 210 200 190 180 170\n")
            f.write("90 91 92 93 94 95 96 97 98 99 100 101\n")
            f.write("110 109 108 107 106 105 104 103 102 101 100 99\n")
        print("Created a dummy 'prices.txt' for demonstration.", file=sys.stderr)
    except FileExistsError:
        pass  # File already exists, proceed.

    price_data = load_prices("prices.txt")

    if price_data is not None:
        analyze_data_characteristics(price_data, output_file=output_file_name)
import numpy as np
from statsmodels.tsa.stattools import coint

# --- STRATEGY PARAMETERS ---
Z_SCORE_THRESHOLD = 1.25
REBALANCE_PERIOD = 3
MEAN_RETURN_LOOKBACK = 20
CORR_THRESHOLD = 0.7
COINT_P_VALUE = 0.05
RISK_TARGET = 5000

# --- GLOBAL STATE ---
last_rebalance = -REBALANCE_PERIOD
previous_positions = np.zeros(50)


def compute_z_scores(prices, lookback):
    """Volatility-normalized z-scores with debug prints."""
    print(f"\n[Z-SCORES] Computing with {lookback}-day lookback...")
    returns = np.diff(np.log(prices), axis=1)

    if returns.shape[1] < lookback:
        print(f"[WARNING] Insufficient data ({returns.shape[1]} days < {lookback} lookback)")
        return np.zeros(prices.shape[0])

    mean = np.mean(returns[:, -lookback:], axis=1)
    vol = np.std(returns[:, -lookback:], axis=1) + 1e-6
    z_scores = mean / vol

    extreme_count = np.sum(np.abs(z_scores) > Z_SCORE_THRESHOLD)
    print(f"[Z-SCORES] Found {extreme_count} instruments with |z| > {Z_SCORE_THRESHOLD}")
    print(
        f"[Z-SCORES] Top/Bottom 3: {np.argsort(z_scores)[:3]} (short candidates), {np.argsort(z_scores)[-3:]} (long candidates)")

    return z_scores


def find_pairs_fast(prices):
    """Pair finding with progress tracking."""
    print("\n[PAIRS] Starting pair search...")
    n = prices.shape[0]
    pairs = []
    tested_pairs = 0

    print(f"[PAIRS] Computing correlation matrix (threshold={CORR_THRESHOLD})...")
    corr = np.corrcoef(prices)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) < CORR_THRESHOLD:
                continue

            tested_pairs += 1
            _, p_val, _ = coint(prices[i], prices[j])

            if p_val < COINT_P_VALUE:
                pairs.append((i, j))
                print(f"[PAIRS] Found pair ({i},{j}) p={p_val:.4f} corr={corr[i, j]:.2f}")

    print(f"[PAIRS] Tested {tested_pairs} correlated pairs | Found {len(pairs)} cointegrated pairs")
    return pairs


def log_position_changes(old_pos, new_pos, prices):
    """Detailed position change logging."""
    changed = np.where(new_pos != old_pos)[0]
    if len(changed) == 0:
        print("\n[POSITIONS] No changes from previous day")
        return

    print("\n[POSITIONS] Changes:")
    total_dvol = 0
    for i in changed:
        delta = new_pos[i] - old_pos[i]
        action = "BUY" if delta > 0 else "SELL"
        dvol = abs(delta) * prices[i, -1]
        total_dvol += dvol

        print(f"  Inst {i:2d}: {action:4s} {abs(delta):4d} shares "
              f"(Now: {new_pos[i]:5d}, Price: ${prices[i, -1]:6.2f}) "
              f"| ${dvol:7,.0f}")

    print(f"[POSITIONS] Total $ Volume: ${total_dvol:,.0f}")


def getMyPosition(prices):
    global last_rebalance, previous_positions

    nInst, nDays = prices.shape
    print(f"\n{'=' * 40}")
    print(f"DAY {nDays} | Last rebalance: {last_rebalance} (Period: {REBALANCE_PERIOD} days)")

    # --- Rebalance Check ---
    if nDays - last_rebalance < REBALANCE_PERIOD:
        print("[SKIP] Not a rebalance day")
        return previous_positions

    if nDays < 30:
        print("[SKIP] Insufficient data (<30 days)")
        return previous_positions

    # --- Market Regime ---
    market_vol = np.std(np.diff(np.log(prices[-1, -20:])))
    print(f"[VOLATILITY] 20-day market vol: {market_vol:.4f}")
    if market_vol > 0.02:
        print("[SKIP] High volatility regime")
        return previous_positions

    # --- Signal Generation ---
    z_scores = compute_z_scores(prices, MEAN_RETURN_LOOKBACK)
    pairs = find_pairs_fast(prices.T)

    # --- Position Calculation ---
    new_positions = np.zeros(nInst)

    # Mean Reversion Core
    for i in range(nInst):
        z = z_scores[i]
        if abs(z) > Z_SCORE_THRESHOLD:
            size = int((RISK_TARGET / (market_vol + 1e-6)) * np.tanh(z / Z_SCORE_THRESHOLD) / prices[i, -1])
            new_positions[i] = size if z < 0 else -size

    # Pairs Overlay
    for i, j in pairs:
        spread = prices[i] - prices[j]
        z_spread = (spread[-1] - np.mean(spread)) / (np.std(spread) + 1e-6)

        if abs(z_spread) > Z_SCORE_THRESHOLD:
            size = int((RISK_TARGET / (market_vol + 1e-6)) / prices[i, -1])
            new_positions[i] -= np.sign(z_spread) * size
            new_positions[j] += np.sign(z_spread) * size
            print(f"[PAIRS TRADE] {i}-{j} | z={z_spread:.2f} | Size: {size}")

    # --- Finalize ---
    last_rebalance = nDays
    log_position_changes(previous_positions, new_positions, prices)
    previous_positions = new_positions

    return new_positions.astype(int)
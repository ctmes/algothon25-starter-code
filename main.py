import numpy as np

# Global parameters
MOMENTUM_LOOKBACK = 10
VOL_LOOKBACK = 10
MA_SHORT = 5
MA_LONG = 20
RSI_LOOKBACK = 14
MOMENTUM_THRESHOLD = 0.06
VOL_THRESHOLD = 0.015
RSI_LONG = 60
RSI_SHORT = 40
TRAILING_STOP = 0.05
MAX_POSITION_VALUE = 10000

def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst, dtype=int)

    # Persistent state to track positions and trailing stops
    if not hasattr(getMyPosition, 'state'):
        getMyPosition.state = {}  # {inst: [position, entry_day, peak_price, trough_price]}

    current_day = nt - 1

    def calculate_rsi(prices, period):
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        return 100 - (100 / (1 + rs))

    for inst in range(nInst):
        prices = prcSoFar[inst, :]

        # Skip if insufficient data
        if nt < max(MOMENTUM_LOOKBACK, VOL_LOOKBACK, MA_LONG, RSI_LOOKBACK) + 1:
            continue

        # Calculate momentum
        if prices[-MOMENTUM_LOOKBACK - 1] != 0:
            momentum = (prices[-1] - prices[-MOMENTUM_LOOKBACK - 1]) / prices[-MOMENTUM_LOOKBACK - 1]
        else:
            momentum = 0

        # Calculate volatility
        if np.all(prices[-VOL_LOOKBACK - 1:-1] != 0):
            returns = np.diff(prices[-VOL_LOOKBACK - 1:]) / prices[-VOL_LOOKBACK - 1:-1]
            volatility = np.std(returns) if len(returns) > 0 else np.inf
        else:
            volatility = np.inf

        # Calculate moving averages
        if nt >= MA_LONG + 1:
            ma_short_val = np.mean(prices[-MA_SHORT:])
            ma_long_val = np.mean(prices[-MA_LONG:])
            ma_crossover = ma_short_val > ma_long_val
            ma_bearish = ma_short_val < ma_long_val
        else:
            ma_crossover = ma_bearish = False

        # Calculate RSI
        rsi = calculate_rsi(prices, RSI_LOOKBACK)

        # Update trailing stop for existing positions
        if inst in getMyPosition.state:
            pos, entry_day, peak_price, trough_price = getMyPosition.state[inst]
            if pos > 0:  # Long position
                peak_price = max(peak_price, prices[-1])
                if prices[-1] <= peak_price * (1 - TRAILING_STOP):
                    del getMyPosition.state[inst]
                    continue
            elif pos < 0:  # Short position
                trough_price = min(trough_price, prices[-1])
                if prices[-1] >= trough_price * (1 + TRAILING_STOP):
                    del getMyPosition.state[inst]
                    continue
            positions[inst] = pos
            getMyPosition.state[inst] = [pos, entry_day, peak_price, trough_price]
            continue

        # Trading logic
        if volatility < VOL_THRESHOLD:
            if (momentum > MOMENTUM_THRESHOLD and ma_crossover and rsi > RSI_LONG):
                # Long position
                shares = int(MAX_POSITION_VALUE / prices[-1])
                if shares * prices[-1] <= MAX_POSITION_VALUE:
                    positions[inst] = shares
                    getMyPosition.state[inst] = [shares, current_day, prices[-1], prices[-1]]
            elif (momentum < -MOMENTUM_THRESHOLD and ma_bearish and rsi < RSI_SHORT):
                # Short position
                shares = -int(MAX_POSITION_VALUE / prices[-1])
                if abs(shares * prices[-1]) <= MAX_POSITION_VALUE:
                    positions[inst] = shares
                    getMyPosition.state[inst] = [shares, current_day, prices[-1], prices[-1]]

    return positions
import numpy as np


def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst, dtype=int)

    # Parameters
    momentum_lookback = 10  # Lookback period for momentum
    vol_lookback = 10  # Lookback period for volatility
    ma_short = 5  # Short-term moving average
    ma_long = 20  # Long-term moving average
    rsi_lookback = 14  # RSI lookback period
    momentum_threshold = 0.06  # 6% price change threshold
    vol_threshold = 0.015  # 1.5% volatility threshold
    rsi_long = 60  # RSI threshold for long
    rsi_short = 40  # RSI threshold for short
    trailing_stop = 0.05  # 5% trailing stop
    max_position_value = 10000  # Position limit per instrument

    # Persistent state to track positions and trailing stops
    if not hasattr(getMyPosition, 'state'):
        getMyPosition.state = {}  # {inst: [position, entry_day, peak_price, trough_price]}

    # Current day
    current_day = nt - 1

    # Calculate RSI
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

    # Process each instrument
    for inst in range(nInst):
        prices = prcSoFar[inst, :]

        # Skip if insufficient data
        if nt < max(momentum_lookback, vol_lookback, ma_long, rsi_lookback) + 1:
            continue

        # Calculate momentum
        if prices[-momentum_lookback - 1] != 0:
            momentum = (prices[-1] - prices[-momentum_lookback - 1]) / prices[-momentum_lookback - 1]
        else:
            momentum = 0

        # Calculate volatility
        if np.all(prices[-vol_lookback - 1:-1] != 0):
            returns = np.diff(prices[-vol_lookback - 1:]) / prices[-vol_lookback - 1:-1]
            volatility = np.std(returns) if len(returns) > 0 else np.inf
        else:
            volatility = np.inf

        # Calculate moving averages
        if nt >= ma_long + 1:
            ma_short_val = np.mean(prices[-ma_short:])
            ma_long_val = np.mean(prices[-ma_long:])
            ma_crossover = ma_short_val > ma_long_val
            ma_bearish = ma_short_val < ma_long_val
        else:
            ma_crossover = ma_bearish = False

        # Calculate RSI
        rsi = calculate_rsi(prices, rsi_lookback)

        # Update trailing stop for existing positions
        if inst in getMyPosition.state:
            pos, entry_day, peak_price, trough_price = getMyPosition.state[inst]
            if pos > 0:  # Long position
                peak_price = max(peak_price, prices[-1])
                if prices[-1] <= peak_price * (1 - trailing_stop):
                    del getMyPosition.state[inst]
                    continue
            elif pos < 0:  # Short position
                trough_price = min(trough_price, prices[-1])
                if prices[-1] >= trough_price * (1 + trailing_stop):
                    del getMyPosition.state[inst]
                    continue
            positions[inst] = pos
            getMyPosition.state[inst] = [pos, entry_day, peak_price, trough_price]
            continue

        # Trading logic
        if volatility < vol_threshold:
            if (momentum > momentum_threshold and ma_crossover and rsi > rsi_long):
                # Long position
                shares = int(max_position_value / prices[-1])
                if shares * prices[-1] <= max_position_value:
                    positions[inst] = shares
                    getMyPosition.state[inst] = [shares, current_day, prices[-1], prices[-1]]
            elif (momentum < -momentum_threshold and ma_bearish and rsi < rsi_short):
                # Short position
                shares = -int(max_position_value / prices[-1])
                if abs(shares * prices[-1]) <= max_position_value:
                    positions[inst] = shares
                    getMyPosition.state[inst] = [shares, current_day, prices[-1], prices[-1]]

    return positions
import numpy as np

# Global parameters - simplified and debugged
MOMENTUM_LOOKBACK = 5
VOL_LOOKBACK = 20
MA_SHORT = 5
MA_LONG = 20
RSI_LOOKBACK = 15
MOMENTUM_THRESHOLD = 0.04  # Lowered to increase trading frequency
VOL_THRESHOLD = 0.014  # Relaxed to allow more trades
RSI_LONG = 60  # Relaxed from 60
RSI_SHORT = 40  # Relaxed from 40
TRAILING_STOP = 0.05
MAX_POSITION_VALUE = 10000
MAX_MOMENTUM_FACTOR = 1.5
MIN_MOMENTUM_FACTOR = 0.5

# Simplified parameters to avoid over-optimization
DEBUG_MODE = True  # Enable debug output


def getMyPosition(prcSoFar):
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst, dtype=int)

    # Simplified persistent state
    if not hasattr(getMyPosition, 'state'):
        getMyPosition.state = {}  # {inst: [position, entry_day, peak_price, trough_price]}
        getMyPosition.debug_info = {'trades': 0, 'current_day': 0}

    current_day = nt - 1
    getMyPosition.debug_info['current_day'] = current_day

    def calculate_rsi(prices, period):
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    for inst in range(nInst):
        prices = prcSoFar[inst, :]
        current_price = prices[-1]

        # Skip if insufficient data or invalid price
        if nt < max(MOMENTUM_LOOKBACK, VOL_LOOKBACK, MA_LONG, RSI_LOOKBACK) + 1:
            continue

        if current_price <= 0:
            continue

        # Calculate momentum with safety checks
        momentum = 0
        if nt > MOMENTUM_LOOKBACK and prices[-MOMENTUM_LOOKBACK - 1] > 0:
            momentum = (current_price - prices[-MOMENTUM_LOOKBACK - 1]) / prices[-MOMENTUM_LOOKBACK - 1]

        # Calculate volatility with safety checks
        volatility = np.inf
        if nt > VOL_LOOKBACK:
            recent_prices = prices[-VOL_LOOKBACK - 1:]
            if np.all(recent_prices[:-1] > 0):
                returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = np.std(returns) if len(returns) > 1 else 0

        # Calculate moving averages
        ma_crossover = ma_bearish = False
        if nt >= MA_LONG:
            ma_short_val = np.mean(prices[-MA_SHORT:])
            ma_long_val = np.mean(prices[-MA_LONG:])
            ma_crossover = ma_short_val > ma_long_val
            ma_bearish = ma_short_val < ma_long_val

        # Calculate RSI
        rsi = calculate_rsi(prices, RSI_LOOKBACK)

        # Handle existing positions with trailing stop
        if inst in getMyPosition.state:
            pos, entry_day, peak_price, trough_price = getMyPosition.state[inst]

            if pos > 0:  # Long position
                peak_price = max(peak_price, current_price)
                if current_price <= peak_price * (1 - TRAILING_STOP):
                    # Exit long position
                    del getMyPosition.state[inst]
                    if DEBUG_MODE:
                        print(f"Day {current_day}: Exited long position in instrument {inst}")
                    continue
            elif pos < 0:  # Short position
                trough_price = min(trough_price, current_price)
                if current_price >= trough_price * (1 + TRAILING_STOP):
                    # Exit short position
                    del getMyPosition.state[inst]
                    if DEBUG_MODE:
                        print(f"Day {current_day}: Exited short position in instrument {inst}")
                    continue

            positions[inst] = pos
            getMyPosition.state[inst] = [pos, entry_day, peak_price, trough_price]
            continue

        # Calculate dynamic position sizing
        momentum_factor = min(MAX_MOMENTUM_FACTOR, max(MIN_MOMENTUM_FACTOR,
                                                       abs(momentum) / MOMENTUM_THRESHOLD if MOMENTUM_THRESHOLD > 0 else 1))

        # Trading logic - simplified and more permissive
        if volatility < VOL_THRESHOLD:
            # Long signal
            if (momentum > MOMENTUM_THRESHOLD and ma_crossover and rsi > RSI_LONG):
                adjusted_position_value = MAX_POSITION_VALUE * momentum_factor
                shares = int(adjusted_position_value / current_price)

                if shares > 0:
                    positions[inst] = shares
                    getMyPosition.state[inst] = [shares, current_day, current_price, current_price]
                    getMyPosition.debug_info['trades'] += 1
                    if DEBUG_MODE:
                        print(f"Day {current_day}: Long {shares} shares of instrument {inst} at {current_price:.2f}")

            # Short signal
            elif (momentum < -MOMENTUM_THRESHOLD and ma_bearish and rsi < RSI_SHORT):
                adjusted_position_value = MAX_POSITION_VALUE * momentum_factor
                shares = -int(adjusted_position_value / current_price)

                if shares < 0:
                    positions[inst] = shares
                    getMyPosition.state[inst] = [shares, current_day, current_price, current_price]
                    getMyPosition.debug_info['trades'] += 1
                    if DEBUG_MODE:
                        print(
                            f"Day {current_day}: Short {abs(shares)} shares of instrument {inst} at {current_price:.2f}")

    # Debug output
    if DEBUG_MODE and current_day % 50 == 0:
        active_positions = len(getMyPosition.state)
        total_trades = getMyPosition.debug_info['trades']
        print(f"Day {current_day}: Active positions: {active_positions}, Total trades: {total_trades}")

    return positions
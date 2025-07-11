# Rigan Multi-Factor Trading Strategy Documentation
## Executive Summary

This systematic trading strategy employs a multi-factor approach combining momentum, volatility, moving averages, and RSI to generate high-probability trade signals across multiple instruments. The strategy incorporates dynamic position sizing, adaptive risk management, and sophisticated exit mechanisms to deliver consistent alpha while maintaining strict risk controls.

**Key Features:**
- Multi-signal confluence filtering for enhanced signal quality
- Dynamic position sizing based on signal strength
- Adaptive trailing stop mechanism for capital preservation
- Systematic approach suitable for institutional deployment

## Strategy Architecture

### Core Signal Generation Framework

The strategy employs four primary technical factors, each serving a distinct purpose in the signal generation process:

#### 1. Momentum Factor
- **Calculation:** Percentage price change over `MOMENTUM_LOOKBACK` periods
- **Signal Role:** Primary trend identification and directional bias
- **Rationale:** Captures persistent price movements and exploits continuation patterns inherent in financial markets
- **Implementation:** Normalized momentum scores enable cross-asset comparison and dynamic position sizing

#### 2. Volatility Filter
- **Calculation:** Rolling standard deviation of returns over `VOL_LOOKBACK` periods
- **Signal Role:** Market regime identification and risk-adjusted entry timing
- **Rationale:** Filters out high-uncertainty periods to reduce whipsaw losses and improve risk-adjusted returns
- **Implementation:** Volatility threshold prevents entry during unstable market conditions

#### 3. Moving Average Confirmation
- **Calculation:** Short-term (`MA_SHORT`) vs. long-term (`MA_LONG`) moving average crossover
- **Signal Role:** Trend confirmation and false signal reduction
- **Rationale:** Provides additional trend validation, reducing momentum-only false positives
- **Implementation:** Dual-timeframe approach ensures alignment between short-term signals and medium-term trends

#### 4. RSI Momentum Oscillator
- **Calculation:** Relative Strength Index over `RSI_LOOKBACK` periods
- **Signal Role:** Overbought/oversold condition assessment
- **Rationale:** Prevents entry at extreme price levels, improving entry timing and reducing reversal risk
- **Implementation:** Asymmetric thresholds optimize performance for both long and short positions

### Position Sizing & Risk Management

#### Dynamic Position Sizing
- **Methodology:** Position size scales with momentum strength within defined bounds
- **Parameters:** `MIN_MOMENTUM_FACTOR` to `MAX_MOMENTUM_FACTOR` range
- **Advantage:** Allocates more capital to higher-conviction signals while maintaining risk discipline
- **Risk Control:** Maximum position value cap (`MAX_POSITION_VALUE`) prevents concentration risk

#### Trailing Stop Exit System
- **Mechanism:** Adaptive stop-loss that follows favorable price movements
- **Implementation:** Tracks peak/trough prices since entry with `TRAILING_STOP` threshold
- **Benefits:** Preserves profits during favorable moves while limiting downside exposure
- **Optimization:** Balances profit protection with noise tolerance

## Entry Logic

### Long Position Criteria
All conditions must be satisfied simultaneously:
1. Momentum > positive threshold (trend strength)
2. Short MA > Long MA (trend confirmation)
3. RSI > long threshold (avoiding oversold conditions)
4. Volatility < threshold (stable market conditions)

### Short Position Criteria
All conditions must be satisfied simultaneously:
1. Momentum < negative threshold (downtrend strength)
2. Short MA < Long MA (bearish trend confirmation)
3. RSI < short threshold (avoiding overbought conditions)
4. Volatility < threshold (stable market conditions)

## State Management & Execution

### Position Tracking
- **State Variables:** Position size, entry timestamp, peak/trough tracking
- **Multi-Asset Support:** Independent state management per instrument
- **Persistence:** Maintains position history for accurate exit calculations

### Execution Workflow
1. **Signal Calculation:** Compute all technical indicators
2. **Entry Assessment:** Evaluate confluence conditions
3. **Position Sizing:** Calculate optimal position size based on signal strength
4. **Risk Validation:** Apply position caps and volatility filters
5. **State Update:** Maintain position tracking for exit management
6. **Exit Monitoring:** Continuous trailing stop evaluation

## Risk Management Framework

### Multi-Layer Risk Control
- **Volatility Filtering:** Prevents entry during unstable market regimes
- **Position Concentration Limits:** Caps maximum exposure per instrument
- **Trailing Stop Protection:** Adaptive exit mechanism for loss limitation
- **Signal Confluence:** Reduces false positives through multi-factor validation

### Performance Characteristics
- **Objective:** Consistent alpha generation with controlled drawdowns
- **Approach:** Systematic, rules-based execution eliminates emotional bias
- **Scalability:** Framework supports multiple instruments and timeframes
- **Robustness:** Diversified signal sources reduce single-point-of-failure risk
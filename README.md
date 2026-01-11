# NQ Main Algo

This repository contains the NQ Main Algorithm strategies.

## Included Strategies

### Triple NQ Variant
File: `Triple_NQ_Variant.pine`

A composite strategy combining:
- **Trend NQ**: Trend following with momentum, ADX, and VWAP filters.
- **Long ORB**: Opening Range Breakout (Long).
- **Simple Short**: RSI/VIX based shorting logic.
- **Short ORB**: Opening Range Breakout (Short).

Features:
- **Tiered Scaling**: Adjusts position size based on Volatility (ATR), VWAP Proximity, or Momentum.
- **Risk Management**: Adaptive drawdown control (Caution, Defense, Survival tiers).
- **Limit Entry**: Optional limit order entry with offset tolerance.

### ES Trend
File: `ES Trend.txt`
Existing strategy for ES Trend following.

## Structure
- `Triple_NQ_Variant.pine`: Main Pine Script strategy.
- `StrategyPipeline/`: Python backtesting framework and pipeline tools.
- `data/`: Market data (ignored by git).

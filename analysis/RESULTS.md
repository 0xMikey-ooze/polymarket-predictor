# Threshold Backtest Results

Resolved ETH trades analyzed: 305

## Summary Table

| Strategy | Bets | WR | PnL | Profit Factor | Sharpe | Max Loss Streak |
|---|---|---|---|---|---|---|
| Fixed 0.52 | 305 | 58.7% | $351 | 1.28 | 2.15 | 5 |
| Fixed 0.55 | 301 | 58.8% | $353 | 1.28 | 2.18 | 5 |
| Fixed 0.58 | 250 | 56.0% | $160 | 1.15 | 1.07 | 5 |
| Fixed 0.60 | 203 | 54.7% | $79 | 1.09 | 0.59 | 4 |
| Fixed 0.62 | 165 | 53.3% | $22 | 1.03 | 0.18 | 5 |
| Fixed 0.65 | 103 | 52.4% | $-4 | 0.99 | -0.04 | 5 |
| Adaptive (log) | 166 | 57.8% | $164 | 1.23 | 1.36 | 4 |

## Serial Correlation
- Lag-1 autocorr: **0.0075** (near 0 = independent)
- P(win|prev win): **57.9%**
- P(win|prev loss): **57.1%**

## Recommendation
**Use fixed 0.55 threshold.** Adaptive reacts to noise (lag-1 autocorr=0.008), adding no Sharpe benefit.

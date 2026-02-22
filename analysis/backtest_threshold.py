#!/usr/bin/env python3
"""
Threshold Backtest: Fixed vs Adaptive
Compares WR, Sharpe, streaks, and serial correlation for ETH paper trades.
"""
import json, os, math
import numpy as np
import pandas as pd
from collections import defaultdict

TRADES_FILE = '/home/node/.openclaw/workspace/polymarket-predictor/paper_trades.jsonl'

# ── 1. LOAD & CLEAN ──────────────────────────────────────────
raw = []
with open(TRADES_FILE) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        t = json.loads(line)
        if t['coin'] == 'ETH' and t.get('correct') is not None:
            raw.append(t)

df = pd.DataFrame(raw)
df['candle_time'] = pd.to_datetime(df['candle_time'])
# Deduplicate by candle_time — keep last (final resolution)
df = df.sort_values('candle_time').drop_duplicates(subset='candle_time', keep='last').reset_index(drop=True)

df['outcome']    = df['correct'].astype(int)          # 1=win 0=loss
df['confidence'] = df['confidence'].astype(float)
df['thresh_used']= df['threshold_used'].astype(float)

print(f"Resolved ETH trades (after dedup): {len(df)}")
print(f"Date range: {df['candle_time'].min()} → {df['candle_time'].max()}\n")

# ── 2. METRICS HELPER ────────────────────────────────────────
def metrics(mask, outcomes, label):
    """mask: boolean array of bets placed; outcomes: 0/1 array"""
    bets = outcomes[mask]
    total = len(df)
    n_bets = len(bets)
    n_wins = bets.sum()
    n_skips = total - n_bets

    if n_bets == 0:
        return dict(label=label, bets=0, wins=0, wr=0, skips=total,
                    max_loss_streak=0, profit_factor=0, sharpe=0, pnl=0)

    wr = n_wins / n_bets

    # Max losing streak
    max_streak = cur = 0
    for o in bets:
        cur = cur + 1 if o == 0 else 0
        max_streak = max(max_streak, cur)

    # PnL per bet: +9 win, -10 loss (Polymarket ~90% payout)
    pnl_series = np.where(bets == 1, 9.0, -10.0)
    total_pnl = pnl_series.sum()
    gross_win  = pnl_series[pnl_series > 0].sum()
    gross_loss = abs(pnl_series[pnl_series < 0].sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe (per-bet)
    mean_pnl = pnl_series.mean()
    std_pnl  = pnl_series.std()
    sharpe   = (mean_pnl / std_pnl) * math.sqrt(n_bets) if std_pnl > 0 else 0

    return dict(label=label, bets=n_bets, wins=int(n_wins), wr=wr,
                skips=n_skips, max_loss_streak=max_streak,
                profit_factor=profit_factor, sharpe=sharpe, pnl=total_pnl)

# ── 3. FIXED THRESHOLDS ─────────────────────────────────────
outcomes = df['outcome'].values
confidence = df['confidence'].values
thresholds = [0.52, 0.55, 0.58, 0.60, 0.62, 0.65]

results = []
for t in thresholds:
    mask = confidence >= t
    results.append(metrics(mask, outcomes, f'Fixed {t:.2f}'))

# ── 4. ADAPTIVE (REPLAY FROM LOG) ───────────────────────────
adaptive_mask = confidence >= df['thresh_used'].values
results.append(metrics(adaptive_mask, outcomes, 'Adaptive (log)'))

# ── 5. PRINT TABLE ──────────────────────────────────────────
print("=" * 80)
print(f"{'Strategy':<20} {'Bets':>5} {'WR':>7} {'PnL':>8} {'PF':>6} {'Sharpe':>7} {'MaxLS':>6} {'Skips':>6}")
print("-" * 80)
for r in results:
    pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else " ∞"
    print(f"{r['label']:<20} {r['bets']:>5} {r['wr']:>7.1%} {r['pnl']:>8.1f} {pf_str:>6} {r['sharpe']:>7.2f} {r['max_loss_streak']:>6} {r['skips']:>6}")
print("=" * 80)

# ── 6. SERIAL CORRELATION ───────────────────────────────────
print("\n── Serial Correlation (adaptive bets only) ──")
bet_outcomes = outcomes[adaptive_mask]
n = len(bet_outcomes)

lag1_autocorr = np.corrcoef(bet_outcomes[:-1], bet_outcomes[1:])[0,1] if n > 1 else 0

win_after_win  = ((bet_outcomes[:-1]==1) & (bet_outcomes[1:]==1)).sum()
loss_after_win = ((bet_outcomes[:-1]==1) & (bet_outcomes[1:]==0)).sum()
win_after_loss = ((bet_outcomes[:-1]==0) & (bet_outcomes[1:]==1)).sum()
loss_after_loss= ((bet_outcomes[:-1]==0) & (bet_outcomes[1:]==0)).sum()

total_wins_prev  = win_after_win + loss_after_win
total_losses_prev= win_after_loss + loss_after_loss

p_win_given_win  = win_after_win  / total_wins_prev  if total_wins_prev  > 0 else 0
p_win_given_loss = win_after_loss / total_losses_prev if total_losses_prev > 0 else 0

print(f"  Lag-1 autocorrelation: {lag1_autocorr:.4f}  (0 = pure noise, ±1 = perfectly predictable)")
print(f"  P(win | prev win):  {p_win_given_win:.1%}  (n={total_wins_prev})")
print(f"  P(win | prev loss): {p_win_given_loss:.1%}  (n={total_losses_prev})")
print(f"  Overall WR:         {bet_outcomes.mean():.1%}")

if abs(lag1_autocorr) < 0.10:
    print("\n  ✅ Outcomes are essentially INDEPENDENT — no serial dependency.")
    print("     The adaptive threshold is reacting to noise, not signal.")
elif lag1_autocorr > 0.10:
    print("\n  📈 Mild MOMENTUM detected — wins beget wins. Hot-hand effect.")
else:
    print("\n  📉 Mild MEAN-REVERSION detected — wins beget losses. Fade effect.")

# ── 7. BEST FIXED vs ADAPTIVE ──────────────────────────────
print("\n── Best Fixed Threshold ──")
fixed_results = [r for r in results if r['label'].startswith('Fixed')]
best = max(fixed_results, key=lambda r: r['sharpe'])
adaptive = [r for r in results if 'Adaptive' in r['label']][0]

print(f"  Best fixed:  {best['label']}  WR={best['wr']:.1%}  Sharpe={best['sharpe']:.2f}  PnL=${best['pnl']:.0f}")
print(f"  Adaptive:    WR={adaptive['wr']:.1%}  Sharpe={adaptive['sharpe']:.2f}  PnL=${adaptive['pnl']:.0f}")

if best['sharpe'] > adaptive['sharpe']:
    diff = best['sharpe'] - adaptive['sharpe']
    print(f"\n  🏆 FIXED WINS by {diff:.2f} Sharpe points.")
    print(f"     Recommendation: replace adaptive with fixed {best['label'].split()[1]} threshold.")
elif adaptive['sharpe'] > best['sharpe']:
    diff = adaptive['sharpe'] - best['sharpe']
    print(f"\n  🏆 ADAPTIVE WINS by {diff:.2f} Sharpe points — keep it.")
else:
    print(f"\n  ≈ TIED — adaptive adds complexity for no gain. Prefer fixed {best['label']}.")

# ── 8. WRITE RESULTS ────────────────────────────────────────
out_path = '/home/node/.openclaw/workspace/polymarket-predictor/analysis/RESULTS.md'
with open(out_path, 'w') as f:
    f.write("# Threshold Backtest Results\n\n")
    f.write(f"Resolved ETH trades analyzed: {len(df)}\n\n")
    f.write("## Summary Table\n\n")
    f.write(f"| Strategy | Bets | WR | PnL | Profit Factor | Sharpe | Max Loss Streak |\n")
    f.write(f"|---|---|---|---|---|---|---|\n")
    for r in results:
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "∞"
        f.write(f"| {r['label']} | {r['bets']} | {r['wr']:.1%} | ${r['pnl']:.0f} | {pf} | {r['sharpe']:.2f} | {r['max_loss_streak']} |\n")
    f.write(f"\n## Serial Correlation\n")
    f.write(f"- Lag-1 autocorr: **{lag1_autocorr:.4f}** (near 0 = independent)\n")
    f.write(f"- P(win|prev win): **{p_win_given_win:.1%}**\n")
    f.write(f"- P(win|prev loss): **{p_win_given_loss:.1%}**\n\n")
    f.write(f"## Recommendation\n")
    if best['sharpe'] > adaptive['sharpe']:
        f.write(f"**Use fixed {best['label'].split()[1]} threshold.** Adaptive reacts to noise (lag-1 autocorr={lag1_autocorr:.3f}), adding no Sharpe benefit.\n")
    else:
        f.write(f"**Keep adaptive threshold.** It outperforms all fixed alternatives by Sharpe.\n")

print(f"\n✅ Results written to {out_path}")

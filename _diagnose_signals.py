"""Diagnosis: prediction distributions and raw signal win rates."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

from config.loader import load_config
config = load_config()

from backtest.runner import _load_bars, _load_predictions
bars = _load_bars(config)
preds = _load_predictions(config, bars)

# ── 1. Prediction distributions across all tickers ──
all_top = []
all_bot = []
all_vol = []

for ticker in sorted(preds.keys()):
    df = preds[ticker]
    if 'tp_top' in df.columns:
        all_top.append(df['tp_top'].dropna())
    if 'tp_bottom' in df.columns:
        all_bot.append(df['tp_bottom'].dropna())
    if 'vol_prob_flat' in df.columns:
        all_vol.append(df['vol_prob_flat'].dropna())

top = pd.concat(all_top) if all_top else pd.Series(dtype=float)
bot = pd.concat(all_bot) if all_bot else pd.Series(dtype=float)
vol = pd.concat(all_vol) if all_vol else pd.Series(dtype=float)

print("=" * 60)
print("PREDICTION DISTRIBUTIONS (all tickers pooled)")
print("=" * 60)
for name, s in [("tp_top", top), ("tp_bottom", bot), ("vol_prob_flat", vol)]:
    print(f"\n{name}: n={len(s)}")
    print(f"  min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}, std={s.std():.4f}")
    print(f"  median={s.median():.4f}, p25={s.quantile(0.25):.4f}, p75={s.quantile(0.75):.4f}")
    for thresh in [0.50, 0.55, 0.60, 0.65]:
        n_above = (s > thresh).sum()
        pct = n_above / len(s) * 100
        print(f"  > {thresh}: {n_above} ({pct:.2f}%)")

# ── 2. Raw signal win rates (NO vol/CNN gate, just threshold) ──
# For each bar where tp_top > thresh or tp_bottom > thresh,
# check if price moved in the predicted direction by TP% before hitting SL%
print("\n\n" + "=" * 60)
print("RAW SIGNAL WIN RATES (forward-looking price check)")
print("=" * 60)

model_cfg = config.get("model", {})
tp_pct = model_cfg.get("meta_tp", 0.005)
sl_pct = model_cfg.get("meta_sl", 0.003)
max_bars = int(model_cfg.get("meta_mb", 48))

print(f"\nUsing backtest TP={tp_pct*100:.1f}%, SL={sl_pct*100:.1f}%, max_bars={max_bars}")
print(f"(from config['model'] meta_tp/meta_sl/meta_mb)\n")

for thresh in [0.50, 0.55]:
    print(f"\n--- Threshold = {thresh} ---")

    for direction, pred_col, label in [("SHORT", "tp_top", "top"), ("BUY", "tp_bottom", "bottom")]:
        total = 0
        wins = 0
        losses = 0
        timeouts = 0
        pnls = []

        for ticker in sorted(bars.keys()):
            bar_df = bars[ticker]
            pred_df = preds.get(ticker)
            if pred_df is None or pred_col not in pred_df.columns:
                continue

            # Align predictions to bars
            aligned = pred_df[pred_col].reindex(bar_df.index, method='ffill')

            closes = bar_df['Close'].values
            highs = bar_df['High'].values
            lows = bar_df['Low'].values
            probs = aligned.values

            for i in range(len(closes) - max_bars):
                if np.isnan(probs[i]) or probs[i] <= thresh:
                    continue

                entry = closes[i]
                total += 1
                hit_tp = False
                hit_sl = False

                for j in range(1, min(max_bars + 1, len(closes) - i)):
                    if direction == "SHORT":
                        # TP: price drops by tp_pct
                        if lows[i + j] <= entry * (1 - tp_pct):
                            hit_tp = True
                            break
                        # SL: price rises by sl_pct
                        if highs[i + j] >= entry * (1 + sl_pct):
                            hit_sl = True
                            break
                    else:  # BUY
                        # TP: price rises by tp_pct
                        if highs[i + j] >= entry * (1 + tp_pct):
                            hit_tp = True
                            break
                        # SL: price drops by sl_pct
                        if lows[i + j] <= entry * (1 - sl_pct):
                            hit_sl = True
                            break

                if hit_tp:
                    wins += 1
                    pnls.append(tp_pct)
                elif hit_sl:
                    losses += 1
                    pnls.append(-sl_pct)
                else:
                    timeouts += 1
                    # Timeout: use close at max_bars
                    exit_price = closes[min(i + max_bars, len(closes) - 1)]
                    if direction == "SHORT":
                        pnl = (entry - exit_price) / entry
                    else:
                        pnl = (exit_price - entry) / entry
                    pnls.append(pnl)

        avg_pnl = np.mean(pnls) if pnls else 0
        wr = wins / total * 100 if total > 0 else 0
        print(f"  {direction:5s} (tp_{label} > {thresh}): "
              f"signals={total}, wins={wins}, losses={losses}, timeout={timeouts}, "
              f"WR={wr:.1f}%, avg_pnl={avg_pnl*100:+.3f}%")
